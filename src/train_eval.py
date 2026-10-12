"""Time-based split, baselines, gradient boosting, evaluation on the latest 14 days.

    python -m src.train_eval --source volunteer --horizon 1
    python -m src.train_eval --source official  --horizon 1 --train-from 2025-11-01
    python -m src.train_eval --source volunteer --horizon 1 --test-end 2026-08-28T22:00
    python -m src.run_all          # the whole experiment grid + results/summary.csv

Timeline (forecast times t, hourly):
  [ train ][gap][ validation 14d ][gap][ test 14d ][gap: labels not yet known]

* Everything is anchored to the end of continuous data coverage (coverage_end),
  or to --test-end if given (used to score both sources on the same 14 days).
* The gap is MAX_HORIZON_H = 6 hours for every horizon, so 1h and 6h models are
  scored on exactly the same hours. It must be >= horizon, because the label of
  the last row of a part looks `horizon` hours ahead.
* Number of trees is chosen on validation; the model is then refit on
  train+validation and scored once on test.

Metrics. The pooled ROC-AUC is dominated by the difference between regions
(Kharkiv almost always, Zakarpattia almost never), which the climatology baseline
already knows. So we also report:
  * Brier skill score vs climatology (1 - Brier_model / Brier_clim; > 0 = better);
  * mean ROC-AUC computed inside each region (can the model tell WHEN?);
  * on how many test days the model beats climatology.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src.features import FEATURES, coverage_end, load_alerts, make_dataset

ROOT = Path(__file__).resolve().parents[1]
FILES = {"volunteer": "volunteer_data_en.csv", "official": "official_data_en.csv"}
DEFAULT_TRAIN_FROM = "2022-04-15"  # official data start 2022-03-15 + 30 days of feature lookback
TEST_DAYS, VAL_DAYS = 14, 14
MAX_HORIZON_H = 6
TREE_GRID = (100, 200, 400)
# Results this good are not plausible from alert history alone (honest runs give
# within-region AUC 0.63-0.73 and Brier skill 0.05-0.14). Above these: suspect a leak.
SUSPICIOUS = {"auc_within": 0.90, "brier_skill": 0.50}
# With shuffled training labels there is nothing to learn: within-region AUC on test
# must stay near 0.5. Above this: information about y reaches the model some other way.
SHUFFLE_MAX_AUC = 0.55


def split_bounds(data_end: pd.Timestamp, train_from: pd.Timestamp) -> dict:
    gap = pd.Timedelta(hours=MAX_HORIZON_H)
    hour = pd.Timedelta(hours=1)
    last_label_t = (data_end - gap).floor("h")  # label window fully observed for any horizon <= 6h
    test_from = last_label_t - pd.Timedelta(days=TEST_DAYS) + hour
    val_to = test_from - gap - hour
    val_from = val_to - pd.Timedelta(days=VAL_DAYS) + hour
    train_to = val_from - gap - hour
    return dict(train=(train_from, train_to), val=(val_from, val_to), test=(test_from, last_label_t))


def within_region_auc(df: pd.DataFrame, p: np.ndarray) -> float:
    d = df.assign(p=p)
    aucs = [roc_auc_score(g["y"], g["p"]) for _, g in d.groupby("region") if g["y"].nunique() == 2]
    return float(np.mean(aucs))


def scores(df: pd.DataFrame, p: np.ndarray, p_clim: np.ndarray) -> dict:
    y = df["y"].to_numpy()
    # 24-hour blocks counted from the start of the test period, NOT calendar days:
    # a 14-day test that starts at 02:00 Kyiv time spans 15 calendar dates, the last
    # one only 2 hours long, and such a stub would count as a full "day" in the score.
    day = ((df["t"] - df["t"].min()) // pd.Timedelta(hours=24)).to_numpy()
    daily_m = pd.Series((p - y) ** 2).groupby(day).mean()   # raw p: clipping would distort the comparison
    daily_c = pd.Series((p_clim - y) ** 2).groupby(day).mean()
    p = np.clip(p, 1e-4, 1 - 1e-4)  # only to keep log_loss finite
    return {
        "brier": brier_score_loss(y, p),
        "brier_skill_vs_clim": 1 - brier_score_loss(y, p) / brier_score_loss(y, p_clim),
        "log_loss": log_loss(y, p),
        "roc_auc_pooled": roc_auc_score(y, p),
        "roc_auc_within_region": within_region_auc(df, p),
        "pr_auc": average_precision_score(y, p),
        "days_better_than_clim": f"{int((daily_m < daily_c).sum())}/{len(daily_m)}",
    }


def baseline_climatology(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """P(y | region, local hour) = share of positives in the last 28 days before the test period."""
    recent = train[train["t"] > train["t"].max() - pd.Timedelta(days=28)]
    rate = recent.groupby(["region", "local_hour"])["y"].mean()
    return test.join(rate.rename("p"), on=["region", "local_hour"])["p"].fillna(recent["y"].mean()).to_numpy()


def baseline_recent_start(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Persistence: P(y | was there a start in this region in the last 6h), estimated on train."""
    key = lambda d: (d["starts_6h"] > 0).astype(int)
    rate = train.groupby(key(train))["y"].mean()
    return key(test).map(rate).to_numpy()


def make_model(max_iter: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=max_iter, max_leaf_nodes=31, min_samples_leaf=200,
        l2_regularization=1.0, categorical_features=[FEATURES.index("region_id")], random_state=0,
        # sklearn's default early_stopping="auto" silently holds out a RANDOM 10% of rows
        # (> 10k samples) and may stop before max_iter. We tune max_iter on a time-based
        # validation set instead, so early stopping must be off.
        early_stopping=False)


def run_tag(source: str, horizon: int, train_from: pd.Timestamp, test_end: str | None) -> str:
    tag = f"{source}_h{horizon}_from{train_from:%Y%m%d}"
    return tag + (f"_test{pd.Timestamp(test_end):%Y%m%d}" if test_end else "")


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(FILES), default="volunteer")
    ap.add_argument("--horizon", type=int, default=1, choices=range(1, MAX_HORIZON_H + 1),
                    help="hours ahead: predict a new alert in [t, t+horizon)")
    ap.add_argument("--train-from", default=DEFAULT_TRAIN_FROM,
                    help="first forecast time used for training, e.g. 2025-11-01")
    ap.add_argument("--test-end", default=None,
                    help="UTC moment treated as the end of data (default: end of continuous coverage)")
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="sanity check: shuffle y in train+val; test AUC must drop to ~0.5")
    args = ap.parse_args(argv)
    h, src = args.horizon, args.source
    train_from = pd.Timestamp(args.train_from, tz="UTC")
    tag = run_tag(src, h, train_from, args.test_end) + ("_SHUFFLED" if args.shuffle_labels else "")

    alerts = load_alerts(ROOT / "data" / "raw" / FILES[src], src)
    data_end = pd.Timestamp(args.test_end, tz="UTC") if args.test_end else coverage_end(alerts)
    b = split_bounds(data_end, train_from)
    print(f"\n=== {tag} ===\ndata end used: {data_end}")
    for k, (lo, hi) in b.items():
        print(f"{k:5s}: {lo} .. {hi}")

    ds = make_dataset(alerts, b["train"][0], b["test"][1], horizon=h)
    part = lambda k: ds[(ds["t"] >= b[k][0]) & (ds["t"] <= b[k][1])].reset_index(drop=True)
    train, val, test = part("train"), part("val"), part("test")
    if args.shuffle_labels:  # break the link between features and answers; test keeps true y
        rng = np.random.default_rng(0)
        train["y"] = rng.permutation(train["y"].to_numpy())
        val["y"] = rng.permutation(val["y"].to_numpy())
    print(f"rows: train={len(train)} val={len(val)} test={len(test)}; "
          f"positive rate train={train['y'].mean():.3f} val={val['y'].mean():.3f} test={test['y'].mean():.3f}")

    best, best_ll, best_p_val = None, np.inf, None
    for n in TREE_GRID:
        mm = make_model(n).fit(train[FEATURES], train["y"])
        p_val = mm.predict_proba(val[FEATURES])[:, 1]
        ll = log_loss(val["y"], p_val)
        print(f"  val log_loss with {n} trees: {ll:.4f}")
        if ll < best_ll:
            best, best_ll, best_p_val = n, ll, p_val
    print("chosen n_trees:", best)
    # Validation scores of the chosen model (trained on train only). Variants are compared
    # on THESE numbers (run_all), so that the test period is never used to choose anything.
    val_scores = scores(val, best_p_val, baseline_climatology(train, val))
    print(f"VAL: log_loss {val_scores['log_loss']:.4f}, brier_skill {val_scores['brier_skill_vs_clim']:.4f}, "
          f"auc_within {val_scores['roc_auc_within_region']:.4f}")

    trval = pd.concat([train, val])
    model = make_model(best).fit(trval[FEATURES], trval["y"])
    p_clim = baseline_climatology(trval, test)
    preds = {
        "climatology (region x hour, 28d)": p_clim,
        "persistence (start in last 6h)": baseline_recent_start(trval, test),
        "gradient boosting": model.predict_proba(test[FEATURES])[:, 1],
    }
    table = pd.DataFrame({k: scores(test, p, p_clim) for k, p in preds.items()}).T
    print(f"\nTEST {b['test'][0]:%Y-%m-%d %H:%M} .. {b['test'][1]:%Y-%m-%d %H:%M} UTC")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(table.to_string())

    out = ROOT / "results" / tag
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "metrics.csv")
    gb = test.assign(p=preds["gradient boosting"], c=p_clim)
    per_region = gb.groupby("region").apply(lambda d: pd.Series({
        "positive_rate": d["y"].mean(),
        "mean_pred": d["p"].mean(),
        "brier_gb": brier_score_loss(d["y"], d["p"]),
        "brier_clim": brier_score_loss(d["y"], d["c"]),
        "auc_gb": roc_auc_score(d["y"], d["p"]) if d["y"].nunique() == 2 else np.nan,
        "auc_clim": roc_auc_score(d["y"], d["c"]) if d["y"].nunique() == 2 else np.nan,
    }), include_groups=False)
    per_region.sort_values("positive_rate").round(4).to_csv(out / "metrics_per_region.csv")

    fig, ax = plt.subplots(figsize=(5, 5))
    for k, p in preds.items():
        fy, fx = calibration_curve(test["y"], np.clip(p, 0, 1), n_bins=10, strategy="quantile")
        ax.plot(fx, fy, marker="o", label=k)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="predicted probability", ylabel="observed frequency", title=f"Calibration: {tag}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "calibration.png", dpi=120)
    plt.close(fig)

    bounds = {k: [str(x) for x in v] for k, v in b.items()}
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "source": src, "horizon": h, "bounds": bounds},
                ROOT / "models" / f"model_{tag}.joblib")
    (out / "split.json").write_text(json.dumps(bounds, indent=2))

    gbs = table.loc["gradient boosting"]
    row = {"run": tag, "source": src, "horizon_h": h, "train_from": f"{train_from:%Y-%m-%d}",
           "test_from": f"{b['test'][0]:%Y-%m-%d %H:%M}", "test_to": f"{b['test'][1]:%Y-%m-%d %H:%M}",
           "pos_rate_train": round(train["y"].mean(), 3), "pos_rate_test": round(test["y"].mean(), 3),
           "mean_pred_test": round(float(preds["gradient boosting"].mean()), 3),
           "n_trees": best,
           "brier_clim": round(table.loc["climatology (region x hour, 28d)", "brier"], 4),
           "brier_gb": round(gbs["brier"], 4),
           "brier_skill": round(gbs["brier_skill_vs_clim"], 4),
           "auc_within_clim": round(table.loc["climatology (region x hour, 28d)", "roc_auc_within_region"], 4),
           "auc_within_gb": round(gbs["roc_auc_within_region"], 4),
           "days_better": gbs["days_better_than_clim"],
           "val_log_loss": round(val_scores["log_loss"], 4),
           "val_brier_skill": round(val_scores["brier_skill_vs_clim"], 4),
           "val_auc_within": round(val_scores["roc_auc_within_region"], 4),
           "shuffled": bool(args.shuffle_labels)}

    # automatic alarms
    if args.shuffle_labels:
        ok = gbs["roc_auc_within_region"] <= SHUFFLE_MAX_AUC
        row["check"] = "ok" if ok else "LEAK SUSPECTED"
        print(f"\nSHUFFLE CHECK: within-region AUC {gbs['roc_auc_within_region']:.3f} "
              f"({'OK, near 0.5' if ok else '!!! ABOVE ' + str(SHUFFLE_MAX_AUC) + ': LEAK SUSPECTED !!!'})")
    else:
        bad = (gbs["roc_auc_within_region"] > SUSPICIOUS["auc_within"]
               or gbs["brier_skill_vs_clim"] > SUSPICIOUS["brier_skill"])
        row["check"] = "SUSPICIOUSLY GOOD" if bad else "ok"
        if bad:
            print("\n" + "!" * 70 + "\n!!! SUSPICIOUSLY GOOD RESULT: check for leakage before trusting it !!!\n"
                  + "!" * 70)
    summ_path = ROOT / "results" / "summary.csv"
    summ = pd.read_csv(summ_path) if summ_path.exists() else pd.DataFrame()
    if not summ.empty:
        summ = summ[summ["run"] != tag]
    pd.concat([summ, pd.DataFrame([row])]).to_csv(summ_path, index=False)
    print(f"\nsaved results/{tag}/, models/model_{tag}.joblib, row in results/summary.csv")
    return row


if __name__ == "__main__":
    main()
