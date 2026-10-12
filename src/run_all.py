"""Run the whole experiment grid, choose variants on VALIDATION, report TEST once.

    python -m src.run_all            # 10 runs + 1 shuffle sanity check (~3-5 min each)
    python -m src.run_all --only-h1  # skip the 6h runs

Grid:
  source      : volunteer (eTryvoga), official
  train from  : 2022-04-15 (all history)  |  2025-11-01 (raion-level regime only)
  horizon     : 1h (main task)  |  6h (weaker variant, kept for comparison)
Each source is tested on its own latest 14 days. The official data stop on
2026-08-29, the volunteer data on 2026-10-12, so these test periods differ.
For an apples-to-apples source comparison the volunteer 1h models are also
scored on the official test period (--test-end = official coverage end).

How the "train from" variant is chosen
  For each (source, horizon, test period) the variant with the lowest VALIDATION
  log-loss wins. Only then its TEST metrics are reported. Choosing on test would
  inflate the reported number (we would pick the variant that got lucky on those
  14 days). Horizon and source are NOT chosen by metrics: they are different tasks
  (different base rates), chosen by task logic (see README).

Sanity check
  One extra run with shuffled training labels (volunteer, 1h, from 2025-11-01).
  Nothing can be learned from shuffled labels, so its test within-region AUC must
  be ~0.5. If it is clearly higher, information about the answer leaks into the
  pipeline outside the features (something tests/test_leakage.py cannot see).
"""
import argparse
from pathlib import Path

import pandas as pd

from src.features import coverage_end, load_alerts
from src.train_eval import FILES, main as run_one

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FROMS = ["2022-04-15", "2025-11-01"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-h1", action="store_true")
    args = ap.parse_args()

    off_end = coverage_end(load_alerts(ROOT / "data" / "raw" / FILES["official"], "official"))
    runs = []
    for h in ([1] if args.only_h1 else [1, 6]):
        for src in ("volunteer", "official"):
            for tf in TRAIN_FROMS:
                runs.append(["--source", src, "--horizon", str(h), "--train-from", tf])
    for tf in TRAIN_FROMS:  # same test period for both sources
        runs.append(["--source", "volunteer", "--horizon", "1", "--train-from", tf,
                     "--test-end", f"{off_end:%Y-%m-%dT%H:%M}"])
    runs.append(["--source", "volunteer", "--horizon", "1", "--train-from", "2025-11-01", "--shuffle-labels"])

    rows, failed = [], []
    for i, argv in enumerate(runs, 1):
        print(f"\n######## run {i}/{len(runs)}: {' '.join(argv)}")
        try:
            rows.append(run_one(argv))
        except Exception as e:  # keep going, report at the end
            failed.append((argv, repr(e)))
            print("FAILED:", repr(e))

    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "results" / "summary_last_run_all.csv", index=False)
    real = table[~table["shuffled"]].copy()

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cols = ["run", "pos_rate_train", "pos_rate_test", "mean_pred_test",
            "val_log_loss", "val_auc_within", "brier_skill", "auc_within_clim", "auc_within_gb",
            "days_better", "check"]
    print("\n================ ALL RUNS (validation and test) ================")
    print(real[cols].to_string(index=False))

    # choose train_from on VALIDATION within each (source, horizon, test period)
    group = ["source", "horizon_h", "test_from"]
    winners = real.loc[real.groupby(group)["val_log_loss"].idxmin()]
    winners = winners.sort_values(["horizon_h", "source", "test_from"])
    print("\n======== CHOSEN ON VALIDATION -> honest TEST numbers ========")
    print(winners[["source", "horizon_h", "test_from", "train_from", "val_log_loss",
                   "brier_skill", "auc_within_clim", "auc_within_gb", "days_better"]].to_string(index=False))
    winners.to_csv(ROOT / "results" / "chosen_on_validation.csv", index=False)

    sh = table[table["shuffled"]]
    if not sh.empty:
        r = sh.iloc[0]
        print(f"\n======== SHUFFLE SANITY CHECK ========\n{r['run']}: test within-region AUC "
              f"{r['auc_within_gb']} (expected ~0.5) -> {r['check']}")
    for argv, err in failed:
        print("FAILED:", " ".join(argv), err)


if __name__ == "__main__":
    main()
