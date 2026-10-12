"""Forecast for one moment: P(new alert within `horizon` hours) for every oblast.

    python -m src.predict                                      # volunteer, 1h, latest moment
    python -m src.predict --at "2026-10-09 15:00"              # UTC, simulated past moment
    python -m src.predict --source official --horizon 6 --train-from 2025-11-01 --at "2026-08-20 18:00"

Real-time simulation: the RAW rows are first cut to what was known at --at
(started before it, ends after it unknown), and only then merged into oblast
episodes, exactly as a live system would do. For a past --at the actual outcome
is shown next to the forecast.
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.features import last_observed, load_raw, make_dataset, merge_episodes, truncate_at
from src.train_eval import DEFAULT_TRAIN_FROM, FILES, run_tag

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(FILES), default="volunteer")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--train-from", default=DEFAULT_TRAIN_FROM)
    ap.add_argument("--at", help="forecast time, UTC, e.g. '2026-10-05 18:00'")
    args = ap.parse_args()
    h = args.horizon

    tag = run_tag(args.source, h, pd.Timestamp(args.train_from, tz="UTC"), None)
    bundle = joblib.load(ROOT / "models" / f"model_{tag}.joblib")
    raw = load_raw(ROOT / "data" / "raw" / FILES[args.source], args.source)
    full = merge_episodes(raw)
    end = last_observed(full)
    t = (pd.Timestamp(args.at, tz="UTC") if args.at else end).floor("h")
    if t <= pd.Timestamp(bundle["bounds"]["val"][1]):
        print("warning: this moment is inside the model's training period; the forecast is in-sample.")

    known = merge_episodes(truncate_at(raw, t))
    row = make_dataset(known, t, t, with_target=False, horizon=h)
    row["p_alert"] = bundle["model"].predict_proba(row[bundle["features"]])[:, 1]

    if t + pd.Timedelta(hours=h) <= end:  # outcome already known -> show it
        fut = full[(full["start"] >= t) & (full["start"] < t + pd.Timedelta(hours=h))]
        row["actual"] = row["region"].isin(fut["region"]).astype(int)

    local = t.tz_convert("Europe/Kyiv")
    print(f"[{tag}] forecast at {t:%Y-%m-%d %H:%M} UTC ({local:%H:%M} Kyiv): new alert within {h}h\n")
    cols = ["region", "active_now", "p_alert"] + (["actual"] if "actual" in row else [])
    print(row[cols].sort_values("p_alert", ascending=False).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
