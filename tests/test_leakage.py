"""No-leakage and correctness tests.

1. test_features_use_only_past: for several forecast times t, features computed
   from the FULL dataset must equal features computed the way a live system
   would compute them at t: cut the raw rows (truncate_at: started before t,
   ends after t unknown), then merge into oblast episodes. Any peek into the
   future (in features OR in the episode merging) makes the two differ.
2. test_target_is_future_start: y(t) = 1 exactly when an episode starts in [t, t+h).
3. test_merge_episodes_toy: hand-checked example of interval merging.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURES, last_observed, load_raw, make_dataset, merge_episodes, truncate_at

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FILES = {"volunteer": "volunteer_data_en.csv", "official": "official_data_en.csv"}


@pytest.fixture(scope="module", params=list(FILES))
def raw(request):
    path = RAW / FILES[request.param]
    if not path.exists():
        pytest.skip("run `python -m src.fetch_data` first")
    return load_raw(path, request.param)


@pytest.mark.parametrize("horizon", [1, 6])
def test_features_use_only_past(raw, horizon):
    full_ep = merge_episodes(raw)
    end = last_observed(full_ep).floor("h")
    # 12 moments: fresh (near the end of data) and old, different hours of the day, so that
    # some fall inside running alerts and some in quiet periods. A leak only shows up when
    # there is something to peek at, so one moment is not enough.
    offsets_h = (7, 9, 13, 30, 47, 24 * 3 + 5, 24 * 5 + 13, 24 * 8 + 20, 24 * 12 + 2, 24 * 15 + 17,
                 24 * 20 + 2, 24 * 22 + 11)
    times = [end - pd.Timedelta(hours=h) for h in offsets_h]
    full = make_dataset(full_ep, times[-1] - pd.Timedelta(days=3), end - pd.Timedelta(hours=7), horizon=horizon)
    # the test is only meaningful if some moments fall inside running alerts
    assert full[full["t"].isin(times)]["active_now"].sum() > 0, "no running alert at tested moments"
    for t in times:
        a = full[full["t"] == t].set_index("region")[FEATURES]
        live = merge_episodes(truncate_at(raw, t))
        b = make_dataset(live, t, t, with_target=False, horizon=horizon).set_index("region")[FEATURES]
        diff = (a - b).abs().max()
        assert np.allclose(a.to_numpy(), b.to_numpy()), f"leak at {t}: {diff[diff > 0].to_dict()}"


@pytest.mark.parametrize("horizon", [1, 6])
def test_target_is_future_start(raw, horizon):
    ep = merge_episodes(raw)
    t = last_observed(ep).floor("h") - pd.Timedelta(days=2)
    row = make_dataset(ep, t, t, horizon=horizon).set_index("region")["y"]
    fut = ep[(ep["start"] >= t) & (ep["start"] < t + pd.Timedelta(hours=horizon))]
    expected = row.index.isin(fut["region"]).astype(int)
    assert (row.to_numpy() == expected).all()


def test_merge_episodes_toy():
    T = lambda h: pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=h)
    raw = pd.DataFrame({
        "region": ["Kyiv City"] * 4 + ["Lvivska oblast"],
        "start": [T(0), T(1), T(5), T(6), T(0)],
        "end": [T(2), T(3), T(7), pd.NaT, T(1)],
    })
    ep = merge_episodes(raw).sort_values(["region", "start"]).reset_index(drop=True)
    # Kyiv: [0,2]+[1,3] -> [0,3];  [5,7]+[6,still running] -> [5, NaT];  Lviv unchanged
    assert list(ep["region"]) == ["Kyiv City", "Kyiv City", "Lvivska oblast"]
    assert list(ep["start"]) == [T(0), T(5), T(0)]
    assert ep["end"].iloc[0] == T(3) and pd.isna(ep["end"].iloc[1]) and ep["end"].iloc[2] == T(1)
