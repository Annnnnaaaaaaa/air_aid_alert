"""Turn raw alert intervals into a leakage-free (forecast time x region) table.

Task: at forecast time t (a whole hour, UTC), for every oblast, predict
    y = 1  if a NEW oblast-level alert episode starts in [t, t + horizon)  else 0.

Two sources (see load_raw):
  * volunteer (eTryvoga): already one row per oblast alert;
  * official: rows are oblast / raion / hromada alerts. Since mid-2025 almost all
    are raion-level, and 73% of rows overlap another alert in the same oblast.
    merge_episodes() unions overlapping intervals per oblast, so "an alert in the
    oblast" means the same thing before and after the switch to raion alerts.

Leakage rules enforced here (and checked by tests/test_leakage.py):
  * a start is used only if started_at < t;
  * an end is used only if finished_at < t (an alert still running at t is
    "active", its future end time is never looked at);
  * alert-coverage features use only hour buckets that closed before t;
  * label-based features (historical rates) use only labels whose window
    closed at or before t.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HORIZON_H = 6
LOOKBACK_DAYS = 30
CAP_H = 24 * 30  # cap for "hours since" features
EXCLUDED = {"Luhanska oblast"}  # permanent alert since 2022-04-04, not in data

_ADJ = {
    "Cherkaska oblast": ["Kyivska oblast", "Poltavska oblast", "Kirovohradska oblast", "Vinnytska oblast"],
    "Chernihivska oblast": ["Kyivska oblast", "Sumska oblast", "Poltavska oblast"],
    "Chernivetska oblast": ["Ivano-Frankivska oblast", "Ternopilska oblast", "Khmelnytska oblast", "Vinnytska oblast"],
    "Dnipropetrovska oblast": ["Poltavska oblast", "Kharkivska oblast", "Donetska oblast", "Zaporizka oblast",
                               "Khersonska oblast", "Mykolaivska oblast", "Kirovohradska oblast"],
    "Donetska oblast": ["Kharkivska oblast", "Dnipropetrovska oblast", "Zaporizka oblast"],
    "Ivano-Frankivska oblast": ["Lvivska oblast", "Zakarpatska oblast", "Chernivetska oblast", "Ternopilska oblast"],
    "Kharkivska oblast": ["Sumska oblast", "Poltavska oblast", "Dnipropetrovska oblast", "Donetska oblast"],
    "Khersonska oblast": ["Mykolaivska oblast", "Dnipropetrovska oblast", "Zaporizka oblast"],
    "Khmelnytska oblast": ["Rivnenska oblast", "Zhytomyrska oblast", "Vinnytska oblast", "Chernivetska oblast",
                           "Ternopilska oblast"],
    "Kirovohradska oblast": ["Cherkaska oblast", "Poltavska oblast", "Dnipropetrovska oblast", "Mykolaivska oblast",
                             "Odeska oblast", "Vinnytska oblast"],
    "Kyiv City": ["Kyivska oblast"],
    "Kyivska oblast": ["Kyiv City", "Zhytomyrska oblast", "Chernihivska oblast", "Poltavska oblast",
                       "Cherkaska oblast", "Vinnytska oblast"],
    "Lvivska oblast": ["Volynska oblast", "Rivnenska oblast", "Ternopilska oblast", "Ivano-Frankivska oblast",
                       "Zakarpatska oblast"],
    "Mykolaivska oblast": ["Odeska oblast", "Kirovohradska oblast", "Dnipropetrovska oblast", "Khersonska oblast"],
    "Odeska oblast": ["Mykolaivska oblast", "Kirovohradska oblast", "Vinnytska oblast"],
    "Poltavska oblast": ["Kyivska oblast", "Chernihivska oblast", "Sumska oblast", "Kharkivska oblast",
                         "Dnipropetrovska oblast", "Kirovohradska oblast", "Cherkaska oblast"],
    "Rivnenska oblast": ["Volynska oblast", "Lvivska oblast", "Ternopilska oblast", "Khmelnytska oblast",
                         "Zhytomyrska oblast"],
    "Sumska oblast": ["Chernihivska oblast", "Poltavska oblast", "Kharkivska oblast"],
    "Ternopilska oblast": ["Lvivska oblast", "Rivnenska oblast", "Khmelnytska oblast", "Chernivetska oblast",
                           "Ivano-Frankivska oblast"],
    "Vinnytska oblast": ["Zhytomyrska oblast", "Kyivska oblast", "Cherkaska oblast", "Kirovohradska oblast",
                         "Odeska oblast", "Chernivetska oblast", "Khmelnytska oblast"],
    "Volynska oblast": ["Lvivska oblast", "Rivnenska oblast"],
    "Zakarpatska oblast": ["Lvivska oblast", "Ivano-Frankivska oblast"],
    "Zaporizka oblast": ["Dnipropetrovska oblast", "Donetska oblast", "Khersonska oblast"],
    "Zhytomyrska oblast": ["Rivnenska oblast", "Khmelnytska oblast", "Vinnytska oblast", "Kyivska oblast"],
}
REGIONS = sorted(_ADJ)

FEATURES = [
    "region_id", "local_hour", "dow",
    "active_now", "h_since_start", "h_since_end",
    "starts_1h", "starts_3h", "starts_6h", "starts_24h", "starts_7d",
    "cover_6h", "cover_24h",
    "nb_active_now", "nb_starts_1h", "nb_starts_3h",
    "nat_active_now", "nat_starts_1h", "nat_starts_3h", "nat_starts_24h",
    "rate_same_hour_28d", "rate_7d",
]


def adjacency() -> np.ndarray:
    idx = {r: i for i, r in enumerate(REGIONS)}
    a = np.zeros((len(REGIONS), len(REGIONS)), dtype=np.float32)
    for r, nbs in _ADJ.items():
        for n in nbs:
            a[idx[r], idx[n]] = a[idx[n], idx[r]] = 1.0
    return a


def load_raw(path, source: str = "volunteer") -> pd.DataFrame:
    """CSV -> raw intervals: region, start, end (UTC). end may be NaT (= still running)."""
    df = pd.read_csv(path)
    region_col = {"volunteer": "region", "official": "oblast"}[source]
    df = df.rename(columns={region_col: "region", "started_at": "start", "finished_at": "end"})
    df["start"] = pd.to_datetime(df["start"], utc=True, format="ISO8601")
    df["end"] = pd.to_datetime(df["end"], utc=True, format="ISO8601")
    df = df[df["region"].isin(REGIONS) & ~df["region"].isin(EXCLUDED)].copy()
    # In a historical file every alert has ended, so a missing end is a data error, not
    # "still running". Left as NaT, merge_episodes would treat it as +infinity and swallow
    # every later alert in that oblast into one endless episode. Impute start + 30 min
    # (the dataset's own convention for alerts without an end message).
    missing = df["end"].isna()
    if missing.any():
        print(f"warning: {int(missing.sum())} alerts without end in {Path(path).name}; using start + 30 min")
        df.loc[missing, "end"] = df.loc[missing, "start"] + pd.Timedelta(minutes=30)
    df["end"] = df["end"].where(df["end"] >= df["start"], df["start"])
    return df[["region", "start", "end"]].sort_values("start").reset_index(drop=True)


def merge_episodes(raw: pd.DataFrame) -> pd.DataFrame:
    """Union of overlapping intervals per oblast -> one row per oblast-level alert episode.

    An episode starts when the oblast goes from "no alert anywhere" to "alert somewhere".
    NaT end (still running) is treated as +infinity and stays NaT in the result.
    """
    if raw.empty:
        return raw.copy()
    d = raw.sort_values(["region", "start"]).reset_index(drop=True)
    big = pd.Timestamp.max.tz_localize("UTC")
    end = d["end"].fillna(big)
    prev_end = end.groupby(d["region"]).transform(lambda e: e.cummax().shift())
    new_ep = prev_end.isna() | (d["start"] > prev_end)
    ep_id = new_ep.cumsum()
    out = d.assign(end=end).groupby(ep_id).agg(region=("region", "first"), start=("start", "min"), end=("end", "max"))
    out.loc[out["end"] == big, "end"] = pd.NaT
    return out.sort_values("start").reset_index(drop=True)


def load_alerts(path, source: str = "volunteer") -> pd.DataFrame:
    return merge_episodes(load_raw(path, source))


def truncate_at(alerts: pd.DataFrame, t) -> pd.DataFrame:
    """What an observer knows at time t: alerts started before t; ends after t are unknown."""
    t = pd.Timestamp(t)
    out = alerts[alerts["start"] < t].copy()
    out.loc[out["end"] >= t, "end"] = pd.NaT
    return out


def last_observed(alerts: pd.DataFrame) -> pd.Timestamp:
    """Latest moment the dataset is known to cover (last recorded event)."""
    return max(alerts["start"].max(), alerts["end"].max())


def coverage_end(alerts: pd.DataFrame, max_gap_h: int = 24, recent_days: int = 60) -> pd.Timestamp:
    """End of continuous coverage.

    Nationwide there is at least one alert a day, so a gap of more than `max_gap_h`
    hours between consecutive alert starts in the last `recent_days` days means the
    dataset stopped being updated, not that the sky was quiet. In that case data
    after the gap is too short to use and coverage ends where the gap begins
    (official data: 2026-08-29 04:55 UTC). Otherwise it is the last observed moment.
    """
    s = alerts["start"].sort_values().reset_index(drop=True)
    gaps = s.diff()
    end = last_observed(alerts)
    big = gaps > pd.Timedelta(hours=max_gap_h)
    recent = s > end - pd.Timedelta(days=recent_days)
    hit = big & recent
    if hit.any():
        return s[hit[hit].index[0] - 1]
    return end


def _ns(x) -> np.ndarray:
    return pd.DatetimeIndex(x).as_unit("ns").asi8


def make_dataset(alerts: pd.DataFrame, t_from, t_to, with_target: bool = True,
                 horizon: int = HORIZON_H) -> pd.DataFrame:
    """Rows for every whole hour t in [t_from, t_to] and every region.

    Target y(t) = 1 if a new alert starts in [t, t + horizon hours).
    """
    times = pd.date_range(pd.Timestamp(t_from).floor("h"), pd.Timestamp(t_to).floor("h"), freq="h")
    g0 = times[0] - pd.Timedelta(days=LOOKBACK_DAYS)
    g_end = times[-1] + pd.Timedelta(hours=horizon + 1)
    n_h = int((g_end - g0) / pd.Timedelta(hours=1))
    i_t = ((times - g0) / pd.Timedelta(hours=1)).astype(int).to_numpy()  # grid index of each t
    R = len(REGIONS)
    tt = _ns(times)
    one_h = 3_600_000_000_000

    starts = np.zeros((n_h, R), dtype=np.float32)      # starts per hour bucket
    cover = np.zeros((n_h, R), dtype=np.float32)       # share of the hour under alert
    active = np.zeros((len(times), R), dtype=np.float32)
    h_start = np.full((len(times), R), CAP_H, dtype=np.float32)
    h_end = np.full((len(times), R), CAP_H, dtype=np.float32)

    for r, reg in enumerate(REGIONS):
        a = alerts[alerts["region"] == reg]
        if a.empty:
            continue
        s = _ns(a["start"])
        e = a["end"].fillna(g_end).clip(upper=g_end)
        e = _ns(e)
        # hourly start counts
        b = (s - g0.value) // one_h
        ok = (b >= 0) & (b < n_h)
        np.add.at(starts[:, r], b[ok], 1)
        # minute-level coverage -> hourly share
        n_m = n_h * 60
        m0 = np.clip((s - g0.value) // 60_000_000_000, 0, n_m)
        m1 = np.clip(-(-(e - g0.value) // 60_000_000_000), 0, n_m)  # ceil
        diff = np.zeros(n_m + 1, dtype=np.int32)
        np.add.at(diff, m0, 1)
        np.add.at(diff, m1, -1)
        cover[:, r] = (np.cumsum(diff[:-1]) > 0).reshape(n_h, 60).mean(axis=1)
        # point-in-time state at each t
        s_sorted = np.sort(s)
        e_known = np.sort(_ns(a["end"].dropna()))
        n_started = np.searchsorted(s_sorted, tt, side="left")
        n_ended = np.searchsorted(e_known, tt, side="left")
        active[:, r] = (n_started - n_ended) > 0
        last_s = np.where(n_started > 0, s_sorted[np.maximum(n_started - 1, 0)], np.iinfo(np.int64).min)
        last_e = np.where(n_ended > 0, e_known[np.maximum(n_ended - 1, 0)], np.iinfo(np.int64).min)
        h_start[:, r] = np.where(n_started > 0, np.minimum((tt - last_s) / one_h, CAP_H), CAP_H)
        h_end[:, r] = np.where(n_ended > 0, np.minimum((tt - last_e) / one_h, CAP_H), CAP_H)

    C = np.vstack([np.zeros((1, R)), np.cumsum(starts, axis=0, dtype=np.float64)])   # C[i] = starts in buckets < i
    V = np.vstack([np.zeros((1, R)), np.cumsum(cover, axis=0, dtype=np.float64)])

    def past(M, k):  # sum over buckets [i-k, i)
        return M[i_t] - M[np.maximum(i_t - k, 0)]

    # label per grid hour i: any start in [i, i+horizon); known only at i+horizon
    lab_all = ((C[np.minimum(np.arange(n_h) + horizon, n_h)] - C[np.arange(n_h)]) > 0).astype(np.float32)
    L = np.vstack([np.zeros((1, R)), np.cumsum(lab_all, axis=0, dtype=np.float64)])
    # labels with windows closed by t: grid indices j <= i-horizon
    rate_7d = (L[np.maximum(i_t - horizon + 1, 0)] - L[np.maximum(i_t - horizon + 1 - 24 * 7, 0)]) / (24 * 7)
    same_hour = np.mean([lab_all[np.maximum(i_t - 24 * d, 0)] for d in range(1, 29)], axis=0)

    A = adjacency()
    s1, s3 = past(C, 1), past(C, 3)
    cols = {
        "active_now": active,
        "h_since_start": h_start, "h_since_end": h_end,
        "starts_1h": s1, "starts_3h": s3, "starts_6h": past(C, 6),
        "starts_24h": past(C, 24), "starts_7d": past(C, 24 * 7),
        "cover_6h": past(V, 6) / 6, "cover_24h": past(V, 24) / 24,
        "nb_active_now": active @ A, "nb_starts_1h": s1 @ A, "nb_starts_3h": s3 @ A,
        "nat_active_now": np.repeat(active.sum(1, keepdims=True), R, 1),
        "nat_starts_1h": np.repeat(s1.sum(1, keepdims=True), R, 1),
        "nat_starts_3h": np.repeat(s3.sum(1, keepdims=True), R, 1),
        "nat_starts_24h": np.repeat(past(C, 24).sum(1, keepdims=True), R, 1),
        "rate_same_hour_28d": same_hour, "rate_7d": rate_7d,
    }
    out = pd.DataFrame({
        "t": np.repeat(times, R),
        "region": np.tile(REGIONS, len(times)),
        "region_id": np.tile(np.arange(R), len(times)),
    })
    local = times.tz_convert("Europe/Kyiv")
    out["local_hour"] = np.repeat(local.hour, R)
    out["dow"] = np.repeat(local.dayofweek, R)
    for k, v in cols.items():
        out[k] = v.reshape(-1)
    if with_target:
        out["y"] = lab_all[i_t].reshape(-1).astype(int)
    return out
