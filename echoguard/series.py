"""Turning detections into an evenly spaced hourly presence series."""
import numpy as np
import pandas as pd

from . import config as C


def detections_to_hourly(times, probs, threshold, start=None, end=None, offline=None):
    """Count windows with prob >= threshold per hour (UTC).

    times    : array-like of window start timestamps
    probs    : detector probability for each window
    offline  : optional list of (start, end) buoy outages -> those hours become NaN
    Returns a DataFrame indexed by hour with columns calls, mean_prob, present, missing.
    """
    t = pd.to_datetime(pd.Series(times), utc=True)
    df = pd.DataFrame({"prob": np.asarray(probs, float)}, index=t)
    df["call"] = (df.prob >= threshold).astype(int)
    start = pd.Timestamp(start, tz="UTC") if start is not None and pd.Timestamp(start).tzinfo is None else (start or t.min())
    end = pd.Timestamp(end, tz="UTC") if end is not None and pd.Timestamp(end).tzinfo is None else (end or t.max())
    idx = pd.date_range(pd.Timestamp(start).floor("h"), pd.Timestamp(end).floor("h"), freq="h")
    hourly = pd.DataFrame(index=idx)
    hourly["calls"] = df.call.resample("h").sum().reindex(idx).fillna(0)
    hourly["mean_prob"] = df.prob.resample("h").mean().reindex(idx)
    for a, b in (offline or []):
        a, b = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
        hourly.loc[(hourly.index >= a.floor("h")) & (hourly.index <= b.floor("h")), ["calls", "mean_prob"]] = np.nan
    return finalize(hourly)


def finalize(hourly):
    """Add present/missing/calendar columns (missing stays NaN, never zero)."""
    h = hourly.copy()
    h["missing"] = h.calls.isna().astype(int)
    h["present"] = (h.calls >= C.PRESENT_MIN_CALLS).astype(float)
    h.loc[h.missing == 1, "present"] = np.nan
    h["hour"] = h.index.hour
    h["dow"] = h.index.day_of_week
    return h


def fill_for_model(calls, max_gap=C.MAX_INTERP_GAP_H):
    """Values used for model fitting. Short gaps (<= max_gap hours) are the
    officially interpolated ones; longer outages are bridged linearly only so
    the models can run, and they stay flagged in the `missing` column."""
    y = calls.astype(float)
    return y.interpolate(limit_direction="both")


def gap_report(calls, max_gap=C.MAX_INTERP_GAP_H):
    """List buoy gaps as (start, length_h, treatment)."""
    isna = calls.isna()
    out = []
    for _, grp in calls[isna].groupby((~isna).cumsum()[isna]):
        n = len(grp)
        out.append((grp.index[0], n, "interpolated" if n <= max_gap else "kept missing"))
    return out


def simulate_season(days=60, seed=C.SEED, start="2016-03-01"):
    """Simulated hourly buoy season with the structure seen in real data:
    night-time calling peak, rise with sea-surface temperature, AR(1)
    persistence, Poisson counts and two buoy outages (3 h and 12 h)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h", tz="UTC")
    d = np.arange(days)
    sst_day = 6.0 + 4.0 * d / days + 0.8 * np.sin(2 * np.pi * d / 11) + rng.normal(0, 0.25, days)
    sst = np.repeat(sst_day, 24)
    hour = idx.hour.values
    log_rate = -1.2 + 0.9 * np.cos(2 * np.pi * (hour - 2) / 24) + 0.45 * (sst - 6.0)
    ar = np.zeros(len(idx))
    for t in range(1, len(idx)):
        ar[t] = 0.85 * ar[t - 1] + rng.normal(0, 0.18)
    calls = rng.poisson(np.exp(log_rate + ar)).astype(float)
    h = pd.DataFrame({"calls": calls, "sst_c": sst.round(2)}, index=idx)
    h["mean_prob"] = np.clip(0.15 + 0.08 * calls + rng.normal(0, 0.03, len(idx)), 0, 1).round(3)
    h.iloc[300:303, [0, 2]] = np.nan          # 3 h outage  -> interpolated
    h.iloc[900:912, [0, 2]] = np.nan          # 12 h outage -> stays missing
    h.index.name = "timestamp"
    return finalize(h)


def apply_detector(h, recall, precision, seed=C.SEED):
    """What a detector with this recall and precision would report each hour.

    Each true call is found with probability `recall`; false alarms arrive at a
    constant hourly rate (they do not depend on whales being there), set so that
    on average a fraction `precision` of all detections are real. Assumes one
    detection per call. Keeps the true counts in `calls_true`.
    """
    rng = np.random.default_rng(seed + 1)
    true = h["calls"].to_numpy()
    ok = ~np.isnan(true)
    fp_rate = np.nanmean(true) * recall * (1 - precision) / precision
    det = np.full(len(h), np.nan)
    det[ok] = rng.binomial(true[ok].astype(int), recall) + rng.poisson(fp_rate, ok.sum())
    out = h.drop(columns=["missing", "present", "hour", "dow"]).assign(calls=det, calls_true=h["calls"])
    return finalize(out)


def load_hourly_csv(path):
    """Load a real hourly table (columns: timestamp, calls[, sst_c]) and tidy it."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated()]
    full = pd.date_range(df.index.min().floor("h"), df.index.max().floor("h"), freq="h")
    df = df.reindex(full)
    df.index.name = "timestamp"
    if "sst_c" in df:
        df["sst_c"] = df["sst_c"].interpolate(limit_direction="both")
    keep = [c for c in ("calls", "sst_c", "mean_prob") if c in df]
    return finalize(df[keep])
