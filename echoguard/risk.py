"""Turn a forecast into hourly risk levels, alert windows and change alarms."""
import numpy as np

from . import config as C

LEVELS = ["LOW", "MEDIUM", "HIGH"]


def thresholds(history=None, recall=1.0, precision=1.0):
    """(high, medium) on the scale of the series being forecast.

    The config thresholds are in real calls/h. A detector reports
    recall x real calls + false alarms, and the false-alarm rate equals
    (1 - precision) x mean detections, estimated from `history`. With
    recall = precision = 1 the config values are returned unchanged."""
    fp = 0.0 if precision >= 1 else (1 - precision) * float(np.nanmean(np.asarray(history, float)))
    return (round(recall * C.HIGH_THRESHOLD + fp, 1), round(recall * C.MEDIUM_THRESHOLD + fp, 1))   # 0.1 = dashboard slider step


def risk_levels(upper, high, medium):
    """Risk from the UPPER bound of the 80 % interval (a missed whale costs more
    than an extra alert)."""
    u = np.asarray(upper, float)
    return np.where(u >= high, "HIGH", np.where(u >= medium, "MEDIUM", "LOW"))


def alert_windows(steps, levels, level="HIGH", max_gap=1):
    """Merge `level` hours into alert windows (first_step, last_step, length).

    Windows separated by at most `max_gap` lower-risk hours are joined, so a
    crew gets a few clear alerts instead of many fragments.
    """
    hits = [st for st, lv in zip(steps, levels) if lv == level]
    out = []
    for st in hits:
        if out and st - out[-1][1] <= max_gap + 1:
            out[-1][1] = st
        else:
            out.append([st, st])
    return [(a, b, b - a + 1) for a, b in out]


def cusum(actual, forecast, sigma, k=C.CUSUM_K, h=C.CUSUM_H):
    """One-sided upper CUSUM on standardised forecast errors.
    Returns the statistic and the index of the first alarm (or None)."""
    z = (np.asarray(actual, float) - np.asarray(forecast, float)) / max(float(sigma), 1e-9)
    s, stats, alarm = 0.0, [], None
    for i, v in enumerate(z):
        if not np.isnan(v):                      # buoy offline: no evidence, the sum stays put
            s = max(0.0, s + v - k)
        stats.append(s)
        if alarm is None and s > h:
            alarm = i
    return np.array(stats), alarm


def advice(level):
    return {"LOW": "Normal operations. Keep monitoring the buoy feed.",
            "MEDIUM": "Post lookouts and be ready to slow to 10 knots or less.",
            "HIGH": "Slow to 10 knots or less or reroute around the area."}[level]
