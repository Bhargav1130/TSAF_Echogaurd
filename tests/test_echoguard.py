"""Quick checks for each stage.  Run:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echoguard import analysis, audio, detect, forecast, risk, series  # noqa: E402
from echoguard import config as C  # noqa: E402


def test_windowing_and_filter():
    x = np.random.default_rng(0).normal(size=10 * C.SAMPLE_RATE)
    W = audio.window_signal(x)
    assert W.shape == (9, 4000)                      # 2-s windows, 1-s hop
    t = np.arange(4000) / C.SAMPLE_RATE
    low, inband = np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 150 * t)
    out = audio.bandpass(np.stack([low, inband]))
    assert out[0].std() < 0.1 * out[1].std()          # 10 Hz removed, 150 Hz kept


def test_detector_learns_upcalls():
    X, y = audio.simulate_clips(n=600, seed=1)
    rows, _ = detect.evaluate_detectors(X, y, [detect.MiniRocketDetector()])
    assert rows[0]["pr_auc"] > 0.6                    # well above the 0.3 base rate


def test_hourly_series_keeps_outage_missing():
    times = pd.date_range("2016-03-01", periods=6 * 3600, freq="s", tz="UTC")[::60]
    probs = np.zeros(len(times)); probs[:5] = 0.9      # 5 calls in the first hour
    h = series.detections_to_hourly(times, probs, 0.5, offline=[("2016-03-01 03:00", "2016-03-01 04:30")])
    assert h.calls.iloc[0] == 5 and h.present.iloc[0] == 1
    assert h.missing.iloc[3] == 1 and np.isnan(h.calls.iloc[3])   # outage is NaN, not 0


def test_gap_handling():
    h = series.simulate_season(days=60)
    gaps = series.gap_report(h.calls)
    assert [g[2] for g in gaps] == ["interpolated", "kept missing"]
    assert not series.fill_for_model(h.calls).isna().any()


def test_analysis_finds_daily_cycle():
    h = series.simulate_season()
    y = series.fill_for_model(h.calls)
    ac = analysis.autocorrelation(y)
    assert ac["acf"][24] > ac["acf"][12]
    _, strength = analysis.stl_decompose(y)
    assert strength["seasonal_strength"] > 0.2


def test_walk_forward_has_no_leakage_and_beats_naive():
    h = series.simulate_season()
    y = series.fill_for_model(h.calls).to_numpy()
    origins = forecast.fold_origins(len(y))
    assert all(o + C.HORIZON <= len(y) for o in origins)
    summary, _, _ = forecast.walk_forward(y, models=["Naive", "Seasonal naive", "Holt-Winters"], folds=2)
    mae = dict(zip(summary.model, summary.mae))
    assert mae["Holt-Winters"] < mae["Naive"]


def test_order_search_picks_lowest_aic():
    y = series.fill_for_model(series.simulate_season(days=20).calls).to_numpy()
    table = forecast.order_search(y, p=(0, 1), q=(0, 1), P=(1,), Q=(0,))
    assert len(table) == 4 and table.aic.is_monotonic_increasing
    best = forecast.best_order(table)
    assert best["order"] == tuple(table[table.converged].iloc[0].order) and best["seasonal_order"][3] == 24
    r = forecast.sarima(y, 24, **best)                 # chosen orders plug straight into the model
    assert len(r["mean"]) == 24 and (r["hi"] >= r["mean"]).all()


def test_sarimax_backtest_never_sees_future_sst():
    h = series.simulate_season(days=12)
    y = series.fill_for_model(h.calls).to_numpy()
    X = h.sst_c.to_numpy()
    s = len(y) - 72
    X_changed = X.copy(); X_changed[s:] += 10.0                   # tamper with SST after the origin
    o = {"order": (1, 0, 0), "seasonal_order": (1, 0, 0, 24)}
    a = forecast.model_runner("SARIMAX (SST)", y, X, kw=o)(s)["mean"]
    b = forecast.model_runner("SARIMAX (SST)", y, X_changed, kw=o)(s)["mean"]
    assert np.allclose(a, b)


def test_offline_hours_are_never_scored():
    y = np.full(1000, 3.0)
    obs = y.copy(); obs[928:940] = np.nan                          # 12 h outage inside the test block
    filled = y.copy(); filled[928:940] = 50.0                      # a wild interpolation must not matter
    s1, _, _ = forecast.walk_forward(y, models=["Naive"], folds=1, actual=obs)
    s2, _, _ = forecast.walk_forward(filled, models=["Naive"], folds=1, actual=obs)
    assert s1.mae[0] == 0 and np.isfinite(s2.mae[0])
    assert forecast.coverage(obs[928:1000], np.full(72, 2.0), np.full(72, 4.0)) == 1.0
    _, alarm = risk.cusum([np.nan] * 5 + [9.0], [0.0] * 6, sigma=1)
    assert alarm == 5                                             # NaN hours do not reset or add to the sum


def test_empirical_interval_calibrates_from_past_errors():
    rng = np.random.default_rng(5)
    y = 3 + rng.normal(0, 1, 2000)
    calls = []
    def run(s):                                                    # a forecaster that is biased low by 1
        calls.append(s)
        return {"mean": np.full(72, 2.0), "lo": None, "hi": None, "resid": None}
    r = forecast.empirical_interval(run, y, 1500)
    assert sorted(calls[:-1]) == [1500 - 72 * j for j in range(5, 0, -1)]   # only past origins
    assert r["lo"][0] < 3 < r["hi"][0] and r["hi"][0] - 2.0 > 2.0     # interval shifted up to cover the bias
    cov = np.mean((y[1500:1572] >= r["lo"]) & (y[1500:1572] <= r["hi"]))
    assert 0.65 < cov < 0.95 and abs(r["err_sd"] - 1) < 0.15


def test_apply_detector_matches_recall_and_precision():
    h = series.simulate_season()
    same = series.apply_detector(h, 1.0, 1.0)
    assert np.array_equal(same.calls.fillna(-1), h.calls.fillna(-1))
    d = series.apply_detector(h, 0.6, 0.8)
    assert d.calls.isna().equals(h.calls.isna())                  # outages stay missing
    hits = 0.6 * np.nansum(h.calls)
    assert abs(np.nansum(d.calls) - hits / 0.8) / (hits / 0.8) < 0.05   # detections = hits + false alarms
    assert (d.calls_true.fillna(-1) == h.calls.fillna(-1)).all()


def test_residual_diagnostics():
    rng = np.random.default_rng(3)
    _, white = forecast.residual_diagnostics(rng.normal(size=600))
    assert white["white_noise"] and white["normal"] and white["n"] == 600 - 25
    t = np.arange(600)                                    # leftover daily cycle + skewed noise
    _, bad = forecast.residual_diagnostics(np.sin(2 * np.pi * t / 24) + rng.exponential(1, 600))
    assert not bad["white_noise"] and not bad["normal"] and 24 in bad["acf_outside_band"]


def test_dclde_selection_table_to_hourly(tmp_path):
    import fetch_dclde
    f = tmp_path / "20190301.selections.txt"
    rows = ["Selection\tChannel\tBegin Time (s)\tEnd Time (s)\tBegin File\tFile Offset (s)\tScore\tTags"]
    for i, (ch, off, sc) in enumerate([(1, 10.0, 0.9), (1, 3599.0, 0.8), (1, 3601.0, 0.76), (2, 20.0, 0.99), (1, 30.0, 0.5)]):
        rows.append(f"{i + 1}\t{ch}\t{off}\t{off + 2}\tX_002K_M05_multi_20190301_000000Z.flac\t{off}\t{sc}\tnarw")
    f.write_text("\n".join(rows))
    det = fetch_dclde.read_selections(f)
    assert str(det.time[0]) == "2019-03-01 00:00:10+00:00" and len(det) == 5
    h = fetch_dclde.to_hourly(det, channel=1, min_score=0.75)
    assert h.loc["2019-03-01 00:00"].calls == 2 and h.loc["2019-03-01 01:00"].calls == 1   # other channel + low score dropped


def test_whoi_parser_and_hourly():
    import fetch_whoi
    row = ('<tr><td><a id="{id}">x</a></td><td align="center">0</td><td align="center" style="b">1</td>'
           '<td align="center">{r}</td><td align="center" style="b">0</td><td align="center">5</td>'
           '<td align="center" style="b">{d}</td><td align="center"></td><td align="center"></td></tr>\n')
    rows = [("20261125090120", 1, 900), ("20261125091620", 6, 900), ("20261125093120", 0, 900),
            ("20261125094620", 1, 900),                       # 09 UTC: 8 calls, full hour
            ("20261125100120", 2, 600),                       # 10 UTC: buoy barely listening
            ("20261125110120", 3, 900), ("20261125111620", 0, 900),   # 11 UTC: 3 calls in 30 min
            ("20261125120120", 2, 900), ("20261125121620", 2, 900), ("20261125123120", 0, 900),
            ("20261125124620", 0, 900),                       # 12 UTC: 4 calls, full hour
            ("20261125130120", 5, 900)]                       # 13 UTC: still filling in -> dropped
    tally = fetch_whoi.parse_tally("".join(row.format(id=i, r=r, d=d) for i, r, d in rows))
    assert len(tally) == 12 and str(tally.time.iloc[0]) == "2026-11-25 09:01:20+00:00"
    h = fetch_whoi.to_hourly(tally)
    assert h.calls.iloc[0] == 8 and np.isnan(h.calls.iloc[1])
    assert h.calls.iloc[2] == 6 and h.calls.iloc[3] == 4 and len(h) == 4    # 3 calls in half an hour = 6/h


def test_risk_rules():
    assert risk.thresholds() == (5.0, 3.5)                          # real calls/h
    assert risk.thresholds([1.0] * 10, recall=0.6, precision=0.8) == (3.2, 2.3)   # 0.6x + 0.2x1.0 false alarms
    lv = risk.risk_levels([1, 3.6, 5.2, 6, 2, 5], high=5.0, medium=3.5)
    assert list(lv) == ["LOW", "MEDIUM", "HIGH", "HIGH", "LOW", "HIGH"]
    assert risk.alert_windows(range(1, 7), lv, max_gap=0) == [(3, 4, 2), (6, 6, 1)]
    assert risk.alert_windows(range(1, 7), lv, max_gap=1) == [(3, 6, 4)]
    _, alarm = risk.cusum([0, 0, 5, 5, 5], [0, 0, 1, 1, 1], sigma=1)
    assert alarm == 3                                  # z = 4 per step, k = 0.5 -> 3.5, 7.0 > h = 4
