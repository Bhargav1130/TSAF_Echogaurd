"""Run the full EchoGuard pipeline and write everything to outputs/.

    python run_all.py                       # simulated data, full pipeline
    python run_all.py --skip-detector       # forecasting part only (faster)
    python run_all.py --hourly-csv data/my_buoy_hourly.csv
    python run_all.py --kaggle-dir data/kaggle_whale   # real Kaggle clips (needs soundfile)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from echoguard import analysis, audio, detect, forecast, plots, risk, series
from echoguard import config as C


def _num_round(df, n=3):
    out = df.copy()
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(n)
    return out


def step(msg):
    print(f"\n=== {msg} ===", flush=True)


def run_detector(args):
    step("1/5  Up-call detection")
    if args.kaggle_dir:
        X, y = audio.load_kaggle_clips(args.kaggle_dir, max_clips=args.clips)
        source = "Kaggle Whale Detection Challenge"
    else:
        X, y = audio.simulate_clips(n=args.clips)
        source = "simulated clips"
    print(f"{len(X)} clips from {source}, {y.mean():.0%} up-calls")
    t0 = time.time()
    rows, fitted = detect.evaluate_detectors(X, y)
    table = pd.DataFrame(rows)
    table.to_csv(C.OUT_DIR / "detector_metrics.csv", index=False)
    print(table.to_string(index=False))
    # precision-recall curves on the test split
    _, _, te = detect.split_indices(len(X), y)
    curves = {}
    for name, (det, _) in fitted.items():
        prob = det.predict_proba(X[te])
        p, r, _ = precision_recall_curve(y[te], prob)
        curves[name] = (r, p, average_precision_score(y[te], prob))
    plots.detector_pr(curves, C.FIG_DIR / "detector_pr.png")
    print(f"detection stage took {time.time() - t0:.0f}s")
    return {"source": source, "metrics": rows}


def _minirocket_rates(det):
    """(recall, precision) of MiniRocket on the test clips: from this run, or the last saved run."""
    if det:
        rows = det["metrics"]
    elif (C.OUT_DIR / "detector_metrics.csv").exists():
        rows = pd.read_csv(C.OUT_DIR / "detector_metrics.csv").to_dict(orient="records")
    else:
        return None
    row = next((r for r in rows if r["detector"] == "MiniRocket"), None)
    return (float(row["recall"]), float(row["precision"])) if row else None


def run_series(args, det=None):
    step("2/5  Hourly presence series")
    rates = (1.0, 1.0)              # (recall, precision) between real calls and the series
    if args.hourly_csv:
        h = series.load_hourly_csv(args.hourly_csv)
        source = f"real data ({args.hourly_csv})"
        if args.detector_rates:     # the CSV holds detector output: risk thresholds are converted with these
            rates = tuple(args.detector_rates)
            source += f", detector recall {rates[0]:.3f}, precision {rates[1]:.3f}"
    else:
        h = series.simulate_season()
        source = "SIMULATED 60-day season"
        rates = _minirocket_rates(det) or (1.0, 1.0)
        if rates != (1.0, 1.0):   # the forecaster sees what the detector would report, not the true calls
            h = series.apply_detector(h, *rates)
            source += f", counted by MiniRocket (recall {rates[0]:.2f}, precision {rates[1]:.2f})"
            print(f"simulated calls passed through the detector: recall {rates[0]:.2f}, precision {rates[1]:.2f} "
                  f"-> {int(np.nansum(h.calls_true))} true calls, {int(np.nansum(h.calls))} detections")
        else:
            print("no detector results yet: using the true simulated calls (run once without --skip-detector)")
    h.to_csv(C.OUT_DIR / "hourly_series.csv", index_label="timestamp")
    y = series.fill_for_model(h["calls"])
    for start, n, how in series.gap_report(h["calls"]):
        print(f"buoy gap at {start:%Y-%m-%d %H:%M}: {n} h -> {how}")
    print(f"{len(h)} hours ({source}), {int(h.missing.sum())} missing, "
          f"{int(np.nansum(h.present))} hours with whales present")
    return h, y, source, rates


def run_analysis(h, y):
    step("3/5  Time series analysis")
    st = analysis.stationarity(y)
    ac = analysis.autocorrelation(y)
    stl_res, strengths = analysis.stl_decompose(y)
    prof = analysis.daily_profile(y)
    corr = analysis.sst_correlation(y, h["sst_c"]) if "sst_c" in h else None
    print(f"ADF p = {st['adf_p']:.3f}, KPSS p = {st['kpss_p']:.3f}  ->  {st['verdict']}")
    print(f"STL seasonal strength {strengths['seasonal_strength']:.2f}, trend strength {strengths['trend_strength']:.2f}")
    print("ACF at 24/48/72 h: " + ", ".join(f"{ac['acf'][k]:.3f}" for k in (24, 48, 72)))
    print(f"peak hour {prof.idxmax():02d}:00 ({prof.max():.2f} calls/h), low {prof.idxmin():02d}:00 ({prof.min():.2f})")
    if corr is not None:
        print(f"correlation of daily calls with SST: r = {corr:.2f}")
        plots.daily_and_sst(y, h["sst_c"], C.FIG_DIR / "daily_calls_sst.png", corr)
    plots.hourly_profile(prof, C.FIG_DIR / "hourly_profile.png")
    plots.acf_plot(np.array(ac["acf"]), ac["ci"], C.FIG_DIR / "acf.png")
    plots.stl_plot(stl_res, C.FIG_DIR / "stl.png")
    return {"stationarity": st, "stl": strengths, "acf_24_48_72": [ac["acf"][k] for k in (24, 48, 72)],
            "acf_ci": ac["ci"], "daily_profile": prof.round(3).tolist(), "sst_corr": corr}


def run_forecasting(h, y, args, rates=(1.0, 1.0)):
    step("4/5  Forecasting with walk-forward validation")
    exog = h["sst_c"].to_numpy() if "sst_c" in h else None
    models = [m for m in forecast.MODELS if exog is not None or m != "SARIMAX (SST)"]
    first = 44 * 24 if (not args.hourly_csv and len(y) == 60 * 24) else None   # same folds as the report
    folds = C.N_FOLDS
    if args.hourly_csv and len(y) >= 75 * 24:
        # a long real season: test on every 72-h block from day 30 on, so the folds cover arrival,
        # peak and departure instead of only the last 15 days
        first = 30 * 24
        folds = (len(y) - first) // C.HORIZON

    # SARIMA order selection by AIC, on the hours before the first test fold only (no peeking)
    origin0 = first if first is not None else forecast.fold_origins(len(y))[0]
    t0 = time.time()
    aic = forecast.order_search(y.to_numpy()[:origin0])
    orders = forecast.best_order(aic)
    aic.insert(0, "model", [f"SARIMA{tuple(o)}{tuple(s)}" for o, s in zip(aic.order, aic.seasonal_order)])
    aic = aic.drop(columns=["order", "seasonal_order"])
    aic.round(1).to_csv(C.OUT_DIR / "sarima_aic.csv", index=False)
    print(f"AIC grid search: {len(aic)} SARIMA(p,0,q)(P,0,Q)24 models fitted on the first {origin0} h "
          f"({time.time() - t0:.0f}s). Best 5:")
    print(aic.head(5).round(1).to_string(index=False))
    chosen = f"SARIMA{orders['order']}{orders['seasonal_order']}"
    print(f"chosen (lowest AIC, converged): {chosen}")
    model_kw = {"SARIMA": orders, "SARIMAX (SST)": orders}

    t0 = time.time()
    print(f"walk-forward: {folds} folds x {C.HORIZON} h from hour {origin0}")
    summary, per_fold, last = forecast.walk_forward(y.to_numpy(), exog=exog, models=models, first_origin=first, folds=folds,
                                                    model_kw=model_kw, actual=h["calls"].to_numpy())
    summary.round(3).to_csv(C.OUT_DIR / "model_metrics.csv", index=False)
    per_fold.round(3).to_csv(C.OUT_DIR / "model_metrics_per_fold.csv", index=False)
    show = summary.copy()
    show["coverage80"] = show.coverage80.map(lambda v: "-" if pd.isna(v) else f"{v:.0%}")
    print(show.round(2).to_string(index=False))
    print(f"forecasting took {time.time() - t0:.0f}s")
    plots.mase_bars(summary, C.FIG_DIR / "model_mase.png")

    # dashboard model: lowest walk-forward MAE among the fitted TSAF models (baselines excluded)
    best = next(m for m in summary.model if m not in ("Naive", "Seasonal naive"))
    print(f"dashboard model (lowest walk-forward MAE): {best}")
    lf = last[best]
    r, diag = forecast.residual_diagnostics(lf["resid"])
    lb = diag["ljung_box_p_24"]
    fmt = lambda p: "< 0.001" if p < 0.001 else f"= {p:.3f}"
    print(f"Residual diagnostics ({best}, last-fold fit, {diag['n']} h): "
          f"Ljung-Box p {fmt(lb)} (lag 24), {fmt(diag['ljung_box_p_48'])} (lag 48); "
          f"Jarque-Bera p {fmt(diag['jarque_bera_p'])}; skew {diag['skew']:.2f}, "
          f"excess kurtosis {diag['excess_kurtosis']:.2f}")
    print("  -> " + diag["verdict"])
    plots.residual_plot(r, np.array(diag["acf"]), diag["acf_ci"], C.FIG_DIR / "residuals.png",
                        title=f"{best} residual diagnostics (last-fold fit)")
    s = lf["origin"]
    idx = y.index
    t = idx[s:s + C.HORIZON]
    fc = pd.DataFrame({"timestamp": t, "actual": lf["actual"], "forecast": lf["mean"], "lo80": lf["lo"], "hi80": lf["hi"]})
    th_bt = risk.thresholds(y.to_numpy()[:s], *rates)
    print(f"risk thresholds: HIGH >= {th_bt[0]}, MEDIUM >= {th_bt[1]} calls/h on this series "
          f"(= {C.HIGH_THRESHOLD} / {C.MEDIUM_THRESHOLD} real calls/h, recall {rates[0]:.2f}, precision {rates[1]:.2f})")
    fc["risk"] = risk.risk_levels(fc.hi80, *th_bt)
    fc["hour_ahead"] = np.arange(1, len(fc) + 1)
    _num_round(fc).to_csv(C.OUT_DIR / "forecast_backtest.csv", index=False)
    plots.forecast_plot(idx[s - 48:s], y.to_numpy()[s - 48:s], t, lf["actual"], lf["mean"], lf["lo"], lf["hi"],
                        C.FIG_DIR / "forecast_backtest.png", title=f"72-hour {best} forecast vs actual (last fold)")
    inside = int(((fc.actual >= fc.lo80) & (fc.actual <= fc.hi80)).sum())
    print(f"last fold: {inside} of {len(fc)} actual hours inside the 80% band")

    # live forecast beyond the last observation: same model, same SST rule, same interval calibration
    run = forecast.model_runner(best, y.to_numpy(), exog, C.HORIZON, model_kw.get(best))
    live_r = forecast.empirical_interval(run, h["calls"].to_numpy(), len(y))   # calibrate on observed hours only
    t_live = pd.date_range(idx[-1] + pd.Timedelta("1h"), periods=C.HORIZON, freq="h")
    live = pd.DataFrame({"timestamp": t_live, "actual": np.nan, "forecast": live_r["mean"], "lo80": live_r["lo"], "hi80": live_r["hi"]})
    th_live = risk.thresholds(y.to_numpy(), *rates)
    live["risk"] = risk.risk_levels(live.hi80, *th_live)
    live["hour_ahead"] = np.arange(1, C.HORIZON + 1)
    _num_round(live).to_csv(C.OUT_DIR / "forecast_live.csv", index=False)
    search = {"chosen": chosen, "fit_hours": origin0, "n_models": len(aic),
              "top5": aic.head(5).round(2).to_dict(orient="records")}
    return summary, best, lb, fc, inside, s, search, diag, lf["err_sd"]


def run_risk(h, y, fc, best, s, sigma, rates=(1.0, 1.0)):
    step("5/5  Risk levels and alerts")
    counts = fc.risk.value_counts().reindex(risk.LEVELS, fill_value=0)
    print("hours by risk: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    wins = risk.alert_windows(list(fc.hour_ahead), list(fc.risk))
    alerts = pd.DataFrame([{"from_hour": a, "to_hour": b, "hours": n,
                            "from_time": fc.timestamp.iloc[a - 1], "to_time": fc.timestamp.iloc[b - 1]} for a, b, n in wins])
    alerts.to_csv(C.OUT_DIR / "alerts.csv", index=False)
    for a, b, n in wins:
        print(f"HIGH-risk window: +{a} h to +{b} h ({n} h)")
    # errors standardised by the spread of this model's past 72-h forecast errors (from interval calibration)
    stat, alarm = risk.cusum(fc.actual, fc.forecast, sigma)
    print(f"CUSUM (sigma = {sigma:.2f}, std of past 72-h forecast errors): "
          + (f"sudden rise detected at +{alarm + 1} h" if alarm is not None else "no sudden change"))
    plots.risk_strip(list(fc.risk), C.FIG_DIR / "risk_strip.png")
    return {"risk_counts": counts.to_dict(), "high_windows": [[a, b] for a, b, _ in wins],
            "cusum_alarm_hour": None if alarm is None else int(alarm + 1), "cusum_sigma": sigma,
            "thresholds": {"backtest": risk.thresholds(y.to_numpy()[:s], *rates),
                           "live": risk.thresholds(y.to_numpy(), *rates)},
            "calls_last_24h": float(np.nansum(h["calls"].to_numpy()[s - 24:s])), "expected_next_72h": float(fc.forecast.sum()),
            "issued": str(fc.timestamp.iloc[0])}


def main():
    ap = argparse.ArgumentParser(description="EchoGuard pipeline")
    ap.add_argument("--hourly-csv", help="real hourly series (timestamp, calls[, sst_c])")
    ap.add_argument("--kaggle-dir", help="folder with Kaggle train.csv and train/*.aiff")
    ap.add_argument("--clips", type=int, default=2000, help="number of clips for the detector stage")
    ap.add_argument("--skip-detector", action="store_true", help="skip the audio detection stage")
    ap.add_argument("--detector-rates", nargs=2, type=float, metavar=("RECALL", "PRECISION"),
                    help="with --hourly-csv: recall and precision of the detector that produced the counts")
    args = ap.parse_args()

    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    det = None if args.skip_detector else run_detector(args)
    h, y, source, rates = run_series(args, det)
    an = run_analysis(h, y)
    summary, best, lb, fc, inside, s, search, diag, err_sd = run_forecasting(h, y, args, rates)
    rk = run_risk(h, y, fc, best, s, err_sd, rates)
    out = {"data_source": source, "detector": det, "analysis": an, "sarima_order_search": search,
           "residual_diagnostics": diag,
           "models": summary.round(3).replace({np.nan: None}).to_dict(orient="records"),
           "dashboard_model": best, "ljung_box_p_lag24": lb, "last_fold_inside_80": inside, "risk": rk,
           "detector_rates": list(rates),
           "folds": int(pd.read_csv(C.OUT_DIR / "model_metrics_per_fold.csv").fold.max())}
    with open(C.OUT_DIR / "summary.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nDone in {time.time() - t0:.0f}s. Results are in {C.OUT_DIR}")
    print("Open the dashboard with:  streamlit run app.py")


if __name__ == "__main__":
    main()
