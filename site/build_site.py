"""Build the EchoGuard showcase page from the pipeline outputs.

    python run_all.py ...          # first, so outputs/ is current
    python site/build_site.py      # writes site/index.html

The page itself is site/template.html; this script only injects the data.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from echoguard import analysis, config as C, series  # noqa: E402

OUT = ROOT / "outputs"


def nums(a, nd=3):
    """List of floats rounded, NaN -> None (valid JSON)."""
    return [None if pd.isna(v) else round(float(v), nd) for v in a]


def forecast_block(df):
    return {"t": [str(t)[:16] for t in df.timestamp], "actual": nums(df.actual), "forecast": nums(df.forecast),
            "lo": nums(df.lo80), "hi": nums(df.hi80), "rb": nums(df.risk_bound)}


def main():
    s = json.loads((OUT / "summary.json").read_text())
    h = pd.read_csv(OUT / "hourly_series.csv", parse_dates=["timestamp"])
    y = series.fill_for_model(h.set_index("timestamp")["calls"])
    bt = pd.read_csv(OUT / "forecast_backtest.csv", parse_dates=["timestamp"])
    live = pd.read_csv(OUT / "forecast_live.csv", parse_dates=["timestamp"])
    det = s.get("detector") or {"source": "not run", "metrics": []}
    mr = next((d for d in det["metrics"] if d["detector"] == "MiniRocket"), None)
    ac = analysis.autocorrelation(y, nlags=72)
    # rates that link the series to real calls (None = the series is real calls already)
    rates = s.get("detector_rates") or ([mr["recall"], mr["precision"]] if mr else [1, 1])
    rates = [None, None] if list(rates) == [1, 1] else rates
    m = re.search(r"real data \((.+?)\)", s["data_source"])          # sidecar written by fetch_dclde.py
    side = (ROOT / m.group(1)).with_suffix(".source.json") if m else None
    src = json.loads(side.read_text()) if side and side.exists() else {}
    sst_daily = h.set_index("timestamp")["sst_c"].resample("D").mean()
    data = {
        "meta": {"source": s["data_source"], "simulated": "SIMULATED" in s["data_source"],
                 "detector_source": det["source"], "issued": s["risk"]["issued"][:16],
                 "model": s["dashboard_model"], "inside": s["last_fold_inside_80"],
                 "thresholds": s["risk"]["thresholds"], "real_thresholds": [C.HIGH_THRESHOLD, C.MEDIUM_THRESHOLD],
                 "cusum": {"sigma": round(s["risk"]["cusum_sigma"], 3), "k": C.CUSUM_K, "h": C.CUSUM_H,
                           "alarm": s["risk"]["cusum_alarm_hour"]},
                 "recall": rates[0], "precision": rates[1], "source_name": src.get("name"), "source_note": src.get("note"),
                 "horizon": C.HORIZON, "risk_q": C.RISK_QUANTILE, "folds": s.get("folds", C.N_FOLDS), "n_cal": C.N_CAL, "band": list(C.BAND)},
        "hourly": {"t0": str(h.timestamp.iloc[0])[:16], "calls": nums(h.calls, 1),
                   "calls_true": nums(h.calls_true, 1) if "calls_true" in h else None,
                   "sst_daily": nums(sst_daily, 2)},
        "detectors": det["metrics"],
        "analysis": {**{k: s["analysis"]["stationarity"][k] for k in ("adf_p", "kpss_p", "verdict")},
                     "stl": s["analysis"]["stl"], "acf": nums(ac["acf"], 3), "acf_ci": round(ac["ci"], 4),
                     "profile": nums(s["analysis"]["daily_profile"], 3), "sst_corr": round(s["analysis"]["sst_corr"], 3)},
        "aic": s["sarima_order_search"],
        "models": s["models"],
        "per_fold": pd.read_csv(OUT / "model_metrics_per_fold.csv").round(3).to_dict(orient="records"),
        "backtest": {**forecast_block(bt), "origin": int(np.searchsorted(h.timestamp, bt.timestamp.iloc[0]))},
        "live": {**forecast_block(live), "origin": len(h)},
        "resid": {k: v for k, v in s["residual_diagnostics"].items() if k != "acf_outside_band"},
        "location": s.get("location"),
    }
    loc = s.get("location")
    coast = json.loads((ROOT / "site" / "coast_ccb.json").read_text())
    bb = coast["bbox"]
    if loc and bb[0] <= loc["lon"] <= bb[2] and bb[1] <= loc["lat"] <= bb[3]:   # coastline only covers Cape Cod Bay
        data["coast"] = coast
    data["resid"]["acf"] = nums(data["resid"]["acf"], 3)
    aic = pd.read_csv(OUT / "sarima_aic.csv")                      # model like "SARIMA(1, 0, 1)(1, 0, 1, 24)"
    o = aic.model.str.extractall(r"(\d+)").unstack().astype(int)   # p, d, q, P, D, Q, s
    data["aic_all"] = [{"p": int(r[0]), "q": int(r[2]), "P": int(r[3]), "Q": int(r[5]), "aic": round(a, 1),
                        "bic": round(b, 1), "converged": bool(c)}
                       for r, a, b, c in zip(o.to_numpy(), aic.aic, aic.bic, aic.converged)]
    js = json.dumps(data, separators=(",", ":"), allow_nan=False)
    page = (ROOT / "site" / "template.html").read_text(encoding="utf-8").replace("__ECHOGUARD_DATA__", js)
    # echoguard.html = page body for the Artifact (its host adds <head>); index.html = standalone copy
    (ROOT / "site" / "echoguard.html").write_text(page, encoding="utf-8")
    head = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')
    standalone = head + page.replace("<style>", "<style>\nbody{margin:0}", 1) + "\n</html>\n"
    (ROOT / "site" / "index.html").write_text(standalone, encoding="utf-8")
    print(f"site/echoguard.html + site/index.html written ({len(page) / 1024:.0f} KB, data {len(js) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
