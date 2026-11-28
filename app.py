"""EchoGuard risk dashboard.  Run:  streamlit run app.py
(run `python run_all.py` first so the outputs/ folder exists)."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from echoguard import config as C  # noqa: E402
from echoguard import risk  # noqa: E402

st.set_page_config(page_title="EchoGuard", page_icon="🐋", layout="wide")
COLORS = {"LOW": "#3a9d5d", "MEDIUM": "#e3a008", "HIGH": "#d64545"}


def full_width(fn, *args, **kw):
    """Full-width element on both new (width="stretch") and older Streamlit."""
    try:
        return fn(*args, width="stretch", **kw)
    except TypeError:
        return fn(*args, use_container_width=True, **kw)


@st.cache_data
def load(name):
    path = C.OUT_DIR / name
    if not path.exists():
        return None
    if name.endswith(".json"):
        return json.loads(path.read_text())
    return pd.read_csv(path, parse_dates=["timestamp"] if name.startswith(("forecast", "hourly")) else None)


summary = load("summary.json")
if summary is None:
    st.error("No results yet. Run `python run_all.py` in this folder first, then refresh this page.")
    st.stop()

# ---------------------------------------------------------------- sidebar
st.sidebar.title("EchoGuard")
st.sidebar.caption("North Atlantic right whale presence forecast")
mode = st.sidebar.radio("Forecast to show", ["Backtest (last fold, with actuals)", "Live (next 72 h after the data)"])
fc = load("forecast_backtest.csv") if mode.startswith("Backtest") else load("forecast_live.csv")
th = summary["risk"].get("thresholds", {}).get("backtest" if mode.startswith("Backtest") else "live", [5.0, 3.5])
high_t = st.sidebar.slider("HIGH if risk bound ≥ (calls/h)", 0.5, 15.0, round(float(th[0]), 1), 0.1,
                           help=f"Default = {C.HIGH_THRESHOLD} real calls/h, converted to detector counts when the series is detector output")
med_t = st.sidebar.slider("MEDIUM if risk bound ≥ (calls/h)", 0.1, high_t, min(round(float(th[1]), 1), high_t), 0.1,
                          help=f"Default = {C.MEDIUM_THRESHOLD} real calls/h, converted to detector counts when the series is detector output")
gap = st.sidebar.slider("Join HIGH windows separated by ≤ (hours)", 0, 3, 1)
st.sidebar.markdown("---")
loc = summary.get("location")
st.sidebar.caption(f"Data: {summary['data_source']}")
if loc:
    st.sidebar.caption(f"Where: {loc['place']} · {loc['station']} · {loc['lat']:.3f}, {loc['lon']:.3f}"
                       f"{' (approx.)' if loc['approx'] else ''} · forecast covers ≈ {loc['radius_km']} km around it")
st.sidebar.caption(f"Model: {summary['dashboard_model']}")

fc = fc.copy()
fc["risk"] = risk.risk_levels(fc.risk_bound, high=high_t, medium=med_t)
wins = risk.alert_windows(list(fc.hour_ahead), list(fc.risk), max_gap=gap)
hourly = load("hourly_series.csv")

# ---------------------------------------------------------------- header
st.title("🐋 EchoGuard — whale presence risk")
issued = fc.timestamp.iloc[0]
st.caption(f"Forecast issued {issued:%d %b %Y %H:%M} UTC · 72-hour horizon · risk = 75% quantile of forecast + past errors at that hour of day")
if "SIMULATED" in summary["data_source"]:
    st.info("Results are from a simulated buoy season. Run `python run_all.py --hourly-csv <file>` with real data to replace them.")

now = fc.risk.iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"<div style='background:{COLORS[now]};color:white;padding:14px;border-radius:8px'>"
            f"<div style='font-size:13px'>Risk next hour</div><div style='font-size:30px;font-weight:700'>{now}</div></div>",
            unsafe_allow_html=True)
hist = hourly.set_index("timestamp")["calls"]
before = hist[hist.index < fc.timestamp.iloc[0]].tail(24)
c2.metric("Calls detected, last 24 h", f"{before.sum():.0f}")
c3.metric("Expected detections, next 72 h", f"{fc.forecast.sum():.0f}")
c4.metric("Next HIGH window", f"+{wins[0][0]} h" if wins else "none")

# ---------------------------------------------------------------- risk strip
st.subheader("Hour-by-hour risk")
cells = "".join(f"<div title='+{h} h: {lv}' style='flex:1;height:34px;margin-right:1px;background:{COLORS[lv]}'></div>"
                for h, lv in zip(fc.hour_ahead, fc.risk))
st.markdown(f"<div style='display:flex'>{cells}</div>"
            "<div style='display:flex;justify-content:space-between;font-size:12px;color:#666'>"
            "<span>+1 h</span><span>+24 h</span><span>+48 h</span><span>+72 h</span></div>", unsafe_allow_html=True)
counts = fc.risk.value_counts().reindex(risk.LEVELS, fill_value=0)
st.caption(" · ".join(f"{k}: {v} h" for k, v in counts.items()))

left, right = st.columns([3, 2])
with left:
    st.subheader("Forecast")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    h48 = hist[hist.index < fc.timestamp.iloc[0]].tail(48)
    ax.plot(h48.index, h48.values, color="#9aa3ad", lw=1, label="History")
    ax.fill_between(fc.timestamp, fc.lo80, fc.hi80, color="#2a6fdb", alpha=0.18, label="80% interval")
    ax.plot(fc.timestamp, fc.forecast, color="#2a6fdb", lw=1.6, label="Forecast")
    ax.plot(fc.timestamp, fc.risk_bound, color="#2a6fdb", lw=0.9, ls="--", label="Risk bound")
    if fc.actual.notna().any():
        ax.plot(fc.timestamp, fc.actual, color="#222", lw=0.9, marker=".", ms=3, label="Actual")
    ax.axhline(high_t, color=COLORS["HIGH"], ls=":", lw=1)
    ax.axhline(med_t, color=COLORS["MEDIUM"], ls=":", lw=1)
    ax.set_ylabel("Up-calls per hour"); ax.legend(fontsize=7, ncol=4, loc="upper left")
    fig.autofmt_xdate(); fig.tight_layout()
    st.pyplot(fig)
with right:
    st.subheader("Planned alerts")
    if wins:
        full_width(st.dataframe, pd.DataFrame([{"Window": f"+{a} h to +{b} h", "Hours": n,
                                    "From (UTC)": f"{fc.timestamp.iloc[a - 1]:%d %b %H:%M}",
                                    "To (UTC)": f"{fc.timestamp.iloc[b - 1]:%d %b %H:%M}"} for a, b, n in wins]),
                     hide_index=True)
    else:
        st.write("No HIGH-risk windows in the next 72 hours.")
    st.subheader("What ships should do")
    for lv in risk.LEVELS[::-1]:
        st.markdown(f"<span style='color:{COLORS[lv]};font-weight:700'>{lv}</span> — {risk.advice(lv)}", unsafe_allow_html=True)
    cu = summary["risk"].get("cusum_alarm_hour")
    if mode.startswith("Backtest") and cu:
        st.warning(f"CUSUM change alarm: calls rose faster than forecast at +{cu} h.")

# ---------------------------------------------------------------- details
st.markdown("---")
tab0, tab1, tab2, tab3 = st.tabs(["Where", "Time series analysis", "Model comparison", "Up-call detector"])
with tab0:
    if not loc:
        st.write("Simulated season: there is no real buoy position. Run with real data (fetch_whoi.py or fetch_dclde.py).")
    else:
        import pydeck as pdk
        st.markdown(f"**{loc['place']}** — {loc['station']}. The forecast is for right whales calling within hearing "
                    f"range of this recorder, about {loc['radius_km']} km around it (assumed; 10–20 km depending on noise). "
                    "One recorder cannot say where inside the circle a whale is, only whether whales are there.")
        rec = pd.DataFrame([{"lat": a, "lon": b, "used": abs(a - loc["lat"]) < 1e-4 and abs(b - loc["lon"]) < 1e-4}
                            for a, b in loc.get("recorders", [[loc["lat"], loc["lon"]]])])
        rec["color"] = rec.used.map({True: [201, 116, 20, 230], False: [86, 106, 115, 200]})
        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(latitude=loc["lat"], longitude=loc["lon"], zoom=8.6),
            layers=[pdk.Layer("ScatterplotLayer", data=pd.DataFrame([loc]), get_position="[lon, lat]",
                              get_radius=loc["radius_km"] * 1000, get_fill_color=[13, 107, 136, 40],
                              get_line_color=[13, 107, 136, 200], stroked=True, line_width_min_pixels=1),
                    pdk.Layer("ScatterplotLayer", data=rec, get_position="[lon, lat]", get_radius=700,
                              get_fill_color="color", pickable=True)],
            tooltip={"text": "{lat}, {lon}"}))
        if loc["approx"]:
            st.caption("WHOI does not publish the exact mooring position on the buoy page; the marker is the position "
                       "EchoGuard uses for sea temperature.")
an = summary["analysis"]
with tab1:
    s = an["stationarity"]
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("ADF p-value", f"{s['adf_p']:.3f}")
    a2.metric("KPSS p-value", f"{s['kpss_p']:.3f}")
    a3.metric("STL seasonal strength", f"{an['stl']['seasonal_strength']:.2f}")
    a4.metric("STL trend strength", f"{an['stl']['trend_strength']:.2f}")
    st.caption(f"Verdict: {s['verdict']}")
    for fname in ["daily_calls_sst.png", "hourly_profile.png", "acf.png", "stl.png"]:
        p = C.FIG_DIR / fname
        if p.exists():
            full_width(st.image, str(p))
with tab2:
    m = pd.DataFrame(summary["models"])
    m["coverage80"] = m.coverage80.map(lambda v: "-" if v is None or pd.isna(v) else f"{v:.0%}")
    full_width(st.dataframe, m.rename(columns={"model": "Model", "mae": "MAE", "rmse": "RMSE", "mase": "MASE", "coverage80": "80% coverage"}).round(2),
                 hide_index=True)
    lb = summary["ljung_box_p_lag24"]
    st.caption(f"Walk-forward validation, {summary.get('folds', C.N_FOLDS)} folds × {C.HORIZON} h. Ljung-Box p at lag 24: "
               f"{'< 0.001' if lb < 0.001 else f'{lb:.3f}'}.")
    if (C.FIG_DIR / "model_mase.png").exists():
        full_width(st.image, str(C.FIG_DIR / "model_mase.png"))
    srch = summary.get("sarima_order_search")
    if srch:
        st.markdown(f"**SARIMA order selection by AIC** — {srch['n_models']} models fitted on the first "
                    f"{srch['fit_hours']} h (before any test fold). Chosen: `{srch['chosen']}`")
        full_width(st.dataframe, pd.DataFrame(srch["top5"]).rename(
            columns={"model": "Model", "aic": "AIC", "bic": "BIC", "converged": "Converged"}), hide_index=True)
    rd = summary.get("residual_diagnostics")
    if rd:
        pf = lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"
        st.markdown(f"**Residual diagnostics** — {summary['dashboard_model']}, last-fold fit, {rd['n']} hours")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Ljung-Box p (lag 24)", pf(rd["ljung_box_p_24"]))
        d2.metric("Ljung-Box p (lag 48)", pf(rd["ljung_box_p_48"]))
        d3.metric("Jarque-Bera p", pf(rd["jarque_bera_p"]))
        d4.metric("Skew / excess kurtosis", f"{rd['skew']:.2f} / {rd['excess_kurtosis']:.2f}")
        st.caption(rd["verdict"])
        if (C.FIG_DIR / "residuals.png").exists():
            full_width(st.image, str(C.FIG_DIR / "residuals.png"))
with tab3:
    det = summary.get("detector")
    if det:
        st.caption(f"Clips: {det['source']}")
        if "SIMULATED" not in summary["data_source"]:
            st.info("This tab shows how up-calls are found, on labelled clips. The hourly counts being forecast "
                    "come from the data source's own automatic detector, not from these models.")
        full_width(st.dataframe, pd.DataFrame(det["metrics"]), hide_index=True)
        if (C.FIG_DIR / "detector_pr.png").exists():
            st.image(str(C.FIG_DIR / "detector_pr.png"), width=520)
    else:
        st.write("Detector stage was skipped (run without --skip-detector to include it).")
