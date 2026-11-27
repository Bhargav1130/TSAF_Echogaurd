"""Figures for the report and dashboard (saved as PNG)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, GREY, RED, AMBER, GREEN = "#2a6fdb", "#9aa3ad", "#d64545", "#e3a008", "#3a9d5d"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def daily_and_sst(y, sst, path, corr):
    daily, s = y.resample("D").sum(), sst.resample("D").mean()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 4), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.bar(daily.index, daily.values, color=BLUE, width=0.8)
    a1.set_ylabel("Up-calls per day")
    a1.set_title(f"Daily up-calls rise as the sea warms (correlation r = {corr:.2f})", loc="left")
    a2.plot(s.index, s.values, color="#333")
    a2.set_ylabel("SST (°C)")
    _save(fig, path)


def hourly_profile(profile, path):
    fig, ax = plt.subplots(figsize=(7, 2.8))
    colors = [BLUE if v == profile.max() else GREY for v in profile.values]
    ax.bar(profile.index, profile.values, color=colors)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hour of day (UTC)"); ax.set_ylabel("Mean calls / hour")
    ax.set_title("Calling peaks in the early hours of the night", loc="left")
    _save(fig, path)


def acf_plot(acf_vals, ci, path):
    lags = np.arange(len(acf_vals))
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.axhspan(-ci, ci, color="#dde5f3")
    colors = [BLUE if l in (24, 48, 72) else GREY for l in lags[1:]]
    ax.bar(lags[1:], acf_vals[1:], color=colors, width=0.7)
    ax.set_xlabel("Lag (hours)"); ax.set_ylabel("ACF")
    ax.set_title("Autocorrelation peaks every 24 h: a daily cycle (shaded = 95% band)", loc="left")
    _save(fig, path)


def stl_plot(res, path):
    fig, axes = plt.subplots(4, 1, figsize=(7, 5.5), sharex=True)
    for ax, comp, name in zip(axes, [res.observed, res.trend, res.seasonal, res.resid],
                              ["Observed", "Trend", "Seasonal (24 h)", "Remainder"]):
        ax.plot(comp.index, comp.values, color=BLUE if name != "Remainder" else GREY, lw=0.8)
        ax.set_ylabel(name, fontsize=8)
    axes[0].set_title("STL decomposition of hourly calls", loc="left")
    _save(fig, path)


def mase_bars(summary, path):
    s = summary.sort_values("mase", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    colors = [BLUE if m in ("SARIMA", "Holt-Winters", "SARIMAX (SST)") else GREY for m in s.model]
    ax.barh(s.model, s.mase, color=colors)
    for i, v in enumerate(s.mase):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlabel("MASE averaged over walk-forward folds (lower is better)")
    ax.set_title("Seasonal models beat both baselines", loc="left")
    _save(fig, path)


def forecast_plot(hist_t, hist_y, t, actual, mean, lo, hi, path, title="72-hour SARIMAX forecast"):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(hist_t, hist_y, color=GREY, lw=1, label="History")
    ax.fill_between(t, lo, hi, color=BLUE, alpha=0.18, label="80% interval")
    ax.plot(t, mean, color=BLUE, lw=1.6, label="Forecast")
    if actual is not None:
        ax.plot(t, actual, color="#222", lw=0.9, marker=".", ms=3, label="Actual")
    ax.axvline(t[0], color="#666", ls="--", lw=0.8)
    ax.set_ylabel("Up-calls per hour"); ax.legend(fontsize=7, ncol=4, loc="upper left")
    ax.set_title(title, loc="left")
    fig.autofmt_xdate()
    _save(fig, path)


def risk_strip(levels, path):
    cmap = {"LOW": GREEN, "MEDIUM": AMBER, "HIGH": RED}
    fig, ax = plt.subplots(figsize=(7, 0.9))
    for i, lv in enumerate(levels):
        ax.add_patch(plt.Rectangle((i, 0), 0.92, 1, color=cmap[lv]))
    ax.set_xlim(0, len(levels)); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_xticks(range(0, len(levels) + 1, 12)); ax.set_xticklabels([f"+{h}h" for h in range(0, len(levels) + 1, 12)])
    for s in ax.spines.values():
        s.set_visible(False)
    _save(fig, path)


def residual_plot(r, acf_vals, ci, path, title="Residual diagnostics"):
    """Residuals over time, their ACF, and a normal Q-Q plot."""
    from scipy import stats
    fig = plt.figure(figsize=(7, 5))
    a1 = fig.add_subplot(2, 1, 1)
    a1.plot(r.index, r.values, color=GREY, lw=0.6)
    a1.axhline(0, color="#333", lw=0.8)
    a1.set_xlabel("Hour (training sample)"); a1.set_ylabel("Residual")
    a1.set_title(title, loc="left")
    a2 = fig.add_subplot(2, 2, 3)
    lags = np.arange(1, len(acf_vals))
    a2.axhspan(-ci, ci, color="#dde5f3")
    a2.bar(lags, acf_vals[1:], color=[RED if abs(v) > ci else BLUE for v in acf_vals[1:]], width=0.7)
    a2.set_xlabel("Lag (hours)"); a2.set_ylabel("Residual ACF")
    a3 = fig.add_subplot(2, 2, 4)
    (osm, osr), (slope, icpt, _) = stats.probplot(r.values, dist="norm")
    a3.scatter(osm, osr, s=4, color=BLUE)
    a3.plot(osm, slope * osm + icpt, color=RED, lw=1)
    a3.set_xlabel("Normal quantiles"); a3.set_ylabel("Residual quantiles"); a3.set_title("Normal Q-Q", fontsize=8)
    _save(fig, path)


def detector_pr(curves, path):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for name, (r, p, ap) in curves.items():
        ax.plot(r, p, label=f"{name} (PR-AUC {ap:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1.02)
    ax.set_title("Up-call detectors on the test clips", loc="left"); ax.legend(fontsize=7)
    _save(fig, path)
