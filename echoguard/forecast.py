"""Forecasting models and walk-forward (rolling-origin) evaluation."""
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf

from . import config as C

MAXITER = 200                                # optimiser steps; the default 50 often stops before converging


def _result(mean, lo=None, hi=None, resid=None):
    mean = np.clip(np.asarray(mean, float), 0, None)  # call counts cannot be negative
    if lo is not None:
        lo, hi = np.clip(np.asarray(lo, float), 0, None), np.asarray(hi, float)
    return {"mean": mean, "lo": lo, "hi": hi, "resid": resid}


def naive(y, h, **_):
    return _result(np.repeat(y[-1], h))


def seasonal_naive(y, h, m=C.SEASON, **_):
    reps = int(np.ceil(h / m))
    return _result(np.tile(y[-m:], reps)[:h])


def holt_winters(y, h, m=C.SEASON, **_):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(y, trend=None, seasonal="add", seasonal_periods=m).fit()
    return _result(fit.forecast(h), resid=fit.resid)


def sarima(y, h, exog=None, exog_future=None, order=(1, 0, 1), seasonal_order=(1, 0, 1, C.SEASON), **_):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal_order, trend="c").fit(disp=False, maxiter=MAXITER)
        fc = fit.get_forecast(h, exog=exog_future)
    ci = np.asarray(fc.conf_int(alpha=C.INTERVAL_ALPHA))
    return _result(fc.predicted_mean, ci[:, 0], ci[:, 1], resid=fit.resid)


def sarimax_sst(y, h, exog=None, exog_future=None, **kw):
    if exog is None:
        raise ValueError("SARIMAX needs the SST exogenous series")
    return sarima(y, h, exog=exog, exog_future=exog_future, **kw)


MODELS = {
    "Naive": naive,
    "Seasonal naive": seasonal_naive,
    "Holt-Winters": holt_winters,
    "SARIMA": sarima,
    "SARIMAX (SST)": sarimax_sst,
}


EXOG_MODELS = {"SARIMAX (SST)"}      # only these models receive the SST input


def order_search(y, p=(0, 1, 2), q=(0, 1, 2), P=(0, 1), Q=(0, 1), m=C.SEASON):
    """Fit SARIMA(p,0,q)(P,0,Q)m for every combination (d = D = 0, as the
    ADF/KPSS tests suggest) and return them sorted by AIC. Pass only training
    data, i.e. the hours before the first walk-forward origin, so the choice
    does not peek at the test folds."""
    rows = []
    for pp, qq, PP, QQ in itertools.product(p, q, P, Q):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit = SARIMAX(y, order=(pp, 0, qq), seasonal_order=(PP, 0, QQ, m), trend="c").fit(disp=False, maxiter=MAXITER)
            except (ValueError, np.linalg.LinAlgError):
                continue
        rows.append({"order": (pp, 0, qq), "seasonal_order": (PP, 0, QQ, m), "aic": float(fit.aic),
                     "bic": float(fit.bic), "converged": bool(fit.mle_retvals.get("converged", True))})
    return pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)


def best_order(table):
    """Lowest-AIC model among those whose optimiser converged."""
    ok = table[table.converged]
    row = (ok if len(ok) else table).iloc[0]
    return {"order": tuple(row.order), "seasonal_order": tuple(row.seasonal_order)}

# ------------------------------------------------------------- metrics
# Metrics skip NaN actuals (buoy offline), so interpolated hours are never scored.
def mae(a, f):
    return float(np.nanmean(np.abs(np.asarray(a, float) - f)))


def rmse(a, f):
    return float(np.sqrt(np.nanmean((np.asarray(a, float) - f) ** 2)))


def mase_scale(y_train, m=C.SEASON):
    """In-sample MAE of the seasonal naive forecast (MASE denominator)."""
    y_train = np.asarray(y_train, float)
    return float(np.nanmean(np.abs(y_train[m:] - y_train[:-m])))


def coverage(a, lo, hi):
    a = np.asarray(a, float)
    ok = ~np.isnan(a)
    return float(np.mean((a[ok] >= lo[ok]) & (a[ok] <= hi[ok])))


def fold_origins(n, h=C.HORIZON, folds=C.N_FOLDS):
    """Origins for the last `folds` non-overlapping 72-h test blocks."""
    return [n - h * (folds - k) for k in range(folds)]


def model_runner(name, y, exog=None, h=C.HORIZON, kw=None):
    """Return run(origin) -> forecast of the h hours after `origin`, using only
    data before it. Results are cached per origin, so walk-forward folds and
    interval calibration can share fits."""
    y = np.asarray(y, float)
    X = None if exog is None else np.asarray(exog, float).reshape(-1, 1)
    if name in EXOG_MODELS and X is None:
        raise ValueError(f"{name} needs the SST exogenous series")
    cache = {}

    def run(s):
        if s not in cache:
            k = dict(kw or {})
            if name in EXOG_MODELS:
                # SST after the origin is not known yet: hold its last value, exactly as in live use
                k.update(exog=X[:s], exog_future=np.repeat(X[s - 1:s], h, axis=0))
            cache[s] = MODELS[name](y[:s], h, **k)
        return cache[s]
    return run


def empirical_interval(run, y, s, h=C.HORIZON, n_cal=C.N_CAL, alpha=C.INTERVAL_ALPHA, min_train=7 * C.SEASON):
    """Forecast from origin s with an (1-alpha) interval built from real past
    errors: the h-step forecasts from the n_cal previous origins (s-h, s-2h, ...)
    are compared with what happened, and the alpha/2 and 1-alpha/2 error
    quantiles are added to the point forecast (split-conformal style). Unlike the
    Gaussian model interval this also covers bias and skewed count errors.
    Also returns err_sd, the spread of those past forecast errors (used by CUSUM)."""
    y = np.asarray(y, float)                         # observed values: NaN where the buoy was offline
    cal = [c for c in range(s - h, s - h * n_cal - 1, -h) if c >= min_train]
    errs = np.concatenate([y[c:c + h] - run(c)["mean"] for c in cal]) if cal else np.array([])
    errs = errs[~np.isnan(errs)]                     # interpolated hours are not real errors
    if not len(errs):
        raise ValueError("Not enough history before the origin to calibrate the interval")
    q_lo, q_hi = np.quantile(errs, [alpha / 2, 1 - alpha / 2])
    r = dict(run(s))
    # calls are whole numbers: round the band to whole calls, else a lower bound of 0.14 calls/h
    # leaves every zero-call hour outside it (sparse live WHOI data: 80% band covered 41%).
    # Nearest, not outwards: floor/ceil over-covered small counts (simulated season 93%, live 82%;
    # nearest gives 88% / 78%).
    r.update(lo=np.round(np.clip(r["mean"] + q_lo, 0, None)), hi=np.round(r["mean"] + q_hi),
             err_sd=float(errs.std()), n_cal=len(cal),
             risk=risk_bound(r["mean"], y, run, cal, h))
    return r


def risk_bound(mean, y, run, cal, h=C.HORIZON, q=C.RISK_QUANTILE):
    """Per forecast hour: the q-quantile of forecast + past 72-h errors made at the
    SAME hour of day (origins are 72 h = 3 days apart, so step i of every past
    forecast falls on the same clock hour). Calls cluster at certain hours, so a
    pooled band is as wide at 14:00 as at 01:00; on bursty live WHOI data its upper
    edge passed the HIGH threshold in every hour and risk was HIGH for 72 h."""
    E = np.array([y[c:c + h] - run(c)["mean"] for c in cal])           # (n_cal, h)
    pooled = E[~np.isnan(E)]
    out = []
    for i in range(h):
        e = E[:, [j for j in range(h) if j % C.SEASON == i % C.SEASON]].ravel()
        e = e[~np.isnan(e)]
        out.append(mean[i] + np.quantile(e if len(e) >= 5 else pooled, q))   # too few: fall back to all hours
    return np.round(np.clip(out, 0, None), 1)


def walk_forward(y, exog=None, models=None, h=C.HORIZON, folds=C.N_FOLDS, first_origin=None, model_kw=None,
                 actual=None):
    """Train on all hours before each origin, forecast the next h hours.

    y is the gap-filled series the models train on; `actual` (default y) is what
    forecasts are scored against, with NaN for buoy-offline hours so that
    interpolated values never count as errors, coverage or calibration.
    Every model gets an empirical 80% interval (`empirical_interval`).
    model_kw: optional {model name: extra keyword arguments}, e.g. the SARIMA
    orders picked by `order_search`.
    Returns (summary DataFrame, per-fold DataFrame, details of the last fold).
    The MASE scale is fixed from the data before the first origin.
    """
    model_kw = model_kw or {}
    models = models or list(MODELS)
    y = np.asarray(y, float)
    obs = y if actual is None else np.asarray(actual, float)
    origins = ([first_origin + h * k for k in range(folds)] if first_origin is not None
               else fold_origins(len(y), h, folds))
    if origins[-1] + h > len(y):
        raise ValueError("Not enough data for the requested folds")
    scale = mase_scale(y[:origins[0]])
    runners = {name: model_runner(name, y, exog, h, model_kw.get(name)) for name in models}
    rows, last = [], {}
    for k, s in enumerate(origins):
        te = obs[s:s + h]
        for name in models:
            r = empirical_interval(runners[name], obs, s, h)
            row = {"fold": k + 1, "model": name, "mae": mae(te, r["mean"]), "rmse": rmse(te, r["mean"]),
                   "coverage80": coverage(te, r["lo"], r["hi"])}
            rows.append(row)
            if k == len(origins) - 1:
                last[name] = {**r, "origin": s, "actual": te}
    per_fold = pd.DataFrame(rows)
    summary = per_fold.groupby("model", sort=False).agg(mae=("mae", "mean"), rmse=("rmse", "mean"),
                                                        coverage80=("coverage80", "mean")).reset_index()
    summary["mase"] = summary.mae / scale
    summary = summary[["model", "mae", "rmse", "mase", "coverage80"]].sort_values("mae").reset_index(drop=True)
    return summary, per_fold, last


def residual_diagnostics(resid, burn=C.SEASON + 1, nlags=48):
    """Check a fitted model's one-step residuals (TSAF diagnostic step).

    The first `burn` residuals come from the filter's start-up with a seasonal
    model and are dropped. Returns (trimmed residuals, dict of results):
    Ljung-Box p at lags 24 and 48 (H0: no autocorrelation left), Jarque-Bera p
    (H0: normal), skew, excess kurtosis, residual ACF and plain-language verdicts.
    """
    r = pd.Series(np.asarray(resid, float)).iloc[burn:].dropna().reset_index(drop=True)
    lb = acorr_ljungbox(r, lags=[C.SEASON, 2 * C.SEASON], return_df=True)["lb_pvalue"]
    jb = stats.jarque_bera(r)
    d = {"n": int(len(r)), "mean": float(r.mean()), "std": float(r.std()),
         "ljung_box_p_24": float(lb.iloc[0]), "ljung_box_p_48": float(lb.iloc[1]),
         "jarque_bera_p": float(jb.pvalue), "skew": float(stats.skew(r)),
         "excess_kurtosis": float(stats.kurtosis(r)),
         "acf": acf(r, nlags=nlags).tolist(), "acf_ci": float(1.96 / np.sqrt(len(r)))}
    d["white_noise"] = bool(d["ljung_box_p_24"] >= 0.05 and d["ljung_box_p_48"] >= 0.05)
    d["normal"] = bool(d["jarque_bera_p"] >= 0.05)
    big = [k for k in range(1, nlags + 1) if abs(d["acf"][k]) > d["acf_ci"]]
    d["acf_outside_band"] = big
    d["verdict"] = (
        ("Residuals look like white noise. " if d["white_noise"] else
         f"Residuals still autocorrelated (Ljung-Box p < 0.05; ACF outside the 95% band at lags "
         f"{', '.join(map(str, big[:8]))}{'...' if len(big) > 8 else ''}), so some structure is not captured. ")
        + ("Residuals look normal." if d["normal"] else
           f"Residuals are not normal (Jarque-Bera p < 0.05, skew {d['skew']:.2f}, excess kurtosis "
           f"{d['excess_kurtosis']:.2f}): counts are skewed, so a Gaussian interval would under-cover; EchoGuard therefore uses empirical intervals built from past forecast errors."))
    return r, d
