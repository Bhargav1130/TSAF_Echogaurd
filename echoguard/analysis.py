"""Exploratory time series analysis: stationarity, autocorrelation, STL."""
import warnings

import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from . import config as C


def stationarity(y):
    """ADF (H0: unit root) and KPSS (H0: stationary) together."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # KPSS warns when p is outside its table
        adf_stat, adf_p, *_ = adfuller(y, autolag="AIC")
        kpss_stat, kpss_p, *_ = kpss(y, regression="c", nlags="auto")
    if adf_p < 0.05 and kpss_p >= 0.05:
        verdict = "stationary"
    elif adf_p >= 0.05 and kpss_p < 0.05:
        verdict = "non-stationary (difference it)"
    elif adf_p < 0.05 and kpss_p < 0.05:
        verdict = "stationary around a moving level (keep d=0, model the trend)"
    else:
        verdict = "inconclusive"
    return {"adf_stat": float(adf_stat), "adf_p": float(adf_p),
            "kpss_stat": float(kpss_stat), "kpss_p": float(kpss_p), "verdict": verdict}


def autocorrelation(y, nlags=72):
    return {"acf": acf(y, nlags=nlags).tolist(), "pacf": pacf(y, nlags=min(nlags, 48)).tolist(),
            "ci": float(1.96 / np.sqrt(len(y)))}


def stl_decompose(y, period=C.SEASON):
    res = STL(y, period=period, robust=True).fit()
    fs = max(0.0, 1 - np.var(res.resid) / np.var(res.seasonal + res.resid))
    ft = max(0.0, 1 - np.var(res.resid) / np.var(res.trend + res.resid))
    return res, {"seasonal_strength": float(fs), "trend_strength": float(ft)}


def daily_profile(y):
    return y.groupby(y.index.hour).mean()


def sst_correlation(y, sst):
    """Correlation between daily call totals and daily mean SST."""
    daily = y.resample("D").sum()
    s = sst.resample("D").mean().reindex(daily.index)
    ok = s.notna()
    return float(np.corrcoef(daily[ok], s[ok])[0, 1])
