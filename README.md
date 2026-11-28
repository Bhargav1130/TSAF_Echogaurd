# EchoGuard

Forecasting North Atlantic right whale presence from underwater sound, 72 hours
ahead, and turning the forecast into Low / Medium / High risk for ships.
TSAF course project (CS3216), RV University.

```
buoy audio ─► windows + 50–250 Hz band-pass ─► up-call detector (MiniRocket)
          ─► hourly detection-count series (gaps kept missing; simulated season is passed through MiniRocket's recall / precision)
          ─► ADF / KPSS · ACF / PACF · STL (period 24)
          ─► SARIMA orders chosen by AIC grid search (training hours only)
          ─► Naive · Seasonal naive · Holt-Winters · SARIMA · SARIMAX(SST)
          ─► walk-forward validation (5 × 72 h) · MAE · RMSE · MASE · empirical 80 % intervals from past errors · no future SST
          ─► residual checks: Ljung-Box (24, 48) · Jarque-Bera · residual ACF · Q-Q
          ─► risk from hour-of-day risk bound (5 / 3.5 real calls/h) · alert windows · CUSUM ─► dashboard
```

## Setup (Windows)

```bat
cd echogaurd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

(macOS / Linux: `python3 -m venv .venv` and `source .venv/bin/activate`.)

## Run

```bat
python run_all.py            :: full pipeline, about 2 minutes
streamlit run app.py         :: opens the dashboard in your browser
python -m pytest -q          :: 19 checks
```

Faster run without the audio stage: `python run_all.py --skip-detector`.

Real, near real-time data from a WHOI buoy (see `data/README.md`):

```bat
python fetch_whoi.py
python run_all.py --skip-detector --hourly-csv data/buoy_hourly.csv
```

If your network cannot reach whoi.edu, let a cloud machine fetch it:

- **GitHub Actions (automatic, hourly).** Push this folder to a GitHub repository. The workflow
  `.github/workflows/live_whoi.yml` runs `fetch_whoi.py` every hour on GitHub's servers and commits
  `data/live_hourly.csv`. Start it once from the Actions tab ("live WHOI data" > Run workflow).
  On the laptop, `git pull` (or download the file) and run
  `python run_all.py --skip-detector --hourly-csv data/live_hourly.csv`.
  The same job then runs the pipeline and publishes the website on GitHub Pages
  (https://bhargav1130.github.io/TSAF_Echogaurd/), so the page always shows the latest forecast.
  One-time setup: Settings > Pages > Source: GitHub Actions.
- **Google Colab (manual).** Open `colab_live_whoi.ipynb` in Colab (File > Upload notebook),
  Run all, put the downloaded `live_hourly.csv` in `data/`, then run the same command.

## What you get in `outputs/`

| File | Contents |
|---|---|
| `summary.json` | every number the pipeline printed |
| `detector_metrics.csv` | PR-AUC, precision, recall of the three detectors |
| `hourly_series.csv` | hourly table: calls, sst_c, mean_prob, present, missing, hour, dow |
| `sarima_aic.csv` | AIC / BIC of all 36 SARIMA order combinations tried (lowest first) |
| `model_metrics.csv` | MAE, RMSE, MASE, 80 % coverage per model (walk-forward average) |
| `model_metrics_per_fold.csv` | the same per fold |
| `forecast_backtest.csv` | last fold: actual, forecast, 80 % interval, risk per hour |
| `forecast_live.csv` | next 72 h after the end of the data |
| `alerts.csv` | merged HIGH-risk windows |
| `figures/*.png` | daily calls vs SST, hourly profile, ACF, STL, MASE, forecast, risk strip, residual diagnostics, PR curves |

## Current results (27 Nov 2026)

Use these numbers when updating the report and paper. 80 % coverage = share of actual hours inside the 80 % band
(target 0.80). Risk = LOW / MEDIUM / HIGH from the risk bound (75 % quantile of forecast + past errors at the same
hour of day), thresholds 5 / 3.5 real calls/h.

| Data | Best model | MAE | MASE | 80 % coverage | Risk check |
|---|---|---|---|---|---|
| **Live** WHOI Cape Cod Bay buoy, 29 Sep - 27 Nov 2026 12:00 UTC, 5 folds | SARIMA(2,0,2) | 6.97 | 2.15 | 87 % | last fold LOW 24 / MEDIUM 6 / HIGH 42 h; all 12 hours with >= 5 calls flagged |
| **Real** Cape Cod Bay 2019 (DCLDE), 28 folds | SARIMAX with SST | 24.55 | 0.60 | 85 % | over all 28 folds: 75 % of HIGH hours were busy, 98 % of busy hours caught |
| **Simulated** 60-day season through MiniRocket (illustration only) | Holt-Winters | 1.01 | 1.22 | 88 % | last fold LOW 54 / MEDIUM 6 / HIGH 12 h |

Detectors on 5000 real Kaggle clips: MiniRocket PR-AUC 0.76 (precision 0.75, recall 0.52), MFCC + logistic
regression 0.60, energy threshold 0.26. On the bursty live data an hourly forecast 72 h ahead is only modestly
better than the average daily pattern (about 39 % of HIGH hours busy against a 29 % base rate); the naive baseline (MAE 6.62)
is as good as the fitted models there, and the live numbers change every hour. On the 2019 season, with a strong daily cycle and steady presence, it works well (MASE 0.60).

## Data

With no data supplied, everything runs on a **simulated** buoy season and
simulated clips, so the numbers illustrate the method. See `data/README.md` to
plug in the real Kaggle clips and buoy series.

**Where.** The real data comes from Cape Cod Bay, Massachusetts (North Atlantic): the live WHOI buoy `maccb2606`
(about 41.95 N, 70.30 W; WHOI does not publish the exact mooring position) and, for 2019, seafloor recorder 1 of 5
(41.934 N, 70.229 W). A forecast covers right whales calling within hearing range of that one recorder, about 15 km
(assumed; 10-20 km depending on noise). It says whether whales are there, not where inside the circle. The website
("Where" section) and the dashboard ("Where" tab) show the map.

## Project layout

```
echoguard/
  config.py     settings (band, horizon, thresholds …)
  audio.py      clip simulation / Kaggle loader, windowing, band-pass, STFT, MFCC, Welch
  detect.py     energy threshold, MFCC + logistic regression, MiniRocket (NumPy implementation)
  series.py     detections → hourly series, gap handling, season simulator, CSV loader
  analysis.py   ADF, KPSS, ACF/PACF, STL strengths, daily profile, SST correlation
  forecast.py   models, walk-forward validation, metrics, Ljung-Box
  risk.py       risk levels, alert windows, CUSUM, advice text
  plots.py      figures
run_all.py      runs the whole pipeline
fetch_whoi.py   downloads real buoy detections (WHOI) + SST (NOAA ERDDAP)
colab_live_whoi.ipynb  same fetch in Google Colab (for networks that block whoi.edu)
.github/workflows/live_whoi.yml  same fetch hourly on GitHub Actions -> data/live_hourly.csv
fetch_dclde.py  real 2019 Cape Cod Bay detections from NOAA Google Cloud + SST (OISST)
app.py          Streamlit dashboard
tests/          pytest checks
```
