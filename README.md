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
          ─► risk from upper 80 % bound (5 / 3.5 real calls/h) · alert windows · CUSUM ─► dashboard
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
python -m pytest -q          :: 15 checks
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

## Data

With no data supplied, everything runs on a **simulated** buoy season and
simulated clips, so the numbers illustrate the method. See `data/README.md` to
plug in the real Kaggle clips and buoy series.

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
