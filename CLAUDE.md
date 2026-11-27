# EchoGuard — project context for Claude Code

Handoff notes from a Claude chat session. Read this first.

## What this is
EchoGuard is the team's TSAF (Time Series Analysis and Forecasting, course CS3216) project at RV University.
It detects North Atlantic right whale up-calls in hydrophone-buoy audio, builds an hourly call-count
time series, forecasts it 72 h ahead and turns the forecast into LOW / MEDIUM / HIGH risk for ships
on a Streamlit dashboard.

- Team (exactly these 4 authors): P Venkata Bhargav Reddy (1RUA24CSE0304), Gireesh Patil (1RUA24CSE0145),
  M Vishwanth (1RUA24CSE0233), A Naga Sai Karthik (1RUA24CSE0050).
- Guide (kept from the course template, confirm with the team): Prof. Prashant Adhikari, School of CSE, RV University.
- The user works on a Windows laptop with Python.

## Hard rules from the user
- This must read as a **TSAF project only**. Do NOT add or mention anything from an earlier
  deep-learning project (no CNNs, no Grad-CAM, no "upgraded from DL" framing).
- All results come from a **simulated** 60-day buoy season unless real data is supplied.
  Always label them as simulated; never present them as real measurements.

## How to run (Windows)
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_all.py              :: full pipeline, ~2 min (incl. ~35 s AIC grid), writes outputs/
python run_all.py --skip-detector
streamlit run app.py           :: dashboard
python -m pytest -q            :: 17 tests
python fetch_whoi.py           :: real near real-time WHOI buoy data -> data/buoy_hourly.csv
```
Real data: `--hourly-csv data/buoy_hourly.csv` (timestamp, calls[, sst_c]) and
`--kaggle-dir data/kaggle_whale` (needs `pip install soundfile`). See data/README.md.

## Code map
- `echoguard/config.py` – all settings (band 50-250 Hz, 2-s windows / 1-s hop, horizon 72, 5 folds,
  80% interval, N_CAL 5, HIGH >= 5 / MEDIUM >= 3.5 REAL calls/h on the risk bound (RISK_QUANTILE 0.75), CUSUM k=0.5 h=4, seed 7).
- `audio.py` – simulated labelled clips, Kaggle loader, windowing, Butterworth band-pass, STFT, MFCC (own impl), Welch.
- `detect.py` – Energy threshold, MFCC + logistic regression, MiniRocket (own NumPy implementation, no sktime).
  MiniRocket runs on each window's **time-frequency contour** (dominant frequency + log band power per STFT frame),
  because on the raw waveform it could not tell rising from falling sweeps (PR-AUC 0.42 -> ~0.84).
  Threshold chosen on validation for >= 80% precision.
- `series.py` – detections -> hourly series, gaps kept NaN (<=3 h interpolated), season simulator, CSV loader.
  `apply_detector(h, recall, precision)`: links detector and forecaster. Each true call found with prob = recall,
  plus constant-rate false alarms so precision matches; keeps `calls_true`. run_all applies MiniRocket's test
  recall/precision to the simulated season (from this run, or outputs/detector_metrics.csv with --skip-detector,
  so both give identical results). Real CSV data is not altered (it is already detector output, e.g. WHOI's).
- `analysis.py` – ADF, KPSS, ACF/PACF, STL strengths, hourly profile, SST correlation.
- `forecast.py` – Naive, Seasonal naive, Holt-Winters, SARIMA, SARIMAX with SST; MAE, RMSE, MASE, coverage.
  `model_runner(name, y, exog, h, kw)` -> cached run(origin). SARIMAX gets SST up to the origin and then the LAST
  value held for 72 h (future SST is unknown; same as live). The backtest used to pass true future SST (look-ahead).
  `empirical_interval(run, y, s)`: 80% interval = point forecast + 10%/90% quantiles of the errors of the 5 previous
  72-h forecasts (origins s-72 ... s-360; split-conformal style); also returns err_sd. Used by every model in
  walk_forward and for the live forecast. Replaced Gaussian model intervals, which under-covered (66-74%) because
  SARIMA (d=0) under-forecasts a rising season and counts are skewed; sqrt/log1p transforms did not help.
  `order_search`/`best_order`: AIC grid over SARIMA(p,0,q)(P,0,Q)24, p,q 0-2, P,Q 0-1 (36 fits) on hours before
  the first fold; lowest AIC among converged fits -> SARIMA and SARIMAX. outputs/sarima_aic.csv.
  `residual_diagnostics`: drops 25 start-up residuals; Ljung-Box 24 & 48, Jarque-Bera, skew, kurtosis, ACF,
  verdict; figures/residuals.png. All SARIMAX-class fits use maxiter=200 (default 50 left most fits unconverged).
  Dashboard model = lowest walk-forward MAE among Holt-Winters / SARIMA / SARIMAX.
  Offline hours: models TRAIN on the gap-filled series, but walk_forward(actual=raw calls with NaN) SCORES only
  observed hours: MAE/RMSE/coverage, interval-calibration errors, live calibration and risk.cusum all skip NaN.
  (Before 27 Nov 2026 fold 1's calibration counted the 12 interpolated outage hours as real.)
- `risk.py` – risk from the risk bound (`forecast.risk_bound`: 75% quantile of forecast + past 72-h errors at the
  SAME hour of day, i.e. HIGH = at least 25% chance of the threshold; columns risk_bound / site `rb`). Replaced the
  pooled upper 80% bound on 27 Nov 2026: on bursty live data (57% zero hours) that bound was ~21-27 every hour, so
  risk was HIGH for all 72 h (last fold: 72 HIGH, only 12 of 70 hours had >= 5 calls). Tested over 8 live folds:
  q 0.75 -> HIGH 39% of hours, precision 0.39 (base 0.29), HIGH+MEDIUM catch 72% of >= 5-call hours. Hourly
  72-h-ahead skill is modest (Brier ~ climatology). `thresholds(history, recall, precision)` converts the real-call
  thresholds to the series scale: recall x T + (1 - precision) x mean detections (false alarms); rounded to 0.1
  (= dashboard slider step). HIGH windows merged across gaps <= 1 h. One-sided CUSUM on 72-h forecast errors
  standardised by err_sd (std of past 72-h forecast errors), not by the raw series std. Advice text.
  A threshold relative to the recent mean was tried and rejected: it rises in busy periods and hides alerts.
- `plots.py`, `run_all.py` (orchestrates 5 stages, writes outputs/*.csv, summary.json, figures/), `app.py` (Streamlit;
  `full_width()` helper; slider defaults come from summary.json risk.thresholds), `tests/test_echoguard.py` (17).
- `site/` – interactive showcase website. `template.html` (page + D3 7.8.5 from cdnjs with SRI) and
  `build_site.py`, which injects outputs/ data and writes `echoguard.html` (Artifact body, published at
  https://claude.ai/artifact/C775C6y5j4wpnjGLhD6shn — republish that file to keep the URL) and `index.html`
  (standalone; serve with the `echoguard-site` config in .claude/launch.json, port 8610). Rebuild after any
  run_all: `python site/build_site.py`. Sections: live spectrogram hero (+ WebAudio up-call), pipeline stepper,
  detector lab (real FFT in JS, contour slope), season chart with brush, day x hour map, ACF, AIC grid,
  walk-forward bars/folds, 72-h forecast with threshold sliders/scrubber/play, CUSUM sliders, residual ACF.
- `fetch_dclde.py` – REAL data reachable from the user's network: NOAA NCEI passive-acoustic bucket on Google
  Cloud (gs://noaa-passive-bioacoustic/dclde/2027/dclde2027_narw_detections_in_cape_cod_bay/), Garcia & Tolkova
  et al. 2025 (Endang Species Res 56:101-115). 118 daily Raven selection tables (336 MB, cached in data/dclde_ccb/,
  git-ignored) of automated up-call detections on 5 recorders, 17 Feb - 12 Jun 2019 (archived, not live).
  Time = file start in "Begin File" name + "File Offset (s)". Default: channel 1, score >= 0.75 (paper's threshold:
  precision 0.857, recall 0.896; 495k detections >= 0.75 vs paper's 485,880). One channel = one buoy (no double
  counting). SST from NOAA OISST v2.1 (MUR ERDDAP does not answer for 2019 within minutes). Output
  data/buoy_hourly.csv: 2757 h, 115,564 calls. Run: `python run_all.py --skip-detector --hourly-csv
  data/buoy_hourly.csv --detector-rates 0.896 0.857` (new flag: converts the 5 / 3.5 real-call thresholds).
  Don't describe the source detector's architecture (hard rule: no deep-learning framing); call it "the
  dataset's automated up-call detector".
  Real-data folds: run_all spreads them for any --hourly-csv of >= 75 days: every 72-h block from day 30
  (here 28 folds, hour 720-2736: arrival, peak, departure). With only the last 5 folds (June, whales gone,
  1.8 calls/h vs 88.7 peak) Naive won; that test was meaningless. summary.json now has "folds" and
  "detector_rates"; data/buoy_hourly.source.json (from fetch_dclde) gives the site its source note.

## Verified results — LIVE WHOI Cape Cod Bay (CURRENT outputs/, 27 Nov 2026)
- Data: GitHub Actions run 45996739028 (12:12 UTC) -> data/live_hourly.csv, 29 Sep - 27 Nov 2026 08:00 UTC,
  1421 h, 4593 calls, 17 h offline. Partly heard hours scaled to calls/h, unfinished newest hour dropped.
  Run #2 failed on a NOAA MUR SST 503; fetch_whoi.get_sst now falls back to OISST, then the previous CSV's SST.
  Command: `python run_all.py --skip-detector --hourly-csv data/live_hourly.csv` (50 s; counts used as-is, rates 1/1).
- ADF 0.001 / KPSS 0.01, STL seasonal 0.22 / trend 0.13, ACF24 0.250, peak 01:00 UTC, daily calls vs SST r = -0.65.
- AIC best SARIMA(2,0,2)(0,0,0)24. 5 folds from hour 1061: SARIMA MAE 7.09 MASE 2.18 cov 82% | SARIMAX 8.24 / 66% |
  Seasonal naive 8.86 / 72% | HW 10.38 / 51% | Naive 13.54 / 61%. MASE > 1: bursty calls.
- Live forecast from 27 Nov 09:00 UTC with the risk bound: LOW 26 / MEDIUM 3 / HIGH 43 h, HIGH windows around the
  night-time call peak each day. Backtest last fold: LOW 23 / MEDIUM 6 / HIGH 43, every >= 5-call hour flagged
  HIGH or MEDIUM, HIGH precision 0.28. Last fold 67/72 inside band. (Old upper-bound rule: HIGH all 72 h.)
- Interval fix (27 Nov): empirical_interval floors lo / ceils hi (integer counts); before, SARIMA covered 41%.
  The 2019 numbers below predate it; 2019 outputs backed up in the scratchpad (real2019_outputs).
- Website v6 = this run; hero "risk next hour" now uses the live forecast (was the backtest fold).
  Hourly workflow also runs run_all + build_site and deploys site/index.html to GitHub Pages
  (https://bhargav1130.github.io/TSAF_Echogaurd/; needs Settings > Pages > Source: GitHub Actions).
  The Claude Artifact copy still needs a manual republish. With --skip-detector, run_all now reads
  outputs/detector_metrics.csv (kept in git, has a source column) so the detector section is filled.

## Verified results — REAL Cape Cod Bay 2019 (main results until 27 Nov 2026)
- Command: `python run_all.py --kaggle-dir whale-detection-challenge/whale_data/data --clips 5000
  --hourly-csv data/buoy_hourly.csv --detector-rates 0.896 0.857` (302 s). Detector stage = MiniRocket on Kaggle.
- 2757 h, 115,564 calls, 0 missing. ADF 0.018 / KPSS 0.01 -> d=0. STL seasonal 0.52, trend 0.53. ACF 24/48/72 =
  0.642 / 0.614 / 0.568. Peak 23:00 UTC (116 calls/h), low 16:00 (20). Daily calls vs SST r = -0.70 (whales
  leave the bay as it warms; the simulator assumed the opposite).
- AIC (first 720 h): best SARIMA(1,0,0)(1,0,1)24, AIC 7442.7.
- 28 folds: SARIMAX (SST) MAE 24.55 RMSE 31.79 MASE 0.60 cov 83% | Seasonal naive 28.39 / 38.82 / 0.70 / 87% |
  SARIMA 28.41 / 35.43 / 0.70 / 84% | Naive 33.14 / 42.90 / 0.82 / 78% | Holt-Winters 41.46 / 53.48 / 1.02 / 81%.
  On real data SST helps clearly; dashboard model = SARIMAX.
- Thresholds 10.7 / 9.3 det/h (= 5 / 3.5 real calls/h). Backtest = last fold (issued 08 Jun 2019 18:00 UTC,
  whales gone): LOW 72 h, 70/72 inside band, CUSUM alarm +10 h. Peak-season folds are only visible in the fold chart.
- Website republished (version 4) from this run. Kaggle-run outputs are backed up in the scratchpad.
- `fetch_whoi.py` – real data: scrapes WHOI Robots4Whales daily tally pages (15-min right whale counts;
  the row anchor id is the UTC time, visible text is EST), sums per UTC hour, <30 min listening -> NaN,
  adds NOAA MUR SST (ERDDAP jplMURSST41), caches days in data/whoi_cache/. Default buoy: Cape Cod Bay
  `maccb2606/maccb2606_maccb` (deployed 16 Jun 2026). Counts are automatic, often not analyst-reviewed.
  dcs.whoi.edu was unreachable from the user's home network (DNS + route); ERDDAP works.
  Parser tested with real markup; live WHOI fetch not yet run from this machine.
  Cloud fetch (whoi.edu blocked here): `.github/workflows/live_whoi.yml` (hourly cron, commits
  data/live_hourly.csv; needs the folder pushed to a GitHub repo) and `colab_live_whoi.ipynb`
  (upload or clone fetch_whoi.py, run, download live_hourly.csv). Neither run yet. Live file is
  data/live_hourly.csv so the 2019 data/buoy_hourly.csv is kept. 60 days < 75 -> last 5 folds.

## Verified results (simulated season seen through MiniRocket: recall 0.67, precision 0.79)
- 1551 true calls -> 1296 detections. ADF p=0.020, KPSS p<=0.01 -> stationary around a moving level (d=0).
  STL seasonal 0.33, trend 0.25. ACF 24/48/72 = 0.341 / 0.249 / 0.197. Peak 01:00 UTC 1.78 det/h, low 14:00 0.41.
  Daily detections vs SST r = 0.84.
- AIC grid (first 1056 h): best SARIMA(1,0,1)(1,0,1)24, AIC 2867.9 / BIC 2897.7; all top 5 converged.
- Walk-forward 5 x 72 h (folds from hour 44*24), empirical 80% intervals:
  Holt-Winters MAE 1.01 RMSE 1.33 MASE 1.22 cov 85% | SARIMAX 1.04 / 1.34 / 1.27 cov 76% |
  SARIMA 1.05 / 1.37 / 1.27 cov 82% | Seasonal naive 1.29 / 1.79 / 1.57 cov 85% | Naive 1.80 / 2.10 / 2.19 cov 76%.
  With no look-ahead, SST adds almost nothing over SARIMA; Holt-Winters is the dashboard model.
- Residuals (Holt-Winters, last-fold fit, 1319 h): Ljung-Box p < 0.001 (24, 48); Jarque-Bera p < 0.001,
  skew 1.01, excess kurtosis 3.10 -> some structure left, skewed errors (handled by the empirical interval).
- Detectors (simulated clips): MiniRocket PR-AUC 0.836 (P 0.79, R 0.67), MFCC logistic 0.62, energy 0.33.
- Detectors on REAL Kaggle clips (5000 of 30 000, 22% up-calls; `--kaggle-dir whale-detection-challenge/whale_data/data`,
  data in the project folder, git-ignored, soundfile 0.14 installed in .venv): MiniRocket PR-AUC 0.760 (P 0.75, R 0.52),
  MFCC logistic 0.599 (R 0.04), energy 0.264. Full run 191 s. Season through those rates: thresholds 2.8 / 2.0 det/h,
  HW best (MAE 0.93, cov 82%; SARIMA 0.95 / 79%, SARIMAX 0.95 / 69%), LOW 0 / MEDIUM 51 / HIGH 21, CUSUM +27 h. The numbers above are the simulated-clip run.
- Former main run (26 Nov 2026) = the Kaggle-clip run on the simulated season. Reproduce with
  `python run_all.py --kaggle-dir whale-detection-challenge/whale_data/data --clips 5000`; `--skip-detector` reuses
  its saved rates. Plain `python run_all.py` reruns on simulated clips and overwrites it.
- Dashboard last fold (issued 26 Apr 2016 00:00 UTC): thresholds HIGH >= 3.5, MEDIUM >= 2.5 detections/h
  (= 5 / 3.5 real calls/h); 34 detections last 24 h, ~96 expected next 72 h; LOW 3 / MEDIUM 57 / HIGH 12 h;
  HIGH windows +1-5, +25-29, +49-53 h; CUSUM (sigma 1.23) alarm at +25 h; 53/72 inside band.
- History of changes: earlier numbers came from (a) unconverged fits, (b) SARIMAX seeing true future SST,
  (c) Gaussian intervals, (d) true counts instead of detector output.

## Known mismatches with the written report / paper
- Every forecasting and risk number in the report/paper is outdated (they say 14/48/10, 183 calls, 44/72,
  SARIMA best). Current: see "Verified results". The method also changed: AIC order search, empirical
  intervals, no SST look-ahead, detector-in-the-loop series, residual diagnostics, CUSUM sigma.
- The documents describe MiniRocket on the raw window; the code runs it on the frequency contour.
- The documents contain no detector accuracy numbers.

## Deliverables made in the chat (not in this folder; they were sent as chat attachments)
- Project report in the team's CIE-3 .docx template (cover, certificate, TOC, chapters 1-8, 10 figures).
- IEEE conference paper (.docx + .pdf, 4 authors) and an Overleaf zip (IEEEtran main.tex + figures/).
- An earlier Claude Doc version of the report.

## Housekeeping
- `_to_delete/` holds an old zip and Python cache folders; safe to delete.
- `.venv/` and `outputs/` are git-ignored.
