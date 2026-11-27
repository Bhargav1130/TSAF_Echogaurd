# data/

The project runs without any downloads: if this folder is empty, `run_all.py`
uses the built-in simulator for both the audio clips and the hourly series.

To use real data:

**Near real-time buoy detections (WHOI Robots4Whales)** – `fetch_whoi.py`
downloads the 15-minute right whale call tallies from a live WHOI DMON buoy,
sums them per UTC hour (hours where the buoy listened < 30 min stay missing),
adds daily NOAA MUR sea-surface temperature from ERDDAP, and writes
`data/buoy_hourly.csv`:

    python fetch_whoi.py                    # Cape Cod Bay buoy, last 60 days
    python run_all.py --skip-detector --hourly-csv data/buoy_hourly.csv

Other buoys: pass `--platform <deployment>/<platform>` from the buoy's
dcs.whoi.edu URL plus `--lat/--lon` for SST. Downloaded days are cached in
`data/whoi_cache/`, so reruns only fetch the newest pages. Notes: counts are
automatic detections (many not yet analyst-reviewed), and the risk thresholds
in `echoguard/config.py` (5 / 3.5 calls/h) are applied to WHOI's counts as they
are, so re-check them for that buoy (or use the dashboard sliders). Some networks cannot reach whoi.edu; use another
network or a VPN if the script says so, or fetch in the cloud: the GitHub Actions workflow
`.github/workflows/live_whoi.yml` (hourly, writes `data/live_hourly.csv`) or the Colab
notebook `colab_live_whoi.ipynb` (see the main README).

**Hourly buoy series** – a CSV with columns `timestamp` (UTC) and `calls`
(up-calls per hour), optionally `sst_c` (daily sea-surface temperature, °C):

    python run_all.py --hourly-csv data/buoy_hourly.csv

Sources: Baumgartner et al. 2019 buoy dataset (Dryad, doi:10.5061/dryad.qt10r04)
and NOAA ERDDAP for sea-surface temperature.

**Labelled clips** – the Kaggle Whale Detection Challenge training set
(30 000 clips of 2 s at 2 kHz, 23 % up-calls). `--kaggle-dir` must point at the
folder that holds `train.csv` and `train/*.aiff`. In this project it is already
unzipped at `whale-detection-challenge/whale_data/data/` (git-ignored):

    pip install soundfile
    python run_all.py --kaggle-dir whale-detection-challenge/whale_data/data --clips 5000

With Kaggle clips the detector's measured recall/precision are also what the
simulated season is passed through (stage 2), so the series is still labelled
SIMULATED. Verified 26 Nov 2026 (5000 clips): MiniRocket PR-AUC 0.760,
precision 0.75, recall 0.52; MFCC logistic 0.599; energy 0.264.
