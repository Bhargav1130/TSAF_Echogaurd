"""Project-wide settings for EchoGuard."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

# ---- audio / detection ----
SAMPLE_RATE = 2000          # Hz (Kaggle clips and buoy audio)
WINDOW_SEC = 2.0            # analysis window length
HOP_SEC = 1.0               # hop between windows
BAND = (50.0, 250.0)        # right whale up-call band, Hz
TARGET_PRECISION = 0.80     # detector threshold is chosen to reach this precision

# ---- hourly series ----
PRESENT_MIN_CALLS = 2       # an hour is "present" if at least this many calls
MAX_INTERP_GAP_H = 3        # buoy gaps up to this length are interpolated

# ---- forecasting ----
SEASON = 24                 # daily cycle in hours
HORIZON = 72                # forecast horizon in hours
N_FOLDS = 5                 # walk-forward folds
INTERVAL_ALPHA = 0.20       # 80 % prediction interval
N_CAL = 5                   # past 72-h forecasts used to calibrate the interval

# ---- risk ----
# In REAL calls per hour. When the series is what a detector reports, risk.thresholds()
# converts them with the detector's recall and precision.
HIGH_THRESHOLD = 5.0        # upper bound >= this -> HIGH
MEDIUM_THRESHOLD = 3.5      # upper bound >= this -> MEDIUM
CUSUM_K = 0.5               # CUSUM slack (in units of the std of past 72-h forecast errors)
CUSUM_H = 4.0               # CUSUM alarm level

SEED = 7
