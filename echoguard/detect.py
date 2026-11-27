"""Up-call detectors: energy threshold, MFCC logistic regression and MiniRocket.

MiniRocket follows Dempster, Schmidt & Webb (KDD 2021): 84 fixed length-9
kernels (three weights of +2, six of -1) at several dilations, biases taken
from quantiles of the convolution output, and PPV (proportion of positive
values) features, followed by a linear classifier. This is a compact NumPy
implementation so the project needs no heavy extra packages (no sktime).
"""
from itertools import combinations

import numpy as np
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import audio
from . import config as C


class EnergyDetector:
    """Baseline: score = log energy in the 50-250 Hz band. No training."""
    name = "Energy threshold"

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        e = audio.band_energy(X)
        return 1 / (1 + np.exp(-(e - e.mean()) / (e.std() + 1e-9)))  # squash to 0-1


class MFCCLogistic:
    """Baseline: logistic regression on MFCC summary features."""
    name = "Logistic regression (MFCC)"

    def __init__(self):
        self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))

    def fit(self, X, y):
        self.model.fit(audio.mfcc_features(audio.bandpass(X)), y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(audio.mfcc_features(audio.bandpass(X)))[:, 1]


class MiniRocketDetector:
    """Main detector: MiniRocket on the window's time-frequency contour.

    Each band-passed 2-s window is turned into a 2-channel time series with one
    value per STFT frame: the dominant frequency in 50-250 Hz and the log band
    power. An up-call is a *rising* frequency track, which the MiniRocket
    kernels pick up as a local slope pattern. MiniRocket features are computed
    per channel, concatenated and fed to a regularised logistic regression.
    """
    name = "MiniRocket"

    def __init__(self, n_dilations=4, quantiles=(0.25, 0.5, 0.75), seed=C.SEED):
        self.n_dilations, self.quantiles, self.seed = n_dilations, quantiles, seed
        self.kernels = list(combinations(range(9), 3))       # 84 kernels
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.01))

    # -- input series
    @staticmethod
    def contour(X, sr=C.SAMPLE_RATE):
        """(n, 2, frames): dominant-frequency track and log band power."""
        out = []
        for x in audio.bandpass(X, sr):
            f, _, S = signal.spectrogram(x, fs=sr, nperseg=256, noverlap=224)
            keep = (f >= C.BAND[0]) & (f <= C.BAND[1])
            S, fb = S[keep], f[keep]
            peak = fb[S.argmax(0)]
            logp = np.log(S.sum(0) + 1e-12)
            mid, half = (C.BAND[0] + C.BAND[1]) / 2, (C.BAND[1] - C.BAND[0]) / 4
            out.append(np.stack([(peak - mid) / half, (logp - logp.mean()) / (logp.std() + 1e-9)]))
        return np.asarray(out, dtype=np.float32)

    # -- MiniRocket transform (univariate, applied per channel)
    @staticmethod
    def _shifted(X, d):
        """Nine shifted copies of X for a length-9 kernel at dilation d (zero padded)."""
        L = X.shape[1]
        pad = 4 * d
        Xp = np.pad(X, ((0, 0), (pad, pad)))
        return [Xp[:, k * d: k * d + L] for k in range(9)]

    def _conv_all(self, X, d):
        S = self._shifted(X, d)
        base = -sum(S)                     # all nine weights at -1
        for idx in self.kernels:           # then add +3 at the three chosen taps (weight +2)
            yield base + 3 * (S[idx[0]] + S[idx[1]] + S[idx[2]])

    def _fit_params(self, Z):
        L = Z.shape[2]
        max_exp = np.log2((L - 1) / 8)
        self.dilations = np.unique(np.floor(2 ** np.linspace(0, max_exp, self.n_dilations)).astype(int))
        rng = np.random.default_rng(self.seed)
        self.biases = {}
        for ch in range(Z.shape[1]):
            for d in self.dilations:
                ex = Z[rng.integers(len(Z)), ch][None, :]
                self.biases[ch, d] = [np.quantile(c[0], self.quantiles) for c in self._conv_all(ex, d)]

    def transform(self, Z):
        feats = []
        for ch in range(Z.shape[1]):
            X = Z[:, ch]
            for d in self.dilations:
                for c, b in zip(self._conv_all(X, d), self.biases[ch, d]):
                    feats.append((c[:, :, None] > b[None, None, :]).mean(axis=1))   # PPV
        return np.concatenate(feats, axis=1).astype(np.float32)

    def fit(self, X, y):
        Z = self.contour(X)
        self._fit_params(Z)
        self.clf.fit(self.transform(Z), y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(self.transform(self.contour(X)))[:, 1]


def choose_threshold(y_true, prob, target_precision=C.TARGET_PRECISION):
    """Lowest threshold whose precision reaches the target (max recall at that precision)."""
    p, r, t = precision_recall_curve(y_true, prob)
    ok = np.where(p[:-1] >= target_precision)[0]
    if len(ok) == 0:
        return 0.5
    best = ok[np.argmax(r[:-1][ok])]
    return float(t[best])


def split_indices(n, y, seed=C.SEED, frac=(0.6, 0.2, 0.2)):
    """Stratified train / validation / test split of clip indices."""
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        a, b = int(frac[0] * len(idx)), int((frac[0] + frac[1]) * len(idx))
        parts[0] += list(idx[:a]); parts[1] += list(idx[a:b]); parts[2] += list(idx[b:])
    return [np.array(sorted(p)) for p in parts]


def evaluate_detectors(X, y, detectors=None):
    """Fit each detector on train, pick threshold on validation, score on test."""
    detectors = detectors or [EnergyDetector(), MFCCLogistic(), MiniRocketDetector()]
    tr, va, te = split_indices(len(X), y)
    rows, fitted = [], {}
    for det in detectors:
        det.fit(X[tr], y[tr])
        thr = choose_threshold(y[va], det.predict_proba(X[va]))
        prob = det.predict_proba(X[te])
        pred = (prob >= thr).astype(int)
        rows.append({"detector": det.name, "pr_auc": round(float(average_precision_score(y[te], prob)), 3),
                     "threshold": round(thr, 3),
                     "precision": round(float(precision_score(y[te], pred, zero_division=0)), 3),
                     "recall": round(float(recall_score(y[te], pred, zero_division=0)), 3)})
        fitted[det.name] = (det, thr)
    return rows, fitted
