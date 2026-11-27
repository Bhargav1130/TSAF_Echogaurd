"""Audio side of the pipeline: clip loading/simulation, filtering and features."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import dct

from . import config as C


# ----------------------------------------------------------------- clips
def simulate_clips(n=2000, pos_frac=0.3, sr=C.SAMPLE_RATE, sec=C.WINDOW_SEC, seed=C.SEED):
    """Labelled 2-s clips that mimic the Kaggle data: rising up-calls in ocean noise.

    Negatives contain only noise, ship tones or down-sweeps (distractors),
    so the detector has to learn the *rising* shape, not just energy.
    """
    rng = np.random.default_rng(seed)
    n_samp = int(sr * sec)
    t = np.arange(n_samp) / sr
    X = np.empty((n, n_samp), dtype=np.float32)
    y = (rng.random(n) < pos_frac).astype(int)
    for i in range(n):
        # coloured background noise (more energy at low frequency) + optional ship tone
        noise = signal.lfilter([1.0], [1.0, -0.9], rng.normal(0, 1, n_samp))
        noise /= noise.std()
        x = noise
        if rng.random() < 0.5:
            f_ship = rng.uniform(20, 60)
            x = x + rng.uniform(0.3, 1.5) * np.sin(2 * np.pi * f_ship * t + rng.uniform(0, 6.28))
        dur = rng.uniform(0.6, 1.2)
        start = rng.uniform(0.1, sec - dur - 0.1)
        f0, f1 = rng.uniform(60, 100), rng.uniform(150, 220)
        snr = rng.uniform(0.25, 1.2)
        if y[i] == 1:                       # up-call: rising sweep
            x = x + snr * _sweep(t, start, dur, f0, f1)
        elif rng.random() < 0.35:           # distractor: falling sweep
            x = x + snr * _sweep(t, start, dur, f1, f0)
        X[i] = x / np.abs(x).max()
    return X, y


def _sweep(t, start, dur, f0, f1):
    m = (t >= start) & (t < start + dur)
    tt = t[m] - start
    phase = 2 * np.pi * (f0 * tt + 0.5 * (f1 - f0) / dur * tt ** 2)
    out = np.zeros_like(t)
    out[m] = np.sin(phase) * np.hanning(m.sum())
    return out


def load_kaggle_clips(folder, max_clips=None):
    """Load the Kaggle Whale Detection Challenge training set if present.

    Expects <folder>/train.csv (clip_name,label) and <folder>/train/*.aiff.
    Needs the optional `soundfile` package to read .aiff files.
    """
    folder = Path(folder)
    labels = pd.read_csv(folder / "train.csv")
    if max_clips:
        labels = labels.sample(n=min(max_clips, len(labels)), random_state=C.SEED)
    import soundfile as sf  # optional dependency
    n_samp = int(C.SAMPLE_RATE * C.WINDOW_SEC)
    X = np.zeros((len(labels), n_samp), dtype=np.float32)
    for i, name in enumerate(labels.clip_name):
        a, _ = sf.read(folder / "train" / name, dtype="float32")
        a = a[:n_samp]
        X[i, :len(a)] = a / (np.abs(a).max() + 1e-9)
    return X, labels.label.to_numpy().astype(int)


# --------------------------------------------------------------- signal
def window_signal(x, sr=C.SAMPLE_RATE, win=C.WINDOW_SEC, hop=C.HOP_SEC):
    """Cut a long recording into overlapping windows -> (n_windows, win*sr)."""
    w, h = int(win * sr), int(hop * sr)
    if len(x) < w:
        return np.empty((0, w), dtype=np.float32)
    starts = np.arange(0, len(x) - w + 1, h)
    return np.stack([x[s:s + w] for s in starts]).astype(np.float32)


def bandpass(X, sr=C.SAMPLE_RATE, band=C.BAND, order=4):
    """4th-order Butterworth band-pass, applied forward-backward (zero phase)."""
    sos = signal.butter(order, band, btype="bandpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, X, axis=-1).astype(np.float32)


def spectrogram(x, sr=C.SAMPLE_RATE, nperseg=256, noverlap=192):
    """STFT power spectrogram limited to the up-call band."""
    f, t, S = signal.spectrogram(x, fs=sr, nperseg=nperseg, noverlap=noverlap)
    keep = (f >= C.BAND[0]) & (f <= C.BAND[1])
    return f[keep], t, S[keep]


def welch_noise_level(x, sr=C.SAMPLE_RATE):
    """Median in-band power from a Welch spectrum (background noise floor, dB)."""
    f, P = signal.welch(x, fs=sr, nperseg=512)
    keep = (f >= C.BAND[0]) & (f <= C.BAND[1])
    return float(10 * np.log10(np.median(P[keep]) + 1e-12))


def _mel_filterbank(n_mels, n_fft, sr, fmin, fmax):
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = inv(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            fb[m - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[m - 1, k] = (r - k) / max(r - c, 1)
    return fb


def mfcc_features(X, sr=C.SAMPLE_RATE, n_mels=24, n_mfcc=13, n_fft=256, hop=64):
    """MFCC summary features per clip: mean, std and slope of each coefficient,
    plus the trend of the spectral centroid (rising for up-calls)."""
    fb = _mel_filterbank(n_mels, n_fft, sr, C.BAND[0], C.BAND[1])
    feats = []
    for x in X:
        f, t, Z = signal.stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        P = np.abs(Z) ** 2
        logmel = np.log(fb @ P + 1e-10)
        M = dct(logmel, type=2, axis=0, norm="ortho")[:n_mfcc]
        tt = np.arange(M.shape[1])
        slope = np.polyfit(tt, M.T, 1)[0]
        band = (f >= C.BAND[0]) & (f <= C.BAND[1])
        Pb = P[band]
        cent = (f[band, None] * Pb).sum(0) / (Pb.sum(0) + 1e-12)
        w = Pb.sum(0)
        top = w >= np.quantile(w, 0.75)            # loudest frames only
        cent_slope = np.polyfit(tt[top], cent[top], 1)[0] if top.sum() > 2 else 0.0
        feats.append(np.concatenate([M.mean(1), M.std(1), slope, [cent_slope]]))
    return np.asarray(feats, dtype=np.float32)


def band_energy(X, sr=C.SAMPLE_RATE):
    """Log energy in the up-call band (input to the energy-threshold baseline)."""
    Xf = bandpass(X, sr)
    return np.log((Xf ** 2).mean(axis=1) + 1e-12)
