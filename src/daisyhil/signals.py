"""Stimulus generators.

Every stimulus is deterministic (seeded) so a capture can be reproduced from a
config file alone, and every generator leaves pre/post roll silence so the
analysis code can estimate the noise floor and the round-trip latency.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

__all__ = [
    "impulse",
    "sine",
    "exponential_sweep",
    "inverse_sweep",
    "noise",
    "silence",
    "pad",
    "rms",
    "peak",
    "dbfs",
    "dbfs_peak",
    "amplitude",
]


def amplitude(level_dbfs: float) -> float:
    return float(10.0 ** (level_dbfs / 20.0))


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def peak(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.max(np.abs(x))) if x.size else 0.0


def dbfs(x: np.ndarray) -> float:
    value = rms(x)
    return -np.inf if value <= 0 else float(20.0 * np.log10(value))


def dbfs_peak(x: np.ndarray) -> float:
    value = peak(x)
    return -np.inf if value <= 0 else float(20.0 * np.log10(value))


def _fade(x: np.ndarray, sr: int, fade_ms: float) -> np.ndarray:
    n = int(round(fade_ms / 1000.0 * sr))
    if n <= 0 or 2 * n >= len(x):
        return x
    window = np.ones(len(x))
    ramp = np.sin(np.linspace(0, np.pi / 2, n)) ** 2
    window[:n] = ramp
    window[-n:] = ramp[::-1]
    return x * window


def pad(x: np.ndarray, sr: int, pre_roll_s: float, post_roll_s: float) -> np.ndarray:
    return np.concatenate(
        [np.zeros(int(pre_roll_s * sr)), np.asarray(x, dtype=np.float64), np.zeros(int(post_roll_s * sr))]
    )


def impulse(sr: int, tail_s: float, level: float = 1.0) -> np.ndarray:
    """A single band-limited click followed by silence.

    Band limiting keeps the click from aliasing in the converters; the analysis
    only needs a compact, repeatable transient with a stable peak.
    """
    n = int(tail_s * sr)
    x = np.zeros(n)
    x[0] = level
    sos = butter(4, 0.45, btype="lowpass", output="sos")
    x = sosfilt(sos, x)
    # normalise so the dry peak is exactly ``level`` (keeps dB maths predictable)
    x /= peak(x) or 1.0
    return x * level


def sine(
    sr: int,
    freq_hz: float,
    duration_s: float,
    level: float = 0.25,
    fade_ms: float = 20.0,
) -> np.ndarray:
    t = np.arange(int(duration_s * sr)) / sr
    x = level * np.sin(2 * np.pi * freq_hz * t)
    return _fade(x, sr, fade_ms)


def exponential_sweep(
    sr: int,
    f0_hz: float,
    f1_hz: float,
    duration_s: float,
    level: float = 0.25,
    fade_ms: float = 5.0,
) -> np.ndarray:
    """Farina exponential sine sweep (constant amplitude, log time axis)."""
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    T = n / sr
    k = (2 * np.pi * f0_hz * T) / np.log(f1_hz / f0_hz)
    x = level * np.sin(k * (np.exp(t / T * np.log(f1_hz / f0_hz)) - 1.0))
    return _fade(x, sr, fade_ms)


def inverse_sweep(sweep: np.ndarray, sr: int, f0_hz: float, f1_hz: float) -> np.ndarray:
    """Amplitude-modulated time reverse of ``sweep``: its deconvolution filter.

    The 6 dB/octave modulation flattens the raw |X(f)|^2 tilt of a swept sine.
    The rig additionally deconvolves the *stimulus* as a reference and divides,
    so any residual tilt cancels out of the measured response.
    """
    n = len(sweep)
    t = np.arange(n) / sr
    T = n / sr
    modulation = np.exp(-t / T * np.log(f1_hz / f0_hz))
    return sweep[::-1] * modulation


def noise(sr: int, duration_s: float, level: float = 0.25, seed: int = 1234) -> np.ndarray:
    """Decorrelated white noise - the stutter test needs no repeating content."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(duration_s * sr))
    return level * x / (peak(x) or 1.0)


def silence(sr: int, duration_s: float) -> np.ndarray:
    return np.zeros(int(duration_s * sr))
