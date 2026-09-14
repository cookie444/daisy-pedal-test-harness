"""Measurement algorithms.

Each function takes a capture (and, where needed, its stimulus) and returns a
:class:`Measurement`: the headline value plus a ``details`` dict carrying
everything the report plots and everything a failing engineer would want to
inspect. Every function is pure - no hardware, no globals - so the algorithms
themselves are unit-testable against synthetic ground truth
(see ``tests/unit/test_measurements.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import fftconvolve, find_peaks
from scipy.signal.windows import blackmanharris

from .signals import dbfs, inverse_sweep, rms

__all__ = [
    "Measurement",
    "estimate_floor",
    "find_impulses",
    "delay_and_decay",
    "thd",
    "frequency_response",
    "stutter",
    "noise_floor",
    "impulse_arrival",
]


@dataclass
class Measurement:
    value: float
    unit: str = ""
    details: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def estimate_floor(x: np.ndarray, sr: int, window_s: float = 0.1, start_s: float = 0.0) -> float:
    """Noise floor (RMS) estimated from a silent region at the start of a capture."""
    a = int(start_s * sr)
    b = int((start_s + window_s) * sr)
    segment = np.asarray(x)[a:max(a + 1, b)]
    return rms(segment)


def _parabolic_peak(mag: np.ndarray, index: int) -> tuple[float, float]:
    """Refine a spectral peak: returns (fractional_bin_offset, interpolated_mag)."""
    if index <= 0 or index >= len(mag) - 1:
        return 0.0, float(mag[index])
    a, b, c = float(mag[index - 1]), float(mag[index]), float(mag[index + 1])
    denom = a - 2 * b + c
    offset = 0.5 * (a - c) / denom if denom != 0 else 0.0
    offset = float(np.clip(offset, -0.5, 0.5))
    return offset, float(b - 0.25 * (a - c) * offset)


def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return np.asarray(x, dtype=np.float64)
    kernel = np.ones(n) / n
    return np.convolve(np.asarray(x, dtype=np.float64), kernel, mode="same")


def _edge_window(length: int, sr: int, fade_ms: float) -> np.ndarray:
    """Flat window with raised-cosine edges - isolates an IR without shaping it."""
    n = int(round(fade_ms / 1000.0 * sr))
    window = np.ones(length)
    if n > 0 and 2 * n < length:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
        window[:n] = ramp
        window[-n:] = ramp[::-1]
    return window


def _log_freq_grid(f_lo: float, f_hi: float, points_per_octave: int = 24) -> np.ndarray:
    octaves = np.log2(f_hi / f_lo)
    n = int(round(octaves * points_per_octave)) + 1
    return np.logspace(np.log2(f_lo), np.log2(f_hi), num=n, base=2)


# --------------------------------------------------------------------------- #
# delay + repeat decay
# --------------------------------------------------------------------------- #


def find_impulses(
    x: np.ndarray,
    sr: int,
    *,
    floor: float | None = None,
    floor_margin: float = 6.0,
    min_separation_ms: float = 1.0,
    dynamic_range_db: float = 65.0,
    contrast: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate decaying impulses in a capture.

    Returns ``(times_ms, amplitudes)``. The detection floor is the larger of the
    measured noise floor (times ``floor_margin``) and ``dynamic_range_db`` below
    the largest impulse, so a quiet pedal is still measured over a usable range
    while a noisy one does not produce phantom repeats.

    Height alone is not enough: every DC-blocking filter leaves a slow
    exponential tail after a click, and thresholding reports that tail (and the
    noise wiggling on top of it) as a repeat. So a candidate also has to stand
    out from its immediate surroundings by ``contrast`` - a real repeat rises
    out of silence, a filter tail does not.
    """
    x = np.asarray(x, dtype=np.float64)
    env = np.abs(x)
    if floor is None:
        floor = estimate_floor(x, sr)
    height = max(floor * floor_margin, env.max() * 10 ** (-dynamic_range_db / 20.0), 1e-9)
    distance = max(1, int(min_separation_ms / 1000.0 * sr))
    candidates, _ = find_peaks(env, height=height, distance=distance)

    hold = max(1, int(0.001 * sr))  # ignore the click's own decay
    window = max(hold + 1, min(int(0.006 * sr), max(2, int(0.3 * distance))))
    keep = []
    for index in candidates:
        before = x[max(0, index - window) : max(0, index - hold)]
        after = x[min(len(x), index + hold) : min(len(x), index + window)]
        background = max(rms(before), rms(after), floor)
        if env[index] > contrast * background:
            keep.append(index)

    indices = np.asarray(keep, dtype=int)
    return indices / sr * 1000.0, env[indices]


def delay_and_decay(
    cap: np.ndarray,
    sr: int,
    *,
    pre_roll_s: float,
    expect_dry: bool = True,
    min_delay_ms: float = 1.0,
    floor_margin: float = 6.0,
) -> tuple[Measurement, Measurement]:
    """Measure delay time and repeat decay from a single click capture.

    Returns ``(delay_ms, decay_db_per_repeat)``. ``decay_time_ms`` (time to
    -60 dB) and the fit quality ride along in the details.
    """
    cap = np.asarray(cap, dtype=np.float64)
    floor = estimate_floor(cap, sr, start_s=max(0.0, pre_roll_s * 0.2))
    times_ms, amps = find_impulses(
        cap, sr, floor=floor, floor_margin=floor_margin, min_separation_ms=min_delay_ms * 0.4
    )

    dry_time = None
    if expect_dry and times_ms.size >= 2:
        dry_time = float(times_ms[0])
        times_ms, amps = times_ms[1:], amps[1:]

    if times_ms.size < 2:
        details = {"times_ms": times_ms, "amps": amps, "floor": floor, "n_repeats": int(times_ms.size)}
        return Measurement(float("nan"), "ms", details), Measurement(float("nan"), "dB", details)

    gaps = np.diff(times_ms)
    delay_ms = float(np.median(gaps))
    jitter_ms = float(np.max(np.abs(gaps - delay_ms))) if gaps.size else 0.0

    # level fit over the repeats that stay clear of the noise floor
    usable = amps > max(floor * floor_margin, amps[0] * 10 ** (-65.0 / 20.0))
    idx = np.arange(len(amps))[usable]
    if idx.size < 2:
        slope, r2, n_used = float("nan"), float("nan"), int(idx.size)
    else:
        db = 20.0 * np.log10(amps[idx])
        slope, intercept = np.polyfit(idx, db, 1)
        fitted = slope * idx + intercept
        ss_res = float(np.sum((db - fitted) ** 2))
        ss_tot = float(np.sum((db - db.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        n_used = int(idx.size)

    decay_db = float(slope) if slope == slope else float("nan")
    if decay_db < 0:
        repeats_to_60db = 60.0 / abs(decay_db)
        decay_time_ms = repeats_to_60db * delay_ms
    else:
        repeats_to_60db = float("nan")
        decay_time_ms = float("nan")

    details = {
        "times_ms": times_ms,
        "amps": amps,
        "dry_time_ms": dry_time,
        "floor": floor,
        "floor_dbfs": float(20 * np.log10(floor)) if floor > 0 else -np.inf,
        "gaps_ms": gaps,
        "jitter_ms": jitter_ms,
        "n_repeats": int(times_ms.size),
        "fit_repeats": n_used,
        "fit_r2": float(r2) if r2 == r2 else float("nan"),
        "decay_time_ms": float(decay_time_ms),
        "repeats_to_60db": float(repeats_to_60db),
        "first_repeat_dbfs": dbfs(np.array([amps[0] / np.sqrt(2)])) if amps.size else -np.inf,
    }
    return Measurement(delay_ms, "ms", details), Measurement(decay_db, "dB", details)


# --------------------------------------------------------------------------- #
# distortion
# --------------------------------------------------------------------------- #


def thd(
    cap: np.ndarray,
    sr: int,
    f0_hz: float,
    *,
    start_s: float = 0.1,
    tail_s: float = 0.05,
    max_harmonic: int = 10,
    search_tol_hz: float = 5.0,
) -> Measurement:
    """THD, THD+N and SNR of a steady sine capture (Blackman-Harris windowed).

    ``start_s`` must clear the pre-roll *and* one delay time: before the wet
    path joins in, the capture is not yet the steady state worth measuring.
    """
    cap = np.asarray(cap, dtype=np.float64)
    start = int(start_s * sr)
    end = max(start + sr // 10, len(cap) - int(tail_s * sr))
    if end - start < sr // 10:
        raise ValueError(f"capture too short for start_s={start_s}s (len={len(cap) / sr:.2f}s)")
    segment = cap[start:end]
    n = len(segment)
    # 4-term Blackman-Harris: -92 dB sidelobes, so a truncated record does not
    # leak the fundamental's skirts into the noise term (a Hann window does).
    window = blackmanharris(n)
    spectrum = np.fft.rfft(segment * window)
    mag = np.abs(spectrum) / (np.sum(window) / 2.0)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    power = (mag**2) / 2.0
    bin_hz = sr / n

    def band_peak(target_hz: float, tol_hz: float) -> tuple[float, float]:
        lo = int(np.searchsorted(freqs, max(0.0, target_hz - tol_hz)))
        hi = int(np.searchsorted(freqs, min(freqs[-1], target_hz + tol_hz)))
        if hi <= lo:
            return target_hz, 0.0
        local = int(np.argmax(mag[lo:hi])) + lo
        offset, amp = _parabolic_peak(mag, local)
        return (local + offset) * bin_hz, amp

    fund_hz, fund_amp = band_peak(f0_hz, search_tol_hz)

    # exclude DC and the main lobe of the fundamental and every harmonic
    excluded = np.zeros(len(freqs), dtype=bool)
    excluded[freqs < 20.0] = True
    for k in range(1, max_harmonic + 1):
        centre = k * fund_hz
        lo = int(np.searchsorted(freqs, max(0.0, centre - 8 * bin_hz)))
        hi = int(np.searchsorted(freqs, min(freqs[-1], centre + 8 * bin_hz)))
        if hi > lo:
            excluded[lo:hi] = True
    harmonic_amps = []
    for k in range(2, max_harmonic + 1):
        _, amp = band_peak(k * fund_hz, max(search_tol_hz, k * f0_hz * 0.02))
        harmonic_amps.append(amp)

    harmonic_power = float(np.sum(np.array(harmonic_amps) ** 2) / 2.0)
    fund_power = float(fund_amp**2 / 2.0)
    # "N" is everything that is not DC, not the fundamental and not a harmonic
    noise_power = float(np.sum(power[~excluded]))

    thd_pct = 100.0 * np.sqrt(harmonic_power / fund_power) if fund_power > 0 else float("nan")
    thd_n_pct = (
        100.0 * np.sqrt((harmonic_power + noise_power) / fund_power) if fund_power > 0 else float("nan")
    )
    snr_db = -20.0 * np.log10(max(thd_n_pct, 1e-12) / 100.0)

    return Measurement(
        float(thd_pct),
        "%",
        {
            "thd_pct": float(thd_pct),
            "thd_n_pct": float(thd_n_pct),
            "snr_db": float(snr_db),
            "fundamental_hz": float(fund_hz),
            "fundamental_dbfs": float(20 * np.log10(fund_amp)) if fund_amp > 0 else -np.inf,
            "harmonics_db": [float(20 * np.log10(a / fund_amp)) if a > 0 and fund_amp > 0 else -np.inf for a in harmonic_amps],
            "freqs": freqs,
            "mag_db": 20 * np.log10(np.maximum(mag, 1e-12)),
            "n_samples": int(n),
        },
    )


# --------------------------------------------------------------------------- #
# frequency response
# --------------------------------------------------------------------------- #


def frequency_response(
    cap: np.ndarray,
    stimulus: np.ndarray,
    sr: int,
    f0_hz: float,
    f1_hz: float,
    *,
    ir_pre_ms: float = 5.0,
    ir_post_ms: float = 170.0,
    points_per_octave: int = 24,
) -> Measurement:
    """Magnitude response by swept-sine deconvolution.

    The capture and the *stimulus* are both deconvolved with the same inverse
    filter; dividing the two removes the sweep's own tilt and any absolute level
    error, leaving the pedal's true gain vs frequency.
    """
    cap = np.asarray(cap, dtype=np.float64)
    stimulus = np.asarray(stimulus, dtype=np.float64)
    inv = inverse_sweep(stimulus, sr, f0_hz, f1_hz)
    ir_dut = fftconvolve(cap, inv)
    ir_ref = fftconvolve(stimulus, inv)

    pre = int(ir_pre_ms / 1000.0 * sr)
    post = int(ir_post_ms / 1000.0 * sr)
    n_fft = 1 << int(np.ceil(np.log2(max(pre + post, 2))))

    def windowed_spectrum(ir: np.ndarray) -> np.ndarray:
        """Spectrum of the IR around its peak, on a shared frequency grid."""
        centre = int(np.argmax(np.abs(ir)))
        start = max(0, centre - pre)
        chunk = ir[start : start + pre + post]
        chunk = chunk * _edge_window(len(chunk), sr, 1.0)
        return np.abs(np.fft.rfft(chunk, n=n_fft))

    mag_dut = windowed_spectrum(ir_dut)
    mag_ref = windowed_spectrum(ir_ref)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    grid = _log_freq_grid(max(f0_hz, freqs[1]), min(f1_hz, freqs[-1]), points_per_octave)
    edges = np.sqrt(grid[:-1] * grid[1:])
    bands = list(zip(np.r_[freqs[1], edges], np.r_[edges, min(f1_hz, freqs[-1])]))

    def band_power(mag: np.ndarray) -> np.ndarray:
        power = mag**2
        out = np.empty(len(bands))
        for i, (lo, hi) in enumerate(bands):
            lo_i = int(np.searchsorted(freqs, lo))
            hi_i = int(np.searchsorted(freqs, hi))
            out[i] = np.mean(power[lo_i:max(hi_i, lo_i + 1)]) if hi_i > lo_i else power[min(lo_i, len(power) - 1)]
        return out

    resp_db = 10.0 * np.log10(np.maximum(band_power(mag_dut), 1e-30) / np.maximum(band_power(mag_ref), 1e-30))

    def at(freq: float) -> float:
        return float(np.interp(freq, grid, resp_db))

    mid_db = at(1000.0)
    band = (grid >= 100.0) & (grid <= 10000.0)
    ripple_db = float(np.max(resp_db[band]) - np.min(resp_db[band])) if band.any() else float("nan")

    def crossing(target_db: float, freqs_: np.ndarray, resp: np.ndarray) -> float:
        """First -3 dB crossing; the band edge if the response never gets there."""
        for i in range(len(resp) - 1):
            a, b = resp[i], resp[i + 1]
            if (a - target_db) * (b - target_db) <= 0 and a != b:
                frac = (target_db - a) / (b - a)
                return float(freqs_[i] * (freqs_[i + 1] / freqs_[i]) ** frac)
        # no crossing: the corner lies beyond the measured band, so report the
        # edge it lies beyond (both arrays are ordered away from mid-band)
        return float(freqs_[-1] if resp[-1] >= target_db else freqs_[0])

    below = grid <= 1000.0
    above = grid >= 1000.0
    corner_low = crossing(mid_db - 3.0, grid[below][::-1], resp_db[below][::-1])
    corner_high = crossing(mid_db - 3.0, grid[above], resp_db[above])

    return Measurement(
        mid_db,
        "dB",
        {
            "gain_1khz_db": mid_db,
            "ripple_db": ripple_db,
            "corner_low_hz": corner_low,
            "corner_high_hz": corner_high,
            "grid_hz": grid,
            "resp_db": resp_db,
            "ir_dut": ir_dut,
            "ir_ref": ir_ref,
        },
    )


# --------------------------------------------------------------------------- #
# stutter
# --------------------------------------------------------------------------- #


def stutter(
    cap: np.ndarray,
    sr: int,
    *,
    trigger_s: float,
    max_slice_ms: float = 600.0,
    min_slice_ms: float = 5.0,
) -> Measurement:
    """Slice length and gate length of a stutter/repeat effect.

    A stutter re-broadcasts the last ``slice`` of audio ``n`` times, so the gated
    region is *periodic*. The slice length is the first strong autocorrelation
    lag; the gate length comes from the last instant where ``x[t]`` still equals
    ``x[t + slice]``, which is sample accurate even when the material is noise.
    """
    cap = np.asarray(cap, dtype=np.float64)
    trig = int(trigger_s * sr)
    guard = int(0.005 * sr)
    seg = cap[trig + guard :]
    if seg.size < 4 * sr // 100:
        return Measurement(float("nan"), "ms", {"error": "capture too short after trigger"})

    min_lag = int(min_slice_ms / 1000.0 * sr)
    max_lag = int(min(max_slice_ms / 1000.0 * sr, seg.size // 2))

    # unbiased normalised autocorrelation over [min_lag, max_lag]
    n_fft = 1 << int(np.ceil(np.log2(2 * seg.size)))
    spec = np.fft.rfft(seg, n=n_fft)
    acf = np.fft.irfft(spec * np.conj(spec), n=n_fft)[:max_lag]
    energy = np.concatenate([[0.0], np.cumsum(seg**2)])
    total = energy[-1]
    lags = np.arange(max_lag)
    head = energy[np.maximum(seg.size - lags, 0)]  # sum(seg[:-lag]^2)
    tail = total - energy[lags]  # sum(seg[lag:]^2)
    norm = np.sqrt(np.maximum(head * tail, 1e-30))
    corr = np.zeros(max_lag)
    corr[min_lag:] = acf[min_lag:] / norm[min_lag:]

    best = int(np.argmax(corr))
    strong = np.flatnonzero(corr >= 0.95 * corr[best])
    lag = int(strong[0]) if strong.size else best
    if 0 < lag < max_lag - 1:
        offset, _ = _parabolic_peak(corr, lag)
        lag_f = lag + float(np.clip(offset, -1.0, 1.0))
    else:
        lag_f = float(lag)
    slice_ms = lag_f / sr * 1000.0

    lag_i = int(round(lag_f))
    # sample-accurate gate: |x[t] - x[t + lag]| is ~0 while the gate is open
    diff = np.abs(cap[trig + lag_i :] - cap[trig : cap.size - lag_i]) if lag_i > 0 else np.abs(cap[trig:])
    smooth = _moving_average(diff, max(1, int(0.0005 * sr)))
    reference = cap[trig + lag_i : trig + lag_i + int(0.1 * sr)]
    level = rms(reference) if reference.size else 0.0
    threshold = max(0.15 * level, 1e-6)
    inside = smooth < threshold
    if not inside.any():
        return Measurement(slice_ms, "ms", {"error": "no periodic region found", "corr": corr, "slice_ms": slice_ms})

    first = int(np.flatnonzero(inside)[0])
    last = int(np.flatnonzero(inside)[-1])
    gate_start_s = (trig + first) / sr
    gate_end_s = (trig + last + lag_i) / sr
    gate_ms = (gate_end_s - trigger_s) * 1000.0
    repeats = gate_ms / slice_ms if slice_ms > 0 else float("nan")

    return Measurement(
        slice_ms,
        "ms",
        {
            "slice_ms": slice_ms,
            "gate_ms": gate_ms,
            "gate_start_ms": (gate_start_s - trigger_s) * 1000.0,
            "repeats": repeats,
            "corr": corr,
            "corr_peak": float(corr[lag]) if lag < len(corr) else float("nan"),
            "diff": smooth,
            "diff_threshold": float(threshold),
            "trigger_s": trigger_s,
        },
    )


# --------------------------------------------------------------------------- #
# noise + latency
# --------------------------------------------------------------------------- #


def noise_floor(cap: np.ndarray, sr: int, *, skip_s: float = 0.05) -> Measurement:
    """RMS and peak level of a capture taken with a silent input."""
    cap = np.asarray(cap, dtype=np.float64)
    skip = int(skip_s * sr)
    segment = cap[skip : len(cap) - skip] if len(cap) > 4 * skip else cap
    floor_dbfs = dbfs(segment)
    return Measurement(
        float(floor_dbfs),
        "dBFS",
        {
            "rms_dbfs": float(floor_dbfs),
            "peak_dbfs": float(20 * np.log10(max(np.max(np.abs(segment)), 1e-12))),
            "n_samples": int(len(segment)),
        },
    )


def impulse_arrival(
    cap: np.ndarray,
    stimulus: np.ndarray,
    sr: int,
    *,
    min_lag_s: float = 0.0,
    max_lag_s: float = 2.0,
) -> Measurement:
    """Arrival time (ms) of a stimulus in a capture, by cross-correlation.

    The search window has to cover the rig's own round-trip latency, which is
    why it defaults to two seconds rather than a few milliseconds.
    """
    cap = np.asarray(cap, dtype=np.float64)
    stimulus = np.asarray(stimulus, dtype=np.float64)
    corr = fftconvolve(cap, stimulus[::-1], mode="full")
    offset = len(stimulus) - 1
    lo, hi = int(min_lag_s * sr), int(max_lag_s * sr)
    search = corr[offset + lo : offset + hi]
    if search.size == 0:
        return Measurement(float("nan"), "ms", {"error": "no correlation window"})
    lag = int(np.argmax(np.abs(search))) + lo
    return Measurement(
        lag / sr * 1000.0,
        "ms",
        {"arrival_samples": lag, "corr_peak": float(np.max(np.abs(search)))},
    )
