"""The measurement algorithms are themselves under test.

Each test builds a signal whose answer is known by construction, then asserts the
algorithm recovers it. If these fail, no hardware result can be trusted, so they
run on every commit with no device attached.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import lfilter

from daisyhil.measure import (
    delay_and_decay,
    estimate_floor,
    find_impulses,
    frequency_response,
    impulse_arrival,
    noise_floor,
    stutter,
    thd,
)
from daisyhil.signals import amplitude, exponential_sweep, impulse, sine

SR = 48000


def _click_train(
    delay_ms: float = 250.0,
    decay_db: float = -6.0,
    repeats: int = 8,
    pre_roll_s: float = 0.25,
    tail_s: float = 1.0,
    noise_rms: float = 1e-5,
    seed: int = 3,
) -> np.ndarray:
    x = np.zeros(int((pre_roll_s + tail_s) * SR))
    rng = np.random.default_rng(seed)
    x += rng.standard_normal(len(x)) * noise_rms
    level = 0.5
    for k in range(1, repeats + 1):
        index = int((pre_roll_s + k * delay_ms / 1000.0) * SR)
        if index < len(x):
            x[index] += level
        level *= 10 ** (decay_db / 20.0)
    return x


# --------------------------------------------------------------------------- #
# delay + decay
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("delay_ms", [50.0, 125.0, 250.0, 375.0])
def test_delay_time_recovers_known_spacing(delay_ms: float):
    x = _click_train(delay_ms=delay_ms)
    delay, _ = delay_and_decay(x, SR, pre_roll_s=0.25, expect_dry=False, min_delay_ms=10.0)
    assert delay.value == pytest.approx(delay_ms, abs=0.1)


@pytest.mark.parametrize("decay_db", [-3.0, -6.0, -12.0])
def test_decay_slope_matches_constructed_repeats(decay_db: float):
    x = _click_train(decay_db=decay_db, repeats=6, tail_s=2.0)
    _, decay = delay_and_decay(x, SR, pre_roll_s=0.25, expect_dry=False, min_delay_ms=10.0)
    assert decay.value == pytest.approx(decay_db, abs=0.1)
    assert decay.details["fit_r2"] > 0.999


def test_dry_hit_is_excluded_when_expected():
    x = _click_train(delay_ms=250.0)
    # add a louder dry hit before the repeats
    x[int(0.25 * SR)] += 1.0
    with_dry, _ = delay_and_decay(x, SR, pre_roll_s=0.25, expect_dry=True, min_delay_ms=10.0)
    without, _ = delay_and_decay(x, SR, pre_roll_s=0.25, expect_dry=False, min_delay_ms=10.0)
    assert with_dry.value == pytest.approx(250.0, abs=0.1)
    # without dropping the dry hit the first gap is 0.25 s - 250 ms... it is not
    assert without.value == pytest.approx(250.0, abs=0.1)


def test_filter_tail_is_not_reported_as_a_repeat():
    """A DC blocker leaves an exponential tail: it must not look like repeats."""
    x = _click_train(delay_ms=250.0, repeats=4, tail_s=2.0, noise_rms=0.0)
    tau = 0.008
    start = int((0.25 + 1 * 0.25) * SR)  # decay tail after the first click only
    n = min(int(0.05 * SR), len(x) - start)
    x[start : start + n] += -0.002 * np.exp(-np.arange(n) / (tau * SR))
    times, amps = find_impulses(x, SR, min_separation_ms=10.0)
    # the four clicks and nothing from the decaying tail
    assert len(times) == 4, f"detected {len(times)} impulses: {times}"
    assert np.all(np.diff(times) == pytest.approx(250.0, abs=0.5))
    assert amps.max() > 0.4


def test_estimate_floor_reads_the_silent_region():
    x = np.concatenate([np.full(SR // 10, 1e-3), np.full(SR // 10, 1.0)])
    assert estimate_floor(x, SR, window_s=0.1) == pytest.approx(1e-3, rel=1e-6)


# --------------------------------------------------------------------------- #
# distortion
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("harmonic,level_pct", [(3, 1.0), (2, 2.5), (5, 0.5)])
def test_thd_recovers_injected_harmonic(harmonic: int, level_pct: float):
    f0, duration = 1000.0, 0.5
    t = np.arange(int(duration * SR)) / SR
    a = amplitude(-12.0)
    x = a * np.sin(2 * np.pi * f0 * t) + a * (level_pct / 100.0) * np.sin(
        2 * np.pi * harmonic * f0 * t + 0.7
    )
    result = thd(x, SR, f0, start_s=0.05, tail_s=0.05)
    assert result.value == pytest.approx(level_pct, rel=0.1), result.details
    assert result.details["thd_n_pct"] == pytest.approx(level_pct, rel=0.2)


def test_thd_of_a_clean_sine_is_near_zero():
    x = sine(SR, 1000.0, 0.5, amplitude(-12.0))
    result = thd(x, SR, 1000.0, start_s=0.05, tail_s=0.05)
    assert result.value < 0.01
    assert result.details["snr_db"] > 70.0


def test_thd_rejects_a_transient_window():
    """Analysing across a level step would poison THD+N; the skip must prevent it."""
    f0 = 1000.0
    steady = sine(SR, f0, 0.5, amplitude(-12.0))
    quiet = sine(SR, f0, 0.5, amplitude(-24.0))
    x = np.concatenate([quiet, steady])
    in_steady = thd(x, SR, f0, start_s=0.55, tail_s=0.05)
    across_step = thd(x, SR, f0, start_s=0.05, tail_s=0.05)
    assert in_steady.details["thd_n_pct"] < 0.05
    assert across_step.details["thd_n_pct"] > in_steady.details["thd_n_pct"] * 10


# --------------------------------------------------------------------------- #
# frequency response
# --------------------------------------------------------------------------- #


def _one_pole(x: np.ndarray, corner_hz: float, sr: int = SR) -> np.ndarray:
    a = 1.0 - np.exp(-2 * np.pi * corner_hz / sr)
    return lfilter([a], [1.0, -(1.0 - a)], x)


def test_frequency_response_recovers_a_known_lowpass():
    f0, f1, duration, corner = 20.0, 20000.0, 2.0, 5000.0
    sweep = exponential_sweep(SR, f0, f1, duration, amplitude(-12.0))
    filtered = _one_pole(sweep, corner)
    capture = np.concatenate([np.zeros(int(0.25 * SR)), filtered, np.zeros(int(0.7 * SR))])

    result = frequency_response(capture, sweep, SR, f0, f1)
    details = result.details
    assert details["gain_1khz_db"] == pytest.approx(0.0, abs=0.5)
    assert details["corner_high_hz"] == pytest.approx(corner, rel=0.15)
    # one octave above the corner (a one-pole is -6 dB/oct there)
    assert np.interp(10000.0, details["grid_hz"], details["resp_db"]) == pytest.approx(-7.0, abs=1.0)


def test_frequency_response_is_flat_for_a_wire():
    f0, f1, duration = 20.0, 20000.0, 1.5
    sweep = exponential_sweep(SR, f0, f1, duration, amplitude(-12.0))
    capture = np.concatenate([np.zeros(int(0.2 * SR)), sweep, np.zeros(int(0.5 * SR))])
    result = frequency_response(capture, sweep, SR, f0, f1)
    assert abs(result.details["gain_1khz_db"]) < 0.5
    assert result.details["ripple_db"] < 1.0


def test_frequency_response_reports_a_collapsed_bandwidth():
    f0, f1, duration = 20.0, 20000.0, 1.5
    sweep = exponential_sweep(SR, f0, f1, duration, amplitude(-12.0))
    capture = np.concatenate([np.zeros(int(0.2 * SR)), _one_pole(sweep, 4000.0), np.zeros(int(0.5 * SR))])
    result = frequency_response(capture, sweep, SR, f0, f1)
    assert result.details["corner_high_hz"] == pytest.approx(4000.0, rel=0.2)


# --------------------------------------------------------------------------- #
# stutter
# --------------------------------------------------------------------------- #


def _synthetic_stutter(slice_ms: float, repeats: int, trigger_s: float = 0.5, seed: int = 11):
    rng = np.random.default_rng(seed)
    n = int(2.0 * SR)
    x = rng.standard_normal(n) * 0.2
    slice_n = int(slice_ms / 1000.0 * SR)
    trig = int(trigger_s * SR)
    # the pedal captured the audio just before the trigger and replays it
    captured = x[trig - slice_n : trig].copy()
    for k in range(repeats):
        x[trig + k * slice_n : trig + (k + 1) * slice_n] = captured
    return x


@pytest.mark.parametrize("slice_ms,repeats", [(125.0, 4), (80.0, 3), (250.0, 2)])
def test_stutter_measures_slice_and_gate(slice_ms: float, repeats: int):
    x = _synthetic_stutter(slice_ms, repeats)
    result = stutter(x, SR, trigger_s=0.5)
    assert result.details["slice_ms"] == pytest.approx(slice_ms, rel=0.02)
    assert result.details["gate_ms"] == pytest.approx(slice_ms * repeats, rel=0.02)
    assert result.details["repeats"] == pytest.approx(repeats, abs=0.1)


def test_stutter_detects_a_wrong_slice_length():
    x = _synthetic_stutter(140.0, 4)  # pedal asked for 125 ms, delivered 140 ms
    result = stutter(x, SR, trigger_s=0.5)
    assert result.details["slice_ms"] == pytest.approx(140.0, rel=0.02)
    assert abs(result.details["slice_ms"] - 125.0) > 10.0


# --------------------------------------------------------------------------- #
# noise + latency
# --------------------------------------------------------------------------- #


def test_noise_floor_reads_a_known_rms():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(SR) * 1e-3
    assert noise_floor(x, SR).value == pytest.approx(-60.0, abs=0.2)


def test_impulse_arrival_finds_a_known_delay():
    stimulus = impulse(SR, 0.5, amplitude(-6.0))
    shift = 1234
    capture = np.concatenate([np.zeros(shift), stimulus])[: len(stimulus)]
    result = impulse_arrival(capture, stimulus, SR)
    assert result.value == pytest.approx(shift / SR * 1000.0, abs=0.05)
