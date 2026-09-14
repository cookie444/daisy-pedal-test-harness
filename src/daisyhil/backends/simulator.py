"""Reference DSP model of the pedal under test.

This is the "simulated DSP" that lets CI exercise the whole rig with no
hardware: same stimuli, same measurements, same thresholds, same report.

The model is a *plant*, not a re-implementation of the measurement code: a
block-based delay/repeat/stutter engine with a converter-ish output stage, so
the rig measures it exactly the way it would measure a Pod.

``faults`` inject known regressions (delay drift, decay error, extra THD,
collapsed bandwidth, slow stutter, extra latency) so the rig's ability to *fail*
is covered by tests too (``tests/unit/test_fault_detection.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import PedalControls, RigConfig
from .base import Backend

MAX_DELAY_S = 2.5
SOFT_CLIP_K = 0.3  # output stage: ~0.05 % THD at -12 dBFS, unity gain at low level


@dataclass
class Faults:
    delay_drift_pct: float = 0.0
    decay_error_pct: float = 0.0
    thd_add_pct: float = 0.0
    noise_floor_dbfs: float = -96.0
    stutter_slice_error_pct: float = 0.0
    latency_extra_ms: float = 0.0
    high_corner_hz: float = 21000.0

    @classmethod
    def from_config(cls, cfg: RigConfig) -> "Faults":
        return cls(
            delay_drift_pct=cfg.fault("delay_drift_pct"),
            decay_error_pct=cfg.fault("decay_error_pct"),
            thd_add_pct=cfg.fault("thd_add_pct"),
            noise_floor_dbfs=cfg.fault("noise_floor_dbfs", -96.0),
            stutter_slice_error_pct=cfg.fault("stutter_slice_error_pct"),
            latency_extra_ms=cfg.fault("latency_extra_ms"),
            high_corner_hz=cfg.fault("high_corner_hz", 21000.0),
        )

    def as_dict(self) -> dict[str, float]:
        return dict(self.__dict__)

    def any_active(self) -> bool:
        defaults = Faults()
        return any(abs(getattr(self, k) - getattr(defaults, k)) > 1e-12 for k in defaults.as_dict())


class SimulatedPedal:
    """Block-based delay + repeats + stutter, as a Daisy audio callback runs it."""

    def __init__(self, sample_rate: int, block_size: int, faults: Faults | None = None) -> None:
        self.sr = sample_rate
        self.block_size = block_size
        self.faults = faults or Faults()
        self.buffer = np.zeros(int(MAX_DELAY_S * sample_rate) + 8 * block_size)
        self.reset()

    # -- state ------------------------------------------------------------- #
    def reset(self) -> None:
        self.buffer[:] = 0.0
        self.write = 0
        self.tone_state = 0.0
        self.dc_state = 0.0
        self.lp_state = 0.0
        self.slice_ring = np.zeros(1)
        self.stutter_buf = np.zeros(1)
        self.stutter_pos = 0
        self.stutter_left = 0
        self.triggered = False
        self.n = 0
        self.taps: list[float] = []
        self.tap_mode = False
        self.delay_samples = 1.0
        self.extra_latency = 0
        self.controls: PedalControls | None = None

    # -- control ----------------------------------------------------------- #
    def configure(self, controls: PedalControls) -> None:
        self.controls = controls
        sr = self.sr
        self.tap_mode = bool(controls.tap_interval_ms)
        delay_ms = controls.tap_interval_ms if self.tap_mode else controls.delay_ms
        self._set_delay_ms(delay_ms)
        self.feedback = controls.feedback * (1.0 + self.faults.decay_error_pct / 100.0)
        self.mix = controls.mix
        self.bypass = controls.bypass

        # feedback damping ("tone") - one pole, bypassed when tone_hz is None
        self.tone_a = 1.0 - math.exp(-2 * math.pi * controls.tone_hz / sr) if controls.tone_hz else 0.0
        # converter model: DC block @ 20 Hz, reconstruction pole, 24-bit quantiser
        self.dc_a = 1.0 - math.exp(-2 * math.pi * 20.0 / sr)
        self.lp_a = 1.0 - math.exp(-2 * math.pi * self.faults.high_corner_hz / sr)
        self.noise_rms = 10.0 ** (self.faults.noise_floor_dbfs / 20.0)
        self.extra_latency = int(self.faults.latency_extra_ms / 1000.0 * sr)
        self.soft_k = SOFT_CLIP_K + 6.0 * controls.drive
        self.cubic = self.faults.thd_add_pct / 100.0

        # stutter
        slice_ms = controls.stutter_slice_ms * (1.0 + self.faults.stutter_slice_error_pct / 100.0)
        self.slice_samples = max(1, int(round(slice_ms / 1000.0 * sr)))
        self.stutter_repeats = controls.stutter_repeats
        self.trigger_sample = int(controls.stutter_trigger_s * sr) if controls.stutter_enabled else None
        self.slice_ring = np.zeros(self.slice_samples)
        self.stutter_left = 0
        self.triggered = False
        self.taps = []

    def _set_delay_ms(self, delay_ms: float) -> None:
        delay_ms *= 1.0 + self.faults.delay_drift_pct / 100.0
        self.delay_samples = max(1.0, delay_ms / 1000.0 * self.sr)

    def tap(self, now_s: float) -> None:
        """Footswitch tap; two taps set the delay time (tap tempo)."""
        self.taps.append(now_s)
        self.taps = self.taps[-2:]
        if self.tap_mode and len(self.taps) >= 2:
            self._set_delay_ms((self.taps[-1] - self.taps[0]) * 1000.0)

    # -- dsp --------------------------------------------------------------- #
    def _read(self, delay_samples: float) -> float:
        pos = self.write - delay_samples
        base = int(math.floor(pos))
        frac = pos - base
        n = len(self.buffer)
        return float(self.buffer[base % n] * (1.0 - frac) + self.buffer[(base + 1) % n] * frac)

    def process_block(self, block: np.ndarray) -> np.ndarray:
        out = np.empty_like(block)
        fb, mix, n = self.feedback, self.mix, len(self.buffer)
        for i, x in enumerate(block):
            x = float(x)
            if self.stutter_left > 0:
                y = float(self.stutter_buf[self.stutter_pos % len(self.stutter_buf)])
                self.stutter_pos += 1
                self.stutter_left -= 1
            else:
                d = self._read(self.delay_samples)
                if self.tone_a:
                    self.tone_state += self.tone_a * (d - self.tone_state)
                    d = self.tone_state
                self.buffer[self.write % n] = x + fb * d
                self.write = (self.write + 1) % n
                y = (1.0 - mix) * x + mix * d

            # converter-ish output stage
            y = math.tanh(self.soft_k * y) / self.soft_k
            if self.cubic:
                y = y + self.cubic * (y**3)
            self.dc_state += self.dc_a * (y - self.dc_state)
            y -= self.dc_state
            self.lp_state += self.lp_a * (y - self.lp_state)
            out[i] = self.lp_state

            # stutter: keep filling the slice ring, fire once at the trigger
            self.slice_ring[self.n % len(self.slice_ring)] = x
            if self.trigger_sample is not None and not self.triggered and self.n >= self.trigger_sample:
                self.triggered = True
                self.stutter_buf = self.slice_ring.copy()
                self.stutter_pos = 0
                self.stutter_left = self.slice_samples * self.stutter_repeats
            self.n += 1
        return out

    def render(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.empty_like(x)
        for start in range(0, len(x), self.block_size):
            block = x[start : start + self.block_size]
            y[start : start + len(block)] = self.process_block(block)
        return y


class SimulatorBackend(Backend):
    """Renders stimuli through :class:`SimulatedPedal` offline."""

    name = "sim"

    def __init__(self, cfg: RigConfig, faults: Faults | None = None, seed: int = 7) -> None:
        self.cfg = cfg
        self.faults = faults or Faults.from_config(cfg)
        self.sample_rate = cfg.sample_rate
        self.pedal = SimulatedPedal(cfg.sample_rate, cfg.block_size, self.faults)
        self.rng = np.random.default_rng(seed)
        self.controls: PedalControls | None = None

    # -- Backend ----------------------------------------------------------- #
    def configure(self, controls: PedalControls) -> None:
        self.controls = controls
        self.pedal.reset()
        self.pedal.configure(controls)
        if controls.tap_interval_ms and not controls.bypass:
            # two footswitch taps one interval apart
            interval = controls.tap_interval_ms / 1000.0
            self.pedal.tap(0.0)
            self.pedal.tap(interval)

    def render(self, stimulus: np.ndarray, sample_rate: int, name: str | None = None) -> np.ndarray:
        if sample_rate != self.sample_rate:
            raise ValueError(f"simulator runs at {self.sample_rate} Hz, stimulus is {sample_rate} Hz")
        controls = self.controls
        if controls is None:
            raise RuntimeError("configure() must be called before render()")
        stimulus = np.asarray(stimulus, dtype=np.float64)
        y = stimulus.copy() if controls.bypass else self.pedal.render(stimulus)

        # one block of callback latency (present in bypass too) plus any extra
        # DSP latency - a bypass is a wire, so extra latency must not appear there
        shift = self.cfg.block_size + (0 if controls.bypass else self.pedal.extra_latency)
        if shift:
            y = np.concatenate([np.zeros(shift), y])[: len(y)]

        # converter noise floor + 24-bit quantisation
        if self.pedal.noise_rms > 0:
            y = y + self.rng.standard_normal(len(y)) * self.pedal.noise_rms
        step = 2.0**-23
        return np.round(y / step) * step

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "sample_rate": self.sample_rate,
            "block_size": self.cfg.block_size,
            "faults": self.faults.as_dict(),
            "faults_active": self.faults.any_active(),
        }


def simulate_reference(cfg: RigConfig, controls: PedalControls, stimulus: np.ndarray) -> np.ndarray:
    """One-shot helper used by tests: render a stimulus through a fresh model."""
    backend = SimulatorBackend(cfg)
    backend.configure(controls)
    return backend.render(stimulus, cfg.sample_rate)


__all__ = ["Faults", "SimulatedPedal", "SimulatorBackend", "simulate_reference", "MAX_DELAY_S"]
