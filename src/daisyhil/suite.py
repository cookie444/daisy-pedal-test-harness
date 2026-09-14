"""The spec suite: stimuli -> pedal -> measurements -> verdicts.

``SpecSuite.run()`` is the single entry point used by the CLI, by pytest and by
GitHub Actions. Adding a spec means adding one method here plus one threshold
block in ``config/rig.yaml``; nothing else in the pipeline changes.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .backends.base import Backend, Capture
from .backends.simulator import SimulatorBackend
from .config import PedalControls, RigConfig
from .measure import (
    delay_and_decay,
    frequency_response,
    impulse_arrival,
    noise_floor,
    stutter,
    thd,
)
from .reporting import plots
from .results import RunReport, SpecResult, evaluate, host_environment
from .signals import amplitude, exponential_sweep, impulse, noise, pad, silence, sine

__all__ = ["SpecSuite", "run_suite", "make_backend"]


class SpecSuite:
    """Runs every spec against one backend and returns a :class:`RunReport`."""

    def __init__(
        self,
        cfg: RigConfig,
        backend: Backend | None = None,
        *,
        with_plots: bool = True,
        save_captures: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.backend = backend or SimulatorBackend(cfg)
        self.with_plots = with_plots
        self.save_captures = save_captures
        self.captures: dict[str, Capture] = {}
        self.started = 0.0

    # -- plumbing ---------------------------------------------------------- #
    @property
    def sr(self) -> int:
        return self.cfg.sample_rate

    def _capture(
        self,
        name: str,
        controls: PedalControls,
        stimulus: np.ndarray,
        *,
        extra_post_s: float = 0.0,
    ) -> Capture:
        """Run one stimulus through the pedal and store the capture.

        ``extra_post_s`` extends the tail when the pedal holds the signal back
        (a delayed sweep loses its high-frequency end if the capture stops
        before the delay line has drained).
        """
        self.backend.configure(controls)
        padded = pad(stimulus, self.sr, self.cfg.pre_roll_s, self.cfg.post_roll_s + extra_post_s)
        data = np.asarray(self.backend.render(padded, self.sr, name=name), dtype=np.float64)
        capture = Capture(
            name=name,
            data=data,
            sample_rate=self.sr,
            stimulus=stimulus,
            controls=controls,
            meta={"controls": controls.as_dict()},
        )
        self.captures[name] = capture
        if self.save_captures is not None:
            from .backends.wavfile import save_capture

            save_capture(Path(self.save_captures) / f"{name}.wav", data, self.sr)
        return capture

    def _plot(self, spec: SpecResult, title: str, png: str, caption: str = "") -> SpecResult:
        if self.with_plots:
            spec.add_figure(title, png, caption)
        return spec

    # -- specs ------------------------------------------------------------- #
    def _spec_latency_calibration(self) -> float:
        """Impulse arrival with the pedal bypassed: the rig's own latency."""
        controls = PedalControls(bypass=True)
        stimulus = impulse(self.sr, 0.5, amplitude(self.cfg.stimulus_level_dbfs + 6))
        cap = self._capture("bypass", controls, stimulus)
        arrival = impulse_arrival(cap.data, cap.stimulus, self.sr)
        self._bypass_capture = cap
        return float(arrival.value)

    def _spec_delay(self) -> list[SpecResult]:
        cfg = self.cfg
        controls = PedalControls(
            delay_ms=cfg.pedal.delay_ms,
            feedback=cfg.pedal.feedback,
            mix=cfg.pedal.mix,
            tone_hz=cfg.pedal.tone_hz,
        )
        level = amplitude(cfg.stimulus_level_dbfs + 6)
        stimulus = impulse(self.sr, float(cfg.stimulus.get("impulse_tail_s", 5.0)), level)
        cap = self._capture("delay", controls, stimulus)

        delay, decay = delay_and_decay(
            cap.data, self.sr, pre_roll_s=cfg.pre_roll_s, expect_dry=controls.mix < 1.0
        )

        expected_delay = float(cfg.pedal.delay_ms)
        result = evaluate(
            "delay_time_ms",
            "Delay",
            delay.value,
            cfg.threshold("delay_time_ms"),
            expected_delay,
            details=delay.details,
            notes=f"knob={expected_delay:g} ms, mix={controls.mix:g}, feedback={controls.feedback:g}",
        )
        self._plot(
            result,
            "Delay: click capture with detected repeats",
            plots.plot_delay(cap.data, self.sr, delay.details, expected_ms=expected_delay),
            "Vertical red X = dry hit, markers = detected repeats, teal lines = expected repeat times.",
        )

        expected_decay = 20.0 * math.log10(max(controls.feedback, 1e-9))
        decay_result = evaluate(
            "decay_db_per_repeat",
            "Repeats",
            decay.value,
            cfg.threshold("decay_db_per_repeat"),
            expected_decay,
            details=decay.details,
            notes=f"theoretical 20*log10(feedback) = {expected_decay:.2f} dB",
        )
        self._plot(
            decay_result,
            "Repeat decay vs fitted slope",
            plots.plot_decay(
                decay.details,
                expected_db=expected_decay,
                tol_db=(cfg.threshold("decay_db_per_repeat").tol_abs or 0.0),
                slope_db=decay.value,
            ),
            "Each repeat should sit on the fitted line; the band is the spec window.",
        )

        expected_time = (
            60.0 / abs(expected_decay) * expected_delay if expected_decay < 0 else float("nan")
        )
        time_result = evaluate(
            "decay_time_ms",
            "Repeats",
            decay.details.get("decay_time_ms", float("nan")),
            cfg.threshold("decay_time_ms"),
            expected_time,
            details=decay.details,
            notes="time for the repeat train to fall 60 dB",
        )
        return [result, decay_result, time_result]

    def _spec_tap_tempo(self) -> list[SpecResult]:
        cfg = self.cfg
        interval = float(cfg.pedal.tap_interval_ms or cfg.pedal.delay_ms)
        # wet only: no dry hit, so the delay is the spacing between repeats
        controls = PedalControls(
            tap_interval_ms=interval,
            feedback=cfg.pedal.feedback,
            mix=1.0,
            tone_hz=cfg.pedal.tone_hz,
        )
        stimulus = impulse(self.sr, 2.5, amplitude(cfg.stimulus_level_dbfs + 6))
        cap = self._capture("tap", controls, stimulus)
        delay, _ = delay_and_decay(cap.data, self.sr, pre_roll_s=cfg.pre_roll_s, expect_dry=False)

        result = evaluate(
            "tap_delay_ms",
            "Delay",
            delay.value,
            cfg.threshold("tap_delay_ms"),
            interval,
            details=delay.details,
            notes=f"tap interval={interval:g} ms, wet-only (mix=1.0)",
        )
        self._plot(
            result,
            "Tap tempo: wet-only repeat train",
            plots.plot_delay(
                cap.data,
                self.sr,
                delay.details,
                expected_ms=interval,
                title=f"Tap tempo ({interval:g} ms between taps)",
            ),
            "No dry hit with mix=1.0, so every marker is a repeat.",
        )
        return [result]

    def _spec_distortion(self) -> list[SpecResult]:
        cfg = self.cfg
        f0 = float(cfg.stimulus.get("sine_freq_hz", 1000.0))
        duration = float(cfg.stimulus.get("sine_duration_s", 1.0))
        controls = PedalControls(delay_ms=cfg.pedal.delay_ms, feedback=0.0, mix=cfg.pedal.mix)
        stimulus = sine(self.sr, f0, duration, amplitude(cfg.stimulus_level_dbfs))
        cap = self._capture("thd", controls, stimulus)

        # the wet path only joins in after one delay time, so the steady state
        # (and therefore the FFT window) starts after pre-roll + delay
        skip_s = cfg.pre_roll_s + controls.delay_ms / 1000.0 + 0.15
        result = thd(cap.data, self.sr, f0, start_s=skip_s, tail_s=cfg.post_roll_s + 0.05)
        thd_result = evaluate(
            "thd_pct",
            "Distortion",
            result.details["thd_pct"],
            cfg.threshold("thd_pct"),
            details=result.details,
            notes=f"{f0:.0f} Hz at {cfg.stimulus_level_dbfs:g} dBFS",
        )
        self._plot(
            thd_result,
            f"Distortion spectrum at {f0:.0f} Hz",
            plots.plot_spectrum(result.details, f0=f0),
            "Harmonics 2-10 are marked; the rest of the floor sets THD+N.",
        )

        thdn_result = evaluate(
            "thd_n_pct",
            "Distortion",
            result.details["thd_n_pct"],
            cfg.threshold("thd_n_pct"),
            details=result.details,
            notes="THD + noise over 20 Hz - Nyquist",
        )
        snr_result = evaluate(
            "snr_db",
            "Distortion",
            result.details["snr_db"],
            cfg.threshold("snr_db"),
            details=result.details,
            notes="signal to noise+distortion",
        )
        return [thd_result, thdn_result, snr_result]

    def _spec_frequency_response(self) -> list[SpecResult]:
        cfg = self.cfg
        f0 = float(cfg.stimulus.get("sweep_f0_hz", 20.0))
        f1 = float(cfg.stimulus.get("sweep_f1_hz", 20000.0))
        duration = float(cfg.stimulus.get("sweep_duration_s", 2.0))
        # wet only, no feedback: the delay adds phase, not comb filtering
        controls = PedalControls(delay_ms=cfg.pedal.delay_ms, feedback=0.0, mix=1.0)
        sweep = exponential_sweep(self.sr, f0, f1, duration, amplitude(cfg.stimulus_level_dbfs))
        cap = self._capture("sweep", controls, sweep, extra_post_s=controls.delay_ms / 1000.0 + 0.05)

        result = frequency_response(cap.data, sweep, self.sr, f0, f1)
        details = result.details

        gain = evaluate(
            "gain_1khz_db",
            "FrequencyResponse",
            details["gain_1khz_db"],
            cfg.threshold("gain_1khz_db"),
            0.0,
            details=details,
            notes="unity gain check",
        )
        self._plot(
            gain,
            "Frequency response (swept sine deconvolution)",
            plots.plot_response(details, ripple_limit_db=cfg.threshold("fr_ripple_db").upper or 3.0),
            "Measured with mix=1.0, feedback=0 so the delay line cannot comb the response.",
        )

        ripple = evaluate(
            "fr_ripple_db",
            "FrequencyResponse",
            details["ripple_db"],
            cfg.threshold("fr_ripple_db"),
            details=details,
            notes="peak-to-peak, 100 Hz - 10 kHz",
        )
        low = evaluate(
            "fr_corner_low_hz",
            "FrequencyResponse",
            details["corner_low_hz"],
            cfg.threshold("fr_corner_low_hz"),
            details=details,
            notes="-3 dB corner below 1 kHz",
        )
        high = evaluate(
            "fr_corner_high_hz",
            "FrequencyResponse",
            details["corner_high_hz"],
            cfg.threshold("fr_corner_high_hz"),
            details=details,
            notes="-3 dB corner above 1 kHz",
        )
        return [gain, ripple, low, high]

    def _spec_stutter(self) -> list[SpecResult]:
        cfg = self.cfg
        slice_ms = float(cfg.pedal.stutter_slice_ms)
        repeats = int(cfg.pedal.stutter_repeats)
        controls = PedalControls(
            delay_ms=cfg.pedal.delay_ms,
            feedback=0.0,
            mix=1.0,
            stutter_enabled=True,
            stutter_slice_ms=slice_ms,
            stutter_repeats=repeats,
            stutter_trigger_s=float(cfg.pedal.stutter_trigger_s),
        )
        duration = float(cfg.stimulus.get("noise_duration_s", 2.0))
        stimulus = noise(self.sr, duration, amplitude(cfg.stimulus_level_dbfs))
        cap = self._capture("stutter", controls, stimulus)

        result = stutter(cap.data, self.sr, trigger_s=controls.stutter_trigger_s)
        details = result.details

        slice_result = evaluate(
            "stutter_slice_ms",
            "Stutter",
            details.get("slice_ms", float("nan")),
            cfg.threshold("stutter_slice_ms"),
            slice_ms,
            details=details,
            notes=f"autocorrelation of the gated region (requested {slice_ms:g} ms)",
        )
        self._plot(
            slice_result,
            "Stutter: noise capture with detected gate",
            plots.plot_stutter(cap.data, self.sr, details),
            "Red = trigger, shaded = detected gate, teal = slice boundaries from the measured period.",
        )

        gate_result = evaluate(
            "stutter_gate_ms",
            "Stutter",
            details.get("gate_ms", float("nan")),
            cfg.threshold("stutter_gate_ms"),
            slice_ms * repeats,
            details=details,
            notes=f"expected slice x repeats = {slice_ms * repeats:g} ms",
        )
        repeats_result = evaluate(
            "stutter_repeats",
            "Stutter",
            details.get("repeats", float("nan")),
            cfg.threshold("stutter_repeats"),
            float(repeats),
            details=details,
            notes="gate / slice should be integral",
        )
        return [slice_result, gate_result, repeats_result]

    def _spec_noise(self) -> list[SpecResult]:
        cfg = self.cfg
        controls = PedalControls(delay_ms=cfg.pedal.delay_ms, feedback=0.0, mix=cfg.pedal.mix)
        duration = float(cfg.stimulus.get("silence_duration_s", 1.0))
        cap = self._capture("noise", controls, silence(self.sr, duration))

        result = noise_floor(cap.data, self.sr)
        spec = evaluate(
            "noise_floor_dbfs",
            "Noise",
            result.value,
            cfg.threshold("noise_floor_dbfs"),
            details=result.details,
            notes="silent input, pedal engaged",
        )
        self._plot(
            spec,
            "Noise floor (silent input)",
            plots.plot_waveform(cap.data, self.sr, title="Output noise with a silent input"),
            "RMS of this capture is the noise floor spec.",
        )
        return [spec]

    def _spec_latency(self, bypass_ms: float) -> list[SpecResult]:
        cfg = self.cfg
        # dry only: the wet path would arrive one delay time later and win the
        # correlation, which would report the delay as latency
        controls = PedalControls(delay_ms=cfg.pedal.delay_ms, feedback=0.0, mix=0.0)
        stimulus = impulse(self.sr, 0.5, amplitude(cfg.stimulus_level_dbfs + 6))
        cap = self._capture("latency", controls, stimulus)
        arrival = impulse_arrival(
            cap.data,
            cap.stimulus,
            self.sr,
            min_lag_s=max(0.0, (bypass_ms - 20.0) / 1000.0),
            max_lag_s=(bypass_ms + 100.0) / 1000.0,
        )
        latency = float(arrival.value) - bypass_ms

        spec = evaluate(
            "processing_latency_ms",
            "Latency",
            latency,
            cfg.threshold("processing_latency_ms"),
            details={
                "bypass_ms": bypass_ms,
                "active_ms": float(arrival.value),
                "block_size": cfg.block_size,
            },
            notes="arrival through the pedal minus arrival bypassed",
        )
        bypass_cap = getattr(self, "_bypass_capture", None)
        if bypass_cap is not None:
            self._plot(
                spec,
                "Latency: bypassed vs through the pedal",
                plots.plot_latency(bypass_cap.data, cap.data, self.sr),
                "Both captures share the same click; the offset is the pedal's added latency.",
            )
        return [spec]

    # -- entry point -------------------------------------------------------- #
    def run(self) -> RunReport:
        self.started = time.time()
        report = RunReport(
            backend=self.backend.name,
            sample_rate=self.sr,
            config=self.cfg.raw,
            environment=host_environment(),
        )
        report.environment.update(self.backend.describe())

        bypass_ms = self._spec_latency_calibration()
        for spec in self._spec_delay():
            report.add(spec)
        for spec in self._spec_tap_tempo():
            report.add(spec)
        for spec in self._spec_distortion():
            report.add(spec)
        for spec in self._spec_frequency_response():
            report.add(spec)
        for spec in self._spec_stutter():
            report.add(spec)
        for spec in self._spec_noise():
            report.add(spec)
        for spec in self._spec_latency(bypass_ms):
            report.add(spec)

        report.duration_s = time.time() - self.started
        self.backend.close()
        return report


def make_backend(name: str, cfg: RigConfig, **kwargs: Any) -> Backend:
    """Instantiate a backend by name (``sim``, ``audio``, ``wavfile``)."""
    if name == "sim":
        return SimulatorBackend(cfg, **kwargs)
    if name == "audio":
        from .backends.audio import AudioBackend

        return AudioBackend(cfg, **kwargs)
    if name == "wavfile":
        from .backends.wavfile import WavFileBackend

        return WavFileBackend(cfg, **kwargs)
    raise ValueError(f"unknown backend '{name}'")


def run_suite(
    cfg: RigConfig,
    backend: Backend | str = "sim",
    *,
    with_plots: bool = True,
    save_captures: Path | None = None,
) -> RunReport:
    """Convenience wrapper used by the CLI and by the pytest HIL suite."""
    backend_instance = make_backend(backend, cfg) if isinstance(backend, str) else backend
    return SpecSuite(cfg, backend_instance, with_plots=with_plots, save_captures=save_captures).run()
