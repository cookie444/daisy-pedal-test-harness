"""Real hardware backend: play a stimulus, record the pedal, return the capture.

Designed for a Daisy Pod (or Seed) wired to an audio interface: pedal input <-
interface out, pedal out -> interface in. ``sounddevice`` gives sample-aligned
play/record, which is what makes the delay and latency measurements meaningful.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import PedalControls, RigConfig
from ..signals import amplitude
from .base import Backend
from .control import NullTransport, ScheduledTransport, Transport

__all__ = ["AudioBackend", "list_devices", "resolve_device"]


def list_devices() -> list[dict[str, Any]]:
    """All PortAudio devices, as dicts (safe to call without deps beyond sd)."""
    import sounddevice as sd

    return [dict(d) for d in sd.query_devices()]


def resolve_device(spec: Any, kind: str = "input") -> int | None:
    """Resolve a device name/index/substring from the config to an index."""
    if spec is None or spec == "":
        return None
    import sounddevice as sd

    devices = sd.query_devices()
    if isinstance(spec, (int, float)) or (isinstance(spec, str) and spec.lstrip("-").isdigit()):
        return int(spec)
    needle = str(spec).lower()
    matches = [
        i
        for i, d in enumerate(devices)
        if needle in d["name"].lower()
        and (d[f"max_{kind}_channels"] > 0 if kind in ("input", "output") else True)
    ]
    if not matches:
        raise RuntimeError(f"no {kind} audio device matching {spec!r}")
    return matches[0]


class AudioBackend(Backend):
    """Play/record loopback through an attached pedal."""

    name = "audio"

    def __init__(self, cfg: RigConfig, transport: Transport | None = None) -> None:
        self.cfg = cfg
        audio = cfg.audio
        self.transport: ScheduledTransport = ScheduledTransport(transport or NullTransport())
        try:
            import sounddevice  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise RuntimeError(
                "sounddevice is required for the audio backend (pip install '.[audio]')"
            ) from exc

        self.input_device = resolve_device(audio.get("input_device"), "input")
        self.output_device = resolve_device(audio.get("output_device"), "output")
        self.input_channel = int(audio.get("input_channel", 1))
        self.output_channel = int(audio.get("output_channel", 1))
        self.output_gain = amplitude(float(audio.get("output_gain_db", -12.0)))
        self.input_gain = amplitude(float(audio.get("input_gain_db", 0.0)))
        self.latency = audio.get("latency", "low")
        self.settle_s = float(audio.get("settle_s", 0.35))
        self.timeout_s = float(audio.get("timeout_s", 120.0))
        self.retries = int(audio.get("retries", 2))
        self.controls: PedalControls | None = None

    # -- Backend ----------------------------------------------------------- #
    def configure(self, controls: PedalControls) -> None:
        self.controls = controls
        self.transport.set_bypass(controls.bypass)
        if controls.tap_interval_ms and not controls.bypass:
            import time

            interval = controls.tap_interval_ms / 1000.0
            self.transport.tap()
            time.sleep(interval)
            self.transport.tap()

    def render(self, stimulus: np.ndarray, sample_rate: int, name: str | None = None) -> np.ndarray:
        import sounddevice as sd

        controls = self.controls
        if controls is None:
            raise RuntimeError("configure() must be called before render()")

        stimulus = np.asarray(stimulus, dtype=np.float64) * self.output_gain
        settle = int(self.settle_s * sample_rate)
        padded = np.concatenate([np.zeros(settle), stimulus])
        data = np.ascontiguousarray(padded[:, None], dtype=np.float32)

        if controls.stutter_enabled:
            self.transport.schedule(controls.stutter_trigger_s + self.settle_s)

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                device = (self.input_device, self.output_device)
                if self.input_device is None or self.output_device is None:
                    device = self.output_device or self.input_device
                recording = sd.playrec(
                    data,
                    samplerate=sample_rate,
                    channels=1,
                    dtype="float32",
                    device=device,
                    input_mapping=[self.input_channel],
                    output_mapping=[self.output_channel],
                    latency=self.latency,
                    blocking=True,
                )
                sd.wait()
                break
            except Exception as exc:  # xruns, device busy, dropped frames
                last_error = exc
                if attempt == self.retries:
                    raise RuntimeError(f"audio capture failed after {self.retries + 1} attempts: {exc}")
        else:  # pragma: no cover - defensive
            raise RuntimeError(str(last_error))

        captured = np.asarray(recording[:, 0], dtype=np.float64)[settle:]
        captured = captured[: len(stimulus)] * self.input_gain
        self.transport.cancel()
        return captured

    def close(self) -> None:
        self.transport.cancel()
        self.transport.close()

    def describe(self) -> dict[str, Any]:
        import sounddevice as sd

        default_in, default_out = sd.default.device
        return {
            "backend": self.name,
            "input_device": self.input_device if self.input_device is not None else default_in,
            "output_device": self.output_device if self.output_device is not None else default_out,
            "input_channel": self.input_channel,
            "output_channel": self.output_channel,
            "output_gain_db": 20 * np.log10(self.output_gain),
            "settle_s": self.settle_s,
            "transport": self.transport._inner.name,
        }
