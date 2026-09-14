"""Offline backend: analyse captures that were recorded earlier.

Useful for triage (attach a WAV to a bug) and for running the full report chain
on hardware captures produced outside the rig, e.g. by a bench recorder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from ..config import PedalControls, RigConfig
from .base import Backend

__all__ = ["WavFileBackend", "load_captures", "save_capture"]


class WavFileBackend(Backend):
    """Serves pre-recorded captures by test name."""

    name = "wavfile"

    def __init__(self, cfg: RigConfig, captures: dict[str, Path] | None = None, directory: Path | None = None):
        self.cfg = cfg
        if captures is None:
            captures = load_captures(directory or Path("captures"))
        self.captures = {k: Path(v) for k, v in captures.items()}
        self.controls: PedalControls | None = None
        self.used: list[str] = []

    def configure(self, controls: PedalControls) -> None:
        self.controls = controls

    def render(self, stimulus: np.ndarray, sample_rate: int, name: str | None = None) -> np.ndarray:
        if name is None:
            raise ValueError("the wavfile backend needs a capture name")
        if name not in self.captures:
            raise KeyError(f"no capture named '{name}' (have: {sorted(self.captures)})")
        path = self.captures[name]
        data, file_sr = sf.read(str(path), dtype="float64", always_2d=False)
        if data.ndim > 1:
            data = data[:, 0]
        if file_sr != sample_rate:
            from math import gcd

            ratio = sample_rate / file_sr
            up = int(round(ratio * 1000))
            down = 1000
            divisor = gcd(up, down)
            data = resample_poly(data, up // divisor, down // divisor)
        self.used.append(name)
        return np.asarray(data, dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "captures": {k: str(v) for k, v in self.captures.items()}}


def load_captures(directory: Path) -> dict[str, Path]:
    """Map ``<name>.wav`` files in ``directory`` to capture names."""
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    return {p.stem: p for p in sorted(directory.glob("*.wav"))}


def save_capture(path: Path, data: np.ndarray, sample_rate: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(data, dtype=np.float64), sample_rate, subtype="FLOAT")
    return path
