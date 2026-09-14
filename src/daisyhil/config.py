"""Rig configuration: sample rate, pedal settings, tolerances, fault injection.

Everything the harness needs to know about the *expected* behaviour lives in one
YAML file (``config/rig.yaml``) so that thresholds are reviewable in PRs without
touching Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .results import Threshold

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "rig.yaml"


@dataclass
class PedalControls:
    """State applied to the pedal before a capture."""

    delay_ms: float = 375.0
    tap_interval_ms: float | None = None
    feedback: float = 0.55
    mix: float = 0.5
    tone_hz: float | None = None
    drive: float = 0.0
    bypass: bool = False
    stutter_enabled: bool = False
    stutter_slice_ms: float = 125.0
    stutter_repeats: int = 4
    stutter_trigger_s: float = 0.6

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def label(self) -> str:
        if self.bypass:
            return "bypass"
        bits = [f"dly={self.delay_ms:g}ms", f"fb={self.feedback:g}", f"mix={self.mix:g}"]
        if self.tap_interval_ms:
            bits[0] = f"tap={self.tap_interval_ms:g}ms"
        if self.stutter_enabled:
            bits.append(f"stut={self.stutter_slice_ms:g}ms x{self.stutter_repeats}")
        return " ".join(bits)


@dataclass
class RigConfig:
    sample_rate: int = 48000
    block_size: int = 48
    pre_roll_s: float = 0.25
    post_roll_s: float = 0.25
    stimulus_level_dbfs: float = -12.0
    pedal: PedalControls = field(default_factory=PedalControls)
    thresholds: dict[str, Threshold] = field(default_factory=dict)
    faults: dict[str, float] = field(default_factory=dict)
    audio: dict[str, Any] = field(default_factory=dict)
    stimulus: dict[str, Any] = field(default_factory=dict)
    plugins: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def threshold(self, name: str) -> Threshold:
        if name not in self.thresholds:
            raise KeyError(f"no threshold configured for '{name}'")
        return self.thresholds[name]

    def fault(self, name: str, default: float = 0.0) -> float:
        return float(self.faults.get(name, default))

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "RigConfig":
        rig = dict(data.get("rig", {}))
        pedal = dict(data.get("pedal", {}))
        # tap interval defaults to the knob delay so delay/tap agree unless stated
        pedal.setdefault("tap_interval_ms", pedal.get("delay_ms"))
        thresholds = {
            name: Threshold.from_mapping(name, spec) for name, spec in (data.get("thresholds") or {}).items()
        }
        return cls(
            sample_rate=int(rig.get("sample_rate", 48000)),
            block_size=int(rig.get("block_size", 48)),
            pre_roll_s=float(rig.get("pre_roll_s", 0.25)),
            post_roll_s=float(rig.get("post_roll_s", 0.25)),
            stimulus_level_dbfs=float(rig.get("stimulus_level_dbfs", -12.0)),
            pedal=PedalControls(**pedal),
            thresholds=thresholds,
            faults=dict(data.get("faults") or {}),
            audio=dict(data.get("audio") or {}),
            stimulus=dict(data.get("stimulus") or {}),
            plugins=dict(data.get("plugins") or {}),
            raw=data,
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "RigConfig":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        with path.open("r", encoding="utf-8") as fh:
            return cls.from_mapping(yaml.safe_load(fh) or {})

    def with_overrides(self, **kwargs: Any) -> "RigConfig":
        """Return a shallow copy with selected fields replaced (used by fault tests)."""
        import copy

        clone = copy.deepcopy(self)
        for key, value in kwargs.items():
            if not hasattr(clone, key):
                raise AttributeError(f"unknown config field '{key}'")
            setattr(clone, key, value)
        return clone


def load_config(path: str | Path | None = None) -> RigConfig:
    return RigConfig.load(path)
