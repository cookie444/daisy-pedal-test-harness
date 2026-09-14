"""Backend and transport interfaces.

A backend answers exactly one question: *given this stimulus and these pedal
settings, what comes out?*  Three implementations ship with the rig:

``sim``     - a reference DSP model of the pedal (CI, no hardware)
``audio``   - real play/record loopback through an attached Daisy Pod
``wavfile`` - pre-recorded captures on disk (offline analysis / triage)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import PedalControls

__all__ = ["Capture", "Backend", "Transport", "NullTransport"]


@dataclass
class Capture:
    """One recorded pass through the pedal."""

    name: str
    data: np.ndarray
    sample_rate: int
    stimulus: np.ndarray | None = None
    controls: PedalControls | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return len(self.data) / self.sample_rate


class Backend(ABC):
    """Plays a stimulus into the pedal and returns what came back."""

    name: str = "abstract"

    @abstractmethod
    def configure(self, controls: PedalControls) -> None:
        """Apply pedal settings before the next capture."""

    @abstractmethod
    def render(
        self, stimulus: np.ndarray, sample_rate: int, name: str | None = None
    ) -> np.ndarray:
        """Return the pedal output for ``stimulus`` (same length, aligned)."""

    def close(self) -> None:  # pragma: no cover - optional
        return None

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name}


class Transport(ABC):
    """Out-of-band pedal control: tap tempo, bypass, stutter trigger."""

    @abstractmethod
    def tap(self) -> None: ...

    @abstractmethod
    def set_bypass(self, enabled: bool) -> None: ...

    @abstractmethod
    def trigger_stutter(self) -> None: ...

    def close(self) -> None:  # pragma: no cover - optional
        return None


class NullTransport(Transport):
    """No-op control surface (simulator and file backends)."""

    name = "null"

    def tap(self) -> None:
        return None

    def set_bypass(self, enabled: bool) -> None:
        return None

    def trigger_stutter(self) -> None:
        return None
