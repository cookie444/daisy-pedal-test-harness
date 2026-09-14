"""Result types and pass/fail evaluation for the Daisy HIL rig.

A single :class:`SpecResult` is one measurable spec (e.g. ``delay_time_ms``)
with a measured value and a tolerance window resolved from the rig config.
:class:`RunReport` is the collection for one backend run and is the only thing
the JUnit / HTML writers need.
"""

from __future__ import annotations

import math
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "Threshold",
    "SpecResult",
    "RunReport",
    "evaluate",
    "resolve_window",
]


@dataclass(frozen=True)
class Threshold:
    """Tolerance window for one spec.

    Either absolute bounds (``lower`` / ``upper``) or a relative window around
    an expected value (``tol_pct`` / ``tol_abs``). ``tol_pct`` is resolved at
    evaluation time because the expected value is often derived from the pedal
    settings (delay knob, tap interval, feedback coefficient...).
    """

    lower: float | None = None
    upper: float | None = None
    tol_pct: float | None = None
    tol_abs: float | None = None
    unit: str = ""
    description: str = ""

    @classmethod
    def from_mapping(cls, name: str, data: dict[str, Any] | None) -> "Threshold":
        data = data or {}
        known = {"lower", "upper", "tol_pct", "tol_abs", "unit", "description"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"threshold '{name}' has unknown keys: {sorted(unknown)}")
        return cls(**data)


def resolve_window(
    threshold: Threshold, expected: float | None = None
) -> tuple[float | None, float | None]:
    """Resolve (lower, upper) for a threshold, given an optional expected value."""
    lower, upper = threshold.lower, threshold.upper
    if threshold.tol_pct is not None or threshold.tol_abs is not None:
        if expected is None:
            raise ValueError("relative tolerance requires an expected value")
        if threshold.tol_pct is not None:
            delta = abs(expected) * threshold.tol_pct / 100.0
        else:
            delta = float(threshold.tol_abs or 0.0)
        lower = expected - delta if lower is None else max(lower, expected - delta)
        upper = expected + delta if upper is None else min(upper, expected + delta)
    return lower, upper


@dataclass
class SpecResult:
    """One measured spec and its verdict."""

    name: str
    group: str
    value: float
    unit: str = ""
    expected: float | None = None
    lower: float | None = None
    upper: float | None = None
    passed: bool = True
    notes: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    figures: list[dict[str, str]] = field(default_factory=list)

    @property
    def classname(self) -> str:
        return self.group

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    @property
    def margin(self) -> float | None:
        """Signed distance to the nearest limit. Negative means out of spec."""
        bounds = [b for b in (self.lower, self.upper) if b is not None]
        if not bounds or not math.isfinite(self.value):
            return None
        return min(abs(self.value - b) for b in bounds)

    def summary_line(self) -> str:
        window = f"[{_fmt(self.lower)} .. {_fmt(self.upper)}]"
        return f"{self.status} {self.group}.{self.name}: {_fmt(self.value)}{self.unit} {window}"

    def add_figure(self, title: str, png_base64: str, caption: str = "") -> None:
        self.figures.append({"title": title, "caption": caption, "png": png_base64})


@dataclass
class RunReport:
    """Everything produced by one pass of the spec suite."""

    backend: str
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    sample_rate: int = 0
    config: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    results: list[SpecResult] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> list[SpecResult]:
        return [r for r in self.results if not r.passed]

    @property
    def counts(self) -> dict[str, int]:
        return {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
        }

    def by_group(self) -> dict[str, list[SpecResult]]:
        groups: dict[str, list[SpecResult]] = {}
        for r in self.results:
            groups.setdefault(r.group, []).append(r)
        return groups

    def add(self, result: SpecResult) -> SpecResult:
        self.results.append(result)
        return result


def evaluate(
    name: str,
    group: str,
    value: float,
    threshold: Threshold,
    expected: float | None = None,
    *,
    details: dict[str, Any] | None = None,
    notes: str = "",
) -> SpecResult:
    """Build a :class:`SpecResult`, verdict included."""
    lower, upper = resolve_window(threshold, expected)
    if value is None or not math.isfinite(float(value)):
        passed, value_out = False, float("nan")
    else:
        value_out = float(value)
        passed = (lower is None or value_out >= lower) and (upper is None or value_out <= upper)
    return SpecResult(
        name=name,
        group=group,
        value=value_out,
        unit=threshold.unit,
        expected=expected,
        lower=lower,
        upper=upper,
        passed=passed,
        notes=notes,
        details=details or {},
    )


def host_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if not math.isfinite(value):
        return "nan"
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.4g}"
