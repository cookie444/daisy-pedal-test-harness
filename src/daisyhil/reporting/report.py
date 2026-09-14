"""HTML report: summary table plus the waveform/spectrum figures per group."""

from __future__ import annotations

import math
from pathlib import Path

import yaml
from jinja2 import Environment, StrictUndefined

from ..results import RunReport

_TEMPLATE_DIR = Path(__file__).parent / "templates"

__all__ = ["write_report", "render_report"]


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


def render_report(report: RunReport) -> str:
    env = Environment(undefined=StrictUndefined, autoescape=True, keep_trailing_newline=True)
    template = env.from_string((_TEMPLATE_DIR / "report.html.j2").read_text(encoding="utf-8"))

    def clean(data):
        """Drop numpy arrays / non-serialisable values from the config dump."""
        if isinstance(data, dict):
            return {k: clean(v) for k, v in data.items() if not hasattr(v, "dtype")}
        if isinstance(data, (list, tuple)):
            return [clean(v) for v in data if not hasattr(v, "dtype")]
        return data

    return template.render(
        report=report,
        counts=report.counts,
        fmt=_fmt,
        env_yaml=yaml.safe_dump(clean(report.environment), sort_keys=False, default_flow_style=False),
        config_yaml=yaml.safe_dump(clean(report.config), sort_keys=False, default_flow_style=False),
    )


def write_report(report: RunReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(report), encoding="utf-8")
    return path
