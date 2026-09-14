"""The spec suite, expressed as pytest cases so `pytest --junit-xml` drives CI.

One testcase per measured spec. The report is built once at collection time and
cached; the backend comes from ``HIL_BACKEND`` (default ``sim`` so a normal
``pytest`` run needs no hardware). A hardware runner sets ``HIL_BACKEND=audio``.
"""

from __future__ import annotations

import os

import pytest

from daisyhil import load_config, make_backend, run_suite

BACKEND = os.environ.get("HIL_BACKEND", "sim")
pytestmark = pytest.mark.hil if BACKEND != "sim" else pytest.mark.sim


def _build_report():
    cfg = load_config(os.environ.get("HIL_CONFIG") or None)
    try:
        backend = make_backend(BACKEND, cfg)
    except Exception as exc:  # no device / no backend available
        pytest.skip(f"backend {BACKEND!r} unavailable: {exc}", allow_module_level=True)
    return run_suite(cfg, backend, with_plots=False)


REPORT = _build_report()


@pytest.mark.parametrize("spec", REPORT.results, ids=lambda s: f"{s.group}.{s.name}")
def test_spec(spec):
    assert spec.passed, (
        f"{spec.group}.{spec.name} = {spec.value}{spec.unit} "
        f"outside [{spec.lower} .. {spec.upper}]; {spec.notes}"
    )
