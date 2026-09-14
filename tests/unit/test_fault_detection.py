"""Fault injection proves the rig actually fails.

A test harness that never fails is a harness nobody trusts. Each of these tests
simulates a firmware regression and asserts the corresponding spec moves outside
its window. If any one of them starts passing, the threshold was loosened too far
to catch that bug.

The fault changes the simulator *plant*, not the measurement, so these are
end-to-end detections, not unit mocks.
"""

from __future__ import annotations

import copy

import pytest

from daisyhil import load_config, run_suite
from daisyhil.backends.simulator import Faults, SimulatorBackend

CONFIG = load_config()

# (fault attribute, injected value, spec that must fail, why it matters)
FAULTS = [
    ("delay_drift_pct", 25.0, "delay_time_ms", "delay clock drifts 25%"),
    ("decay_error_pct", 40.0, "decay_db_per_repeat", "feedback gain error (repeats ring too long)"),
    ("thd_add_pct", 100.0, "thd_pct", "output-stage distortion rises past spec"),
    ("noise_floor_dbfs", -60.0, "noise_floor_dbfs", "power-supply hiss floor"),
    ("stutter_slice_error_pct", 20.0, "stutter_slice_ms", "stutter slice length off by 20%"),
    ("latency_extra_ms", 5.0, "processing_latency_ms", "extra buffering latency"),
    ("high_corner_hz", 8000.0, "fr_corner_high_hz", "bandwidth collapses to 8 kHz"),
]


def _run_with(fault: str, value: float):
    cfg = copy.deepcopy(CONFIG)
    faults = Faults.from_config(cfg)
    setattr(faults, fault, value)
    backend = SimulatorBackend(cfg, faults=faults)
    report = run_suite(cfg, backend, with_plots=False)
    return report


def _spec(report, name):
    return next(r for r in report.results if r.name == name)


def test_clean_simulator_passes_every_spec():
    report = run_suite(copy.deepcopy(CONFIG), with_plots=False)
    assert report.passed, "\n".join(r.summary_line() for r in report.failures)
    assert report.counts["failed"] == 0


@pytest.mark.parametrize("fault,value,spec,reason", FAULTS, ids=[f[0] for f in FAULTS])
def test_fault_is_detected(fault: str, value: float, spec: str, reason: str):
    report = _run_with(fault, value)
    target = _spec(report, spec)
    assert not target.passed, (
        f"fault {fault}={value} was NOT caught by {spec} ({reason}). "
        f"measured={target.value}{target.unit} window=[{target.lower} .. {target.upper}]"
    )
