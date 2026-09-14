"""Guard rails on the suite itself.

The contract between the code and the rig YAML is the interesting artefact: a
threshold that nothing measures is dead config, and a spec without a threshold
is unenforceable. These tests keep the two in lockstep.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from daisyhil import load_config, run_suite
from daisyhil.reporting.junit import to_junit_string

EXPECTED_SPECS = {
    "delay_time_ms",
    "tap_delay_ms",
    "processing_latency_ms",
    "decay_db_per_repeat",
    "decay_time_ms",
    "thd_pct",
    "thd_n_pct",
    "snr_db",
    "gain_1khz_db",
    "fr_ripple_db",
    "fr_corner_low_hz",
    "fr_corner_high_hz",
    "stutter_slice_ms",
    "stutter_gate_ms",
    "stutter_repeats",
    "noise_floor_dbfs",
}


def test_every_threshold_in_config_is_measured():
    cfg = load_config()
    report = run_suite(cfg, with_plots=False)
    measured = {r.name for r in report.results}
    orphaned = set(cfg.thresholds) - measured
    assert not orphaned, f"thresholds with no matching spec: {sorted(orphaned)}"


def test_every_spec_has_a_threshold():
    cfg = load_config()
    report = run_suite(cfg, with_plots=False)
    missing = {r.name for r in report.results} - set(cfg.thresholds)
    assert not missing, f"specs with no configured threshold: {sorted(missing)}"


def test_suite_produces_the_full_spec_list():
    cfg = load_config()
    report = run_suite(cfg, with_plots=False)
    names = {r.name for r in report.results}
    assert names == EXPECTED_SPECS, f"spec list changed: {sorted(names)}"


def test_junit_xml_round_trips_all_specs():
    cfg = load_config()
    report = run_suite(cfg, with_plots=False)
    root = ET.fromstring(to_junit_string(report))
    suite = root.find("testsuite")
    assert int(suite.get("tests")) == len(report.results)
    assert int(suite.get("failures")) == report.counts["failed"]
    xml_names = {tc.get("name") for tc in suite.findall("testcase")}
    assert xml_names == EXPECTED_SPECS
    # every case carries its measured value for dashboards
    first = suite.find("testcase")
    props = {p.get("name"): p.get("value") for p in first.find("properties")}
    assert "measured" in props and "unit" in props and "lower" in props
