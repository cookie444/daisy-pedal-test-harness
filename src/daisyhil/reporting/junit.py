"""Minimal, dependency-free JUnit XML writer.

One ``<testsuite>`` per run, one ``<testcase>`` per measured spec. Measured
values, limits and units ride along as ``<property>`` elements so tools like
``mikepenz/action-junit-report`` or a test-report dashboard can show the number,
not just the verdict.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

from ..results import RunReport, SpecResult

__all__ = ["write_junit", "to_junit_string"]


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    if value != value or value in (float("inf"), float("-inf")):
        return "nan"
    return f"{value:.6g}"


def _testcase(parent: ET.Element, spec: SpecResult, suite_name: str) -> None:
    case = ET.SubElement(
        parent,
        "testcase",
        {
            "classname": f"{suite_name}.{spec.classname}",
            "name": spec.name,
            "time": "0",
        },
    )
    properties = ET.SubElement(case, "properties")
    for key, value in (
        ("measured", _fmt(spec.value)),
        ("unit", spec.unit),
        ("expected", _fmt(spec.expected)),
        ("lower", _fmt(spec.lower)),
        ("upper", _fmt(spec.upper)),
        ("group", spec.group),
        ("notes", spec.notes),
    ):
        ET.SubElement(properties, "property", {"name": key, "value": str(value)})

    if not spec.passed:
        window = f"[{_fmt(spec.lower)} .. {_fmt(spec.upper)}] {spec.unit}".strip()
        message = f"{spec.name} = {_fmt(spec.value)}{spec.unit} outside {window}"
        failure = ET.SubElement(case, "failure", {"message": message, "type": "SpecOutOfTolerance"})
        detail = [
            message,
            f"expected: {_fmt(spec.expected)}{spec.unit}",
            f"notes:    {spec.notes}",
        ]
        for key, value in spec.details.items():
            if isinstance(value, (int, float, str)):
                detail.append(f"{key}: {value}")
        failure.text = "\n".join(line for line in detail if line)
    elif spec.notes:
        ET.SubElement(case, "system-out").text = spec.notes


def to_junit_string(report: RunReport, suite_name: str | None = None) -> str:
    """Serialize a :class:`RunReport` to a JUnit XML document."""
    suite_name = suite_name or f"DaisyHIL.{report.backend}"
    root = ET.Element("testsuites")
    counts = report.counts
    suite = ET.SubElement(
        root,
        "testsuite",
        {
            "name": suite_name,
            "tests": str(counts["total"]),
            "failures": str(counts["failed"]),
            "errors": "0",
            "skipped": "0",
            "time": f"{report.duration_s:.3f}",
            "hostname": report.environment.get("platform", ""),
        },
    )
    ET.SubElement(suite, "properties")
    for spec in report.results:
        _testcase(suite, spec, suite_name)

    stdout = [
        f"backend={report.backend} sample_rate={report.sample_rate}",
        f"duration={report.duration_s:.2f}s",
        f"{counts['passed']}/{counts['total']} specs passed",
        "",
    ]
    stdout += [spec.summary_line() for spec in report.results]
    ET.SubElement(suite, "system-out").text = "\n".join(stdout)

    xml_bytes = ET.tostring(root, encoding="utf-8")
    return minidom.parseString(xml_bytes).toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")


def write_junit(report: RunReport, path: str | Path, suite_name: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_junit_string(report, suite_name), encoding="utf-8")
    return path
