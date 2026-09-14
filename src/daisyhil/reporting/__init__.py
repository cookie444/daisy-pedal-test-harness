"""Reporting: JUnit XML for CI, HTML for humans."""

from .junit import write_junit  # noqa: F401
from .report import write_report  # noqa: F401

__all__ = ["write_junit", "write_report"]
