"""Hardware-in-the-loop audio test rig for Daisy pedals.

    from daisyhil import load_config, run_suite
    report = run_suite(load_config(), "sim")
    report.passed
"""

from .config import PedalControls, RigConfig, load_config
from .results import RunReport, SpecResult, Threshold
from .suite import SpecSuite, make_backend, run_suite

__version__ = "0.1.0"

__all__ = [
    "PedalControls",
    "RigConfig",
    "RunReport",
    "SpecResult",
    "SpecSuite",
    "Threshold",
    "load_config",
    "make_backend",
    "run_suite",
    "__version__",
]
