"""Command line entry point.

    daisy-hil run --backend sim --config config/rig.yaml --out build
    daisy-hil run --backend audio            # Pod attached to this machine
    daisy-hil run --fault delay_drift_pct=25 # prove the rig fails
    daisy-hil devices
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import load_config
from .reporting import write_junit, write_report
from .suite import make_backend, run_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="daisy-hil", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version="daisy-hil 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the spec suite and write reports")
    run.add_argument(
        "--backend",
        default=os.environ.get("HIL_BACKEND", "sim"),
        choices=("sim", "audio", "wavfile"),
        help="sim = simulated DSP (CI), audio = attached Pod, wavfile = recorded captures",
    )
    run.add_argument("--config", default=os.environ.get("HIL_CONFIG"), help="rig YAML")
    run.add_argument("--out", default="build", help="output directory")
    run.add_argument("--junit-xml", default=None, help="JUnit XML path (default <out>/junit.xml)")
    run.add_argument("--html", default=None, help="HTML report path (default <out>/report.html)")
    run.add_argument("--no-plots", action="store_true", help="skip figures (faster CI)")
    run.add_argument("--save-captures", action="store_true", help="write each capture as WAV")
    run.add_argument(
        "--fault",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="inject a fault (simulator only), repeatable",
    )
    run.add_argument("--captures", default="captures", help="wavfile backend: capture directory")
    run.add_argument("--quiet", action="store_true")

    sub.add_parser("devices", help="list audio devices (audio backend)")
    return parser


def _run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    for fault in args.fault:
        if "=" not in fault:
            raise SystemExit(f"--fault expects KEY=VALUE, got {fault!r}")
        key, value = fault.split("=", 1)
        try:
            cfg.faults[key] = float(value)
        except ValueError:
            cfg.faults[key] = value

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if args.backend == "wavfile":
        kwargs["directory"] = Path(args.captures)

    backend = make_backend(args.backend, cfg, **kwargs)
    report = run_suite(
        cfg,
        backend,
        with_plots=not args.no_plots,
        save_captures=(out / "captures") if args.save_captures else None,
    )

    junit_path = write_junit(report, args.junit_xml or (out / "junit.xml"))
    html_path = write_report(report, args.html or (out / "report.html"))

    if not args.quiet:
        for spec in report.results:
            print(spec.summary_line())
        counts = report.counts
        print(
            f"\n{counts['passed']}/{counts['total']} specs passed "
            f"({report.duration_s:.1f}s, backend={report.backend})"
        )
        print(f"  JUnit XML: {junit_path}")
        print(f"  HTML:      {html_path}")
    return 0 if report.passed else 1


def _devices(_args: argparse.Namespace) -> int:
    from .backends.audio import list_devices

    for index, device in enumerate(list_devices()):
        print(
            f"{index:>3}  {device['name']:<50} in={device['max_input_channels']:<3} "
            f"out={device['max_output_channels']:<3} {device['default_samplerate']:.0f} Hz"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "devices":
        return _devices(args)
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
