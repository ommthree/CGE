"""Minimal CLI.

cge engines                                     list registered engines
cge run --scenario examples/carbon_price_toy.yaml
cge build --exiobase [--year 2019]              live EXIOBASE build (downloads)
cge build --test                                offline build from pymrio test MRIO
cge data                                         list data builds in the store
cge quality <build_id>                           show a build's quality report
cge validate [--suite io_price] [--strict]       run the model validation suite
cge gui                                          launch the Streamlit web GUI
"""

from __future__ import annotations

import argparse
import json
import sys

import cge.engines  # noqa: F401  (registers engines)
from cge.contracts.engine import registry
from cge.runner import run_scenario
from cge.scenarios.loader import load_scenario


def _cmd_engines(_: argparse.Namespace) -> int:
    for meta in registry.all_meta():
        caps = ", ".join(c.value for c in meta.capabilities)
        print(f"{meta.name} v{meta.version} [{caps}] — {meta.description}")
        print(f"    shocks: {', '.join(meta.supported_shocks)}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    result = run_scenario(scenario, data_source=args.data)
    df = result.data
    print(f"Scenario: {scenario.name}  (engine={scenario.engine}, rows={len(df)})")
    print(f"Scenario hash: {result.manifest.scenario_hash}")
    print("Assumptions:")
    print(json.dumps(result.manifest.assumptions, indent=2))
    print("\nResults (head):")
    print(df.head(12).to_string(index=False))
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    from cge.data.build import build_exiobase, build_test

    if args.test:
        written = build_test()
    elif args.exiobase:
        written = build_exiobase(year=args.year, make_small=not args.no_small)
    else:
        print("Specify --exiobase (live download) or --test (offline).", file=sys.stderr)
        return 2
    for kind, build_id in written.items():
        print(f"built {kind}: {build_id}")
    return 0


def _cmd_data(_: argparse.Namespace) -> int:
    from cge.data.store import default_store

    cat = default_store().catalogue()
    if cat.empty:
        print("No builds yet. Run 'cge build --test' or 'cge build --exiobase'.")
        return 0
    print(cat.to_string(index=False))
    return 0


def _cmd_quality(args: argparse.Namespace) -> int:
    from cge.data.store import default_store

    report = default_store().load_quality(args.build_id)
    if report is None:
        print(f"No quality report for build {args.build_id!r}.", file=sys.stderr)
        return 1
    print(f"Quality report for {report.build_id}  (worst: {report.worst.value})")
    print(f"Summary: {report.summary()}")
    for c in report.checks:
        print(f"  [{c.severity.value:>4}] {c.name}: {c.message}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from cge.validation import run_all
    from cge.validation.report import format_text

    summary = run_all(only=args.suite)
    print(format_text(summary))
    return 1 if (args.strict and not summary.passed) else 0


def _cmd_gui(args: argparse.Namespace) -> int:
    import subprocess

    from cge.gui import APP_PATH

    cmd = ["streamlit", "run", APP_PATH]
    if args.port:
        cmd += ["--server.port", str(args.port)]
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print(
            "Streamlit not installed. Install the GUI extra: pip install -e '.[gui]'",
            file=sys.stderr,
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cge", description="CGE/IAM platform CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("engines", help="list registered engines").set_defaults(func=_cmd_engines)

    run = sub.add_parser("run", help="run a scenario file")
    run.add_argument("--scenario", required=True, help="path to a scenario YAML")
    run.add_argument("--data", default="toy", help="data source: 'toy' or a build id")
    run.set_defaults(func=_cmd_run)

    build = sub.add_parser("build", help="build a dataset into the store")
    build.add_argument("--exiobase", action="store_true", help="live EXIOBASE download build")
    build.add_argument("--test", action="store_true", help="offline build from pymrio test MRIO")
    build.add_argument("--year", type=int, default=2019, help="EXIOBASE year (with --exiobase)")
    build.add_argument("--no-small", action="store_true", help="skip the aggregated small build")
    build.set_defaults(func=_cmd_build)

    sub.add_parser("data", help="list data builds in the store").set_defaults(func=_cmd_data)

    quality = sub.add_parser("quality", help="show a build's quality report")
    quality.add_argument("build_id")
    quality.set_defaults(func=_cmd_quality)

    validate = sub.add_parser("validate", help="run the model validation suite")
    validate.add_argument("--suite", action="append", help="limit to named suite(s)")
    validate.add_argument("--strict", action="store_true", help="exit non-zero on any failure")
    validate.set_defaults(func=_cmd_validate)

    gui = sub.add_parser("gui", help="launch the Streamlit web GUI")
    gui.add_argument("--port", type=int, default=None, help="port for the Streamlit server")
    gui.set_defaults(func=_cmd_gui)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
