"""``undertow-data`` CLI — the entry point for the data pipeline (T14).

Four subcommands: ``pull``, ``verify``, ``snapshot``, ``info``. Each delegates to
``pipeline.py``; this module does argument parsing, exit codes, and human/machine output
and nothing else. The reviewer will check that orchestration logic lives in
``pipeline.py``, not here.

Exit codes: ``0`` ok, ``1`` critical check failed, ``2`` config error, ``3`` fetch error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from undertow.data.config import DataConfig, load_config
from undertow.data.pipeline import (
    info,
    pull,
    snapshot,
    verify,
)
from undertow.data.pipeline_report import (
    InfoReport,
    PullReport,
    SnapshotReport,
)
from undertow.data.types import ConfigError, FetchError, Route, ValidationError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="undertow-data",
        description="Uniswap V3 data pipeline — pull, verify, snapshot, info.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="re-raise exceptions with full tracebacks (default: clean error messages)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- pull ---
    pull_p = sub.add_parser("pull", help="fetch streams and write to parquet")
    pull_p.add_argument("--config", required=True, help="path to TOML config file")
    pull_p.add_argument(
        "--stream",
        action="append",
        dest="streams",
        default=None,
        help="stream to pull (repeatable; default: all)",
    )
    pull_p.add_argument(
        "--route",
        choices=("thegraph", "rpc"),
        default=None,
        help="force a route for log streams (default: thegraph for logs, rpc for fee_growth)",
    )
    pull_p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="re-fetch even if already written",
    )
    pull_p.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="machine-readable JSON output",
    )

    # --- verify ---
    verify_p = sub.add_parser("verify", help="run validation checks on a built dataset")
    verify_p.add_argument("--config", required=True, help="path to TOML config file")
    verify_p.add_argument(
        "--sample-windows",
        type=int,
        default=1,
        help="number of route-sampling windows (default: 1)",
    )
    verify_p.add_argument(
        "--fail-on",
        choices=("warning", "critical"),
        default="critical",
        help="severity threshold for non-zero exit (default: critical)",
    )
    verify_p.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="machine-readable JSON output",
    )

    # --- snapshot ---
    snap_p = sub.add_parser("snapshot", help="pull + build + verify + manifest → one command")
    snap_p.add_argument("--config", required=True, help="path to TOML config file")
    snap_p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory override (default: config.output_dir)",
    )
    snap_p.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="machine-readable JSON output",
    )

    # --- info ---
    info_p = sub.add_parser("info", help="print manifest summary and row counts")
    info_p.add_argument("--config", required=True, help="path to TOML config file")
    info_p.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="machine-readable JSON output",
    )

    return parser


def _load_config(path: str) -> DataConfig:
    """Load and return a DataConfig, translating ConfigError to exit code 2."""
    try:
        return load_config(path)
    except ConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(2)


def _die(message: str, *, exit_code: int) -> int:
    """Print an error to stderr and return an exit code.

    Returns the code rather than raising SystemExit — callers propagate via
    ``return _die(...)``.  This keeps the function testable without
    catching BaseException.
    """
    print(f"Error: {message}", file=sys.stderr)
    return exit_code


def _format_pull_report(report: PullReport, *, json_output: bool) -> str:
    """Render a PullReport as human-readable text or JSON."""
    if json_output:
        obj = {
            "streams": {
                name: {
                    "route": spr.route.value,
                    "row_count": spr.row_count,
                    "n_requests": spr.n_requests,
                    "from_cache": spr.from_cache,
                    "warnings": list(spr.warnings),
                    "content_hash": spr.content_hash,
                }
                for name, spr in report.streams.items()
            },
            "n_total_requests": report.n_total_requests,
            "all_from_cache": report.all_from_cache,
            "started_at_utc": report.started_at_utc.isoformat(),
            "finished_at_utc": report.finished_at_utc.isoformat(),
        }
        return json.dumps(obj, indent=2)

    lines = ["Pull complete."]
    if report.all_from_cache:
        lines.append("  All streams were already cached (0 network requests).")
    else:
        lines.append(f"  Total requests: {report.n_total_requests}")
    lines.append("")
    for name, spr in sorted(report.streams.items()):
        cache_tag = " (cached)" if spr.from_cache else ""
        lines.append(
            f"  {name:12s}  {spr.route.value:10s}  "
            f"{spr.row_count:>8d} rows  "
            f"{spr.n_requests:>4d} req{cache_tag}"
        )
        for w in spr.warnings:
            lines.append(f"           ⚠ {w}")
    return "\n".join(lines)


def _format_checks(
    checks: list, *, json_output: bool, fail_on: str
) -> tuple[str, int]:
    """Render check results; return (text, exit_code).

    Exit code is 0 if all checks at or above the fail_on severity pass, else 1.
    """
    if json_output:
        obj = [
            {
                "name": c.name,
                "passed": c.passed,
                "severity": c.severity,
                "detail": c.detail,
                "metrics": dict(c.metrics),
            }
            for c in checks
        ]
        # Determine exit code based on --fail-on threshold.
        if fail_on == "warning":
            exit_code = 1 if any(not c.passed and c.severity != "info" for c in checks) else 0
        else:
            exit_code = 1 if any(not c.passed and c.severity == "critical" for c in checks) else 0
        return json.dumps(obj, indent=2), exit_code

    lines = []
    failures = [c for c in checks if not c.passed and c.severity != "info"]

    lines.append(f"Validation: {len(checks)} check(s) run.")
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        lines.append(f"  [{c.severity:8s}] {status:4s}  {c.name}")
        if not c.passed and c.detail:
            lines.append(f"           {c.detail}")

    n_fail = len(failures)
    n_crit = len([c for c in checks if c.severity == "critical"])
    n_warn = len([c for c in checks if c.severity == "warning"])
    n_info = len([c for c in checks if c.severity == "info"])
    lines.append("")
    lines.append(
        f"Summary: {n_fail} failed, "
        f"{n_crit} critical, {n_warn} warning, "
        f"{n_info} info."
    )

    # Determine exit code based on --fail-on threshold.
    if fail_on == "warning":
        exit_code = 1 if any(not c.passed and c.severity != "info" for c in checks) else 0
    else:
        exit_code = 1 if any(not c.passed and c.severity == "critical" for c in checks) else 0
    return "\n".join(lines), exit_code


def _format_snapshot_report(report: SnapshotReport, *, json_output: bool) -> str:
    if json_output:
        obj = {
            "manifest_path": str(report.manifest_path),
            "validation_report_path": str(report.validation_report_path),
            "pull": {
                "n_total_requests": report.pull.n_total_requests,
                "all_from_cache": report.pull.all_from_cache,
                "streams": {
                    name: {"route": spr.route.value, "row_count": spr.row_count}
                    for name, spr in report.pull.streams.items()
                },
            },
            "checks": [
                {"name": c.name, "passed": c.passed, "severity": c.severity}
                for c in report.checks
            ],
        }
        return json.dumps(obj, indent=2)

    lines = [
        "Snapshot complete.",
        f"  Manifest:  {report.manifest_path}",
        f"  Report:    {report.validation_report_path}",
        f"  Requests:  {report.pull.n_total_requests}",
    ]
    n_pass = sum(1 for c in report.checks if c.passed)
    n_fail = sum(1 for c in report.checks if not c.passed)
    lines.append(f"  Checks:    {n_pass} passed, {n_fail} failed")
    return "\n".join(lines)


def _format_info_report(report: InfoReport, *, json_output: bool) -> str:
    """Render an InfoReport."""
    m = report.manifest
    if json_output:
        obj = {
            "dataset_id": m.dataset_id,
            "schema_version": m.schema_version,
            "dataset_exists": report.dataset_exists,
            "output_dir": str(report.output_dir),
            "pool": m.pool.address,
            "window": {
                "start_block": m.window.start_block,
                "end_block": m.window.end_block,
                "start_utc": (
                    m.window.start_utc.isoformat()
                    if m.window.start_utc is not None
                    else None
                ),
                "end_utc": (
                    m.window.end_utc.isoformat()
                    if m.window.end_utc is not None
                    else None
                ),
            },
            "streams": {
                name: {
                    "row_count": sm.row_count,
                    "route": sm.route.value if sm.route is not None else None,
                    "min_block": sm.min_block,
                    "max_block": sm.max_block,
                }
                for name, sm in m.streams.items()
            },
            "git_commit": m.git_commit,
            "warnings": list(m.warnings),
        }
        return json.dumps(obj, indent=2)

    lines = [
        f"Dataset: {m.dataset_id}",
        f"  Schema version: {m.schema_version}",
        f"  Pool:           {m.pool.address} "
        f"({m.pool.token0_symbol}/{m.pool.token1_symbol}, "
        f"{m.pool.fee_tier.value} bps)",
    ]
    w = m.window
    if w.start_block is not None:
        lines.append(f"  Window:         blocks [{w.start_block}, {w.end_block}]")
    else:
        lines.append(
            f"  Window:         {w.start_utc} → {w.end_utc} (UTC)"
            if w.start_utc is not None
            else "  Window:         (unknown)"
        )
    if report.dataset_exists:
        lines.append("  Status:         pulled")
    else:
        lines.append("  Status:         NOT PULLED — run 'undertow-data pull' first")
    lines.append("")
    if m.streams:
        lines.append("  Streams:")
        for name, sm in sorted(m.streams.items()):
            block_range = ""
            if sm.min_block is not None:
                block_range = f"  blocks [{sm.min_block}, {sm.max_block}]"
            route_str = sm.route.value if sm.route is not None else "—"
            lines.append(
                f"    {name:14s} {sm.row_count:>10,d} rows  "
                f"{route_str:10s}{block_range}"
            )
    else:
        lines.append("  (no streams pulled yet)")
    if m.warnings:
        lines.append("")
        lines.append("  Warnings:")
        for w_msg in m.warnings:
            lines.append(f"    ⚠ {w_msg}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point registered as ``[project.scripts] undertow-data``.

    Returns an exit code (0-3) rather than calling ``sys.exit`` so tests can
    assert on the return value. ``cli.py`` never touches ``sys.exit`` directly
    — unresolved ``SystemExit`` is caught in the ``__name__ == '__main__'`` guard.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    debug: bool = args.debug

    try:
        if args.command == "pull":
            return _cmd_pull(args)
        elif args.command == "verify":
            return _cmd_verify(args)
        elif args.command == "snapshot":
            return _cmd_snapshot(args)
        elif args.command == "info":
            return _cmd_info(args)
        else:
            parser.print_help()
            return 2
    except ConfigError as e:
        if debug:
            raise
        print(f"Config error: {e}", file=sys.stderr)
        return 2
    except FetchError as e:
        if debug:
            raise
        print(f"Fetch error: {e}", file=sys.stderr)
        return 3
    except ValidationError as e:
        if debug:
            raise
        print(f"Validation error: {e}", file=sys.stderr)
        return 1
    except SystemExit:
        raise  # propagate — _load_config uses this for bad config path
    except Exception:
        if debug:
            raise
        import traceback

        traceback.print_exc()
        return 3


def _cmd_pull(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    route = None
    if args.route is not None:
        route = Route(args.route)
    report = pull(
        config,
        streams=args.streams,
        route=route,
        force=args.force,
    )
    print(_format_pull_report(report, json_output=args.json))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    checks = verify(config)
    output, exit_code = _format_checks(
        checks, json_output=args.json, fail_on=args.fail_on
    )
    print(output)
    return exit_code


def _cmd_snapshot(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    report = snapshot(config, out=args.out)
    print(_format_snapshot_report(report, json_output=args.json))
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    report = info(config)
    print(_format_info_report(report, json_output=args.json))
    return 0 if report.dataset_exists else 1


if __name__ == "__main__":
    sys.exit(main())