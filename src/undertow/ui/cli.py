"""``undertow-ui`` — command-line entry point for the live policy cockpit.

U01 ships the parser and the read-only ``info`` command.  ``serve`` is a
placeholder returning exit code ``2``; U05 replaces it with the FastAPI/uvicorn
composition root (`docs/plans/ui/PLAN.md` §2.1).

The CLI is a thin composition root: it parses arguments (``argparse``), resolves
a frozen :class:`~undertow.ui.config.UiConfig`, and delegates.  Exit codes match
``undertow-sim``: ``0`` ok, ``1`` runtime error, ``2`` config/usage error.

``info`` counts run directories inline rather than importing a registry — U02
owns the real scan (``undertow/ui/registry.py``); this walk is deliberately
local and dependency-free so the package imports without the web stack.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from undertow.ui.config import UiConfig

__all__ = ["main", "serve"]

PROG = "undertow-ui"

#: A directory is a run directory when it directly contains either of these
#: (CONTRACTS §3; U02's registry widens this set to include ``status.json``).
MANIFEST_FILENAME = "manifest.json"
TELEMETRY_FILENAME = "telemetry.jsonl"

DEFAULT_RUNS_DIR = Path("runs")
DEFAULT_RESULTS_DIR = Path("results")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2


class CliError(Exception):
    """A user-facing CLI failure carrying the process exit code."""

    def __init__(self, message: str, exit_code: int = EXIT_ERROR) -> None:
        super().__init__(message)
        self.exit_code = int(exit_code)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _add_runs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runs",
        default=str(DEFAULT_RUNS_DIR),
        help=f"directory tree to scan for run artifacts (default: {DEFAULT_RUNS_DIR})",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``undertow-ui`` argument parser."""
    defaults = UiConfig()
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Undertow live policy cockpit — inspect training runs and serve the "
            "read-only dashboard API."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="re-raise exceptions with full tracebacks (default: clean messages)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    info_p = sub.add_parser(
        "info", help="print the resolved config and the number of discovered runs"
    )
    _add_runs(info_p)

    serve_p = sub.add_parser(
        "serve", help="serve the dashboard API (implemented by U05)"
    )
    _add_runs(serve_p)
    serve_p.add_argument(
        "--results",
        default=str(DEFAULT_RESULTS_DIR),
        help=f"directory holding evaluation artifacts (default: {DEFAULT_RESULTS_DIR})",
    )
    serve_p.add_argument(
        "--host",
        default=defaults.host,
        help=f"interface to bind (default: {defaults.host}; loopback only in v1)",
    )
    serve_p.add_argument(
        "--port",
        type=int,
        default=defaults.port,
        help=f"TCP port to bind (default: {defaults.port})",
    )
    serve_p.add_argument(
        "--web-dist",
        default=None,
        help="built frontend directory to mount (default: none — API only)",
    )
    serve_p.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="reload the server on code changes (development only)",
    )
    return parser


# ---------------------------------------------------------------------------
# Run discovery (U02 supersedes this with the in-memory registry)
# ---------------------------------------------------------------------------
def _discover_run_dirs(runs_dir: Path) -> list[Path]:
    """Return the sorted run directories directly under ``runs_dir``.

    A run directory is any directory containing ``manifest.json`` or
    ``telemetry.jsonl``.  A missing root is not an error (a fresh clone has no
    runs yet); a root that exists but is not a directory is a config error.
    """
    if not runs_dir.exists():
        return []
    if not runs_dir.is_dir():
        raise CliError(f"runs path is not a directory: {runs_dir}", EXIT_CONFIG)

    found: set[Path] = set()
    for marker in (MANIFEST_FILENAME, TELEMETRY_FILENAME):
        for marker_path in runs_dir.rglob(marker):
            if marker_path.is_file():
                found.add(marker_path.parent)
    return sorted(found)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def _cmd_info(config: UiConfig) -> int:
    """Print the resolved config and the number of discovered run directories."""
    run_dirs = _discover_run_dirs(config.runs_dir)
    count = len(run_dirs)
    noun = "run" if count == 1 else "runs"
    origins = ", ".join(config.cors_origins) if config.cors_origins else "same-origin only"
    web_dist = str(config.web_dist) if config.web_dist is not None else "API only (no web/dist)"
    lines = [
        f"{PROG} — configuration summary",
        f"  runs dir:          {config.runs_dir}",
        f"  results dir:       {config.results_dir}",
        f"  host:              {config.host}",
        f"  port:              {config.port}",
        f"  telemetry poll:    {config.telemetry_poll_seconds}s",
        f"  max points:        {config.max_points}",
        f"  cors origins:      {origins}",
        f"  web dist:          {web_dist}",
        f"  discovered runs:   {count} {noun}",
    ]
    print("\n".join(lines))
    return EXIT_OK


def serve(config: UiConfig, *, reload: bool = False) -> int:
    """Start the dashboard with uvicorn (U05).

    ``--reload`` needs uvicorn's import-string mode, so the resolved config is
    serialised into the environment and the reload worker builds its own app via
    :func:`undertow.ui.app.create_app_from_env`.  Without ``--reload`` the app
    instance is passed directly.  Errors are reported as config errors (exit
    ``2``) so a typo never leaves a half-started server.
    """
    # Local imports keep ``undertow-ui info`` independent of the web stack.
    import uvicorn

    from undertow.ui.app import CONFIG_ENV_VAR, config_payload, create_app

    if not 0 < config.port < 65536:
        raise CliError(f"port must be between 1 and 65535, got {config.port}", EXIT_CONFIG)
    if config.web_dist is not None and not config.web_dist.is_dir():
        raise CliError(
            f"web-dist is not a directory: {config.web_dist}", EXIT_CONFIG
        )

    print(f"Undertow UI serving on http://{config.host}:{config.port}")
    print(f"  runs dir:    {config.runs_dir}")
    print(f"  results dir: {config.results_dir}")
    if config.web_dist is not None:
        print(f"  web dist:    {config.web_dist}")
    else:
        print("  web dist:    API only (no built frontend mounted)")

    if reload:
        os.environ[CONFIG_ENV_VAR] = json.dumps(config_payload(config))
        uvicorn.run(
            "undertow.ui.app:create_app_from_env",
            factory=True,
            host=config.host,
            port=config.port,
            reload=True,
        )
    else:
        app = create_app(config)
        uvicorn.run(app, host=config.host, port=config.port, reload=False)
    return EXIT_OK


# ---------------------------------------------------------------------------
# Dispatch / entry point
# ---------------------------------------------------------------------------
def _run(args: argparse.Namespace) -> int:
    """Execute the parsed command; raises :class:`CliError` on user errors."""
    if args.command == "info":
        config = dataclasses.replace(UiConfig(), runs_dir=Path(args.runs))
        return _cmd_info(config)
    if args.command == "serve":
        config = dataclasses.replace(
            UiConfig(),
            runs_dir=Path(args.runs),
            results_dir=Path(args.results),
            host=args.host,
            port=int(args.port),
            web_dist=Path(args.web_dist) if args.web_dist else None,
        )
        return serve(config, reload=bool(args.reload))
    raise CliError(f"unknown command {args.command!r}", EXIT_CONFIG)  # pragma: no cover


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.  Returns the process exit code (never raises by default)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.debug:
        return _run(args)
    try:
        return _run(args)
    except CliError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # noqa: BLE001 - a CLI must not leak tracebacks
        print(f"Error: unexpected failure: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover - exercised via ``python -m``
    raise SystemExit(main())
