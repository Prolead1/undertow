"""Tests for the ``undertow-data`` CLI (T14).

Parser, formatters, and error-handling are tested here without touching the
network — pipeline functions are mocked where needed.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from undertow.data.cli import (
    _build_parser,
    _format_checks,
    _format_info_report,
    _format_pull_report,
    _format_snapshot_report,
    main,
)
from undertow.data.config import PoolConfig, WindowConfig
from undertow.data.logging_setup import configure_logging
from undertow.data.pipeline_report import (
    InfoReport,
    PullReport,
    SnapshotReport,
    StreamPullResult,
)
from undertow.data.schemas import SCHEMA_VERSION
from undertow.data.storage.manifest import DatasetManifest
from undertow.data.types import CheckResult, FeeTier, Route

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check(name: str, passed: bool, severity: str = "critical", detail: str = "") -> CheckResult:
    return CheckResult(name=name, passed=passed, severity=severity, detail=detail, metrics={})


def _stream_pull(stream: str, row_count: int = 100) -> StreamPullResult:
    return StreamPullResult(
        stream=stream,
        route=Route.THEGRAPH,
        row_count=row_count,
        n_requests=1,
        from_cache=False,
        content_hash="abc",
    )


def _pull_report() -> PullReport:
    return PullReport(
        streams={"swap": _stream_pull("swap")},
        n_total_requests=1,
        all_from_cache=False,
        started_at_utc=datetime(2024, 1, 1, tzinfo=UTC),
        finished_at_utc=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
    )


def _pool_config() -> PoolConfig:
    return PoolConfig(
        address="0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",
        token0_symbol="USDC",
        token1_symbol="WETH",
        token0_decimals=6,
        token1_decimals=18,
        fee_tier=FeeTier(3000),
        tick_spacing=60,
        deployment_block=12376729,
    )


def _minimal_manifest(pool: PoolConfig) -> DatasetManifest:
    return DatasetManifest(
        schema_version=SCHEMA_VERSION,
        pool=pool,
        window=WindowConfig(start_block=100, end_block=200),
        streams={},
        created_at_utc=datetime.now(UTC),
        git_commit="abc",
        library_versions={},
    )


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


def test_parser_requires_command() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_pull_requires_config() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["pull"])


def test_parser_pull_default_streams() -> None:
    parser = _build_parser()
    args = parser.parse_args(["pull", "--config", "config.toml"])
    assert args.streams is None  # None = all streams


def test_parser_pull_stream_subset() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        ["pull", "--config", "config.toml", "--stream", "swap", "--stream", "gas"]
    )
    assert args.streams == ["swap", "gas"]


def test_parser_verify_requires_config() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["verify"])


def test_parser_debug_flag() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--debug", "pull", "--config", "c.toml"])
    assert args.debug


def test_parser_pull_route_choices() -> None:
    parser = _build_parser()
    args = parser.parse_args(["pull", "--config", "c.toml", "--route", "rpc"])
    assert args.route == "rpc"
    with pytest.raises(SystemExit):
        parser.parse_args(["pull", "--config", "c.toml", "--route", "invalid"])


# ---------------------------------------------------------------------------
# _format_checks tests
# ---------------------------------------------------------------------------


def test_format_checks_all_pass_json() -> None:
    checks = [_check("ck1", True), _check("ck2", True)]
    text, exit_code = _format_checks(checks, json_output=True, fail_on="critical")
    assert exit_code == 0
    data = json.loads(text)
    assert len(data) == 2


def test_format_checks_critical_fail_json() -> None:
    checks = [_check("ck1", True), _check("ck2", False, "critical", "bad!")]
    _, exit_code = _format_checks(checks, json_output=True, fail_on="critical")
    assert exit_code == 1


def test_format_checks_warning_fail_with_critical_threshold_json() -> None:
    checks = [_check("ck1", True), _check("ck2", False, "warning", "hmm")]
    _, exit_code = _format_checks(checks, json_output=True, fail_on="critical")
    assert exit_code == 0


def test_format_checks_warning_fail_with_warning_threshold_json() -> None:
    checks = [_check("ck1", True), _check("ck2", False, "warning", "hmm")]
    _, exit_code = _format_checks(checks, json_output=True, fail_on="warning")
    assert exit_code == 1


def test_format_checks_info_never_affects_exit_code() -> None:
    checks = [_check("ck1", True), _check("ck2", True, "info", "skipped")]
    _, code = _format_checks(checks, json_output=True, fail_on="warning")
    assert code == 0


def test_format_checks_human_all_pass() -> None:
    checks = [_check("ck1", True, "critical"), _check("ck2", True, "warning")]
    text, code = _format_checks(checks, json_output=False, fail_on="critical")
    assert code == 0
    assert "ck1" in text
    assert "ck2" in text
    assert "PASS" in text


def test_format_checks_human_critical_fail() -> None:
    checks = [_check("ck1", False, "critical", "broken")]
    _, exit_code = _format_checks(checks, json_output=False, fail_on="critical")
    assert exit_code == 1


# ---------------------------------------------------------------------------
# _format_pull_report tests
# ---------------------------------------------------------------------------


def test_format_pull_report_json() -> None:
    report = _pull_report()
    text = _format_pull_report(report, json_output=True)
    data = json.loads(text)
    assert data["streams"]["swap"]["route"] == "thegraph"
    assert data["n_total_requests"] == 1


def test_format_pull_report_human() -> None:
    report = _pull_report()
    text = _format_pull_report(report, json_output=False)
    assert "swap" in text
    assert "100" in text  # row count


def test_format_pull_report_all_cached() -> None:
    report = replace(_pull_report(), all_from_cache=True, n_total_requests=0)
    text = _format_pull_report(report, json_output=False)
    assert "cached" in text
    assert "0 network requests" in text


# ---------------------------------------------------------------------------
# _format_snapshot_report tests
# ---------------------------------------------------------------------------


def test_format_snapshot_report_json() -> None:
    report = SnapshotReport(
        pull=_pull_report(),
        dataset=MagicMock(),
        checks=[_check("ck1", True)],
        manifest_path=Path("/tmp/m.json"),
        validation_report_path=Path("/tmp/r.md"),
    )
    text = _format_snapshot_report(report, json_output=True)
    data = json.loads(text)
    assert data["manifest_path"] == "/tmp/m.json"
    assert data["checks"][0]["passed"] is True


# ---------------------------------------------------------------------------
# _format_info_report tests
# ---------------------------------------------------------------------------


def test_format_info_report_json() -> None:
    pool = _pool_config()
    manifest = _minimal_manifest(pool)
    info_report = InfoReport(manifest=manifest, dataset_exists=True, output_dir=Path("/tmp"))
    text = _format_info_report(info_report, json_output=True)
    data = json.loads(text)
    assert data["dataset_id"] is not None
    # Secrets must never appear.
    raw = json.dumps(data)
    assert "9999" not in raw
    assert "localhost" not in raw


def test_format_info_report_human_unpulled() -> None:
    pool = _pool_config()
    manifest = DatasetManifest(
        schema_version=SCHEMA_VERSION,
        pool=pool,
        window=WindowConfig(start_block=100, end_block=200),
        streams={},
        created_at_utc=datetime.now(UTC),
        git_commit="abc",
        library_versions={},
        warnings=("No manifest found — dataset has not been pulled yet.",),
    )
    info_report = InfoReport(manifest=manifest, dataset_exists=False, output_dir=Path("/tmp"))
    text = _format_info_report(info_report, json_output=False)
    assert "NOT PULLED" in text


# ---------------------------------------------------------------------------
# main() error-handling tests
# ---------------------------------------------------------------------------


def test_main_no_command() -> None:
    """main() with no args → SystemExit (argparse requirement)."""
    with pytest.raises(SystemExit):
        main([])


def test_main_config_error_raises_system_exit() -> None:
    """A bad config path → SystemExit(2) from _load_config."""
    with pytest.raises(SystemExit) as exc_info:
        main(["pull", "--config", "/nonexistent/path.toml"])
    assert exc_info.value.code == 2


def test_main_verify_happy_path(tmp_path: Path) -> None:
    """verify command with a valid config path → mock pipeline returns checks."""
    toml_path = tmp_path / "test.toml"
    toml_path.write_text("""
[pool]
address = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
token0_symbol = "USDC"
token1_symbol = "WETH"
token0_decimals = 6
token1_decimals = 18
fee_tier = 3000
tick_spacing = 60
deployment_block = 12376729

[window]
start_block = 100
end_block = 200

[regime]
# all defaults

[paths]
cache_dir = "cache"
output_dir = "out"

[endpoints]
graph_url = "http://localhost/graph"
rpc_url = "http://localhost/rpc"
reference_base_url = "http://localhost/ref"
""")
    with patch("undertow.data.cli.verify") as mock_verify:
        mock_verify.return_value = [_check("ck1", True)]
        result = main(["verify", "--config", str(toml_path)])
        assert result == 0


def test_main_info_unpulled_dataset_returns_1(tmp_path: Path) -> None:
    """info on an unpulled dataset → exit code 1."""
    toml_path = tmp_path / "test.toml"
    toml_path.write_text("""
[pool]
address = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
token0_symbol = "USDC"
token1_symbol = "WETH"
token0_decimals = 6
token1_decimals = 18
fee_tier = 3000
tick_spacing = 60
deployment_block = 12376729

[window]
start_block = 100
end_block = 200

[regime]
# all defaults

[paths]
cache_dir = "cache"
output_dir = "out"

[endpoints]
graph_url = "http://localhost/graph"
rpc_url = "http://localhost/rpc"
reference_base_url = "http://localhost/ref"
""")
    # No manifest → info returns dataset_exists=False → exit 1
    result = main(["info", "--config", str(toml_path)])
    assert result == 1


def _write_pull_config(tmp_path: Path) -> Path:
    """Minimal valid config whose log_dir points inside ``tmp_path``."""
    toml_path = tmp_path / "test.toml"
    toml_path.write_text(f"""
[pool]
address = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
token0_symbol = "USDC"
token1_symbol = "WETH"
token0_decimals = 6
token1_decimals = 18
fee_tier = 3000
tick_spacing = 60
deployment_block = 12376729

[window]
start_block = 100
end_block = 200

[regime]
# all defaults

[paths]
cache_dir = "{tmp_path / 'cache'}"
output_dir = "{tmp_path / 'out'}"
log_dir = "{tmp_path / 'logs'}"

[endpoints]
graph_url = "http://localhost/graph"
rpc_url = "http://localhost/rpc"
reference_base_url = "http://localhost/ref"
""")
    return toml_path


def _log_and_return(report: object, message: str) -> Callable[..., object]:
    """Mock side effect that emits a package log record, so file capture is real."""

    def _side_effect(*_args: object, **_kwargs: object) -> object:
        logging.getLogger("undertow.data.pipeline").info(message)
        return report

    return _side_effect


def test_main_pull_auto_writes_a_log_file(tmp_path: Path) -> None:
    """Every pull invocation must leave a DEBUG log under config.log_dir."""
    toml_path = _write_pull_config(tmp_path)

    try:
        with patch("undertow.data.cli.pull") as mock_pull:
            mock_pull.side_effect = _log_and_return(_pull_report(), "pull started")
            result = main(["pull", "--config", str(toml_path)])
    finally:
        # add_file_handler raised the package logger to DEBUG; reset the global state.
        configure_logging(force=True)

    assert result == 0
    logs = sorted((tmp_path / "logs").glob("pull-*.log"))
    assert len(logs) == 1  # one unique timestamped file per run
    assert "pull started" in logs[0].read_text(encoding="utf-8")  # capture is live


def test_main_snapshot_auto_writes_a_log_file(tmp_path: Path) -> None:
    """snapshot is a download too, so it must auto-capture a log as well."""
    toml_path = _write_pull_config(tmp_path)
    report = SnapshotReport(
        pull=_pull_report(),
        dataset=MagicMock(),
        checks=[_check("ck1", True)],
        manifest_path=tmp_path / "manifest.json",
        validation_report_path=tmp_path / "report.md",
    )

    try:
        with patch("undertow.data.cli.snapshot") as mock_snapshot:
            mock_snapshot.side_effect = _log_and_return(report, "snapshot started")
            result = main(["snapshot", "--config", str(toml_path)])
    finally:
        configure_logging(force=True)

    assert result == 0
    logs = sorted((tmp_path / "logs").glob("snapshot-*.log"))
    assert len(logs) == 1
    assert "snapshot started" in logs[0].read_text(encoding="utf-8")


def test_main_pull_log_open_failure_is_not_fatal(tmp_path: Path) -> None:
    """An unwritable log dir warns but must never abort the pull."""
    toml_path = _write_pull_config(tmp_path)
    try:
        with (
            patch("undertow.data.cli.pull") as mock_pull,
            patch("undertow.data.cli.add_file_handler", side_effect=OSError("nope")),
        ):
            mock_pull.return_value = _pull_report()
            result = main(["pull", "--config", str(toml_path)])
    finally:
        configure_logging(force=True)

    assert result == 0  # the pull still ran and reported success


# ---------------------------------------------------------------------------
# CLI secrets-leak guard
# ---------------------------------------------------------------------------


def test_pull_report_json_no_secrets() -> None:
    """JSON output of pull must never contain full endpoints."""
    report = _pull_report()
    text = _format_pull_report(report, json_output=True)
    assert "http://" not in text
    assert "localhost" not in text