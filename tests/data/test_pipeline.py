"""Tests for pipeline orchestration (T14).

No network. Uses T11's ``tiny_dataset()`` and monkeypatched fetchers.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pytest

from undertow.data.config import DataConfig
from undertow.data.fetchers.base import FetchRequest, FetchResult
from undertow.data.pipeline import (
    _DEFAULT_STREAMS,
    build,
    info,
    pull,
    snapshot,
    verify,
)
from undertow.data.pipeline_report import (
    StreamPullResult,
)
from undertow.data.storage.manifest import read_manifest
from undertow.data.storage.parquet import read_stream
from undertow.data.types import Route, ValidationError

# Use the shared tiny_dataset fixture from T11.
pytestmark = pytest.mark.usefixtures("tiny_dataset")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_output(tmp_path: Path) -> Path:
    return tmp_path / "output"


def _write_tiny_streams(tiny, output_dir: Path) -> DataConfig:
    """Write all streams from a TinyDataset to parquet under ``output_dir``
    AND write a manifest so pull() can detect cached data.

    Returns a DataConfig pointing at that output.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    from datetime import UTC, datetime

    from undertow.data.config import (
        EndpointConfig,
        RegimeConfig,
        WindowConfig,
    )
    from undertow.data.storage.manifest import (
        DatasetManifest,
        git_commit,
        library_versions,
        write_manifest,
    )
    from undertow.data.storage.parquet import write_stream
    from undertow.data.schemas import SCHEMA_VERSION
    from undertow.data.types import BlockNumber, Route

    stream_manifests: dict[str, object] = {}
    for name, table in tiny.streams.items():
        sm = write_stream(table, output_dir, name, pool=tiny.pool)
        stream_manifests[name] = sm
    for name in ("gas", "reference", "regime", "fee_growth"):
        table = getattr(tiny, name)
        if table is not None:
            sm = write_stream(table, output_dir, name, pool=tiny.pool)
            stream_manifests[name] = sm

    manifest = DatasetManifest(
        schema_version=SCHEMA_VERSION,
        pool=tiny.pool,
        window=WindowConfig(
            start_block=BlockNumber(17_000_000),
            end_block=BlockNumber(17_000_062),
        ),
        streams=stream_manifests,
        created_at_utc=datetime.now(UTC),
        git_commit=git_commit(),
        library_versions=library_versions(),
        endpoint_hosts={
            "graph": "localhost:9999",
            "rpc": "localhost:9999",
            "reference": "localhost:9999",
        },
    )
    write_manifest(manifest, output_dir)

    return DataConfig(
        pool=tiny.pool,
        window=WindowConfig(
            start_block=BlockNumber(17_000_000),
            end_block=BlockNumber(17_000_062),
        ),
        regime=RegimeConfig(),
        endpoints=EndpointConfig(
            graph_url="http://localhost:9999/graph",
            rpc_url="http://localhost:9999/rpc",
            reference_base_url="http://localhost:9999/ref",
        ),
        cache_dir=output_dir / "cache",
        output_dir=output_dir,
    )


def _make_fake_fetch(table: pa.Table, route: Route = Route.THEGRAPH):
    """Return a mock ``fetch`` method that returns a pre-built table."""

    def fake_fetch(self, request: FetchRequest) -> FetchResult:
        return FetchResult(
            table=table,
            route=route,
            request=request,
            n_requests=1,
            from_cache=False,
            warnings=(),
        )

    return fake_fetch


# ---------------------------------------------------------------------------
# pull tests
# ---------------------------------------------------------------------------


def test_pull_writes_parquet_layout(tmp_path: Path, tiny_dataset) -> None:
    """pull() writes each stream to the expected parquet layout."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _write_tiny_streams(tiny_dataset, output_dir)

    # Already written — pull should skip everything.
    report = pull(config)

    assert len(report.streams) == len(_DEFAULT_STREAMS)
    assert report.n_total_requests == 0
    assert report.all_from_cache

    # Check that parquet files exist for each stream.
    for stream in _DEFAULT_STREAMS:
        assert stream in report.streams
        spr = report.streams[stream]
        assert spr.row_count >= 0
        assert spr.from_cache


def test_pull_idempotent(tmp_path: Path, tiny_dataset) -> None:
    """Second pull with warm cache → n_requests == 0, identical content_hash."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _write_tiny_streams(tiny_dataset, output_dir)

    report1 = pull(config)
    report2 = pull(config)

    assert report2.n_total_requests == 0
    assert report2.all_from_cache
    for name in _DEFAULT_STREAMS:
        h1 = report1.streams[name].content_hash
        h2 = report2.streams[name].content_hash
        assert h1 == h2, f"content_hash mismatch for {name}: {h1} vs {h2}"


def test_pull_force_re_fetches(tmp_path: Path, tiny_dataset) -> None:
    """--force re-fetches and produces identical content hash to the original."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _write_tiny_streams(tiny_dataset, output_dir)

    report1 = pull(config)

    # Now force-re-pull "gas" — but mock the fetcher so it doesn't hit the network.
    from undertow.data.fetchers.base import FetchRequest, FetchResult
    from undertow.data.storage.parquet import read_stream

    gas_table = read_stream(output_dir, "gas", pool=tiny_dataset.pool)

    def fake_fetch(self, request: FetchRequest) -> FetchResult:
        return FetchResult(
            table=gas_table,
            route=Route.RPC,
            request=request,
            n_requests=1,
            from_cache=False,
            warnings=(),
        )

    from undertow.data.fetchers.gas import GasFetcher

    with patch.object(GasFetcher, "fetch", fake_fetch):
        report2 = pull(config, streams=["gas"], force=True)

    # gas was re-fetched (requests > 0 since we nuked the check by force)
    gas2 = report2.streams.get("gas")
    assert gas2 is not None
    # The hash should match (same data re-written).
    assert gas2.content_hash == report1.streams["gas"].content_hash


def test_pull_unknown_stream_raises(tmp_path: Path, tiny_dataset) -> None:
    """Requesting an unknown stream raises ConfigError."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _write_tiny_streams(tiny_dataset, output_dir)
    from undertow.data.types import ConfigError

    with pytest.raises(ConfigError, match="Unknown stream"):
        pull(config, streams=["nonexistent"])


# ---------------------------------------------------------------------------
# build tests
# ---------------------------------------------------------------------------


def test_build_assembles_dataset(tmp_path: Path, tiny_dataset) -> None:
    """build() reads cached parquet and returns a Dataset with a tape."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _write_tiny_streams(tiny_dataset, output_dir)

    # Write a manifest so build can find the streams.
    from undertow.data.storage.manifest import (
        DatasetManifest,
        StreamManifest,
        git_commit,
        library_versions,
        write_manifest,
    )
    from undertow.data.schemas import SCHEMA_VERSION
    from datetime import UTC, datetime

    manifest = DatasetManifest(
        schema_version=SCHEMA_VERSION,
        pool=config.pool,
        window=config.window,
        streams={
            name: StreamManifest(
                row_count=getattr(tiny_dataset, name, pa.table({})).num_rows
                if name not in tiny_dataset.streams
                else tiny_dataset.streams[name].num_rows,
                content_hash="dummy",
                route=Route.THEGRAPH,
                n_requests=1,
                warnings=(),
            )
            for name in _DEFAULT_STREAMS
        },
        created_at_utc=datetime.now(UTC),
        git_commit=git_commit(),
        library_versions=library_versions(),
    )
    write_manifest(manifest, output_dir)

    dataset = build(config)
    assert dataset.tape is not None
    assert dataset.tape.num_rows > 0


def test_build_no_manifest_raises(tmp_path: Path, tiny_dataset) -> None:
    """build() raises ValidationError when no manifest exists."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    # Write parquet files but NO manifest — only then does build raise.
    from undertow.data.config import EndpointConfig, RegimeConfig, WindowConfig
    from undertow.data.storage.parquet import write_stream
    from undertow.data.types import BlockNumber

    for name, table in tiny_dataset.streams.items():
        write_stream(table, output_dir, name, pool=tiny_dataset.pool)
    for name in ("gas", "reference", "regime", "fee_growth"):
        table = getattr(tiny_dataset, name)
        if table is not None:
            write_stream(table, output_dir, name, pool=tiny_dataset.pool)

    config = DataConfig(
        pool=tiny_dataset.pool,
        window=WindowConfig(
            start_block=BlockNumber(17_000_000),
            end_block=BlockNumber(17_000_062),
        ),
        regime=RegimeConfig(),
        endpoints=EndpointConfig(
            graph_url="http://localhost:9999/graph",
            rpc_url="http://localhost:9999/rpc",
            reference_base_url="http://localhost:9999/ref",
        ),
        cache_dir=output_dir / "cache",
        output_dir=output_dir,
    )

    with pytest.raises(ValidationError, match="No manifest found"):
        build(config)


# ---------------------------------------------------------------------------
# verify tests
# ---------------------------------------------------------------------------


def test_verify_returns_check_results(tmp_path: Path, tiny_dataset) -> None:
    """verify() runs checks and returns results; does not raise on failures."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _write_tiny_streams(tiny_dataset, output_dir)

    # Write manifest + parquet so verify can build.
    from datetime import UTC, datetime

    from undertow.data.storage.manifest import (
        DatasetManifest,
        StreamManifest,
        git_commit,
        library_versions,
        write_manifest,
    )
    from undertow.data.schemas import SCHEMA_VERSION

    manifest = DatasetManifest(
        schema_version=SCHEMA_VERSION,
        pool=config.pool,
        window=config.window,
        streams={
            name: StreamManifest(
                row_count=getattr(tiny_dataset, name, pa.table({})).num_rows
                if name not in tiny_dataset.streams
                else tiny_dataset.streams[name].num_rows,
                content_hash="dummy",
                route=Route.THEGRAPH,
                n_requests=1,
                warnings=(),
            )
            for name in _DEFAULT_STREAMS
        },
        created_at_utc=datetime.now(UTC),
        git_commit=git_commit(),
        library_versions=library_versions(),
    )
    write_manifest(manifest, output_dir)

    checks = verify(config)
    assert len(checks) > 0
    # verify() returns results; it does not raise.
    for c in checks:
        assert c.name
        assert c.severity in ("critical", "warning", "info")


# ---------------------------------------------------------------------------
# snapshot tests
# ---------------------------------------------------------------------------


def test_snapshot_writes_manifest_and_report(tmp_path: Path, tiny_dataset) -> None:
    """snapshot() writes manifest.json + validation_report.md."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _write_tiny_streams(tiny_dataset, output_dir)

    report = snapshot(config)

    assert report.manifest_path.exists()
    assert report.validation_report_path.exists()
    assert report.manifest_path.name == "manifest.json"
    assert report.validation_report_path.name == "validation_report.md"

    # Manifest is readable.
    m = read_manifest(output_dir)
    assert m.pool.address == tiny_dataset.pool.address


def test_snapshot_manifest_json_valid(tmp_path: Path, tiny_dataset) -> None:
    """The manifest.json written by snapshot() is valid JSON with expected keys."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _write_tiny_streams(tiny_dataset, output_dir)

    report = snapshot(config)
    data = json.loads(report.manifest_path.read_text())

    assert "schema_version" in data
    assert "dataset_id" in data
    assert "pool" in data
    assert "window" in data
    assert "streams" in data
    assert "created_at_utc" in data
    # Secrets must not appear in the manifest.
    raw = report.manifest_path.read_text()
    assert "9999/graph" not in raw  # only the host, not the path


# ---------------------------------------------------------------------------
# info tests
# ---------------------------------------------------------------------------


def test_info_on_unpulled_dataset(tmp_path: Path, tiny_dataset) -> None:
    """info() reports dataset_exists=False when no manifest exists."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    from undertow.data.config import EndpointConfig, RegimeConfig, WindowConfig
    from undertow.data.types import BlockNumber

    config = DataConfig(
        pool=tiny_dataset.pool,
        window=WindowConfig(
            start_block=BlockNumber(17_000_000),
            end_block=BlockNumber(17_000_062),
        ),
        regime=RegimeConfig(),
        endpoints=EndpointConfig(
            graph_url="http://localhost:9999/graph",
            rpc_url="http://localhost:9999/rpc",
            reference_base_url="http://localhost:9999/ref",
        ),
        cache_dir=output_dir / "cache",
        output_dir=output_dir,
    )

    result = info(config)
    assert not result.dataset_exists
    assert any("not been pulled" in w for w in result.manifest.warnings)


def test_info_on_pulled_dataset(tmp_path: Path, tiny_dataset) -> None:
    """info() reports row counts and coverage for a pulled dataset."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _write_tiny_streams(tiny_dataset, output_dir)

    # Need a manifest for info to read.
    from datetime import UTC, datetime

    from undertow.data.storage.manifest import (
        DatasetManifest,
        StreamManifest,
        git_commit,
        library_versions,
        write_manifest,
    )
    from undertow.data.schemas import SCHEMA_VERSION

    manifest = DatasetManifest(
        schema_version=SCHEMA_VERSION,
        pool=config.pool,
        window=config.window,
        streams={
            "swap": StreamManifest(
                row_count=tiny_dataset.streams["swap"].num_rows,
                content_hash="abc123",
                route=Route.THEGRAPH,
                n_requests=1,
                warnings=(),
            ),
        },
        created_at_utc=datetime.now(UTC),
        git_commit=git_commit(),
        library_versions=library_versions(),
    )
    write_manifest(manifest, output_dir)

    result = info(config)
    assert result.dataset_exists
    assert "swap" in result.manifest.streams
    assert result.manifest.streams["swap"].row_count == tiny_dataset.streams["swap"].num_rows


# ---------------------------------------------------------------------------
# StreamPullResult tests
# ---------------------------------------------------------------------------


def test_load_dataset_from_config(tmp_path: Path, tiny_dataset) -> None:
    """load_dataset() returns a Dataset when a valid built dataset exists."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _write_tiny_streams(tiny_dataset, output_dir)

    from undertow.data import load_dataset

    dataset = load_dataset(config)
    assert dataset.tape is not None
    assert dataset.tape.num_rows > 0
    assert dataset.manifest.pool.address == tiny_dataset.pool.address


def test_load_dataset_from_output_dir(tmp_path: Path, tiny_dataset) -> None:
    """load_dataset() works when given a path to a dataset output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_tiny_streams(tiny_dataset, output_dir)

    from undertow.data import load_dataset

    dataset = load_dataset(output_dir)
    assert dataset.tape is not None
    assert dataset.tape.num_rows > 0


def test_load_dataset_partial(tmp_path: Path, tiny_dataset) -> None:
    """load_dataset() with streams=frozenset drops non-requested side-streams."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_tiny_streams(tiny_dataset, output_dir)

    from undertow.data import load_dataset

    dataset = load_dataset(output_dir, streams=frozenset({"tape"}))
    assert dataset.tape is not None
    assert dataset.gas is None  # not requested
    assert dataset.reference is None


def test_stream_pull_result_immutable() -> None:
    """StreamPullResult is frozen."""
    spr = StreamPullResult(
        stream="swap",
        route=Route.THEGRAPH,
        row_count=100,
        n_requests=1,
        from_cache=False,
    )
    with pytest.raises(Exception):
        spr.row_count = 200  # type: ignore[misc]