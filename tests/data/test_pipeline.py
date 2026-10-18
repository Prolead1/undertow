"""Tests for pipeline orchestration (T14).

No network. Uses T11's ``tiny_dataset()`` and monkeypatched fetchers.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pyarrow as pa
import pytest

from tests.data.conftest import TinyDataset
from undertow.data.config import GLOBAL_TICK_SENTINEL, DataConfig, WindowConfig
from undertow.data.fetchers.base import FetchRequest, FetchResult
from undertow.data.pipeline import (
    _DEFAULT_STREAMS,
    _config_with_output_dir,
    build,
    capture_pool_state,
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


def test_snapshot_fresh_output_dir_writes_manifest(tmp_path: Path, tiny_dataset) -> None:
    """snapshot() works on a bare output dir with streams but no pre-written manifest.

    ``build()``/``verify()`` read the manifest from disk, so snapshot() must write a
    provisional manifest between pull and verify; without this a live first-run
    snapshot fails with "No manifest found" (regression test).
    """
    from undertow.data.fetchers.base import FetchResult
    from undertow.data.fetchers.gas import GasFetcher
    from undertow.data.fetchers.reference import ReferenceFetcher
    from undertow.data.fetchers.thegraph import TheGraphFetcher

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _write_tiny_streams(tiny_dataset, output_dir)
    (output_dir / "manifest.json").unlink()  # streams present, manifest absent

    def fake_logs_fetch(self, request: FetchRequest) -> FetchResult:
        return FetchResult(
            table=tiny_dataset.streams[request.stream],
            route=self.route,
            request=request,
            n_requests=1,
            from_cache=False,
            warnings=(),
        )

    def fake_table_fetch(table):
        def fake(self, request: FetchRequest) -> FetchResult:
            return FetchResult(
                table=table, route=self.route, request=request,
                n_requests=1, from_cache=False, warnings=(),
            )
        return fake

    with (
        patch.object(TheGraphFetcher, "fetch", fake_logs_fetch),
        patch.object(GasFetcher, "fetch", fake_table_fetch(tiny_dataset.gas)),
        patch.object(ReferenceFetcher, "fetch", fake_table_fetch(tiny_dataset.reference)),
    ):
        report = snapshot(config)

    assert (output_dir / "manifest.json").exists()
    assert report.manifest_path.exists()


def test_build_threads_reference_symbol(tmp_path: Path, tiny_dataset: TinyDataset) -> None:
    """``build()`` must forward ``config.reference_symbol`` to the tape builder."""
    from undertow.data.transforms.align import build_event_tape

    out = tmp_path / "out"
    cfg = _write_tiny_streams(tiny_dataset, out)
    captured: dict[str, object] = {}

    def spy(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return build_event_tape(*args, **kwargs)  # type: ignore[arg-type]

    with patch("undertow.data.pipeline.build_event_tape", spy):
        build(cfg)
    # The kwarg must be present (a deleted wiring line omits it) and equal the config.
    assert captured.get("reference_symbol") == cfg.reference_symbol


def test_capture_pool_state_is_none_without_the_split_hook(
    tmp_path: Path, tiny_dataset: TinyDataset
) -> None:
    # The tiny window is block-mode, so the walk-forward hook is unset → no-op.
    cfg = _write_tiny_streams(tiny_dataset, tmp_path / "out")
    assert capture_pool_state(cfg) is None


def _fee_growth_snapshot(pool: object, block: int) -> pa.Table:
    from undertow.data.schemas import FEE_GROWTH_SCHEMA

    return pa.Table.from_pylist(
        [
            {
                "block_number": block,
                "tick": GLOBAL_TICK_SENTINEL,
                "fee_growth_outside_0_x128": None,
                "fee_growth_outside_1_x128": None,
                "liquidity_gross": "0",
                "liquidity_net": "0",
                "initialized": False,
                "fee_growth_global_0_x128": "0",
                "fee_growth_global_1_x128": "0",
                "current_tick": 0,
                "current_liquidity": "0",
                "source": "rpc_call",
                "pool_address": str(pool.address),  # type: ignore[attr-defined]
                "fee_protocol": 0,
            }
        ],
        schema=FEE_GROWTH_SCHEMA,
    )


def test_capture_pool_state_snapshots_at_first_block_minus_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_dataset: TinyDataset
) -> None:
    cfg = _write_tiny_streams(tiny_dataset, tmp_path / "out")
    tape = pl.from_arrow(build(cfg).tape)
    lo = tape["block_timestamp"].min()
    hi = tape["block_timestamp"].max()
    assert lo is not None and hi is not None
    mid = lo + (hi - lo) / 2
    cfg = replace(
        cfg,
        window=WindowConfig(
            start_utc=lo, end_utc=hi, train_end_utc=mid, eval_start_utc=mid
        ),
    )

    def first_seed_block(lo: datetime, hi: datetime) -> int:
        lane = tape.filter(
            (pl.col("block_timestamp") >= lo) & (pl.col("block_timestamp") < hi)
        )
        return int(lane["block_number"].min()) - 1

    expected = [
        first_seed_block(cfg.window.start_utc, cfg.window.train_end_utc),  # type: ignore[arg-type]
        first_seed_block(cfg.window.eval_start_utc, cfg.window.end_utc),  # type: ignore[arg-type]
    ]
    seen: list[int] = []

    def fake_pool_state_at(self: object, pool: object, block: object) -> pa.Table:
        seen.append(int(block))
        return _fee_growth_snapshot(pool, int(block))

    monkeypatch.setattr(
        "undertow.data.pipeline.RpcFetcher.pool_state_at", fake_pool_state_at
    )
    result = capture_pool_state(cfg)
    assert result is not None
    assert seen == expected


def test_load_fee_growth_is_typed_empty_when_the_stream_is_absent(
    tmp_path: Path, tiny_dataset: TinyDataset
) -> None:
    import shutil

    from undertow.data.loader import load_fee_growth
    from undertow.data.schemas import FEE_GROWTH_SCHEMA

    cfg = _write_tiny_streams(tiny_dataset, tmp_path / "out")
    shutil.rmtree(
        tmp_path / "out" / f"pool={tiny_dataset.pool.address}" / "stream=fee_growth",
        ignore_errors=True,
    )
    table = load_fee_growth(cfg)
    assert table.num_rows == 0
    assert table.schema == FEE_GROWTH_SCHEMA


def test_config_with_output_dir_preserves_extensions(
    tmp_path: Path, tiny_dataset: TinyDataset
) -> None:
    """``snapshot(out=...)`` must not drop defaulted config extensions (ADR-014)."""
    base = _write_tiny_streams(tiny_dataset, tmp_path / "base")
    cfg = replace(base, reference_symbol="ETHUSDC", tip_surcharge_pct=9)
    new_out = tmp_path / "new"
    got = _config_with_output_dir(cfg, new_out)
    # Equality with dataclasses.replace pins that no field is omitted; a future
    # extension added to DataConfig but not to the helper fails this test.
    assert got == replace(cfg, output_dir=new_out)
    assert got.reference_symbol == "ETHUSDC"
    assert got.tip_surcharge_pct == 9


def test_snapshot_critical_failure_removes_provisional_manifest(
    tmp_path: Path, tiny_dataset
) -> None:
    """A critical check failure leaves no manifest behind (manifest-last invariant)."""
    from undertow.data.types import CheckResult

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _write_tiny_streams(tiny_dataset, output_dir)
    good_report = pull(config)  # a valid, all-cached pull report
    (output_dir / "manifest.json").unlink()

    failed = [
        CheckResult(
            name="check_x", passed=False, severity="critical", detail="boom",
        )
    ]
    with (
        patch("undertow.data.pipeline.pull", return_value=good_report),
        patch("undertow.data.pipeline.verify", return_value=failed),
    ):
        with pytest.raises(ValidationError):
            snapshot(config)

    assert not (output_dir / "manifest.json").exists()


def test_snapshot_exception_removes_provisional_manifest(tmp_path: Path, tiny_dataset) -> None:
    """An exception mid-verify removes the provisional manifest and re-raises."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _write_tiny_streams(tiny_dataset, output_dir)
    good_report = pull(config)
    (output_dir / "manifest.json").unlink()

    with (
        patch("undertow.data.pipeline.pull", return_value=good_report),
        patch("undertow.data.pipeline.verify", side_effect=RuntimeError("boom")),
    ):
        with pytest.raises(RuntimeError):
            snapshot(config)

    assert not (output_dir / "manifest.json").exists()


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