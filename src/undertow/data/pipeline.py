"""Pipeline orchestration for ``undertow.data`` (T14, ``CONTRACTS.md`` §9).

The CLI's *orchestration* layer — every public function here is testable without touching
``sys.argv`` or ``sys.exit``. ``cli.py`` does argument parsing, exit codes, and human output;
nothing else. A CLI module containing any pipeline logic is a god module (PLAN.md §4).

Report types live in ``pipeline_report.py`` — they are in their own module so ``cli.py``
can import them without pulling in the fetcher, storage, and transform dependencies
that this module carries.

Design rules:

- **Resumability.** ``pull`` skips already-fetched and written-to-parquet streams. Progress
  is tracked at bucket granularity so an interrupted multi-day pull continues without restart.
- **Idempotence.** A second ``pull`` with a warm cache makes zero network requests and leaves
  content hashes unchanged — this is a tested acceptance criterion, not an aspiration.
- **Manifest-last.** ``write_manifest`` is called only after all streams AND verification
  succeed. A failed run must never leave a manifest claiming completeness.
- **Secrets never leak.** Endpoints are recorded host-only in the manifest; no URL, path,
  or key ever appears in a report, log or exception message.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa

from undertow.data.config import DataConfig, PoolConfig
from undertow.data.fetchers.base import FetchRequest, FetchResult
from undertow.data.fetchers.gas import GasFetcher
from undertow.data.fetchers.reference import ReferenceFetcher
from undertow.data.fetchers.rpc import RpcFetcher
from undertow.data.fetchers.thegraph import TheGraphFetcher
from undertow.data.pipeline_report import (
    InfoReport,
    PullReport,
    SnapshotReport,
    StreamPullResult,
)
from undertow.data.schemas import SCHEMA_VERSION
from undertow.data.storage.manifest import (
    DatasetManifest,
    StreamManifest,
    endpoint_hosts,
    git_commit,
    library_versions,
    read_manifest,
    write_manifest,
)
from undertow.data.storage.parquet import read_stream, write_stream
from undertow.data.transforms.align import build_event_tape
from undertow.data.transforms.regimes import label_series
from undertow.data.types import (
    BlockNumber,
    CheckResult,
    ConfigError,
    FetchError,
    Route,
    ValidationError,
)
from undertow.data.validation import render_report, run_all_checks

LOGGER = logging.getLogger("undertow.data.pipeline")

# ---------------------------------------------------------------------------
# Stream inventory — which fetcher serves which stream, and is it log or global.
# ---------------------------------------------------------------------------

_LOG_STREAMS: tuple[str, ...] = ("swap", "mint", "burn", "collect")
"""The four log streams the tape unions."""

_DEFAULT_STREAMS: tuple[str, ...] = (
    "swap",
    "mint",
    "burn",
    "collect",
    "gas",
    "reference",
    "regime",
)
"""Streams a bare ``pull`` / ``snapshot`` fetches. Excludes ``fee_growth``: its
block-range sampling strategy (T14 brief: lifecycle boundaries + a regular stride)
is not implemented — :class:`RpcFetcher` exposes per-block ``fee_growth_at`` calls,
not a range, so it cannot flow through the generic ``fetch`` path. Requesting it
explicitly yields a clear error (see ``_pull_one_stream``); the tape and validation
treat it as optional (``check_liquidity_conservation`` / ``reconcile_fees_against_collect``
report ``skipped`` rather than fail)."""

_VALID_STREAMS: frozenset[str] = frozenset(_DEFAULT_STREAMS) | {"fee_growth"}
"""Every stream name ``--stream`` accepts (a superset of the default pull set)."""


# ---------------------------------------------------------------------------
# Window resolution: date → blocks
# ---------------------------------------------------------------------------


def _resolve_window_to_blocks(
    config: DataConfig,
    gas_fetcher: GasFetcher,
) -> tuple[BlockNumber, BlockNumber]:
    """Resolve a date-mode window to concrete block numbers via the gas fetcher."""
    w = config.window
    if w.start_block is not None and w.end_block is not None:
        return w.start_block, w.end_block
    if w.start_utc is not None and w.end_utc is not None:
        start = BlockNumber(gas_fetcher.block_for_timestamp(w.start_utc))
        end = BlockNumber(gas_fetcher.block_for_timestamp(w.end_utc))
        LOGGER.info(
            "Resolved date window %s..%s → blocks %d..%d",
            w.start_utc.isoformat(),
            w.end_utc.isoformat(),
            start,
            end,
        )
        return start, end
    raise ConfigError(
        "WindowConfig has neither block bounds nor date bounds set; cannot resolve"
    )


# ---------------------------------------------------------------------------
# Fetcher factory
# ---------------------------------------------------------------------------


def _make_fetchers(config: DataConfig) -> dict[str, object]:
    """Instantiate the four concrete fetchers for a given config."""
    endpoints = config.endpoints
    cache_dir = config.cache_dir
    return {
        "thegraph": TheGraphFetcher(endpoints, cache_dir),
        "rpc": RpcFetcher(endpoints, cache_dir),
        "gas": GasFetcher(endpoints, cache_dir),
        "reference": ReferenceFetcher(endpoints, cache_dir),
    }


# ---------------------------------------------------------------------------
# Idempotency check
# ---------------------------------------------------------------------------


def _stream_already_pulled(
    output_dir: Path,
    stream: str,
) -> bool:
    """Check whether ``stream`` already has parquet data in the manifest."""
    try:
        manifest = read_manifest(output_dir)
    except (ValidationError, FileNotFoundError):
        return False
    sm = manifest.streams.get(stream)
    if sm is None:
        return False
    return sm.row_count > 0


# ---------------------------------------------------------------------------
# pull — the core fetch orchestration
# ---------------------------------------------------------------------------


def pull(
    config: DataConfig,
    *,
    streams: Sequence[str] | None = None,
    route: Route | None = None,
    force: bool = False,
) -> PullReport:
    """Orchestrate fetching all requested streams and writing them to parquet.

    Resumable: skips streams whose parquet already covers the window. Idempotent:
    a re-run with a warm cache makes zero network requests. The manifest is NOT
    written here — ``snapshot()`` writes it after a successful verify.
    """
    requested = tuple(streams) if streams is not None else _DEFAULT_STREAMS
    for s in requested:
        if s not in _VALID_STREAMS:
            raise ConfigError(
                f"Unknown stream {s!r}; expected one of {sorted(_VALID_STREAMS)}"
            )

    fetchers = _make_fetchers(config)
    gas_fetcher: GasFetcher = fetchers["gas"]  # type: ignore[assignment]
    start_block, end_block = _resolve_window_to_blocks(config, gas_fetcher)
    output_dir = config.output_dir

    report = PullReport(started_at_utc=datetime.now(UTC), all_from_cache=True)
    total_requests = 0
    all_cached = True

    for stream in requested:
        if not force and _stream_already_pulled(output_dir, stream):
            LOGGER.info("pull: %s already pulled — skipping (use --force to re-fetch)", stream)
            try:
                existing = read_manifest(output_dir)
                sm = existing.streams.get(stream)
                if sm is not None:
                    spr = StreamPullResult(
                        stream=stream,
                        route=sm.route or Route.RPC,
                        row_count=sm.row_count,
                        n_requests=0,
                        from_cache=True,
                        warnings=sm.warnings,
                        content_hash=sm.content_hash,
                        min_block=sm.min_block,
                        max_block=sm.max_block,
                    )
                    report = PullReport(
                        streams={**report.streams, stream: spr},
                        n_total_requests=total_requests,
                        all_from_cache=all_cached,
                        started_at_utc=report.started_at_utc,
                        finished_at_utc=datetime.now(UTC),
                    )
                    continue
            except (ValidationError, FileNotFoundError):
                pass

        spr = _pull_one_stream(
            config, fetchers, stream, start_block, end_block, route
        )
        total_requests += spr.n_requests
        if not spr.from_cache:
            all_cached = False
        report = PullReport(
            streams={**report.streams, stream: spr},
            n_total_requests=total_requests,
            all_from_cache=all_cached,
            started_at_utc=report.started_at_utc,
            finished_at_utc=datetime.now(UTC),
        )

    return report


def _pull_one_stream(
    config: DataConfig,
    fetchers: dict[str, object],
    stream: str,
    start_block: BlockNumber,
    end_block: BlockNumber,
    force_route: Route | None,
) -> StreamPullResult:
    """Pull a single stream end-to-end: fetch → write parquet → return result."""
    pool = config.pool
    output_dir = config.output_dir

    if stream == "gas":
        fetcher: GasFetcher = fetchers["gas"]  # type: ignore[assignment]
        used_route = Route.RPC
        request = FetchRequest(
            stream="gas", pool=None,
            start_block=start_block, end_block=end_block,
        )
    elif stream == "reference":
        fetcher = fetchers["reference"]  # type: ignore[assignment]
        used_route = Route.REFERENCE
        w = config.window
        request = FetchRequest(
            stream="reference", pool=None,
            start_block=None, end_block=None,
            start_utc=w.start_utc, end_utc=w.end_utc,
        )
    elif stream == "regime":
        return _compute_regime(config, output_dir)
    elif stream == "fee_growth":
        raise FetchError(
            "fee_growth is not part of the default pull and has no block-range fetch: "
            "the T14 sampling strategy (position-lifecycle boundaries + a regular stride) "
            "is not implemented yet. RpcFetcher.fee_growth_at(pool, block, ticks) provides "
            "per-block snapshots; T10's replay covers the gaps between samples."
        )
    else:
        if force_route == Route.RPC:
            fetcher = fetchers["rpc"]  # type: ignore[assignment]
            used_route = Route.RPC
        else:
            fetcher = fetchers["thegraph"]  # type: ignore[assignment]
            used_route = Route.THEGRAPH
        request = FetchRequest(
            stream=stream, pool=pool,
            start_block=start_block, end_block=end_block,
        )

    LOGGER.info(
        "pull: fetching %s via %s [%d..%d]",
        stream, used_route.value, start_block, end_block,
    )
    result: FetchResult = fetcher.fetch(request)

    sm = write_stream(
        result.table, output_dir, stream,
        pool=pool, route=used_route,
        n_requests=result.n_requests, warnings=result.warnings,
    )
    return StreamPullResult(
        stream=stream, route=used_route,
        row_count=result.table.num_rows,
        n_requests=result.n_requests,
        from_cache=result.from_cache,
        warnings=result.warnings,
        content_hash=sm.content_hash,
        min_block=sm.min_block,
        max_block=sm.max_block,
    )


def _compute_regime(config: DataConfig, output_dir: Path) -> StreamPullResult:
    """Compute regime labels from the cached reference stream and write to parquet."""
    pool = config.pool
    try:
        reference = read_stream(output_dir, "reference", pool=pool)
    except (ValidationError, FileNotFoundError):
        raise FetchError(
            "Cannot compute regime labels: reference stream not yet pulled. "
            "Run 'undertow-data pull --stream reference' first."
        ) from None

    regime_table = label_series(reference, config.regime)
    sm = write_stream(
        regime_table, output_dir, "regime",
        pool=pool, route=Route.REFERENCE,
        n_requests=0, warnings=(),
    )
    return StreamPullResult(
        stream="regime", route=Route.REFERENCE,
        row_count=regime_table.num_rows,
        n_requests=0, from_cache=False, warnings=(),
        content_hash=sm.content_hash,
        min_block=None, max_block=None,
    )


# ---------------------------------------------------------------------------
# build — assemble the event tape
# ---------------------------------------------------------------------------


def build(config: DataConfig) -> "Dataset":
    """Read all cached streams from parquet and assemble a ``Dataset``."""
    from undertow.data.transforms.align import Dataset

    pool = config.pool
    output_dir = config.output_dir

    try:
        manifest = read_manifest(output_dir)
    except (ValidationError, FileNotFoundError):
        raise ValidationError(
            f"No manifest found at {output_dir}. Run 'undertow-data pull' first."
        ) from None

    streams: dict[str, pa.Table] = {}
    for name in _LOG_STREAMS:
        try:
            streams[name] = read_stream(output_dir, name, pool=pool)
        except (ValidationError, FileNotFoundError):
            raise ValidationError(
                f"Stream {name!r} not found in {output_dir}. "
                f"Run 'undertow-data pull --stream {name}' first."
            ) from None

    gas = _read_optional_stream(output_dir, "gas", pool)
    reference = _read_optional_stream(output_dir, "reference", pool)
    regime = _read_optional_stream(output_dir, "regime", pool)
    fee_growth = _read_optional_stream(output_dir, "fee_growth", pool)

    tape = build_event_tape(
        streams=streams,
        gas=gas, reference=reference, regime=regime, fee_growth=fee_growth,
        pool=pool,
    )
    return Dataset(
        tape=tape,
        gas=gas, reference=reference, regime=regime, fee_growth=fee_growth,
        manifest=manifest,
    )


def _read_optional_stream(
    root: Path, stream: str, pool: PoolConfig
) -> pa.Table | None:
    """Read a stream from parquet, returning ``None`` if not found."""
    try:
        return read_stream(root, stream, pool=pool)
    except (ValidationError, FileNotFoundError):
        return None


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def verify(config: DataConfig) -> list[CheckResult]:
    """Build the dataset and run all validation checks."""
    dataset = build(config)
    return run_all_checks(dataset)


# ---------------------------------------------------------------------------
# snapshot — the end-to-end pipeline
# ---------------------------------------------------------------------------


def _build_manifest(
    config: DataConfig,
    pull_report: PullReport,
    warnings: tuple[str, ...] = (),
) -> DatasetManifest:
    """Assemble the manifest from a completed pull and the check-derived warnings."""
    return DatasetManifest(
        schema_version=SCHEMA_VERSION,
        pool=config.pool,
        window=config.window,
        streams={
            name: StreamManifest(
                row_count=spr.row_count,
                content_hash=spr.content_hash,
                min_block=spr.min_block,
                max_block=spr.max_block,
                route=spr.route,
                n_requests=spr.n_requests,
                warnings=spr.warnings,
            )
            for name, spr in pull_report.streams.items()
        },
        created_at_utc=datetime.now(UTC),
        git_commit=git_commit(),
        library_versions=library_versions(),
        warnings=warnings,
        endpoint_hosts=endpoint_hosts(config.endpoints),
    )


def snapshot(
    config: DataConfig,
    out: Path | None = None,
) -> SnapshotReport:
    """Run the full pipeline: pull → build → verify → write manifest + report.

    A provisional manifest is written between pull and verify because ``build()`` and
    ``verify()`` read it from disk — but a fresh output directory has none. The
    provisional manifest is replaced by the final one after checks pass, and deleted
    if a critical check fails, so a failed run never leaves a manifest claiming
    completeness (the manifest-last invariant).
    """
    output_dir = out if out is not None else config.output_dir
    if out is not None:
        config = DataConfig(
            pool=config.pool,
            window=config.window,
            regime=config.regime,
            endpoints=config.endpoints,
            cache_dir=config.cache_dir,
            output_dir=output_dir,
            gas_units=config.gas_units,
        )

    pull_report = pull(config)

    provisional_path = write_manifest(
        _build_manifest(config, pull_report, warnings=()),
        output_dir,
    )
    try:
        dataset = build(config)
        checks = verify(config)
    except BaseException:
        # BaseException (not Exception) is deliberate: a Ctrl-C mid-verify must also
        # clean up, so an unverified provisional manifest never survives.
        provisional_path.unlink(missing_ok=True)
        raise

    critical_failures = [c for c in checks if not c.passed and c.severity == "critical"]
    if critical_failures:
        LOGGER.warning(
            "snapshot: %d critical check(s) failed — manifest will NOT be written",
            len(critical_failures),
        )
        report_path = output_dir / "validation_report.md"
        render_report(checks, report_path, manifest=dataset.manifest)
        provisional_path.unlink(missing_ok=True)
        raise ValidationError(
            f"snapshot: {len(critical_failures)} critical check(s) failed; "
            "manifest not written. See validation_report.md for details."
        )

    all_warnings: list[str] = []
    for spr in pull_report.streams.values():
        all_warnings.extend(spr.warnings)
    for c in checks:
        if not c.passed and c.severity == "warning":
            all_warnings.append(f"{c.name}: {c.detail}")

    manifest = _build_manifest(
        config, pull_report, warnings=tuple(dict.fromkeys(all_warnings))
    )
    manifest_path = write_manifest(manifest, output_dir)

    report_path = output_dir / "validation_report.md"
    render_report(checks, report_path, manifest=manifest)

    LOGGER.info("snapshot: complete — manifest at %s", manifest_path)
    return SnapshotReport(
        pull=pull_report,
        dataset=dataset,
        checks=checks,
        manifest_path=manifest_path,
        validation_report_path=report_path,
    )


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


def info(config: DataConfig) -> InfoReport:
    """Read the manifest and return a summary of the dataset at ``config.output_dir``."""
    output_dir = config.output_dir
    try:
        manifest = read_manifest(output_dir)
        exists = True
    except (ValidationError, FileNotFoundError):
        manifest = DatasetManifest(
            schema_version=SCHEMA_VERSION,
            pool=config.pool,
            window=config.window,
            streams={},
            created_at_utc=datetime.now(UTC),
            git_commit=git_commit(),
            library_versions=library_versions(),
            warnings=("No manifest found — dataset has not been pulled yet.",),
            endpoint_hosts=endpoint_hosts(config.endpoints),
        )
        exists = False
    return InfoReport(
        manifest=manifest,
        dataset_exists=exists,
        output_dir=output_dir,
    )