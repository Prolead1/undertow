"""Partitioned parquet read/write for ``undertow.data`` streams (T09, ``CONTRACTS.md`` §7).

On-disk layout (hive-style directory names so readers can prune whole directories):

- **pool-scoped streams** — exactly the streams in T03's ``POOL_SCOPED``::

      root/pool=<address>/stream=<stream>/block_bucket=<block_number // 100_000>/part.parquet

- **global block streams** (``gas``, the only global stream with a block column)::

      root/stream=<stream>/block_bucket=<block_number // 100_000>/part.parquet

- **global month streams** (``reference``, ``regime`` — no block column)::

      root/stream=<stream>/month=<YYYY-MM>/part.parquet

Nothing here hardcodes a stream name. The pool-scoped classification comes from
``schemas.POOL_SCOPED``, the sort order from ``schemas.SORT_KEYS``, and the bucket
dimension is derived from the stream's own schema: a ``block_number`` column implies
block buckets, otherwise the first timestamp column implies month buckets.

Determinism contract: ``write_stream`` sorts by the stream's sort keys, writes with
fixed options (zstd, fixed level, fixed row group size, statistics on), and computes
``content_hash`` over the **table**, never the file bytes — so the hash is stable
across pyarrow versions even though parquet files legitimately embed a ``created_by``
string. Idempotency: a bucket holds exactly one ``part.parquet``, replaced atomically
(temp file + ``os.replace``), so re-writing a stream never leaves duplicated rows and
an interrupted write never leaves a half-written part that reads as valid.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import struct
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from undertow.data.config import PoolConfig
from undertow.data.schemas import (
    POOL_SCOPED,
    SCHEMA_REGISTRY,
    SORT_KEYS,
    empty_table,
    validate_table,
)
from undertow.data.storage.manifest import StreamManifest
from undertow.data.types import BlockNumber, Route, SchemaViolationError, ValidationError

# Fixed writer options — the determinism contract (T09 brief §1). Changing any of
# these changes part.parquet bytes; they are pinned on purpose.
_COMPRESSION: Final[str] = "zstd"
_COMPRESSION_LEVEL: Final[int] = 7
_ROW_GROUP_SIZE: Final[int] = 65_536
_WRITE_STATISTICS: Final[bool] = True
_PARQUET_VERSION: Final[str] = "2.6"
_DATA_PAGE_VERSION: Final[str] = "2.0"
_USE_DICTIONARY: Final[bool] = True

_BLOCK_BUCKET_SIZE: Final[int] = 100_000
_MONTH_FORMAT: Final[str] = "%Y-%m"
_PART_FILENAME: Final[str] = "part.parquet"


def _sort_keys(keys: Sequence[str]) -> list[tuple[str, str]]:
    """``SORT_KEYS`` names -> pyarrow ``sort_by`` sort keys (ascending)."""
    return [(name, "ascending") for name in keys]


# ---------------------------------------------------------------------------
# write_stream
# ---------------------------------------------------------------------------


def write_stream(
    table: pa.Table,
    root: Path,
    stream: str,
    *,
    pool: PoolConfig,
    route: Route | None = None,
    n_requests: int = 0,
    warnings: Sequence[str] = (),
) -> StreamManifest:
    """Write ``table`` for ``stream`` under ``root`` and return its per-stream manifest.

    Validates against the canonical schema (strict), sorts by ``SORT_KEYS[stream]``,
    partitions into bucket directories, and writes one ``part.parquet`` per bucket
    via an atomic temp-file replace. Re-writing the same bucket replaces its part
    file — one logical dataset, never two part files with duplicated rows.

    ``route`` / ``n_requests`` / ``warnings`` are provenance recorded by the caller
    (T14 passes them through from the fetch result); they default to neutral values.

    Idempotency model: a bucket holds exactly one ``part.parquet``, replaced
    atomically, so re-writing the same data never duplicates rows. Note that
    re-writing a *narrower* block range leaves stale bucket directories on disk;
    a logical full replacement needs a fresh root or explicit cleanup.
    """
    if stream not in SCHEMA_REGISTRY:
        raise ValidationError(
            f"write_stream: unknown stream {stream!r}; expected one of {sorted(SCHEMA_REGISTRY)}"
        )
    schema = SCHEMA_REGISTRY[stream]
    validate_table(table, schema, strict=True)
    if stream in POOL_SCOPED:
        _check_pool_address(table, pool, stream)

    root_path = Path(root)
    ordered = table.sort_by(_sort_keys(SORT_KEYS[stream]))
    checksum = content_hash(ordered)

    if ordered.num_rows == 0:
        return StreamManifest(
            row_count=0,
            content_hash=checksum,
            route=route,
            n_requests=n_requests,
            warnings=tuple(warnings),
        )

    if "block_number" in schema.names:
        min_block = BlockNumber(int(pc.min(ordered.column("block_number")).as_py()))
        max_block = BlockNumber(int(pc.max(ordered.column("block_number")).as_py()))
        _write_block_buckets(ordered, root_path, stream, pool)
    else:
        min_block = None
        max_block = None
        _write_month_buckets(ordered, root_path, stream, pool)

    return StreamManifest(
        row_count=ordered.num_rows,
        min_block=min_block,
        max_block=max_block,
        content_hash=checksum,
        route=route,
        n_requests=n_requests,
        warnings=tuple(warnings),
    )


def _check_pool_address(table: pa.Table, pool: PoolConfig, stream: str) -> None:
    """Every row of a pool-scoped stream must belong to the pool being written —
    otherwise rows land under the wrong ``pool=`` directory and reads silently miss."""
    if "pool_address" not in table.column_names:
        raise ValidationError(
            f"write_stream: pool-scoped stream {stream!r} has no pool_address column"
        )
    addrs = table.column("pool_address").to_pylist()
    if any(a != pool.address for a in addrs):
        raise ValidationError(
            f"write_stream: {stream} table mixes pools; expected every row to have "
            f"pool_address {pool.address!r}"
        )


def _write_block_buckets(table: pa.Table, root: Path, stream: str, pool: PoolConfig) -> None:
    base = _stream_root(root, stream, pool)
    blocks = table.column("block_number").to_pylist()
    keys = [f"block_bucket={int(b) // _BLOCK_BUCKET_SIZE}" for b in blocks]
    _write_buckets(table, base, keys)


def _write_month_buckets(table: pa.Table, root: Path, stream: str, pool: PoolConfig) -> None:
    base = _stream_root(root, stream, pool)
    ts_col = _timestamp_column(table.schema)
    stamps = table.column(ts_col).to_pylist()
    keys = [f"month={st.strftime(_MONTH_FORMAT)}" for st in stamps]
    _write_buckets(table, base, keys)


def _stream_root(root: Path, stream: str, pool: PoolConfig) -> Path:
    if stream in POOL_SCOPED:
        return root / f"pool={pool.address}" / f"stream={stream}"
    return root / f"stream={stream}"


def _timestamp_column(schema: pa.Schema) -> str:
    for field in schema:
        if pa.types.is_timestamp(field.type):
            return field.name
    raise ValidationError(
        "stream schema has neither a block_number column nor a timestamp column; "
        "cannot choose a bucket dimension"
    )


def _write_buckets(table: pa.Table, base: Path, keys: list[str]) -> None:
    """Split the pre-sorted ``table`` into per-bucket subtables and write one part file each."""
    groups: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    for key, indices in groups.items():
        part_path = base / key / _PART_FILENAME
        _write_part(part_path, table.take(pa.array(indices)))


def _write_part(path: Path, table: pa.Table) -> None:
    """Atomic part-file write: temp file in the same directory, then ``os.replace``.

    An interrupted write leaves at worst a dot-prefixed ``.part.parquet.*.tmp`` file
    that no reader matches, never a half-written ``part.parquet``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".part.parquet.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        pq.write_table(
            table,
            tmp,
            compression=_COMPRESSION,
            compression_level=_COMPRESSION_LEVEL,
            row_group_size=_ROW_GROUP_SIZE,
            write_statistics=_WRITE_STATISTICS,
            version=_PARQUET_VERSION,
            data_page_version=_DATA_PAGE_VERSION,
            use_dictionary=_USE_DICTIONARY,
        )
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# read_stream
# ---------------------------------------------------------------------------


def read_stream(
    root: Path,
    stream: str,
    *,
    pool: PoolConfig,
    start_block: BlockNumber | None = None,
    end_block: BlockNumber | None = None,
) -> pa.Table:
    """Read ``stream`` back from ``root``, schema-validated and sorted by its sort keys.

    Predicate pushdown: only buckets overlapping ``[start_block, end_block]`` are
    opened (whole-directory pruning) and a parquet row filter on ``block_number`` is
    applied per file. A range spanning buckets yields exactly the in-range rows with
    no duplicates. Streams without a block column (``reference``, ``regime``) reject
    block ranges — there is nothing to push down on, and silently returning the full
    table would be a lie about the request.
    """
    if stream not in SCHEMA_REGISTRY:
        raise ValidationError(
            f"read_stream: unknown stream {stream!r}; expected one of {sorted(SCHEMA_REGISTRY)}"
        )
    schema = SCHEMA_REGISTRY[stream]
    has_block = "block_number" in schema.names
    if start_block is not None or end_block is not None:
        if not has_block:
            raise ValidationError(
                f"read_stream: stream {stream!r} has no block_number column; "
                "start_block/end_block are not supported for it"
            )
        if (
            start_block is not None
            and end_block is not None
            and start_block > end_block
        ):
            raise ValidationError(
                f"read_stream: start_block {start_block} must be <= end_block {end_block}"
            )

    root_path = Path(root)
    parts = _matching_parts(root_path, stream, pool, start_block, end_block, has_block)
    if not parts:
        return empty_table(schema)

    tables = [_read_part(p, schema, start_block, end_block) for p in parts]
    merged = pa.concat_tables(tables)
    merged = merged.sort_by(_sort_keys(SORT_KEYS[stream]))
    validate_table(merged, schema, strict=True)
    return merged


def _matching_parts(
    root: Path,
    stream: str,
    pool: PoolConfig,
    start_block: BlockNumber | None,
    end_block: BlockNumber | None,
    has_block: bool,
) -> list[Path]:
    base = _stream_root(root, stream, pool)
    if not base.is_dir():
        return []
    if has_block:
        lo = start_block // _BLOCK_BUCKET_SIZE if start_block is not None else -(2**63)
        hi = end_block // _BLOCK_BUCKET_SIZE if end_block is not None else 2**63 - 1
        chosen = [
            d
            for d in base.glob("block_bucket=*")
            if d.is_dir() and lo <= _bucket_number(d.name) <= hi
        ]
    else:
        chosen = [d for d in base.glob("month=*") if d.is_dir()]
    parts: list[Path] = []
    for directory in sorted(chosen):
        part = directory / _PART_FILENAME
        if part.is_file():
            parts.append(part)
    return parts


def _bucket_number(dirname: str) -> int:
    """Parse ``block_bucket=150`` -> 150; a parse failure prunes the directory."""
    raw = dirname.split("=", 1)[1] if "=" in dirname else dirname
    try:
        return int(raw)
    except ValueError:
        return -(2**63)  # never matches any pruning range


def _read_part(
    path: Path,
    schema: pa.Schema,
    start_block: BlockNumber | None,
    end_block: BlockNumber | None,
) -> pa.Table:
    filters: list[tuple[str, str, int]] = []
    if start_block is not None:
        filters.append(("block_number", ">=", int(start_block)))
    if end_block is not None:
        filters.append(("block_number", "<=", int(end_block)))
    table = pq.read_table(path, filters=filters or None)
    if table.column_names != list(schema.names):
        raise SchemaViolationError(
            f"_read_part: {path} has columns {table.column_names}; expected the canonical "
            f"order {list(schema.names)}. Refusing to attach names positionally: a "
            "reordered file would silently mislabel values under the wrong names."
        )
    # Re-attach the canonical schema (metadata included, no data cast) so concat and
    # validation see exactly the canonical schema regardless of what the file carried.
    return pa.Table.from_arrays(table.columns, schema=schema)


# ---------------------------------------------------------------------------
# content_hash
# ---------------------------------------------------------------------------


def content_hash(table: pa.Table) -> str:
    """Stable hash of the TABLE, not the file bytes.

    sha256 over: the schema (``name:type`` per field, in order), then ``num_rows``,
    then each column's values in the stream's sort-key order. Column values are
    packed canonically (per-type, little-endian, null-prefixed) so the digest does
    NOT depend on row-group boundaries, chunking, compression, or the pyarrow
    version. The stream is identified from the table's schema metadata; tables
    without metadata are matched against ``SCHEMA_REGISTRY`` by names and types.

    Two tables with equal logical content — same schema, same rows, any input
    order — hash identically; any difference flips the digest. Ties in the sort
    key are hashed in input order, but every stream's ``SORT_KEYS`` is unique per
    partition, so ties cannot arise for data this pipeline writes. Distinct
    ``-0.0``/``+0.0`` and NaN payloads digest differently (bit patterns differ).
    """
    ordered = table.sort_by(_sort_keys(SORT_KEYS[_stream_name(table)]))
    hasher = hashlib.sha256()
    for field in ordered.schema:
        hasher.update(f"{field.name}:{field.type}".encode())
        hasher.update(b"\x00")
    hasher.update(b"\xff")
    hasher.update(struct.pack("<q", ordered.num_rows))
    for column in ordered.columns:
        hasher.update(_column_digest(column))
    return hasher.hexdigest()


def _stream_name(table: pa.Table) -> str:
    metadata = table.schema.metadata or {}
    if b"stream" in metadata:
        return metadata[b"stream"].decode()
    for name, schema in SCHEMA_REGISTRY.items():
        if table.schema.names == schema.names and all(
            a.type == b.type for a, b in zip(table.schema, schema, strict=True)
        ):
            return name
    raise ValidationError(
        "content_hash: cannot determine the table's stream (no schema metadata and no "
        "SCHEMA_REGISTRY match by names/types)"
    )


def _column_digest(column: pa.ChunkedArray) -> bytes:
    """Digest of one column's values, in logical row order.

    The column is flattened (``combine_chunks``) before digesting so the result is
    independent of how pyarrow chunked the data. This is what keeps ``content_hash``
    stable between a freshly built table and the same table read back from parquet,
    whose chunk layout we do not control; per-row-group digests would leak chunk
    boundaries into the hash.
    """
    return _chunk_digest(column.combine_chunks())


def _chunk_digest(arr: pa.Array) -> bytes:
    arr_type = arr.type
    if pa.types.is_timestamp(arr_type):
        # int64 microseconds since epoch — exact, tz-independent.
        arr = arr.cast(pa.int64())
        arr_type = arr.type
    elif pa.types.is_boolean(arr_type):
        arr = arr.cast(pa.int8())
        arr_type = arr.type

    if pa.types.is_string(arr_type) or pa.types.is_large_string(arr_type):
        values = arr.to_pylist()
        return b"".join(
            b"\x01" + value.encode("utf-8") if value is not None else b"\x00"
            for value in values
        )
    if pa.types.is_floating(arr_type):
        values = arr.to_pylist()
        return b"".join(
            b"\x01" + struct.pack("<d", value) if value is not None else b"\x00"
            for value in values
        )
    if pa.types.is_integer(arr_type):
        fmt = _integer_fmt(arr_type)
        values = arr.to_pylist()
        return b"".join(
            b"\x01" + struct.pack(fmt, value) if value is not None else b"\x00"
            for value in values
        )
    raise ValidationError(f"content_hash: unsupported column type {arr_type}")


def _integer_fmt(arr_type: pa.DataType) -> str:
    signed = "<q" if not pa.types.is_unsigned_integer(arr_type) else "<Q"
    by_width = {8: "<b", 16: "<h", 32: "<i", 64: signed}
    fmt = by_width.get(arr_type.bit_width)
    if fmt is None:
        raise ValidationError(f"content_hash: unsupported integer width {arr_type.bit_width}")
    return fmt