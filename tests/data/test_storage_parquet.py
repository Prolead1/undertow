"""Tests for ``undertow.data.storage.parquet`` (T09) — partitioned read/write.

Covers the thirteen behaviours the T09 brief pins: round-trip for every schema in
``SCHEMA_REGISTRY`` (forcing the global streams, which have neither ``pool_address``
nor ``log_index``), the two on-disk layouts (pool-scoped vs global, one copy of
``reference`` across pools), byte/hash determinism, sort-on-write, bucket-pruning
predicate pushdown, idempotent re-write, partial-write safety, empty tables, and
big-integer fidelity through the string encoding.

Table facts the tests rely on: ``write_stream`` sorts by the stream's ``SORT_KEYS``
before bucketing, writes fixed ``compression="zstd"`` level-7 files of a fixed row
group size, and hashes the sorted table. Byte identity of the part files is asserted
because the writer options are pinned (``version="2.6"``, ``data_page_version="2.0"``,
dictionary on) and the environment is fixed; the *portable* guarantee is
``content_hash`` identity, which holds across pyarrow versions — see the docstring of
``content_hash``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from undertow.data.config import PoolConfig, default_pools
from undertow.data.schemas import (
    REFERENCE_SCHEMA,
    REGIME_SCHEMA,
    SCHEMA_REGISTRY,
    SORT_KEYS,
    decode_uint,
    encode_uint,
)
from undertow.data.storage import parquet as storage_parquet
from undertow.data.storage.parquet import content_hash, read_stream, write_stream
from undertow.data.types import BlockNumber, SchemaViolationError, ValidationError

POOL_A = default_pools()["USDC_WETH_3000"]
POOL_B = default_pools()["USDC_WETH_500"]


def _table_for(
    stream: str, *, blocks: Sequence[int] | None = None, n: int = 4, pool: PoolConfig = POOL_A
) -> pa.Table:
    """A table conforming to the stream's canonical schema, with deterministic values.

    Block numbers land in ``block // 100_000`` buckets; nullable columns carry nulls
    on even rows so null round-tripping is exercised too. ``pool`` picks the
    ``pool_address`` value on pool-scoped stream rows.
    """
    schema = SCHEMA_REGISTRY[stream]
    if blocks is None:
        blocks = [10_000_000 + i * 7 for i in range(n)]
    rows: list[dict[str, object]] = []
    for i, block in enumerate(blocks):
        row: dict[str, object] = {}
        for field in schema:
            name = field.name
            t = field.type
            if name == "block_number":
                row[name] = block
            elif name == "log_index":
                row[name] = i
            elif name == "block_timestamp":
                row[name] = datetime(2022, 1, 1, tzinfo=UTC) + timedelta(
                    seconds=int(block) * 12 + i
                )
            elif name == "tx_hash":
                row[name] = f"0x{i:064x}"
            elif name == "pool_address":
                row[name] = pool.address
            elif name == "event_type":
                row[name] = stream
            elif pa.types.is_string(t) or pa.types.is_large_string(t):
                row[name] = None if field.nullable and i % 2 == 0 else encode_uint(2**128 + i)
            elif pa.types.is_timestamp(t):
                row[name] = datetime(2022, 1, 1, tzinfo=UTC) + timedelta(days=i)
            elif pa.types.is_integer(t):
                row[name] = i
            elif pa.types.is_floating(t):
                row[name] = 3000.0 + i
            elif pa.types.is_boolean(t):
                row[name] = i % 2 == 0
            else:
                raise AssertionError(f"no test value for {stream}.{name}: {t}")
        rows.append(row)
    return pa.Table.from_pylist(rows, schema=schema)


def _reference_table(symbols: Sequence[str] = ("ETHUSDT", "ETHUSDC")) -> pa.Table:
    """Reference bars spanning months 2022-01..2022-03 — the month-bucket layout."""
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        for month in ("2022-01", "2022-02", "2022-03"):
            open_time = datetime(2022, int(month[5:7]), 1, 0, 0, 0, tzinfo=UTC)
            rows.append(
                {
                    "open_time": open_time,
                    "close_time": open_time + timedelta(minutes=1),
                    "open": 2000.0,
                    "high": 2100.0,
                    "low": 1900.0,
                    "close": 2050.0,
                    "volume_base": 100.0,
                    "quote_volume": 200_000.0,
                    "trades": 42,
                    "symbol": symbol,
                    "is_gap_filled": False,
                }
            )
    return pa.Table.from_pylist(rows, schema=REFERENCE_SCHEMA)


def _regime_table() -> pa.Table:
    """Regime rows in months 2022-01..2022-02 — the month-bucket layout."""
    rows = []
    for month in ("2022-01", "2022-02"):
        rows.append(
            {
                "timestamp": datetime(2022, int(month[5:7]), 1, 12, 0, 0, tzinfo=UTC),
                "sigma_rv": 0.5,
                "mu": 0.1,
                "regime": "bull",
                "window_complete": True,
                "symbol": "ETHUSDT",
            }
        )
    return pa.Table.from_pylist(rows, schema=REGIME_SCHEMA)


# ---------------------------------------------------------------------------
# 1 / 1b. Round-trip over every schema; layouts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", sorted(SCHEMA_REGISTRY))
def test_roundtrip_every_schema(stream: str, tmp_path: Path) -> None:
    table = _table_for(stream)
    sorted_expected = table.sort_by([(k, "ascending") for k in SORT_KEYS[stream]])
    manifest = write_stream(table, tmp_path, stream, pool=POOL_A)
    read = read_stream(tmp_path, stream, pool=POOL_A)

    assert manifest.row_count == table.num_rows
    assert read.column_names == SCHEMA_REGISTRY[stream].names
    assert read.num_rows == table.num_rows
    assert read.schema == SCHEMA_REGISTRY[stream]
    assert read.to_pylist() == sorted_expected.to_pylist()
    # The hash is stable across the write side and the read side.
    assert content_hash(read) == manifest.content_hash


def test_layout_pool_scoped_streams_under_pool_dir(tmp_path: Path) -> None:
    write_stream(_table_for("swap", blocks=[1_234_567, 1_234_568]), tmp_path, "swap", pool=POOL_A)
    part = tmp_path / f"pool={POOL_A.address}" / "stream=swap" / "block_bucket=12" / "part.parquet"
    assert part.is_file()
    # A different pool gets its own hierarchy; reads are pool-keyed.
    write_stream(_table_for("swap", blocks=[1_234_567], pool=POOL_B), tmp_path, "swap", pool=POOL_B)
    assert (tmp_path / f"pool={POOL_B.address}" / "stream=swap").is_dir()
    assert read_stream(tmp_path, "swap", pool=POOL_A).num_rows == 2
    assert read_stream(tmp_path, "swap", pool=POOL_B).num_rows == 1


def test_layout_global_streams_have_no_pool_dir(tmp_path: Path) -> None:
    # gas: chain-wide, block-bucketed, never under pool=.
    write_stream(_table_for("gas", blocks=[1_234_567]), tmp_path, "gas", pool=POOL_A)
    assert (tmp_path / "stream=gas" / "block_bucket=12" / "part.parquet").is_file()
    assert not list(tmp_path.glob("pool=*/stream=gas"))

    # reference: writing for two pools must produce ONE copy on disk.
    ref = _reference_table()
    write_stream(ref, tmp_path, "reference", pool=POOL_A)
    write_stream(ref, tmp_path, "reference", pool=POOL_B)
    assert not list(tmp_path.glob("pool=*/stream=reference"))
    month_dirs = sorted(tmp_path.glob("stream=reference/month=*"))
    assert [d.name for d in month_dirs] == ["month=2022-01", "month=2022-02", "month=2022-03"]
    count = read_stream(tmp_path, "reference", pool=POOL_A).num_rows
    assert count == ref.num_rows  # one copy, not two per pool

    # regime: global, month-bucketed on its timestamp column.
    write_stream(_regime_table(), tmp_path, "regime", pool=POOL_A)
    assert (tmp_path / "stream=regime" / "month=2022-02" / "part.parquet").is_file()
    assert not list(tmp_path.glob("pool=*/stream=regime"))


# ---------------------------------------------------------------------------
# 2. Determinism
# ---------------------------------------------------------------------------


def test_determinism_byte_and_hash_identity(tmp_path: Path) -> None:
    """Same table written to two roots yields identical part-file bytes and hashes.

    Byte identity is asserted because the writer options are pinned in this
    environment; it is not the load-bearing guarantee across pyarrow versions —
    ``content_hash`` (computed over the TABLE, not the file bytes) is, so both are
    checked. If a future pyarrow bumps the embedded ``created_by`` string, byte
    identity may break while hash identity must not — do not delete this test, relax
    the byte assertion with this docstring as the reason.
    """
    table = _table_for("swap", blocks=[5_000, 105_000, 205_000])  # three buckets
    root1 = tmp_path / "r1"
    root2 = tmp_path / "r2"
    sm1 = write_stream(table, root1, "swap", pool=POOL_A)
    sm2 = write_stream(table, root2, "swap", pool=POOL_A)
    assert sm1.content_hash == sm2.content_hash

    share = root1 / f"pool={POOL_A.address}" / "stream=swap"
    parts1 = sorted(share.rglob("part.parquet"))
    parts2 = sorted((root2 / f"pool={POOL_A.address}" / "stream=swap").rglob("part.parquet"))
    assert [p.read_bytes() for p in parts1] == [p.read_bytes() for p in parts2]


def test_content_hash_is_order_and_chunk_independent() -> None:
    table = _table_for("swap", blocks=[5_000, 105_000, 205_000, 305_000])
    shuffled = table.sort_by([("log_index", "descending")])  # deliberately unsorted
    assert content_hash(table) == content_hash(shuffled)

    # Same logical content, different chunk layout -> identical hash (contract:
    # no dependence on chunk boundaries / row-group boundaries).
    single = table.combine_chunks()
    chopped = pa.concat_tables([single.slice(0, 1), single.slice(1)])
    assert content_hash(single) == content_hash(chopped)


# ---------------------------------------------------------------------------
# 3. Sorting
# ---------------------------------------------------------------------------


def test_write_sorts_and_read_returns_sorted(tmp_path: Path) -> None:
    shuffled = _table_for("gas", blocks=[5, 4, 3, 2, 1])
    write_stream(shuffled, tmp_path, "gas", pool=POOL_A)
    blocks = read_stream(tmp_path, "gas", pool=POOL_A).column("block_number").to_pylist()
    assert blocks == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# 4. Predicate pushdown and bucket pruning
# ---------------------------------------------------------------------------


def test_range_read_spans_buckets_without_duplicates(tmp_path: Path) -> None:
    table = _table_for("swap", blocks=[50_000, 150_000, 250_000, 350_000])
    write_stream(table, tmp_path, "swap", pool=POOL_A)

    middle = read_stream(
        tmp_path,
        "swap",
        pool=POOL_A,
        start_block=BlockNumber(125_000),
        end_block=BlockNumber(275_000),
    )
    assert middle.column("block_number").to_pylist() == [150_000, 250_000]
    assert middle.num_rows == 2

    full = read_stream(tmp_path, "swap", pool=POOL_A)
    assert full.column("block_number").to_pylist() == [50_000, 150_000, 250_000, 350_000]

    none = read_stream(
        tmp_path,
        "swap",
        pool=POOL_A,
        start_block=BlockNumber(400_000),
        end_block=BlockNumber(500_000),
    )
    assert none.num_rows == 0
    assert none.schema == SCHEMA_REGISTRY["swap"]


def test_range_inside_one_bucket_opens_only_that_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = _table_for("swap", blocks=[50_000, 150_000, 150_010, 250_000])
    write_stream(table, tmp_path, "swap", pool=POOL_A)

    opened: list[str] = []
    real_read = storage_parquet._read_part

    def spy(path: Path, schema: pa.Schema, start: object, end: object) -> pa.Table:
        opened.append(path.parent.name)  # the block_bucket=<n> directory
        return real_read(path, schema, start, end)  # type: ignore[arg-type]

    monkeypatch.setattr(storage_parquet, "_read_part", spy)
    out = read_stream(
        tmp_path,
        "swap",
        pool=POOL_A,
        start_block=BlockNumber(150_001),
        end_block=BlockNumber(150_020),
    )
    assert opened == ["block_bucket=1"]  # the other buckets were pruned by directory
    assert out.column("block_number").to_pylist() == [150_010]  # intra-bucket filter applied


def test_range_read_rejects_streams_without_block_column(tmp_path: Path) -> None:
    write_stream(_reference_table(), tmp_path, "reference", pool=POOL_A)
    with pytest.raises(ValidationError, match="no block_number column"):
        read_stream(tmp_path, "reference", pool=POOL_A, start_block=BlockNumber(1))


# ---------------------------------------------------------------------------
# 5. Idempotent re-write
# ---------------------------------------------------------------------------


def test_rewrite_same_stream_is_idempotent(tmp_path: Path) -> None:
    table = _table_for("swap", blocks=[150_000, 150_010])
    first = write_stream(table, tmp_path, "swap", pool=POOL_A)
    second = write_stream(table, tmp_path, "swap", pool=POOL_A)
    assert second.row_count == first.row_count == 2
    assert second.content_hash == first.content_hash
    parts = list((tmp_path / f"pool={POOL_A.address}" / "stream=swap").rglob("part.parquet"))
    assert len(parts) == 1  # one logical dataset, not duplicated part files
    assert read_stream(tmp_path, "swap", pool=POOL_A).num_rows == 2


def test_rewrite_with_new_data_replaces_not_duplicates(tmp_path: Path) -> None:
    write_stream(
        _table_for("swap", blocks=[150_000, 150_010, 150_020]), tmp_path, "swap", pool=POOL_A
    )
    write_stream(_table_for("swap", blocks=[150_030]), tmp_path, "swap", pool=POOL_A)
    out = read_stream(tmp_path, "swap", pool=POOL_A)
    assert out.num_rows == 1 and out.column("block_number").to_pylist() == [150_030]


# ---------------------------------------------------------------------------
# 6. Partial-write safety
# ---------------------------------------------------------------------------


def test_interrupted_write_leaves_no_part_file_and_preserves_prior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = _table_for("swap", blocks=[150_000, 150_010])
    write_stream(table, tmp_path, "swap", pool=POOL_A)
    prior = read_stream(tmp_path, "swap", pool=POOL_A).to_pylist()

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("simulated disk full")

    monkeypatch.setattr(storage_parquet.pq, "write_table", boom)
    with pytest.raises(OSError):
        write_stream(table, tmp_path, "swap", pool=POOL_A)

    swap_dir = tmp_path / f"pool={POOL_A.address}" / "stream=swap"
    parts = list(swap_dir.rglob("part.parquet"))
    assert len(parts) == 1  # the interrupted write left no second part
    strays = [p for p in swap_dir.rglob("*") if p.is_file() and p.name != "part.parquet"]
    assert strays == []  # the temp file was cleaned up
    assert read_stream(tmp_path, "swap", pool=POOL_A).to_pylist() == prior


# ---------------------------------------------------------------------------
# 7. Empty tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", ["swap", "reference"])
def test_empty_table_roundtrip(stream: str, tmp_path: Path) -> None:
    empty = SCHEMA_REGISTRY[stream].empty_table()
    sm = write_stream(empty, tmp_path, stream, pool=POOL_A)
    assert sm.row_count == 0 and sm.min_block is None and sm.max_block is None
    out = read_stream(tmp_path, stream, pool=POOL_A)
    assert out.num_rows == 0
    assert out.schema == SCHEMA_REGISTRY[stream]  # schema-valid


def test_read_never_written_stream_returns_empty_valid_table(tmp_path: Path) -> None:
    out = read_stream(tmp_path, "swap", pool=POOL_A)
    assert out.num_rows == 0
    assert out.schema == SCHEMA_REGISTRY["swap"]


# ---------------------------------------------------------------------------
# 8. Big-integer fidelity (the whole reason for string encoding)
# ---------------------------------------------------------------------------


def test_bigint_fidelity_through_string_columns(tmp_path: Path) -> None:
    big = encode_uint(2**256 - 1)  # uint256 max
    neg = encode_uint(-(2**255))  # int256 min
    table = _table_for("swap", n=1)
    table = table.set_column(
        table.column_names.index("amount0"), "amount0", pa.array([big], type=pa.string())
    ).set_column(
        table.column_names.index("amount1"), "amount1", pa.array([neg], type=pa.string())
    )
    write_stream(table, tmp_path, "swap", pool=POOL_A)
    out = read_stream(tmp_path, "swap", pool=POOL_A)
    assert decode_uint(out.column("amount0")[0].as_py()) == 2**256 - 1
    assert decode_uint(out.column("amount1")[0].as_py()) == -(2**255)
    assert out.column("amount0")[0].as_py() == big  # byte-exact round-trip


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_unknown_stream_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError, match="unknown stream"):
        write_stream(_table_for("swap"), tmp_path, "bogus", pool=POOL_A)
    with pytest.raises(ValidationError, match="unknown stream"):
        read_stream(tmp_path, "bogus", pool=POOL_A)


def test_mixed_pool_rows_rejected(tmp_path: Path) -> None:
    table = _table_for("swap", n=2)
    table = table.set_column(
        table.column_names.index("pool_address"),
        "pool_address",
        pa.array([POOL_A.address, POOL_B.address], type=pa.string()),
    )
    with pytest.raises(ValidationError, match="mixes pools"):
        write_stream(table, tmp_path, "swap", pool=POOL_A)


def test_written_table_is_validated_against_schema(tmp_path: Path) -> None:
    bad = _table_for("swap").drop_columns(["recipient"])
    from undertow.data.types import SchemaViolationError

    with pytest.raises(SchemaViolationError):
        write_stream(bad, tmp_path, "swap", pool=POOL_A)


def test_range_read_validates_bounds(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="must be <= end_block"):
        read_stream(
            tmp_path, "gas", pool=POOL_A, start_block=BlockNumber(10), end_block=BlockNumber(5)
        )


def test_read_stream_rejects_reordered_column_file(tmp_path: Path) -> None:
    """A part file whose columns are in a different order must be refused, never
    silently re-labelled: same-typed columns (e.g. two string columns swapped)
    would otherwise pass schema validation with values under the wrong names."""
    table = _table_for("swap", blocks=[150_000, 150_010])
    reordered = table.select([c for c in table.column_names if c != "amount0"] + ["amount0"])
    part = (
        tmp_path / f"pool={POOL_A.address}" / "stream=swap" / "block_bucket=1" / "part.parquet"
    )
    part.parent.mkdir(parents=True)
    pq.write_table(reordered, part)
    with pytest.raises(SchemaViolationError, match="expected the canonical order"):
        read_stream(tmp_path, "swap", pool=POOL_A)