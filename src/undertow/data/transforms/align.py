"""Stream alignment -> the block-ordered event tape (T11; ``CONTRACTS.md`` §6.2).

This is the artifact ``undertow.sim``'s backtester consumes, so it is where the
roadmap's hardest rule lives: **no look-ahead, ever** (``PLAN.md`` §4, §8). Every
transform that produces a row for block *n* may only read data from blocks ≤ *n*;
a rolling window, an as-of join, or a forward-fill that touches a later row is
look-ahead bias — the single most common way a backtest lies.

The mechanical expression of that rule is ``assert_no_lookahead``: every as-of
join in this module carries its source timestamp through as a private
``_src_ts_<col>`` / ``_src_blk_<col>`` column, and the final tape is checked —
inside ``build_event_tape``, not just by callers who remember — that no joined
value is known only at a time later than its row's own block time.

Price orientation (quoted from T02's ``sqrt_price_x96_to_price`` docstring,
``CONTRACTS.md`` §3.0): *"Human price of token1 denominated in token0 — USDC per
WETH for the pinned pools (~3000)"*. ``price_pool`` and ``price_reference`` are
both in those units; a systematic 1e12-scale disagreement between them is the
decimals bug that T11 test #9 exists to catch.

The merge key is ``(block_number, log_index)`` and it **must be unique** in the
tape. A duplicate means a fetcher paginated wrong; raising ``ValidationError``
instead of silently de-duplicating keeps that bug loud (T05's boundary de-dup is
T05's job). ``seq`` is the dense 0..N-1 step index the RL environment uses.

This module performs no fetching and no disk I/O beyond what T09 provides; the
enriched ``gas`` table with ``eth_usd_price`` populated is produced by
``attach_reference_prices`` (the ``CONTRACTS.md`` GAS_SCHEMA note says T11 fills
that column, which T07 writes null).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

import polars as pl
import pyarrow as pa

from undertow.data.config import PoolConfig
from undertow.data.fixedpoint import sqrt_price_x96_to_price
from undertow.data.schemas import (
    EVENT_TAPE_SCHEMA,
    GAS_SCHEMA,
    REGIME_SCHEMA,
    SCHEMA_REGISTRY,
    empty_table,
    validate_table,
)
from undertow.data.storage.manifest import DatasetManifest
from undertow.data.types import EventType, Regime, ValidationError

LOGGER = logging.getLogger("undertow.data.transforms.align")

_LOG_STREAMS: tuple[EventType, ...] = (
    EventType.SWAP,
    EventType.MINT,
    EventType.BURN,
    EventType.COLLECT,
)
"""The four log streams the tape unions (CONTRACTS.md §4.0: log streams are keyed by
``(block_number, log_index)``). ``flash`` is a fee-growth contributor, not a liquidity
event, and is deliberately absent from the tape."""

_KEY_COLUMNS: tuple[str, ...] = (
    "block_number",
    "log_index",
    "block_timestamp",
    "tx_hash",
    "pool_address",
    "event_type",
)
"""The six key columns, in ``CONTRACTS.md`` §4.0.1 order."""

_BASE_TAPE_COLUMNS: tuple[str, ...] = tuple(f.name for f in EVENT_TAPE_SCHEMA)[:17]
"""The base tape carries exactly the key columns plus the union of log-stream payload
columns (``CONTRACTS.md`` §4.8, columns 0..16) — the derived columns are added later."""

_BASE_TAPE_SCHEMA: pa.Schema = pa.schema(
    [EVENT_TAPE_SCHEMA.field(name) for name in _BASE_TAPE_COLUMNS]
)
"""The exact 17-column schema of the pre-derivation base tape."""

# Private join-tracking column prefixes, understood by assert_no_lookahead.
_SRC_TS_PREFIX = "_src_ts_"  # source *timestamp* of an as-of join (vs the tape's block time)
_SRC_BLK_PREFIX = "_src_blk_"  # source *block number* of a fee-growth as-of join

# fee_growth_source values (CONTRACTS.md §4.8) — there is deliberately no
# "interpolated_future" value: wanting one means you are about to look ahead.
_FG_EXACT = "exact"
_FG_STALE = "stale_prior"


@dataclass(frozen=True, slots=True)
class Dataset:
    """The handoff artifact (``CONTRACTS.md`` §6.2).

    Every table field is ``pa.Table | None`` on purpose: T16's
    ``load_dataset(streams=[...])`` fast path returns a **partial** dataset. ``None``
    means "not requested / not present"; an empty table means "requested, present,
    zero rows". Collapsing the two would make a missing stream indistinguishable from
    an empty one (test 14 pins this). ``build_event_tape`` does not populate
    ``route_samples`` — T14's route policy does; it is plumbed through here only so
    T13's ``crosscheck_routes`` can read both routes' raw tables.
    """

    tape: pa.Table | None
    gas: pa.Table | None
    reference: pa.Table | None
    regime: pa.Table | None
    fee_growth: pa.Table | None
    manifest: DatasetManifest
    route_samples: Mapping[tuple[str, str], pa.Table] = field(default_factory=dict)

    def require(self, name: str) -> pa.Table:
        """Return the requested stream's table, or raise ``ValidationError`` naming the
        stream and the ``undertow-data pull`` command that would produce it.

        Use this instead of scattering ``assert x is not None`` at every call site —
        the error message a user sees must tell them *how* to fix it.
        """
        field_name = _STREAM_FIELD_BY_NAME.get(name)
        if field_name is None:
            raise ValidationError(
                f"Dataset.require({name!r}): unknown stream {name!r}; expected one of "
                f"{sorted(_STREAM_FIELD_BY_NAME)}. Run 'undertow-data pull' to fetch streams."
            )
        table = getattr(self, field_name)
        if table is None:
            raise ValidationError(
                f"Dataset.require({name!r}): stream {name!r} is absent from this dataset "
                "(caller did not request it). "
                f"Run 'undertow-data pull --stream {name}' to fetch it."
            )
        return table


_STREAM_FIELD_BY_NAME: Mapping[str, str] = {
    "event_tape": "tape",  # the canonical stream name for the tape
    "tape": "tape",  # alias some callers prefer
    "gas": "gas",
    "reference": "reference",
    "regime": "regime",
    "fee_growth": "fee_growth",
}


# ---------------------------------------------------------------------------
# Merge: four log streams -> one block-ordered, key-unique base tape.
# ---------------------------------------------------------------------------


def _cast_stream_to_base(table: pa.Table, event_type: str) -> pa.Table:
    """Project one log stream onto the tape's column superset, null-padding inapplicable
    columns (``CONTRACTS.md`` §4.8: "null where inapplicable, never a zero — zero is a
    value, null is an absence")."""
    available = set(table.column_names)
    arrays: dict[str, pa.ChunkedArray] = {}
    for name in _KEY_COLUMNS:
        if name == "event_type":
            arrays[name] = pa.array([event_type] * table.num_rows, type=pa.string())
        else:
            arrays[name] = table.column(name)
    for name in _BASE_TAPE_COLUMNS:
        if name in _KEY_COLUMNS:
            continue  # handled above
        field_col = EVENT_TAPE_SCHEMA.field(name)
        if name in available:
            arrays[name] = table.column(name).cast(field_col.type)
        else:
            arrays[name] = pa.nulls(table.num_rows, type=field_col.type)
    base = pa.table(arrays)
    if base.schema != _BASE_TAPE_SCHEMA:
        return base.cast(_BASE_TAPE_SCHEMA)
    return base


def _assert_unique_keys(block_numbers: list[int], log_indices: list[int], where: str) -> None:
    """Raise ``ValidationError`` if ``(block_number, log_index)`` is not strictly unique.

    The input must already be sorted ascending by ``(block_number, log_index)`` (every
    caller sorts first), so it suffices to compare each row against its predecessor.
    Same ``block`` + same ``log`` in a row of an adjacent equal block pair is a
    duplicate key. Raises with the offending keys listed — never a silent
    ``drop_duplicates`` (a duplicate means a fetcher paginated wrong, and swallowing
    it hides a real bug).
    """
    offending: list[tuple[int, int]] = []
    for i in range(1, len(block_numbers)):
        if block_numbers[i] == block_numbers[i - 1] and log_indices[i] == log_indices[i - 1]:
            key = (int(block_numbers[i]), int(log_indices[i]))
            if key not in offending:
                offending.append(key)
    if offending:
        example = offending[0]
        raise ValidationError(
            f"{where}: duplicate merge keys in the event tape; a duplicate means a "
            "fetcher paginated wrong and must not be silently de-duplicated. "
            f"{len(offending)} distinct offending key(s), the first is "
            f"(block_number={example[0]}, log_index={example[1]}): "
            f"{offending[:20]}"
        )


def _build_base_tape(streams: Mapping[str, pa.Table]) -> pa.Table:
    """Union the four log streams, sort by ``(block_number, log_index)``, assert the key
    is unique, and stamp the dense ``seq``. Raises ``ValidationError`` listing the
    offending keys on duplicates — never a silent ``drop_duplicates``."""
    parts: list[pa.Table] = []
    for event_type in _LOG_STREAMS:
        stream_schema = SCHEMA_REGISTRY[event_type.value]
        table = streams.get(event_type.value)
        if table is None:
            table = empty_table(stream_schema)
            LOGGER.info(
                "align: stream %r absent from the inputs; proceeding without it "
                "(the tape will simply contain no such rows)",
                event_type.value,
            )
        validate_table(table, stream_schema, strict=True)
        parts.append(_cast_stream_to_base(table, event_type.value))

    base = (
        pa.concat_tables(parts)
        .sort_by([("block_number", "ascending"), ("log_index", "ascending")])
        .select(list(_BASE_TAPE_COLUMNS))  # key columns + payload, in §4.8 order
    )

    _assert_unique_keys(
        base.column("block_number").to_pylist(),
        base.column("log_index").to_pylist(),
        "build_event_tape: union of the four log streams",
    )
    base = base.append_column(
        "seq", pa.array(range(base.num_rows), type=pa.int64())
    )
    return base


# ---------------------------------------------------------------------------
# Derived columns.
# ---------------------------------------------------------------------------


def _price_from_sqrt(sqrt_price_x96: str | None, pool: PoolConfig) -> float | None:
    """``float`` convenience price (USDC per WETH) from a swap row's Q64.96 sqrt price.

    The parsing is exact (decimal string -> int -> Decimal via T02); float64 is only
    applied for the *output* column, which the contract declares ``float64`` — never
    for the accumulator math itself.
    """
    if sqrt_price_x96 is None:
        return None
    return float(
        sqrt_price_x96_to_price(int(sqrt_price_x96), pool.token0_decimals, pool.token1_decimals)
    )


def _join_fee_growth_globals(
    frame: pl.DataFrame,
    fee_growth: pa.Table,
) -> pl.DataFrame:
    """As-of **backward** join of the tape's ``block_number`` onto the fee-growth
    snapshots' ``block_number``, taking each snapshot's denormalized global
    accumulators (``CONTRACTS.md`` §4.8: every ``FEE_GROWTH_SCHEMA`` row carries
    ``fee_growth_global_{0,1}_x128``).

    The direction is backward on principle: a tape row at block *n* must carry the
    state *as of* block *n*, from the nearest snapshot at ``<= n`` — never from a
    later snapshot (look-ahead). ``fee_growth_source`` is ``"exact"`` when a
    snapshot exists at this exact block and ``"stale_prior"`` when the nearest
    prior snapshot is older; a row with no prior snapshot at all keeps nulls.
    """
    if fee_growth.num_rows == 0:
        return frame.with_columns(
            pl.lit(None).cast(pl.String).alias("fee_growth_global_0_x128"),
            pl.lit(None).cast(pl.String).alias("fee_growth_global_1_x128"),
            pl.lit(None).cast(pl.String).alias("fee_growth_source"),
            pl.lit(None).cast(pl.Int64).alias(f"{_SRC_BLK_PREFIX}fee_growth_global_0_x128"),
        )

    # The global accumulators are denormalized on every row, so the per-block
    # distinct set is one row per snapshot block.
    snapshots = (
        pl.from_arrow(
            fee_growth.select(
                ["block_number", "fee_growth_global_0_x128", "fee_growth_global_1_x128"]
            )
        )
        .unique()
        .rename({"block_number": "_snap_block"})
        .sort("_snap_block")
    )
    joined = frame.sort("block_number").join_asof(
        snapshots,
        left_on="block_number",
        right_on="_snap_block",
        strategy="backward",  # only snapshots with block <= the tape row's block
    )
    return joined.with_columns(
        pl.col("_snap_block").alias(f"{_SRC_BLK_PREFIX}fee_growth_global_0_x128"),
        pl.when(pl.col("_snap_block").is_null())
        .then(pl.lit(None).cast(pl.String))
        .when(pl.col("_snap_block") == pl.col("block_number"))
        .then(pl.lit(_FG_EXACT))
        .otherwise(pl.lit(_FG_STALE))
        .alias("fee_growth_source"),
    ).drop("_snap_block")


# ---------------------------------------------------------------------------
# The public transform (CONTRACTS.md §6.2).
# ---------------------------------------------------------------------------


def build_event_tape(
    streams: Mapping[str, pa.Table],
    *,
    pool: PoolConfig,
    gas: pa.Table,
    reference: pa.Table,
    regime: pa.Table,
    fee_growth: pa.Table,
) -> pa.Table:
    """Build the block-ordered, look-ahead-free event tape (``EVENT_TAPE_SCHEMA``).

    ``streams`` holds the four log streams keyed ``"swap" | "mint" | "burn" | "collect"``
    (a missing key is tolerated as an empty stream — the tape simply contains no such
    rows; the handoff notes must say which stream was absent). ``gas``, ``reference``,
    ``regime`` and ``fee_growth`` are the non-log streams, each validated against its
    canonical schema before use.

    Join directions, every one a comment in the code too:

    - ``gas`` — **exact** equi-join on ``block_number`` (§10.2.3 needs *that block's*
      gas price; T07 guarantees no gaps, so a tape block missing from ``gas`` raises).
    - ``reference``/``regime`` — as-of **backward** on ``close_time <=
      block_timestamp`` / ``timestamp <= block_timestamp`` (``polars.join_asof(
      strategy="backward")``). A ``"nearest"`` or ``"forward"`` strategy is look-ahead.
    - ``fee_growth`` globals — as-of **backward** on ``block_number``, forward-filled
      from the nearest *prior* snapshot with ``fee_growth_source`` marking it.

    Rows before the first reference bar / regime label keep null ``price_reference``
    and ``regime = "unknown"`` — they are retained, never back-filled. The final tape
    passes ``assert_no_lookahead`` inside this function before the private source
    columns are dropped.
    """
    for name in ("swap", "mint", "burn", "collect"):
        if name in streams:
            validate_table(streams[name], SCHEMA_REGISTRY[name], strict=True)
    validate_table(gas, GAS_SCHEMA, strict=True)
    validate_table(reference, SCHEMA_REGISTRY["reference"], strict=True)
    validate_table(regime, REGIME_SCHEMA, strict=True)

    base = _build_base_tape(streams)

    if base.num_rows == 0:
        # Nothing to join; still return a conforming (empty) tape. The empty
        # ``EVENT_TAPE_SCHEMA`` table (all 27 columns declared, zero rows) is
        # the honest answer — ``_build_base_tape``'s 18-column base cannot
        # satisfy the full schema.
        return empty_table(EVENT_TAPE_SCHEMA)

    # --- price_pool: derived from the row's OWN sqrt_price_x96 (swap rows only) ---
    sqrt_values = base.column("sqrt_price_x96").to_pylist()
    price_pool = [_price_from_sqrt(s, pool) for s in sqrt_values]
    base = base.append_column("price_pool", pa.array(price_pool, type=pa.float64()))

    frame: pl.DataFrame = pl.from_arrow(base)

    # --- gas: EXACT equi-join on block_number (that block's gas price, §10.2.3) ---
    gas_join = pl.from_arrow(
        gas.select(["block_number", "base_fee_per_gas", "priority_fee_p50_wei"])
    )
    frame = frame.join(gas_join, on="block_number", how="left")
    missing = (
        frame.filter(
            pl.col("base_fee_per_gas").is_null() | pl.col("priority_fee_p50_wei").is_null()
        )
        .select("block_number")
        .unique()
        .to_series()
        .to_list()
    )
    if missing:
        raise ValidationError(
            "build_event_tape: tape rows reference block(s) with no row in the gas "
            "stream; gas is chain-wide one-row-per-block with no gaps, so a missing "
            "block means the gas pull was truncated. Missing block(s) (first 20): "
            f"{missing[:20]}"
        )

    # --- price_reference: as-of BACKWARD on close_time <= block_timestamp ---
    ref_join = pl.from_arrow(reference.select(["close_time", "close"])).sort("close_time")
    if ref_join.height > 0:
        frame = (
            frame.sort("block_timestamp")
            .join_asof(
                ref_join,
                left_on="block_timestamp",
                right_on="close_time",
                strategy="backward",  # last bar with close_time <= block_timestamp
            )
            .rename({"close": "price_reference", "close_time": f"{_SRC_TS_PREFIX}price_reference"})
        )
    else:
        frame = frame.with_columns(
            pl.lit(None).cast(pl.Float64).alias("price_reference"),
            pl.lit(None)
            .cast(pl.Datetime("us", time_zone="UTC"))
            .alias(f"{_SRC_TS_PREFIX}price_reference"),
        )

    # --- regime: as-of BACKWARD on timestamp <= block_timestamp; pre-history = unknown ---
    regime_join = pl.from_arrow(regime.select(["timestamp", "regime"])).sort("timestamp")
    if regime_join.height > 0:
        frame = (
            frame.sort("block_timestamp")
            .join_asof(
                regime_join,
                left_on="block_timestamp",
                right_on="timestamp",
                strategy="backward",  # last label with timestamp <= block_timestamp
            )
            .rename({"timestamp": f"{_SRC_TS_PREFIX}regime"})
            # Never back-filled: before the first label, null -> the 'unknown' sentinel
            # that CONTRACTS.md §4.7/§1 define for exactly this warmup case.
            .with_columns(pl.col("regime").fill_null(Regime.UNKNOWN.value))
        )
    else:
        frame = frame.with_columns(
            pl.lit(Regime.UNKNOWN.value).alias("regime"),
            pl.lit(None)
            .cast(pl.Datetime("us", time_zone="UTC"))
            .alias(f"{_SRC_TS_PREFIX}regime"),
        )

    # --- fee growth globals: as-of BACKWARD on block_number, forward-filled ---
    frame = _join_fee_growth_globals(frame, fee_growth)

    # --- land the tape -----------------------------------------------------
    # The as-of joins above re-sorted the frame (each ``join_asof`` requires the
    # left side sorted by its join key), so the canonical ``(block_number,
    # log_index)`` order is NOT guaranteed any more. Restore it and re-stamp a
    # dense ``seq`` so the RL step index is the row's position in block order.
    frame = frame.sort(["block_number", "log_index"])
    frame = frame.with_columns(pl.int_range(0, pl.len()).alias("seq"))

    tape = frame.to_arrow()
    # Re-assert merge-key uniqueness on the FULLY assembled tape, not just the base
    # four-stream union: a schema-valid ``gas`` table could carry a duplicate
    # ``block_number`` (Arrow schemas express no uniqueness), which the ``left``
    # equi-join would silently fan out into two rows per block with no error here.
    _assert_unique_keys(
        tape.column("block_number").to_pylist(),
        tape.column("log_index").to_pylist(),
        "build_event_tape: assembled tape",
    )
    assert_no_lookahead(tape)  # the §10.2.3 rule, enforced here and not left to callers
    tape = tape.drop_columns(
        [c for c in tape.column_names if c.startswith(("_src_ts_", "_src_blk_"))]
    )
    tape = tape.select([f.name for f in EVENT_TAPE_SCHEMA]).cast(EVENT_TAPE_SCHEMA)
    validate_table(tape, EVENT_TAPE_SCHEMA, strict=True)
    return tape


def attach_reference_prices(gas: pa.Table, reference: pa.Table) -> pa.Table:
    """Populate ``gas.eth_usd_price`` by a backward as-of join on
    ``close_time <= block_timestamp`` (the GAS_SCHEMA note: T07 writes that column
    null; T11 fills it), so a gas cost can be expressed in USD at the block's
    prevailing price. Rows before the first reference bar keep null. Returns a
    schema-validated gas table; the input is not mutated.
    """
    validate_table(gas, GAS_SCHEMA, strict=True)
    validate_table(reference, SCHEMA_REGISTRY["reference"], strict=True)
    frame = pl.from_arrow(gas).sort("block_timestamp")
    ref_join = pl.from_arrow(reference.select(["close_time", "close"])).sort("close_time")
    if ref_join.height > 0:
        joined = frame.join_asof(
            ref_join,
            left_on="block_timestamp",
            right_on="close_time",
            strategy="backward",  # last bar with close_time <= the block's timestamp
        )
    else:
        joined = frame.with_columns(pl.lit(None).cast(pl.Float64).alias("close"))
    # coalesce: the matched bar's close wins; rows without one keep the previous value
    # (which is null for freshly-fetched gas, but enrichment is idempotent).
    enriched = joined.with_columns(
        pl.coalesce("close", "eth_usd_price").alias("eth_usd_price")
    ).drop("close")
    out = enriched.to_arrow()
    out = out.select([f.name for f in GAS_SCHEMA]).cast(GAS_SCHEMA)
    validate_table(out, GAS_SCHEMA, strict=True)
    return out


# ---------------------------------------------------------------------------
# The mechanical no-look-ahead guard (CONTRACTS.md §6.2, §4).
# ---------------------------------------------------------------------------


def assert_no_lookahead(table: pa.Table, time_col: str = "block_timestamp") -> None:
    """Raise ``ValidationError`` if any as-of-joined column could only have been known
    at a later time than the row's own ``time_col``.

    The as-of joins inside ``build_event_tape`` carry each join's source time through
    as a private ``_src_ts_<col>`` (timestamp) or ``_src_blk_<col>`` (block number)
    column — e.g. ``_src_ts_price_reference`` is the matched bar's ``close_time``.
    This function verifies ``_src_ts_<col> <= time_col`` and ``_src_blk_<col> <=
    block_number`` on every row where the joined value is non-null, then returns
    (the caller drops the private columns after the check). A table built with a
    ``"forward"`` or ``"nearest"`` as-of strategy trips it immediately.

    Rows whose joined value is null (no prior bar / label / snapshot) are skipped —
    there is nothing they could have peeked at.
    """
    if time_col not in table.column_names:
        raise ValidationError(
            f"assert_no_lookahead: time column {time_col!r} not found in the table "
            f"(have: {table.column_names})"
        )
    if "block_number" not in table.column_names:
        raise ValidationError(
            "assert_no_lookahead: 'block_number' column not found; the block-ordering "
            "guard is meaningless without it"
        )

    if time_col not in table.column_names:
        raise ValidationError(
            f"assert_no_lookahead: time column {time_col!r} not found in the table "
            f"(have: {table.column_names})"
        )
    if "block_number" not in table.column_names:
        raise ValidationError(
            "assert_no_lookahead: 'block_number' column not found; the block-ordering "
            "guard is meaningless without it"
        )

    # Timestamp-sourced as-of columns. ``as_py()`` yields ``datetime`` / ``None``
    # (NaT); None (no match) compares False against every datetime, so null rows are
    # skipped — exactly what a backward-join check wants.
    time_values = table.column(time_col).to_pylist()
    for name in table.column_names:
        if not name.startswith(_SRC_TS_PREFIX):
            continue
        derived = name[len(_SRC_TS_PREFIX) :]
        if derived not in table.column_names:
            raise ValidationError(
                f"assert_no_lookahead: source column {name!r} has no matching derived "
                f"column {derived!r}"
            )
        src = table.column(name).to_pylist()
        for row, (s, t) in enumerate(zip(src, time_values, strict=True)):
            if s is not None and s > t:
                raise ValidationError(
                    f"assert_no_lookahead: column {derived!r} is look-ahead — the value "
                    f"at row {row} was joined from source time {s} which is later than "
                    f"the row's {time_col} {t}. Every as-of join must be backward-only."
                )

    # Block-number-sourced as-of columns (fee-growth globals). ``None``/``NaN`` (no
    # prior snapshot) compares False against every block, so they are skipped.
    tape_blocks = table.column("block_number").to_pylist()
    for name in table.column_names:
        if not name.startswith(_SRC_BLK_PREFIX):
            continue
        derived = name[len(_SRC_BLK_PREFIX) :]
        if derived not in table.column_names:
            raise ValidationError(
                f"assert_no_lookahead: source column {name!r} has no matching derived "
                f"column {derived!r}"
            )
        src = table.column(name).to_pylist()
        for row, (s, b) in enumerate(zip(src, tape_blocks, strict=True)):
            if s is not None and s > b:
                raise ValidationError(
                    f"assert_no_lookahead: column {derived!r} is look-ahead — the value "
                    f"at row {row} was carried from snapshot block {s} which is later "
                    f"than the row's block_number {b}. Fee-growth globals may only be "
                    "forward-filled from a PRIOR snapshot."
                )


__all__ = [
    "Dataset",
    "build_event_tape",
    "attach_reference_prices",
    "assert_no_lookahead",
]