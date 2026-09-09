"""Canonical pyarrow schemas for every ``undertow.data`` stream (T03).

This module is the data contract (``CONTRACTS.md`` §4): the single place where column
names, types, units, sign conventions and nullability are declared for all ten streams.
Seven other tasks (T05–T09, T11, T12) produce or consume tables that must conform to
these schemas, so a mismatch is a loud ``SchemaViolationError`` at the boundary rather
than a silent wrong number three tasks later.

Design rules (rationale in ``docs/schema_notes.md``):

- **Big integers are decimal strings.** ``uint256``/``int256``/``uint128``/``uint160``
  values do not fit in any Arrow integer type and float64 loses precision above 2**53,
  so they are stored as ``pa.string()`` holding a decimal representation and handled in
  Python as ``int``. ``encode_uint`` / ``decode_uint`` are the only sanctioned
  conversions.
- **Column order is part of the schema.** The six key columns come first, in the
  ``CONTRACTS.md`` §4.0.1 order, in every log stream. ``validate_table(strict=True)``
  fails on wrong order because T09's deterministic-bytes requirement depends on stable
  column order.
- **Nullability is declared, not incidental.** An absent value is null, never ``""`` and
  never a zero; there are no sentinel values anywhere in this contract.
- **Metadata carries the data dictionary.** Every schema carries ``schema_version``, the
  stream name, and a per-column unit/convention string for EVERY field — T16's
  ``docs/data_dictionary.md`` is generated from this metadata.

This module contains schemas and codecs only: no fetcher, no transform, no I/O.
"""

from __future__ import annotations

import re
from typing import Final

import pyarrow as pa

from undertow.data.types import EventType, SchemaViolationError

SCHEMA_VERSION: Final[str] = "1.0.0"
"""Semver of the canonical schemas. Bumped on any change; written into the manifest."""

# ---------------------------------------------------------------------------
# Field builders
# ---------------------------------------------------------------------------


def _key_fields() -> list[pa.Field]:
    """The six mandatory key columns, in CONTRACTS.md §4.0.1 order (log streams only)."""
    return [
        pa.field("block_number", pa.int64(), nullable=False),
        pa.field("log_index", pa.int32(), nullable=False),
        pa.field("block_timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("tx_hash", pa.string(), nullable=False),
        pa.field("pool_address", pa.string(), nullable=False),
        pa.field("event_type", pa.string(), nullable=False),
    ]


def _build_schema(stream: str, fields: list[pa.Field], units: dict[str, str]) -> pa.Schema:
    """Build a canonical schema with the standard metadata envelope.

    ``units`` must cover exactly the fields: T16's data dictionary is generated from this
    metadata, so a missing unit string is a build-time error, not a docs gap.
    """
    names = [f.name for f in fields]
    if set(units) != set(names):
        raise ValueError(
            f"schema {stream!r}: unit metadata must cover exactly the fields; "
            f"missing={sorted(set(names) - set(units))}, "
            f"extra={sorted(set(units) - set(names))}"
        )
    metadata: dict[str, str] = {"schema_version": SCHEMA_VERSION, "stream": stream}
    for name in names:
        metadata[f"unit:{name}"] = units[name]
    return pa.schema(fields, metadata=metadata)


# ---------------------------------------------------------------------------
# Unit / convention strings — one per column, every column (T16's data dictionary).
# ---------------------------------------------------------------------------

_KEY_UNITS: dict[str, str] = {
    "block_number": "block height (int64)",
    "log_index": "index of the log within the block (int32)",
    "block_timestamp": "block time, UTC (timestamp[us])",
    "tx_hash": "0x-prefixed lowercase transaction hash",
    "pool_address": "lowercase 0x-prefixed pool address",
    "event_type": "stream discriminator: swap|mint|burn|collect|flash",
}

_SWAP_UNITS: dict[str, str] = {
    **_KEY_UNITS,
    "amount0": "raw token0 units, signed int256, positive=flowed INTO the pool",
    "amount1": "raw token1 units, signed int256, positive=flowed INTO the pool",
    "sqrt_price_x96": "uint160 pool sqrt price after the swap, Q64.96",
    "liquidity": "uint128 active liquidity after the swap; RPC-only (null on The Graph route)",
    "tick": "pool tick after the swap (int32)",
    "sender": "lowercase 0x-prefixed address",
    "recipient": "lowercase 0x-prefixed address",
}

_MINT_BURN_UNITS: dict[str, str] = {
    **_KEY_UNITS,
    "owner": "lowercase 0x-prefixed address (NFT position manager for retail positions)",
    "tick_lower": "lower tick, on the spacing grid (int32)",
    "tick_upper": "upper tick, on the spacing grid, > tick_lower (int32)",
    "liquidity_amount": "uint128 event amount — the position's L",
    "amount0": "raw token0 units, unsigned uint256",
    "amount1": "raw token1 units, unsigned uint256",
    "sender": "lowercase 0x-prefixed address; null on burn rows (Burn has no sender)",
}

_COLLECT_UNITS: dict[str, str] = {
    **_KEY_UNITS,
    "owner": "lowercase 0x-prefixed address",
    "recipient": "lowercase 0x-prefixed address; RPC-only (null on The Graph route)",
    "tick_lower": "lower tick, on the spacing grid (int32)",
    "tick_upper": "upper tick, on the spacing grid, > tick_lower (int32)",
    "amount0": "raw token0 units withdrawn, unsigned uint128",
    "amount1": "raw token1 units withdrawn, unsigned uint128",
}

_FLASH_UNITS: dict[str, str] = {
    **_KEY_UNITS,
    "sender": "lowercase 0x-prefixed address",
    "recipient": "lowercase 0x-prefixed address",
    "amount0": "raw token0 units borrowed, unsigned uint256",
    "amount1": "raw token1 units borrowed, unsigned uint256",
    "paid0": "raw token0 units repaid, unsigned uint256 (exceeds amount0 by the fee)",
    "paid1": "raw token1 units repaid, unsigned uint256 (exceeds amount1 by the fee)",
}

_FEE_GROWTH_UNITS: dict[str, str] = {
    "block_number": "block the eth_call was made at; state after the block (int64)",
    "tick": "tick whose Tick struct was read; GLOBAL_TICK_SENTINEL marks global-only rows (int32)",
    "fee_growth_outside_0_x128": "uint256 Q128; null on global rows",
    "fee_growth_outside_1_x128": "uint256 Q128; null on global rows",
    "liquidity_gross": "uint128 gross liquidity at the tick",
    "liquidity_net": "signed int128 net liquidity at the tick",
    "initialized": "tick struct initialized flag (bool)",
    "fee_growth_global_0_x128": "uint256 Q128; present on every row (denormalized)",
    "fee_growth_global_1_x128": "uint256 Q128; present on every row",
    "current_tick": "pool slot0.tick at that block (int32)",
    "current_liquidity": "pool liquidity() at that block, uint128",
    "source": "'rpc_call' (exact) or 'interpolated' (flagged, never silently)",
    "pool_address": "lowercase 0x-prefixed pool address",
    "fee_protocol": "packed uint8 from slot0: bits 0-3 = fp0, bits 4-7 = fp1; 0 when off",
}

_GAS_UNITS: dict[str, str] = {
    "block_number": "block height; one row per block, no gaps (int64)",
    "block_timestamp": "block time, UTC (timestamp[us])",
    "base_fee_per_gas": "wei, EIP-1559 base fee (uint256)",
    "gas_used": "block gas used (int64)",
    "gas_limit": "block gas limit (int64)",
    "priority_fee_p50_wei": "median effective priority fee of txs in the block (uint256)",
    "priority_fee_p90_wei": "90th percentile effective priority fee — the spike cost (uint256)",
    "eth_usd_price": "USD price of ETH at the block's prevailing price; null as written by "
    "T07, populated by T11 via a backward as-of join (float64)",
}

_REFERENCE_UNITS: dict[str, str] = {
    "open_time": "kline open, left-closed interval, UTC (timestamp[us])",
    "close_time": "kline close, UTC (timestamp[us])",
    "open": "open price, ETH in USD-stable terms (float64)",
    "high": "high price (float64)",
    "low": "low price (float64)",
    "close": "close price (float64)",
    "volume_base": "ETH volume (float64)",
    "quote_volume": "quote volume (float64)",
    "trades": "number of trades (int64)",
    "symbol": "'ETHUSDT' / 'ETHUSDC'",
    "is_gap_filled": "true if the bar was forward-filled over an exchange outage (bool)",
}

_REGIME_UNITS: dict[str, str] = {
    "timestamp": "end of the backward-looking window; the label applies at this time, UTC",
    "sigma_rv": "annualized realized vol over the lookback (float64)",
    "mu": "total log return over the lookback: log(S_end/S_start) (float64)",
    "regime": "label: bull|bear|sideways|high_vol|unknown",
    "window_complete": "false => regime == 'unknown' (bool)",
    "symbol": "reference symbol the labels were computed from",
}

_TAPE_PAYLOAD_UNITS: dict[str, str] = {
    "amount0": "signed on swap rows, non-negative on mint/burn/collect rows; null where inapplicable",
    "amount1": "signed on swap rows, non-negative on mint/burn/collect rows; null where inapplicable",
    "sqrt_price_x96": "uint160 pool sqrt price, Q64.96; swap rows only",
    "liquidity": "uint128 active liquidity; swap rows only",
    "tick": "pool tick (int32); swap rows only",
    "sender": "lowercase 0x-prefixed address; swap/mint rows only",
    "recipient": "lowercase 0x-prefixed address; swap/collect rows only",
    "owner": "lowercase 0x-prefixed address; mint/burn/collect rows only",
    "tick_lower": "lower tick on the spacing grid (int32); mint/burn/collect rows only",
    "tick_upper": "upper tick on the spacing grid (int32); mint/burn/collect rows only",
    "liquidity_amount": "uint128 position liquidity; mint/burn rows only",
}

_TAPE_DERIVED_UNITS: dict[str, str] = {
    "seq": "0..N-1 dense canonical step index (int64)",
    "price_pool": "pool price derived from sqrt_price_x96, decimals-adjusted (float64)",
    "price_reference": "as-of join: last reference close with close_time <= block_timestamp (float64)",
    "regime": "as-of join from the regime table (bull|bear|sideways|high_vol|unknown)",
    "base_fee_per_gas": "joined from gas on block_number, wei (uint256)",
    "priority_fee_p50_wei": "joined from gas on block_number, wei (uint256)",
    "fee_growth_global_0_x128": "forward-filled from nearest prior snapshot, Q128 (uint256)",
    "fee_growth_global_1_x128": "forward-filled from nearest prior snapshot, Q128 (uint256)",
    "fee_growth_source": "'exact' | 'stale_prior' — never 'interpolated_future'",
}

# ---------------------------------------------------------------------------
# The ten canonical schemas (CONTRACTS.md §4.1–§4.8).
# ---------------------------------------------------------------------------

SWAP_SCHEMA: Final[pa.Schema] = _build_schema(
    "swap",
    _key_fields()
    + [
        pa.field("amount0", pa.string(), nullable=False),
        pa.field("amount1", pa.string(), nullable=False),
        pa.field("sqrt_price_x96", pa.string(), nullable=False),
        pa.field("liquidity", pa.string(), nullable=True),
        pa.field("tick", pa.int32(), nullable=False),
        pa.field("sender", pa.string(), nullable=False),
        pa.field("recipient", pa.string(), nullable=False),
    ],
    _SWAP_UNITS,
)

MINT_SCHEMA: Final[pa.Schema] = _build_schema(
    "mint",
    _key_fields()
    + [
        pa.field("owner", pa.string(), nullable=False),
        pa.field("tick_lower", pa.int32(), nullable=False),
        pa.field("tick_upper", pa.int32(), nullable=False),
        pa.field("liquidity_amount", pa.string(), nullable=False),
        pa.field("amount0", pa.string(), nullable=False),
        pa.field("amount1", pa.string(), nullable=False),
        pa.field("sender", pa.string(), nullable=True),
    ],
    _MINT_BURN_UNITS,
)

BURN_SCHEMA: Final[pa.Schema] = _build_schema(
    "burn",
    _key_fields()
    + [
        pa.field("owner", pa.string(), nullable=False),
        pa.field("tick_lower", pa.int32(), nullable=False),
        pa.field("tick_upper", pa.int32(), nullable=False),
        pa.field("liquidity_amount", pa.string(), nullable=False),
        pa.field("amount0", pa.string(), nullable=False),
        pa.field("amount1", pa.string(), nullable=False),
        pa.field("sender", pa.string(), nullable=True),
    ],
    _MINT_BURN_UNITS,
)

COLLECT_SCHEMA: Final[pa.Schema] = _build_schema(
    "collect",
    _key_fields()
    + [
        pa.field("owner", pa.string(), nullable=False),
        pa.field("recipient", pa.string(), nullable=True),
        pa.field("tick_lower", pa.int32(), nullable=False),
        pa.field("tick_upper", pa.int32(), nullable=False),
        pa.field("amount0", pa.string(), nullable=False),
        pa.field("amount1", pa.string(), nullable=False),
    ],
    _COLLECT_UNITS,
)

FLASH_SCHEMA: Final[pa.Schema] = _build_schema(
    "flash",
    _key_fields()
    + [
        pa.field("sender", pa.string(), nullable=False),
        pa.field("recipient", pa.string(), nullable=False),
        pa.field("amount0", pa.string(), nullable=False),
        pa.field("amount1", pa.string(), nullable=False),
        pa.field("paid0", pa.string(), nullable=False),
        pa.field("paid1", pa.string(), nullable=False),
    ],
    _FLASH_UNITS,
)

FEE_GROWTH_SCHEMA: Final[pa.Schema] = _build_schema(
    "fee_growth",
    [
        pa.field("block_number", pa.int64(), nullable=False),
        pa.field("tick", pa.int32(), nullable=False),
        pa.field("fee_growth_outside_0_x128", pa.string(), nullable=True),
        pa.field("fee_growth_outside_1_x128", pa.string(), nullable=True),
        pa.field("liquidity_gross", pa.string(), nullable=False),
        pa.field("liquidity_net", pa.string(), nullable=False),
        pa.field("initialized", pa.bool_(), nullable=False),
        pa.field("fee_growth_global_0_x128", pa.string(), nullable=False),
        pa.field("fee_growth_global_1_x128", pa.string(), nullable=False),
        pa.field("current_tick", pa.int32(), nullable=False),
        pa.field("current_liquidity", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("pool_address", pa.string(), nullable=False),
        pa.field("fee_protocol", pa.int32(), nullable=False),
    ],
    _FEE_GROWTH_UNITS,
)

GAS_SCHEMA: Final[pa.Schema] = _build_schema(
    "gas",
    [
        pa.field("block_number", pa.int64(), nullable=False),
        pa.field("block_timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("base_fee_per_gas", pa.string(), nullable=False),
        pa.field("gas_used", pa.int64(), nullable=False),
        pa.field("gas_limit", pa.int64(), nullable=False),
        pa.field("priority_fee_p50_wei", pa.string(), nullable=False),
        pa.field("priority_fee_p90_wei", pa.string(), nullable=False),
        pa.field("eth_usd_price", pa.float64(), nullable=True),
    ],
    _GAS_UNITS,
)

REFERENCE_SCHEMA: Final[pa.Schema] = _build_schema(
    "reference",
    [
        pa.field("open_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("close_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("open", pa.float64(), nullable=False),
        pa.field("high", pa.float64(), nullable=False),
        pa.field("low", pa.float64(), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("volume_base", pa.float64(), nullable=False),
        pa.field("quote_volume", pa.float64(), nullable=False),
        pa.field("trades", pa.int64(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("is_gap_filled", pa.bool_(), nullable=False),
    ],
    _REFERENCE_UNITS,
)

REGIME_SCHEMA: Final[pa.Schema] = _build_schema(
    "regime",
    [
        pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("sigma_rv", pa.float64(), nullable=False),
        pa.field("mu", pa.float64(), nullable=False),
        pa.field("regime", pa.string(), nullable=False),
        pa.field("window_complete", pa.bool_(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
    ],
    _REGIME_UNITS,
)

EVENT_TAPE_SCHEMA: Final[pa.Schema] = _build_schema(
    "event_tape",
    _key_fields()
    + [
        # Union of log-stream payload columns, in CONTRACTS.md §4.8 order — null where
        # inapplicable to the row's event_type.
        pa.field("amount0", pa.string(), nullable=True),
        pa.field("amount1", pa.string(), nullable=True),
        pa.field("sqrt_price_x96", pa.string(), nullable=True),
        pa.field("liquidity", pa.string(), nullable=True),
        pa.field("tick", pa.int32(), nullable=True),
        pa.field("sender", pa.string(), nullable=True),
        pa.field("recipient", pa.string(), nullable=True),
        pa.field("owner", pa.string(), nullable=True),
        pa.field("tick_lower", pa.int32(), nullable=True),
        pa.field("tick_upper", pa.int32(), nullable=True),
        pa.field("liquidity_amount", pa.string(), nullable=True),
        # Derived columns, in CONTRACTS.md §4.8 order.
        pa.field("seq", pa.int64(), nullable=False),
        pa.field("price_pool", pa.float64(), nullable=True),
        pa.field("price_reference", pa.float64(), nullable=True),
        pa.field("regime", pa.string(), nullable=True),
        pa.field("base_fee_per_gas", pa.string(), nullable=True),
        pa.field("priority_fee_p50_wei", pa.string(), nullable=True),
        pa.field("fee_growth_global_0_x128", pa.string(), nullable=True),
        pa.field("fee_growth_global_1_x128", pa.string(), nullable=True),
        pa.field("fee_growth_source", pa.string(), nullable=True),
    ],
    {**_KEY_UNITS, **_TAPE_PAYLOAD_UNITS, **_TAPE_DERIVED_UNITS},
)

# ---------------------------------------------------------------------------
# Registry, sort keys, pool scoping (CONTRACTS.md §4.0, §4.9).
# ---------------------------------------------------------------------------

SCHEMA_REGISTRY: Final[dict[str, pa.Schema]] = {
    EventType.SWAP.value: SWAP_SCHEMA,
    EventType.MINT.value: MINT_SCHEMA,
    EventType.BURN.value: BURN_SCHEMA,
    EventType.COLLECT.value: COLLECT_SCHEMA,
    EventType.FLASH.value: FLASH_SCHEMA,
    "fee_growth": FEE_GROWTH_SCHEMA,
    "gas": GAS_SCHEMA,
    "reference": REFERENCE_SCHEMA,
    "regime": REGIME_SCHEMA,
    "event_tape": EVENT_TAPE_SCHEMA,
}

SORT_KEYS: Final[dict[str, tuple[str, ...]]] = {
    EventType.SWAP.value: ("block_number", "log_index"),
    EventType.MINT.value: ("block_number", "log_index"),
    EventType.BURN.value: ("block_number", "log_index"),
    EventType.COLLECT.value: ("block_number", "log_index"),
    EventType.FLASH.value: ("block_number", "log_index"),
    "fee_growth": ("block_number", "tick"),
    "gas": ("block_number",),
    "reference": ("symbol", "open_time"),
    "regime": ("timestamp",),
    "event_tape": ("block_number", "log_index"),
}

POOL_SCOPED: Final[frozenset[str]] = frozenset(
    {
        EventType.SWAP.value,
        EventType.MINT.value,
        EventType.BURN.value,
        EventType.COLLECT.value,
        EventType.FLASH.value,
        "fee_growth",
        "event_tape",
    }
)
"""Streams partitioned by ``pool_address`` (CONTRACTS.md §4.0).

``gas`` (chain-wide), ``reference`` (exchange feed) and ``regime`` (derived from
``reference``) are global and must not be duplicated under each pool directory.
"""

# ---------------------------------------------------------------------------
# Big-integer codecs (CONTRACTS.md §0, §4.9).
# ---------------------------------------------------------------------------

_DECIMAL_RE: Final[re.Pattern[str]] = re.compile(r"-?[0-9]+")


def encode_uint(value: int) -> str:
    """Encode an arbitrary-precision ``int`` as a decimal string for a Parquet string column.

    The only sanctioned int->string conversion in the pipeline. Rejects anything that is
    not an ``int`` — including ``bool`` and ``float`` — because a float sneaking in here
    is how Q128 precision dies (``2**128 ~= 3.4e38`` carries ~70 bits more than a float64
    mantissa).
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaViolationError(
            f"encode_uint: expected an int, got {type(value).__name__} {value!r}; "
            "big integers must never pass through float"
        )
    return str(value)


def decode_uint(value: str) -> int:
    """Decode a decimal string back to an ``int``.

    Handles a leading ``-`` (signed ``int256`` values keep it). Rejects anything that is
    not a plain decimal string — ``"1e5"``, ``"0x10"``, ``"+5"``, whitespace — because
    the pipeline stores canonical decimal representations only.
    """
    if not isinstance(value, str):
        raise SchemaViolationError(
            f"decode_uint: expected a decimal string, got {type(value).__name__} {value!r}"
        )
    if _DECIMAL_RE.fullmatch(value) is None:
        raise SchemaViolationError(
            f"decode_uint: {value!r} is not a decimal string; only plain decimal "
            "representations are accepted (no exponents, no hex, no sign other than a "
            "leading '-')"
        )
    return int(value)


# ---------------------------------------------------------------------------
# Table validation (CONTRACTS.md §4.9).
# ---------------------------------------------------------------------------


def validate_table(table: pa.Table, schema: pa.Schema, *, strict: bool = True) -> None:
    """Validate ``table`` against the canonical ``schema``.

    Collects EVERY mismatch (missing, extra, wrong type, wrong position, wrong
    nullability) and raises a single ``SchemaViolationError`` listing them all.

    - ``strict=True``: the table's columns must be exactly the schema's columns, in
      exactly the same order.
    - ``strict=False``: extra columns are allowed (T11 appends derived columns), but
      every declared column must be present, correctly typed, in the schema's relative
      order, and non-nullable columns must not contain nulls.

    Nullability is checked in the direction that matters: a column declared non-nullable
    must not contain nulls. A column declared nullable may legitimately have no nulls
    (pyarrow infers ``nullable=False`` for a null-free column), so that direction is not
    a violation.
    """
    if not isinstance(table, pa.Table):
        raise SchemaViolationError(
            f"validate_table: expected a pyarrow Table, got {type(table).__name__}"
        )

    violations: list[str] = []
    table_names = list(table.column_names)
    schema_names = [f.name for f in schema]

    # Missing columns.
    for name in schema_names:
        if name not in table_names:
            violations.append(f"missing column: {name!r}")

    if strict:
        # Extra columns.
        for name in table_names:
            if name not in schema_names:
                violations.append(f"extra column: {name!r} (position {table_names.index(name)})")
        # Order: exact match. Report the first divergence only — a permutation would
        # otherwise produce one message per column.
        if table_names != schema_names:
            for pos, name in enumerate(table_names):
                if pos >= len(schema_names) or schema_names[pos] != name:
                    if name in schema_names:
                        violations.append(
                            f"wrong order: column {name!r} found at position {pos}, "
                            f"expected at position {schema_names.index(name)}"
                        )
                    break
    else:
        # Order: declared columns must appear in the schema's relative order.
        declared_in_table = [n for n in table_names if n in schema_names]
        if declared_in_table != schema_names:
            for pos, name in enumerate(schema_names):
                if pos >= len(declared_in_table) or declared_in_table[pos] != name:
                    violations.append(
                        f"wrong order: column {name!r} expected at position {pos} "
                        "among declared columns"
                    )
                    break

    # Types and nullability for every declared column that is present.
    for field in schema:
        name = field.name
        if name not in table_names:
            continue
        column = table.column(name)
        if column.type != field.type:
            violations.append(
                f"wrong type: column {name!r} expected {field.type}, got {column.type}"
            )
        if not field.nullable and column.null_count > 0:
            violations.append(
                f"nullability: column {name!r} is declared non-nullable but contains "
                f"{column.null_count} null value(s)"
            )

    if violations:
        stream = schema.metadata.get(b"stream", b"<unknown>").decode()
        version = schema.metadata.get(b"schema_version", SCHEMA_VERSION.encode()).decode()
        header = (
            f"table does not conform to schema {stream!r} (schema_version={version}); "
            f"{len(violations)} violation(s):"
        )
        body = "\n".join(f"  - {v}" for v in violations)
        raise SchemaViolationError(f"{header}\n{body}")


def empty_table(schema: pa.Schema) -> pa.Table:
    """A 0-row table conforming to ``schema`` — what every fetcher returns for an empty range.

    Built via ``pa.Schema.empty_table()`` so types, nullability and metadata are exactly
    right, and ``validate_table`` accepts the result.
    """
    return schema.empty_table()
