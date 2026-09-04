"""Tests for ``undertow.data.schemas`` (T03) — the canonical pyarrow schemas.

Covers everything ``CONTRACTS.md`` §4 and the T03 brief pin: the ten schemas, the
registry / sort-keys / pool-scoping consistency, the key-column rule, string-encoded
big integers, ``validate_table`` failure modes, ``empty_table``, timestamps and
``SCHEMA_VERSION``.

Note on the "set equality three ways" requirement: ``SORT_KEYS`` must cover exactly the
registry keys (T09 indexes ``SORT_KEYS[stream]``), but ``POOL_SCOPED`` is the
pool-partitioned *subset* — ``CONTRACTS.md`` §4.0 explicitly marks ``gas``,
``reference`` and ``regime`` as global. The tests below assert the three consistency
properties the contract actually requires: registry == sort_keys, pool_scoped is a
subset of both, and the global streams are exactly the complement.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

import pyarrow as pa
import pytest

from undertow.data.schemas import (
    BURN_SCHEMA,
    COLLECT_SCHEMA,
    EVENT_TAPE_SCHEMA,
    FEE_GROWTH_SCHEMA,
    FLASH_SCHEMA,
    GAS_SCHEMA,
    MINT_SCHEMA,
    POOL_SCOPED,
    REFERENCE_SCHEMA,
    REGIME_SCHEMA,
    SCHEMA_REGISTRY,
    SCHEMA_VERSION,
    SORT_KEYS,
    SWAP_SCHEMA,
    decode_uint,
    empty_table,
    encode_uint,
    validate_table,
)
from undertow.data.types import EventType, SchemaViolationError

# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

KEY_COLUMNS = (
    "block_number",
    "log_index",
    "block_timestamp",
    "tx_hash",
    "pool_address",
    "event_type",
)

LOG_STREAMS = {"swap", "mint", "burn", "collect", "flash", "event_tape"}
GLOBAL_STREAMS = {"gas", "reference", "regime"}


def _default_value(field: pa.Field) -> object:
    """A type-correct scalar for one row of ``field``."""
    t = field.type
    if pa.types.is_timestamp(t):
        return datetime(2022, 1, 1, 12, 0, 0, tzinfo=UTC)
    if pa.types.is_boolean(t):
        return True
    if pa.types.is_floating(t):
        return 1.5
    if pa.types.is_integer(t):
        return 1
    if pa.types.is_string(t):
        return "0"
    raise AssertionError(f"no default value for type {t}")


def _conforming_arrays(schema: pa.Schema) -> dict[str, pa.Array]:
    """One type-correct row per schema column, as explicitly-typed arrays."""
    return {f.name: pa.array([_default_value(f)], type=f.type) for f in schema}


def _conforming_table(schema: pa.Schema) -> pa.Table:
    arrays = _conforming_arrays(schema)
    return pa.Table.from_arrays(list(arrays.values()), names=list(arrays.keys()))


# ---------------------------------------------------------------------------
# Registry / sort keys / pool scoping
# ---------------------------------------------------------------------------


def test_registry_keys_are_the_ten_contract_streams() -> None:
    assert set(SCHEMA_REGISTRY) == {
        "swap",
        "mint",
        "burn",
        "collect",
        "flash",
        "fee_growth",
        "gas",
        "reference",
        "regime",
        "event_tape",
    }


def test_log_stream_keys_match_event_type_values() -> None:
    assert {e.value for e in EventType} <= set(SCHEMA_REGISTRY)


def test_registry_sort_keys_pool_scoped_consistency() -> None:
    # SORT_KEYS must cover exactly the registry: T09 indexes SORT_KEYS[stream] for every
    # stream it touches, so a missing key is a KeyError far from here.
    assert set(SORT_KEYS) == set(SCHEMA_REGISTRY)
    # POOL_SCOPED is the pool-partitioned subset (CONTRACTS §4.0): every entry is a
    # registered stream, and the global streams are exactly the complement.
    assert POOL_SCOPED <= set(SCHEMA_REGISTRY)
    assert POOL_SCOPED <= set(SORT_KEYS)
    assert set(SCHEMA_REGISTRY) - POOL_SCOPED == GLOBAL_STREAMS
    assert POOL_SCOPED == {
        "swap",
        "mint",
        "burn",
        "collect",
        "flash",
        "fee_growth",
        "event_tape",
    }


@pytest.mark.parametrize("stream", sorted(SCHEMA_REGISTRY))
def test_sort_keys_reference_existing_columns(stream: str) -> None:
    names = set(SCHEMA_REGISTRY[stream].names)
    for col in SORT_KEYS[stream]:
        assert col in names, f"{stream}: sort key {col!r} is not a column of the schema"


def test_sort_keys_match_contract() -> None:
    assert SORT_KEYS == {
        "swap": ("block_number", "log_index"),
        "mint": ("block_number", "log_index"),
        "burn": ("block_number", "log_index"),
        "collect": ("block_number", "log_index"),
        "flash": ("block_number", "log_index"),
        "event_tape": ("block_number", "log_index"),
        "fee_growth": ("block_number", "tick"),
        "gas": ("block_number",),
        "reference": ("symbol", "open_time"),
        "regime": ("timestamp",),
    }


# ---------------------------------------------------------------------------
# Metadata: stream name, schema_version, per-column unit strings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", sorted(SCHEMA_REGISTRY))
def test_every_field_has_unit_metadata(stream: str) -> None:
    schema = SCHEMA_REGISTRY[stream]
    meta = schema.metadata
    assert meta[b"stream"] == stream.encode()
    assert meta[b"schema_version"] == SCHEMA_VERSION.encode()
    for field in schema:
        key = f"unit:{field.name}".encode()
        assert key in meta, f"{stream}: field {field.name!r} has no unit metadata"
        assert meta[key].strip(), f"{stream}: unit metadata for {field.name!r} is empty"


# ---------------------------------------------------------------------------
# Key-column rule: present (in order) for log streams, absent elsewhere
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", sorted(LOG_STREAMS))
def test_key_columns_present_in_contract_order_for_log_streams(stream: str) -> None:
    schema = SCHEMA_REGISTRY[stream]
    assert schema.names[:6] == list(KEY_COLUMNS)
    assert schema.field("block_number").type == pa.int64()
    assert schema.field("log_index").type == pa.int32()
    assert schema.field("block_timestamp").type == pa.timestamp("us", tz="UTC")
    assert schema.field("tx_hash").type == pa.string()
    assert schema.field("pool_address").type == pa.string()
    assert schema.field("event_type").type == pa.string()
    for col in KEY_COLUMNS:
        assert not schema.field(col).nullable


def test_gas_has_no_log_index() -> None:
    names = set(GAS_SCHEMA.names)
    assert "block_number" in names
    assert "block_timestamp" in names
    assert "log_index" not in names
    assert "tx_hash" not in names
    assert "pool_address" not in names
    assert "event_type" not in names


def test_fee_growth_has_no_log_index() -> None:
    names = set(FEE_GROWTH_SCHEMA.names)
    assert "block_number" in names
    assert "pool_address" in names
    assert "log_index" not in names
    assert "block_timestamp" not in names
    assert "tx_hash" not in names
    assert "event_type" not in names


@pytest.mark.parametrize("stream", ["reference", "regime"])
def test_reference_regime_have_no_block_columns(stream: str) -> None:
    schema = SCHEMA_REGISTRY[stream]
    for col in KEY_COLUMNS:
        assert col not in schema.names, f"{stream}: must not carry {col!r}"


# ---------------------------------------------------------------------------
# Full column lists and types per stream (CONTRACTS §4.1–§4.8)
# ---------------------------------------------------------------------------


def test_swap_schema_columns_and_types() -> None:
    assert SWAP_SCHEMA.names == list(KEY_COLUMNS) + [
        "amount0",
        "amount1",
        "sqrt_price_x96",
        "liquidity",
        "tick",
        "sender",
        "recipient",
    ]
    for col in ("amount0", "amount1", "sqrt_price_x96", "liquidity"):
        assert SWAP_SCHEMA.field(col).type == pa.string()
    assert SWAP_SCHEMA.field("tick").type == pa.int32()


def test_mint_burn_schema_columns() -> None:
    expected = list(KEY_COLUMNS) + [
        "owner",
        "tick_lower",
        "tick_upper",
        "liquidity_amount",
        "amount0",
        "amount1",
        "sender",
    ]
    assert MINT_SCHEMA.names == expected
    assert BURN_SCHEMA.names == expected
    for col in ("liquidity_amount", "amount0", "amount1"):
        assert MINT_SCHEMA.field(col).type == pa.string()
    assert MINT_SCHEMA.field("tick_lower").type == pa.int32()
    assert MINT_SCHEMA.field("tick_upper").type == pa.int32()


def test_collect_schema_columns() -> None:
    assert COLLECT_SCHEMA.names == list(KEY_COLUMNS) + [
        "owner",
        "recipient",
        "tick_lower",
        "tick_upper",
        "amount0",
        "amount1",
    ]
    for col in ("amount0", "amount1"):
        assert COLLECT_SCHEMA.field(col).type == pa.string()


def test_flash_schema_columns() -> None:
    assert FLASH_SCHEMA.names == list(KEY_COLUMNS) + [
        "sender",
        "recipient",
        "amount0",
        "amount1",
        "paid0",
        "paid1",
    ]
    for col in ("amount0", "amount1", "paid0", "paid1"):
        assert FLASH_SCHEMA.field(col).type == pa.string()


def test_fee_growth_schema_columns() -> None:
    assert FEE_GROWTH_SCHEMA.names == [
        "block_number",
        "tick",
        "fee_growth_outside_0_x128",
        "fee_growth_outside_1_x128",
        "liquidity_gross",
        "liquidity_net",
        "initialized",
        "fee_growth_global_0_x128",
        "fee_growth_global_1_x128",
        "current_tick",
        "current_liquidity",
        "source",
        "pool_address",
    ]
    for col in (
        "fee_growth_outside_0_x128",
        "fee_growth_outside_1_x128",
        "liquidity_gross",
        "liquidity_net",
        "fee_growth_global_0_x128",
        "fee_growth_global_1_x128",
        "current_liquidity",
    ):
        assert FEE_GROWTH_SCHEMA.field(col).type == pa.string()
    assert FEE_GROWTH_SCHEMA.field("initialized").type == pa.bool_()


def test_gas_schema_columns() -> None:
    assert GAS_SCHEMA.names == [
        "block_number",
        "block_timestamp",
        "base_fee_per_gas",
        "gas_used",
        "gas_limit",
        "priority_fee_p50_wei",
        "priority_fee_p90_wei",
        "eth_usd_price",
    ]
    for col in ("base_fee_per_gas", "priority_fee_p50_wei", "priority_fee_p90_wei"):
        assert GAS_SCHEMA.field(col).type == pa.string()
    assert GAS_SCHEMA.field("gas_used").type == pa.int64()
    assert GAS_SCHEMA.field("eth_usd_price").type == pa.float64()


def test_reference_schema_columns() -> None:
    assert REFERENCE_SCHEMA.names == [
        "open_time",
        "close_time",
        "open",
        "high",
        "low",
        "close",
        "volume_base",
        "quote_volume",
        "trades",
        "symbol",
        "is_gap_filled",
    ]
    for col in ("open", "high", "low", "close", "volume_base", "quote_volume"):
        assert REFERENCE_SCHEMA.field(col).type == pa.float64()
    assert REFERENCE_SCHEMA.field("trades").type == pa.int64()
    assert REFERENCE_SCHEMA.field("is_gap_filled").type == pa.bool_()


def test_regime_schema_columns() -> None:
    assert REGIME_SCHEMA.names == [
        "timestamp",
        "sigma_rv",
        "mu",
        "regime",
        "window_complete",
        "symbol",
    ]
    assert REGIME_SCHEMA.field("sigma_rv").type == pa.float64()
    assert REGIME_SCHEMA.field("mu").type == pa.float64()
    assert REGIME_SCHEMA.field("window_complete").type == pa.bool_()


def test_event_tape_column_order_is_contractual() -> None:
    names = EVENT_TAPE_SCHEMA.names
    assert names[:6] == list(KEY_COLUMNS)
    assert names[6:17] == [
        "amount0",
        "amount1",
        "sqrt_price_x96",
        "liquidity",
        "tick",
        "sender",
        "recipient",
        "owner",
        "tick_lower",
        "tick_upper",
        "liquidity_amount",
    ]
    assert names[17:] == [
        "seq",
        "price_pool",
        "price_reference",
        "regime",
        "base_fee_per_gas",
        "priority_fee_p50_wei",
        "fee_growth_global_0_x128",
        "fee_growth_global_1_x128",
        "fee_growth_source",
    ]
    assert not EVENT_TAPE_SCHEMA.field("seq").nullable


# ---------------------------------------------------------------------------
# Nullability declarations
# ---------------------------------------------------------------------------


def test_nullability_declarations() -> None:
    # The three contract-mandated nullable fields.
    assert GAS_SCHEMA.field("eth_usd_price").nullable
    assert FEE_GROWTH_SCHEMA.field("fee_growth_outside_0_x128").nullable
    assert FEE_GROWTH_SCHEMA.field("fee_growth_outside_1_x128").nullable
    assert MINT_SCHEMA.field("sender").nullable
    assert BURN_SCHEMA.field("sender").nullable
    # Non-nullable where the contract says so.
    assert not SWAP_SCHEMA.field("amount0").nullable
    assert not GAS_SCHEMA.field("base_fee_per_gas").nullable
    assert not FEE_GROWTH_SCHEMA.field("fee_growth_global_0_x128").nullable
    assert not FEE_GROWTH_SCHEMA.field("fee_growth_global_1_x128").nullable
    # The tape's union payload columns are null where inapplicable to the row's event_type.
    for col in (
        "amount0",
        "sqrt_price_x96",
        "tick",
        "sender",
        "recipient",
        "owner",
        "tick_lower",
        "liquidity_amount",
    ):
        assert EVENT_TAPE_SCHEMA.field(col).nullable


# ---------------------------------------------------------------------------
# encode_uint / decode_uint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        0,
        1,
        2**256 - 1,
        -(2**255),
        1446501726624926496477173928747177,  # a real sqrtPriceX96 (ETH ~ $3000, CONTRACTS §3.0)
        2**128 - 1,  # a Q128 accumulator
        123456789012345678901234567890123456789,  # a large Q128 fee-growth value
    ],
)
def test_encode_decode_roundtrip(value: int) -> None:
    assert encode_uint(value) == str(value)
    assert decode_uint(encode_uint(value)) == value


def test_decode_uint_handles_leading_minus() -> None:
    assert decode_uint("-123") == -123
    assert decode_uint("-0") == 0


@pytest.mark.parametrize("bad", [1.0, True, "5", None, [1]])
def test_encode_uint_rejects_non_int(bad: object) -> None:
    with pytest.raises(SchemaViolationError):
        encode_uint(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad",
    ["1e5", "0x10", "+5", " 5", "5 ", "", "0x", "1.5", "1_000", "١٢٣"],
)
def test_decode_uint_rejects_non_decimal(bad: str) -> None:
    with pytest.raises(SchemaViolationError):
        decode_uint(bad)


def test_decode_uint_rejects_non_string() -> None:
    with pytest.raises(SchemaViolationError):
        decode_uint(5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate_table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", sorted(SCHEMA_REGISTRY))
def test_validate_table_accepts_conforming_table(stream: str) -> None:
    schema = SCHEMA_REGISTRY[stream]
    table = _conforming_table(schema)
    validate_table(table, schema)  # must not raise
    validate_table(table, schema, strict=False)  # must not raise


def test_validate_table_missing_column() -> None:
    table = _conforming_table(SWAP_SCHEMA).drop(["tx_hash"])
    with pytest.raises(SchemaViolationError) as exc:
        validate_table(table, SWAP_SCHEMA)
    assert "missing column: 'tx_hash'" in str(exc.value)


def test_validate_table_extra_column() -> None:
    table = _conforming_table(SWAP_SCHEMA).append_column("bogus", pa.array([1], type=pa.int64()))
    with pytest.raises(SchemaViolationError) as exc:
        validate_table(table, SWAP_SCHEMA)
    assert "extra column: 'bogus'" in str(exc.value)
    # strict=False tolerates extra columns.
    validate_table(table, SWAP_SCHEMA, strict=False)


def test_validate_table_wrong_type() -> None:
    arrays = _conforming_arrays(SWAP_SCHEMA)
    arrays["amount0"] = pa.array([123], type=pa.int64())
    table = pa.Table.from_arrays(list(arrays.values()), names=list(arrays.keys()))
    with pytest.raises(SchemaViolationError) as exc:
        validate_table(table, SWAP_SCHEMA)
    assert "wrong type: column 'amount0'" in str(exc.value)
    # strict=False still rejects a wrong type on a declared column.
    with pytest.raises(SchemaViolationError) as exc:
        validate_table(table, SWAP_SCHEMA, strict=False)
    assert "wrong type: column 'amount0'" in str(exc.value)


def test_validate_table_reordered() -> None:
    table = _conforming_table(SWAP_SCHEMA)
    names = table.column_names
    reordered = pa.Table.from_arrays(
        [table.column(names[-1])] + [table.column(n) for n in names[:-1]],
        names=[names[-1]] + names[:-1],
    )
    with pytest.raises(SchemaViolationError) as exc:
        validate_table(reordered, SWAP_SCHEMA)
    msg = str(exc.value)
    assert "wrong order" in msg
    assert "recipient" in msg  # the offending column is named
    # strict=False still rejects a reordering of declared columns.
    with pytest.raises(SchemaViolationError) as exc:
        validate_table(reordered, SWAP_SCHEMA, strict=False)
    assert "wrong order" in str(exc.value)


def test_validate_table_reports_all_mismatches_at_once() -> None:
    arrays = _conforming_arrays(SWAP_SCHEMA)
    arrays["amount0"] = pa.array([123], type=pa.int64())  # wrong type
    del arrays["tx_hash"]  # missing
    arrays["bogus"] = pa.array([1], type=pa.int64())  # extra
    table = pa.Table.from_arrays(list(arrays.values()), names=list(arrays.keys()))
    with pytest.raises(SchemaViolationError) as exc:
        validate_table(table, SWAP_SCHEMA)
    msg = str(exc.value)
    assert "missing column: 'tx_hash'" in msg
    assert "extra column: 'bogus'" in msg
    assert "wrong type: column 'amount0'" in msg


def test_validate_table_message_format() -> None:
    arrays = _conforming_arrays(SWAP_SCHEMA)
    del arrays["recipient"]  # trailing column: no order shift, isolates the missing-column violation
    table = pa.Table.from_arrays(list(arrays.values()), names=list(arrays.keys()))
    with pytest.raises(SchemaViolationError) as exc:
        validate_table(table, SWAP_SCHEMA)
    msg = str(exc.value)
    assert msg.startswith("table does not conform to schema 'swap'")
    assert "schema_version=1.0.0" in msg
    assert "1 violation(s):" in msg
    assert "- missing column: 'recipient'" in msg


def test_validate_table_rejects_nulls_in_non_nullable_column() -> None:
    arrays = _conforming_arrays(SWAP_SCHEMA)
    arrays["amount0"] = pa.array([None], type=pa.string())
    table = pa.Table.from_arrays(list(arrays.values()), names=list(arrays.keys()))
    with pytest.raises(SchemaViolationError) as exc:
        validate_table(table, SWAP_SCHEMA)
    msg = str(exc.value)
    assert "nullability" in msg
    assert "amount0" in msg


def test_validate_table_accepts_nulls_in_nullable_column() -> None:
    arrays = _conforming_arrays(GAS_SCHEMA)
    arrays["eth_usd_price"] = pa.array([None], type=pa.float64())
    table = pa.Table.from_arrays(list(arrays.values()), names=list(arrays.keys()))
    validate_table(table, GAS_SCHEMA)  # must not raise


def test_validate_table_rejects_naive_timestamp() -> None:
    arrays = _conforming_arrays(SWAP_SCHEMA)
    arrays["block_timestamp"] = pa.array([datetime(2022, 1, 1)], type=pa.timestamp("us"))
    table = pa.Table.from_arrays(list(arrays.values()), names=list(arrays.keys()))
    with pytest.raises(SchemaViolationError) as exc:
        validate_table(table, SWAP_SCHEMA)
    assert "wrong type: column 'block_timestamp'" in str(exc.value)


def test_validate_table_rejects_non_table() -> None:
    batch = pa.RecordBatch.from_arrays([pa.array([1])], names=["a"])
    with pytest.raises(SchemaViolationError):
        validate_table(batch, SWAP_SCHEMA)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", sorted(SCHEMA_REGISTRY))
def test_all_timestamp_fields_are_utc_microsecond(stream: str) -> None:
    for field in SCHEMA_REGISTRY[stream]:
        if pa.types.is_timestamp(field.type):
            assert field.type.unit == "us"
            assert field.type.tz == "UTC"


# ---------------------------------------------------------------------------
# empty_table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", sorted(SCHEMA_REGISTRY))
def test_empty_table_is_valid_and_zero_rows(stream: str) -> None:
    schema = SCHEMA_REGISTRY[stream]
    table = empty_table(schema)
    assert table.num_rows == 0
    assert table.column_names == schema.names
    validate_table(table, schema)
    validate_table(table, schema, strict=False)


def test_empty_table_preserves_metadata() -> None:
    table = empty_table(SWAP_SCHEMA)
    assert table.schema.metadata == SWAP_SCHEMA.metadata


# ---------------------------------------------------------------------------
# SCHEMA_VERSION
# ---------------------------------------------------------------------------

_SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(-(0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(\.(0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*)?"
    r"(\+[0-9a-zA-Z-]+(\.[0-9a-zA-Z-]+)*)?$"
)


def test_schema_version_is_valid_semver() -> None:
    assert _SEMVER_RE.fullmatch(SCHEMA_VERSION) is not None
    assert SCHEMA_VERSION == "1.0.0"
