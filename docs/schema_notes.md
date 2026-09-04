# Schema notes — `undertow.data` canonical schemas

This file documents the non-obvious design choices in `src/undertow/data/schemas.py`
(the canonical pyarrow schemas, `CONTRACTS.md` §4). It is the companion to the
`metadata` attached to each schema, which T16's data dictionary is generated from.

## Why big integers are stored as decimal strings

`uint256`/`int256`/`uint128`/`uint160` values — `amount0`, `amount1`, `liquidity`,
`sqrtPriceX96`, every `feeGrowth*X128` accumulator, `baseFeePerGas` — are stored in
Parquet as `pa.string()` holding a plain decimal representation, and handled in Python
as `int`. The reason is precision: `2**128 ≈ 3.4e38`, which carries ~70 bits more than
a float64 mantissa (53 bits). A Q128 fee-growth accumulator converted to `float` before
the final division silently destroys the low bits that the whole pipeline exists to
preserve. Parquet has no native 256-bit integer type, so the string column is the only
lossless representation. `encode_uint` / `decode_uint` in `schemas.py` are the *only*
sanctioned conversions: `encode_uint` rejects anything that is not an `int` (including
`bool` and `float` — a float sneaking in here is how precision dies), and `decode_uint`
accepts only plain decimal strings (no exponents, no hex, no sign other than a leading
`-` for signed `int256` values). Signed values keep their leading `-` in the string.

## Why the ordering key is `(block_number, log_index)`, not `timestamp`

Multiple events share a block timestamp, and block timestamps are not strictly
increasing across reorged history, so `timestamp` is not a total order. The protocol
itself orders events within a block by `log_index`, and `(block_number, log_index)` is
the unique, monotone key that survives reorgs and deduplication. Every log stream
(`swap`, `mint`, `burn`, `collect`, `flash`, `event_tape`) carries the six key columns
— `block_number`, `log_index`, `block_timestamp`, `tx_hash`, `pool_address`,
`event_type` — in that exact order, first in the schema. Column order is part of the
schema: `validate_table(strict=True)` fails on wrong order, not just wrong membership,
because T09's deterministic-bytes requirement depends on stable column order. Non-log
streams (`gas`, `reference`, `regime`) have their own sort keys (`SORT_KEYS`) and
carry no `log_index`; `reference`/`regime` have no block columns at all. There are no
sentinel values — an absent value is `null`, never `""` and never `0`.

## Why reference-price gaps are forward-filled and flagged

The reference feed (Binance klines, T08) can have missing minutes — exchange outages,
thin pairs, or a kline that never printed. The pipeline does not silently fabricate a
price: a gap is carried forward from the last observed close (backward-looking only, so
no look-ahead) and a flag column records that the value is carried, not observed. This
keeps downstream consumers (regime labels in T12, IL/LVR in the simulator) on a
contiguous time grid while making the provenance of every carried value explicit and
auditable. The flag is what lets T13's validation distinguish "genuinely flat" from
"missing data we papered over".

## Nullability

Nullability is declared per field, not incidental. The notable cases:

- `gas.eth_usd_price` is **nullable** — T07 writes `null` (it does not fetch USD
  prices); T11 joins the value in later.
- `fee_growth.fee_growth_outside_*` are **nullable** — `null` on global-only rows
  where no tick is referenced.
- `mint.sender` is **nullable** and is `null` — not `""` — on burn rows and wherever
  else it does not apply (the `Burn` event has no `sender`).

`validate_table` checks nullability in the direction that matters: a column declared
non-nullable must not contain nulls; a column declared nullable may legitimately have
no nulls (pyarrow infers `nullable=False` for a null-free column), so that direction is
not a violation.

## Metadata

Every schema carries `pa.schema(..., metadata={...})` with `schema_version`, the
stream name, and a per-column unit/convention string for **every** field (e.g.
`amount0 -> "raw token0 units, signed, positive=into pool"`). T16's data dictionary is
generated from this metadata, so a column added without a unit string is a test
failure (`test_schema_metadata_units_complete` parametrizes over the whole registry).
