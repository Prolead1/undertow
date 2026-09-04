# T03 — Canonical schemas (the data contract)

**Wave 2 · size M · depends on: T00, T01 · blocks: T05, T06, T07, T08, T09, T11, T12**
**Branch:** `feature-data-schemas`

## Why this task exists

Seven fetchers/transforms written by seven different agents in parallel must produce tables that
`build_event_tape` (T11) can merge without adaptation. This module is the single point where column
names, types, units and sign conventions are declared, so that a mismatch is a loud
`SchemaViolationError` at the boundary rather than a silent wrong number three tasks later.

## Files you own

```
src/undertow/data/schemas.py
tests/data/test_schemas.py
docs/schema_notes.md            (short: why string-encoded ints, why (block, log_index) ordering)
```

## What to build

Implement `CONTRACTS.md` §4 exactly — all **ten** schemas (the four log streams, `flash`, `fee_growth`,
`gas`, `reference`, `regime`, `event_tape`), plus `SCHEMA_REGISTRY`, **`SORT_KEYS`**, **`POOL_SCOPED`**,
`SCHEMA_VERSION`, `validate_table`, `encode_uint`/`decode_uint`, `empty_table`.

Read `CONTRACTS.md` §4.0 first: streams come in two families. The six mandatory key columns apply to
**log streams only** — `gas`, `reference` and `regime` have no `log_index` and (for the latter two) no
block columns at all, and must not carry sentinel values to fake them. `SORT_KEYS` and `POOL_SCOPED`
exist so T09 and T14 never have to infer any of this.

### Key design points you must honor

1. **String-encoded big integers.** `uint256`/`int256`/`uint128`/`uint160` columns are `pa.string()`
   holding a decimal representation. `encode_uint(v) -> str` / `decode_uint(s) -> int` are the only
   sanctioned conversions. `encode_uint` must reject non-`int` input (including `bool` and `float`)
   with `SchemaViolationError` — a `float` sneaking in here is how precision dies. Signed values keep a
   leading `-`; `decode_uint` must handle it.

2. **Column order is part of the schema.** The six key columns come first, in the `CONTRACTS.md` §4
   order, in every stream table. `validate_table(strict=True)` fails on wrong order, not just wrong
   membership — because T09's deterministic-bytes requirement depends on stable column order.

3. **`validate_table` reports everything at once.** Collect all mismatches (missing column, extra
   column, wrong type, wrong position, wrong nullability) and raise a single `SchemaViolationError`
   whose message lists them. Fifteen agents will be debugging against this message; make it good.
   `strict=False` allows *extra* columns (T11 appends derived ones) but still enforces the declared ones.

4. **Nullability is declared, not incidental.** Be explicit per field, and write the nullability table
   into `docs/schema_notes.md`. Notably:
   - `gas.eth_usd_price` is nullable (T07 writes null; T11 joins it in).
   - `fee_growth.fee_growth_outside_*` are nullable (null on global-only rows).
   - `mint.sender` is **nullable**, and is **null** — not `""` — on burn rows and wherever else it does
     not apply. `CONTRACTS.md` §4.2 makes this a project-wide rule: an absent value is null, never an
     empty string and never a zero. There are no sentinel values in this contract.

5. **Metadata.** Attach `pa.schema(..., metadata={...})` carrying `schema_version`, the stream name, and
   a short per-column unit string (e.g. `amount0 -> "raw token0 units, signed, positive=into pool"`).
   T16's data dictionary is generated from this metadata, so it must be complete — every column gets a
   unit/convention string.

## Tests you must write

- Every schema is in `SCHEMA_REGISTRY` and its `metadata` has a unit string for **every** field
  (parametrize over the registry — this catches the next agent who adds a column and forgets).
- `SCHEMA_REGISTRY`, `SORT_KEYS` and `POOL_SCOPED` have **exactly** the same stream keys — assert set
  equality three ways. A stream present in one and missing from another is a `KeyError` in T09 or T14,
  far from here.
- Every `SORT_KEYS` entry names columns that actually exist in that stream's schema.
- Key columns are asserted present **only** for the log-stream family, and asserted **absent** for
  `reference`/`regime` (no `block_number`, no `log_index`).
- Key columns present, correctly typed, and **in the contract's order** for all stream schemas.
- `encode_uint` / `decode_uint` round-trip on: `0`, `1`, `2**256 - 1`, `-(2**255)`,
  a real `sqrtPriceX96`, a real Q128 accumulator.
- `encode_uint(1.0)`, `encode_uint(True)`, `encode_uint("5")` all raise `SchemaViolationError`.
- `decode_uint("1e5")` and `decode_uint("0x10")` raise — decimal strings only, no clever parsing.
- `validate_table` accepts a conforming table; and for each failure mode (missing / extra / wrong type /
  reordered) it raises and the message **names the offending column**.
- `validate_table(strict=False)` accepts extra columns but still rejects a wrong type on a declared one.
- `empty_table(schema)` produces a 0-row table that `validate_table` accepts — every fetcher returns
  this for an empty range, so it must be exactly right.
- Timestamps: all timestamp fields are `timestamp[us, tz=UTC]`; a naive-timestamp table is rejected.
- `SCHEMA_VERSION` is a valid semver string.

## Acceptance criteria

- `uv run pytest` green; `mypy` clean.
- No fetcher-specific or transform-specific logic in this module — schemas and codecs only. If you find
  yourself importing `requests`, you are in the wrong file.
- `docs/schema_notes.md` explains the three non-obvious choices (string ints, `(block_number,
  log_index)` ordering, forward-fill-and-flag for reference gaps) in a paragraph each.

## Handoff notes for your PR body

- The exact `SCHEMA_REGISTRY` keys (fetchers index into it by stream name).
- The `SchemaViolationError` message format, with an example — the other agents will match on it in
  tests.
- Any column you added beyond `CONTRACTS.md` §4, and why.

## Process (mandatory)

Branch → implement → tests green → `code-reviewer` → fix → commit → push → PR → update `STATUS.md`.
