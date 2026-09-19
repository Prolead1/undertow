# T07 — Gas / block fetcher

**Wave 3 · size M · depends on: T01, T03, T04 · blocks: T11, T14**
**Branch:** `feature-data-gas`

## Why this task exists

Roadmap §10.1.6 lists gas as a required stream because it *"gives the rebalancing cost `G_t` — the
friction that makes RL hard"*, and §10.2.3 is emphatic: *"gas is subtracted at the exact block where the
action fires, using that block's gas price (spikes are real, and they materially change optimal rebalance
frequency)."* The chapter's own exercise Q5 asks why an average will not do, and answers: gas spikes
correlate with exactly the moments an active policy wants to act. Gap **G1** (friction realism) lives or
dies on this table being per-block and complete.

A second, quieter job: this module owns the **timestamp ↔ block** index, which is how every date-based
config window becomes a block range.

## Files you own

```
src/undertow/data/fetchers/gas.py
tests/data/test_gas.py
tests/data/fixtures/blocks_gas.json
```

## What to build

`GasFetcher(BaseHttpFetcher)` per `CONTRACTS.md` §5.3, producing `GAS_SCHEMA` — **one row per block, no
gaps**.

**You own table-building.** Per `CONTRACTS.md` §5.1's layering rule, `BaseHttpFetcher` returns raw rows
and never touches schemas. So *your* module implements `_rows_to_table(rows, stream)`, calls
`schemas.validate_table` on the result, and wraps it in `FetchResult`. Use `empty_table(schema)` for an
empty range.

### 1. Per-block fields
- `base_fee_per_gas` from `eth_feeHistory(blockCount, newestBlock, [])` — one call per ~1024
  blocks (ADR-005). **ADR-006:** `block_timestamp` is no longer fetched — gas is base-fee only; the
  event tape carries block timestamps free from the event-log rows.
- `priority_fee_p50_wei` / `priority_fee_p90_wei` — percentiles of the *effective priority fee* paid by
  transactions in the block. Two ways to get them:
  - **Preferred:** `eth_feeHistory(blockCount, newestBlock, [50, 90])`, which returns exactly this and
    costs one call per ~1024 blocks. Use it. Note that `eth_feeHistory`'s `reward` array is `null` for
    blocks with no transactions — those get `priority_fee_p*_wei = 0`, flagged, not null-propagated.
  - Fallback: `eth_getBlockByNumber(..., full_tx=True)` and compute
    `min(maxPriorityFeePerGas, maxFeePerGas - baseFee)` per tx. Expensive; implement only if
    `eth_feeHistory` is unavailable, and say which you used.
- **EIP-1559 only.** The pinned window starts 2022-01-01, comfortably after the London fork
  (block 12,965,000, Aug 2021), so `baseFeePerGas` is always present. If a caller requests a
  pre-London block, raise `ConfigError` rather than inventing a base fee. Test this boundary.
- `priority_fee_p90_wei` is deliberately **not** consumed by the event tape — T11 joins only `p50`. Fetch
  it anyway: it is what lets `undertow.sim` later ask "what if we had to pay to get included during a
  spike?", and re-pulling three years of blocks to add one column is exactly the waste §10.7 warns about.
  `gas_cost_wei(..., percentile=90)` is its accessor.

### 2. Completeness is the contract
Roughly 2.5M blocks span the three-year window. The stream must contain **every** block in the requested
range — T13 has a `check_no_block_gaps` that will fail the whole dataset otherwise. Concretely:
- fetch in ranges via `eth_feeHistory` + batched `eth_getBlockByNumber`;
- assert, before returning, that `block_number` is a **contiguous** ascending run from `start` to `end`
  with no duplicates, and raise `ValidationError` naming the first missing block if not;
- `gas` is a **non-log, chain-wide** stream (`CONTRACTS.md` §4.0): it has **no** `log_index`, `tx_hash`,
  `pool_address` or `event_type` columns — do not add `-1` sentinels for them. Its sort key is
  `(block_number,)` and it is **not** partitioned per pool; both pinned pools share one gas table.

### 3. The block index
```python
def block_for_timestamp(self, ts: datetime) -> BlockNumber      # last block with timestamp <= ts
def timestamp_for_block(self, block: BlockNumber) -> datetime
```
Binary search over `eth_getBlockByNumber`, memoized in the T04 cache. Exactness at boundaries matters:
if `ts` falls exactly on a block's timestamp, return **that** block. Reject a `ts` before the chain's
genesis or after the latest block with `ConfigError`. Reject a naive datetime.

**You own the public block index.** T06 keeps a *private* resolver for annotating logs with timestamps;
that duplication is deliberate (you are both in wave 3 and cannot negotiate). Do not try to coordinate
with T06 — just make yours correct and public, since T01's date→block resolution and T14 both call it.

### 4. Gas-cost helper
```python
def gas_cost_wei(gas_row: Mapping[str, object], gas_units: int, *, percentile: int = 50) -> int
```
`(base_fee_per_gas + priority_fee_p{percentile}_wei) * gas_units`, integer. `gas_units` comes from
`config.GAS_UNITS` — never hardcode it here. This is the function `undertow.sim` will ultimately call to
price a rebalance, so keep it pure and integer.

## Tests you must write (no network)

1. Fixture → `GAS_SCHEMA` conformance under `validate_table(strict=True)`, and an explicit assertion that
   the table has **no** `log_index` / `tx_hash` / `pool_address` / `event_type` column.
2. Field decode: `base_fee_per_gas` as an exact integer (a real value ~1e10 wei), `gas_used`,
   `gas_limit` (both zero per ADR-005).
3. Percentiles: from a mocked `eth_feeHistory` reward array, `p50` and `p90` match hand-computed values;
   a block with `reward: null` yields `0` for both and is flagged in `FetchResult.warnings`.
4. **Gap detection:** a mocked response missing one block in the middle raises `ValidationError` naming
   that block number. Then assert the happy path over a contiguous range does not raise.
5. Duplicate block in the response → `ValidationError`.
6. `block_for_timestamp`: exact hit returns that block; a timestamp between two blocks returns the
   **earlier** one; before genesis and after latest each raise `ConfigError`; naive datetime raises.
7. `timestamp_for_block` round-trips with `block_for_timestamp` on the fixture's blocks.
8. Pre-London block request → `ConfigError` mentioning EIP-1559.
9. `gas_cost_wei`: hand-computed integer for a known base fee, priority fee and `gas_units`; `p90`
   selection yields a strictly larger cost than `p50` when the percentiles differ; result is `int`, never
   `float`.
10. **Spike preservation:** build a fixture where one block's base fee is 20× its neighbours and assert
    the returned table preserves that value exactly — no smoothing, no averaging, anywhere. This is the
    §10.2.3 requirement expressed as a test.
11. Cache-hit makes zero HTTP calls; no RPC key in `caplog`.

## Acceptance criteria

- `uv run pytest tests/data/test_gas.py` green; `mypy` clean.
- No averaging or interpolation of gas anywhere in the module. If a value is unavailable it is `0` with a
  warning, or the fetch fails — never a smoothed estimate.
- Wei values are integers end-to-end.

## Handoff notes for your PR body

- Whether `eth_feeHistory` or the full-tx fallback was used.
- The confirmed `block_for_timestamp` boundary semantics (T01's date→block resolution and T14 depend on
  it).

- Approximate call count for the full pinned window (T14 needs it to size the pull).

## Process (mandatory)

Branch → implement → tests green → `code-reviewer` → fix → commit → push → PR → update `STATUS.md`.
