# ADR 006 — Drop per-block gas timestamps; GAS_SCHEMA is base-fee only

**Status:** accepted
**Affects:** T03 `schemas.py` · T07 `fetchers/gas.py` · T11 `transforms/align.py` · CONTRACTS.md §4.5
**Raised by:** `feature-gas-feehistory-only`

## Context

CONTRACTS.md §4.5 defines `GAS_SCHEMA` as one row per block carrying
`base_fee_per_gas`, `block_timestamp`, `gas_used`, `gas_limit`, the priority-fee percentiles, and a
nullable `eth_usd_price`. T07 fetches it; T11 populates `eth_usd_price` by a backward as-of join of
`reference.close_time <= gas.block_timestamp`.

ADR-005 already replaced per-block priority-fee resolution with a flat tip surcharge and moved base-fee
fetching to `eth_feeHistory(…, percentiles=[])` (~10 CU per 1024 blocks). Its amendment then re-added
"real timestamps per block" via batched `eth_getBlockByNumber`, estimating the batch at "~20 CU". That
estimate is wrong: Alchemy bills `eth_getBlockByNumber` **16 CU per item inside a batch**, not once per
batch.

## Problem

Fetching `block_timestamp` for every block is the single most expensive operation in the whole pipeline,
and nothing that matters reads it.

Live measurement against the pinned window (7,609,726 blocks, 2022-01-01 → 2024-12-31):

| gas sub-stream | CU | wall-clock @ 10,000 CU/s |
|---|---|---|
| `eth_feeHistory` base fees only | ~75K | ~1 min |
| + `eth_getBlockByNumber` timestamps for every block | ~122M | ~3.5 h |

The dense timestamp column is consumed only by `transforms/align.py:attach_reference_prices`, which
enriches the **standalone** gas table's `eth_usd_price`. The event tape — the artifact the simulator
actually replays — does not read either column from gas:

- the tape joins `gas` on `block_number` only, reading `base_fee_per_gas` and
  `priority_fee_p50_wei` (align.py `build_event_tape`);
- the tape's `price_reference` and `regime` are joined on the **tape's** `block_timestamp`, which comes
  free from the The Graph event rows;
- `attach_reference_prices` is never called by `pull`, `build`, `verify`, or `snapshot` (it is exported
  public API only), so `gas.eth_usd_price` is always null in a produced dataset anyway;
- every validation check on gas (`check_no_block_gaps`) keys on `block_number` only; the timestamp
  ordering checks run against the tape.

The simulator's decision cadence is "every 10 minutes … nearest tape block boundary" (sim PLAN.md §7.8),
so the gas price it needs is always at a block the tape already carries with a free Graph timestamp.

## Decision

`GAS_SCHEMA` becomes base-fee only:

```
block_number       int64      (pk, one row per block, no gaps)
base_fee_per_gas   string     (wei, uint256, exact)
gas_used           int64      (0, kept per ADR-005 compatibility)
gas_limit          int64      (0, kept per ADR-005 compatibility)
priority_fee_p50_wei string   ("0", kept per ADR-005)
priority_fee_p90_wei string   ("0", kept per ADR-005)
```

- `block_timestamp` and `eth_usd_price` are **removed** from `GAS_SCHEMA`.
- The gas fetcher's `_fetch_chunk` performs **one** `eth_feeHistory(blockCount, newestBlock, [])` call
  per chunk of 1024 blocks and emits rows with the six columns above; the batched
  `eth_getBlockByNumber` timestamp path (`_batch_headers`) is deleted.
- The sparse block index (`block_for_timestamp` / `timestamp_for_block`) is **unchanged** — it is the
  date→block window resolver and stays memoized with single-block `eth_getBlockByNumber` calls.
- `transforms/align.py:attach_reference_prices` is deleted; USD conversion happens on the tape via its
  `price_reference` column, which the sim consumes (sim PLAN.md: "Uses the tape's joined gas columns").

## Consequences

- `schemas.py`: GAS_SCHEMA loses `block_timestamp` and `eth_usd_price`; the data dictionary string
  updates.
- `fetchers/gas.py`: `_fetch_chunk`, `_make_row`, and `_batch_headers` simplify; module docstring
  updates.
- `transforms/align.py`: `attach_reference_prices` and its `__all__` entry removed.
- `validation/checks.py`: no change — `check_no_block_gaps` keys on `block_number`.
- CONTRACTS.md §4.5 table rows for `block_timestamp` / `eth_usd_price` removed (or marked superseded),
  with a pointer to this ADR.
- sim side: S08's `gas_cost_usdc(..., eth_usd_price, ...)` is now fed `price_reference` from the tape,
  never a gas-table column — document in sim CONTRACTS §S08 when S08 lands.
- **Vault note flagged to the human** (never edited by a code task): the thesis methods section should
  record that gas-cost conversion uses the event tape's reference price, with per-block base fees from
  `eth_feeHistory`.

### Rejected alternatives

- **Keep fetching all timestamps**: exact, but ~122M CU / ~$64–80 and ~3.5 h at 10,000 CU/s for a phase
  of the dataset nothing consumes; also the source of the live 429 storm (PR #31's fix reduces the
  storm but not the wall-clock/CU bill).
- **Sparse anchor + interpolation**: bounds skipped-slot drift to a chunk but still violates the 30 s
  join-tolerance claim ADR-005 relied on; rejected for the same reason as the original slot-spacing
  idea.
- **Make `block_timestamp` nullable-but-present**: less schema churn, but a permanently-null column is
  misleading to anyone loading GAS_SCHEMA; the column has no reader, so it is dropped outright.

---

**Process:** ADR raised on `feature-gas-feehistory-only`; listed in `docs/plans/data-pipeline/STATUS.md`
under "ADRs raised".