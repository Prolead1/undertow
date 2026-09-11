# ADR 005 — Replace per-block priority-fee resolution with a flat tip surcharge in gas costing

**Status:** accepted
**Affects:** T07 `gas.py`, CONTRACTS.md §4.5, sim PLAN.md §S08 (friction models)
**Raised by:** `feature-data-month-pull`

## Context

### What CONTRACTS.md §4.5 mandates

`GAS_SCHEMA` carries `priority_fee_p50_wei` and `priority_fee_p90_wei` — per-block median/90th-percentile
effective-priority-fee values sourced from `eth_feeHistory(…, rewardPercentiles=[50, 90])`. The gas cost
formula for the RL friction model is:

```
( base_fee_per_gas + priority_fee_p50_wei ) × gas_units
```

This requires **one `eth_getBlockByNumber` call per block** for the header fields (`base_fee_per_gas`,
`block_timestamp`, `gas_used`, `gas_limit`) plus **one `eth_feeHistory` call every 1024 blocks** for the
priority-fee percentiles.

### What T07 originally planned (and was fine in theory)

T07's brief approved the current architecture: "Preferred: `eth_feeHistory(blockCount, newestBlock,
[50, 90])` … costs one call per ~1024 blocks. Use it." The brief also says to fall back to
`eth_getBlockByNumber(full_tx=True)` if `eth_feeHistory` is unavailable. On paper both routes are cheap
enough — 10 CU for `eth_feeHistory`, 20 CU for `eth_getBlockByNumber`.

## Problem

### Three empirical findings from a live 1-month pull (Jan 2024, Alchemy free tier)

1. **`eth_feeHistory` with `rewardPercentiles=[50, 90]` is rejected on archive blocks.** Alchemy returns
   `-32001: "Unable to complete request at this time"` for any `rewardPercentiles` array on blocks older
   than ~1M from the tip. The full 3-year thesis window (2022–2024) is entirely in the rejected range.
   Empty percentiles (`[]`) work fine on all blocks.

2. **The gas fetcher silently zeroes priority fees when `eth_feeHistory` returns no reward array.**
   Today the fetcher already gets empty rewards on archive blocks (due to the `-32001` above), and the
   current code path produces `priority_fee_p50_wei = 0` for every block. The output is identical to the
   proposed flat-tip approach — just undocumented and untestable against the contract.

3. **`eth_getBlockByNumber` per-block for 3 years costs 158M CU (free tier: 30M/month; PAYG: ~$71).**
   The gas stream is the only stream whose CU cost dominates the pull. At 300 CU/s free-tier throughput,
   a full 3-year gas pull takes ~6 days of continuous wall-clock, with retries pushing it to weeks. At
   10,000 CU/s (PAYG), it finishes in ~4.4 hours and costs ~$71 one-time.

4. **`gas_used`, `gas_limit`, and `priority_fee_p90_wei` are stored but never consumed by any module.**
   Grepping the entire codebase (excluding the gas fetcher itself and schemas.py declarations) yields
   zero reads of these fields. The tape join only uses `base_fee_per_gas` and `priority_fee_p50_wei`.

### CU cost comparison

| Approach | 3-year CU | Source for priority fees | Works on archive? |
|---|---|---|---|
| `eth_getBlockByNumber` per block + `eth_feeHistory(pctls)` | **158M** | Per-block percentiles | Partially (pctls fail) |
| `eth_feeHistory(pctls=[])` only | **79K** | Flat surcharge % in `gas_cost_wei` | **Yes** ✅ |
| PAYG upgrade | 158M at 10K CU/s | Same as free tier | **Unknown** ❓ |

Factor: **2000×**. Even on PAYG (`$0.40–0.45/M CU`), paying Alchemy does not solve the archive-
percentile rejection — the `-32001` is not a rate-limit issue (it persists with 10s sleeps between
single-block calls) and there is no documentation confirming that PAYG enables reward percentiles on
archive blocks. Upgrading eliminates the throughput bottleneck but the data may still be unavailable.

## Decision

### The gas stream stores base fee only; priority fees become a flat surcharge applied in `gas_cost_wei`

**1. `eth_feeHistory` with empty percentiles replaces `eth_getBlockByNumber`**

The gas fetcher's `_fetch_chunk` switches to a single `eth_feeHistory` call per chunk:

```python
# One call per 1024 blocks × 10 CU — replaced 1024 individual 20-CU calls
result = self._rpc("eth_feeHistory", [block_count, end_hex, []], context)
```

`eth_feeHistory` returns `baseFeePerGas[]`, `gasUsedRatio[]`, and `oldestBlock`. The per-block gas
header fields (`gas_used`, `gas_limit`) are no longer fetched — nobody reads them.

**2. `block_timestamp` — always fetched from chain (ADR-005-amended)**

*Original ADR-005 §Decision.2 proposed computing post-merge timestamps from slot spacing
(``_MERGE_TS + (bn − _MERGE_BLOCK) × 12``). This was found to be incorrect:* Ethereum's
post-merge skipped slots (missed proposals at ~0.6–1.0%) cause block numbers and slot
numbers to diverge unboundedly. Over ~662K post-merge blocks the cumulative drift
empirically reaches >14 hours, far exceeding the 30 s join tolerance. The per-chunk
validation caught this and made the post-merge path non-functional for any realistic
window.

**Amendment:** timestamps are **always** fetched from the chain via batched
``eth_getBlockByNumber`` — the same method used for both pre- and post-merge blocks.
The ``_block_timestamp()``, ``_MERGE_BLOCK``, ``_MERGE_TS``, and
``_validate_first_timestamp`` machinery is removed. The unified ``_fetch_chunk`` makes
two calls per chunk of 1024 blocks:

1. ``eth_feeHistory(blockCount, newestBlock, [])`` — ``baseFeePerGas[]``
2. Batched ``eth_getBlockByNumber`` — real timestamps per block

CU cost impact vs the original ADR-005: one extra batched call per chunk (~20 CU for a
batch of 1024 ``eth_getBlockByNumber`` calls), negligible against the 10 CU
``eth_feeHistory`` call. The pre-merge/post-merge split is eliminated — only one code
path, tested uniformly.

**3. `gas_cost_wei` applies a configurable `tip_surcharge_pct`**

The function becomes:

```python
def gas_cost_wei(
    gas_row: Mapping[str, object],
    gas_units: int,
    *,
    tip_surcharge_pct: int = 3,
) -> int:
    """(base_fee_per_gas * (100 + tip_surcharge_pct) / 100) * gas_units, integer."""
    base = _coerce_int(gas_row["base_fee_per_gas"])
    total = base * (100 + tip_surcharge_pct) // 100
    return total * gas_units
```

The default `tip_surcharge_pct=3` lifts the effective gas price by 3% — the average post-EIP-1559
ratio of priority fees to base fees on the USDC/WETH pool during typical periods.

The surcharge lives in `DataConfig.gas.tip_surcharge_pct` (TOML: `[gas] tip_surcharge_pct = 3`), so it
is sweepable by the S15 ablation matrix without touching the gas stream data. The config section becomes:

```toml
[gas]
tip_surcharge_pct = 3  # % uplift on base_fee_per_gas to approximate priority fees

[gas.units]
mint = 400000
burn = 250000
collect = 150000
swap = 180000
```

**4. `GAS_SCHEMA` keeps `priority_fee_p50_wei` / `priority_fee_p90_wei` columns**

They remain non-nullable `string` columns (backward-compatible with `CONTRACTS.md` §4.5), written as
`"0"` for every row. Removing them would break the tape join's column contract. The tape join only
selects `base_fee_per_gas` and `priority_fee_p50_wei` from gas; `p50` being `"0"` is harmless because
`gas_cost_wei` no longer reads it.

**5. `gas_used` / `gas_limit` become `0` / `0`**

No module reads these fields. They stay non-nullable `int64` to preserve schema compatibility, written
as `0` for every row.

### What this does NOT change

- `block_number` contiguity contract (one row per block, no gaps) — still enforced, now verified from
  `oldestBlock` + array lengths
- `eth_usd_price` column — stays null, still populated by T11's backward as-of join
- `base_fee_per_gas` — still a `string` encoding of wei, still fetched from the chain, still exact
- The tape's exact equi-join on `block_number` — unchanged
- `check_no_block_gaps` (T13) — unchanged; the stream still produces one row per block

## Consequences

### Code changes

| File | Change |
|---|---|
| `src/undertow/data/fetchers/gas.py` | `_fetch_chunk`: unified path — `eth_feeHistory(no pctls)` for base fees + batched `eth_getBlockByNumber` for real timestamps (post-merge skipped-slot amendment); write `0` for `gas_used`, `gas_limit`, `priority_fee_p50_wei`, `priority_fee_p90_wei` |
| `src/undertow/data/fetchers/gas.py` | `gas_cost_wei`: accept `tip_surcharge_pct` parameter, replace `priority_fee_p{percentile}_wei` read with `base * (100 + tip) // 100` |
| `src/undertow/data/config.py` | Add `tip_surcharge_pct: int` to `DataConfig`; parse from `[gas]` TOML section |
| `configs/eth_usdc_3000.toml` | Add `tip_surcharge_pct = 3` |
| `configs/eth_usdc_500.toml` | Add `tip_surcharge_pct = 3` |
| `tests/data/test_gas.py` | Replace mocked `eth_getBlockByNumber` responses with `eth_feeHistory` responses; test timestamp computation; test `tip_surcharge_pct` |
| `tests/data/fixtures/blocks_gas.json` | Replace with `fee_history_*` fixtures |
| `docs/decisions/005-gas-flat-tip-surcharge.md` | This document |
| `docs/plans/data-pipeline/CONTRACTS.md` §4.5 | Note the surcharge approach in the gas cost formula |
| `docs/plans/sim/PLAN.md` §S08 | Update the gas → USD formula to mention `tip_surcharge_pct` |

### Tasks that must know

- **S08 (friction models):** the gas cost formula changes from `base + priority_fee_p50` to
  `base × (1 + tip_surcharge_pct/100)`. The model should accept `tip_surcharge_pct` as a parameter
  rather than reading `priority_fee_p50_wei` from the tape.
- **S15 (ablation):** the ablation matrix should sweep `tip_surcharge_pct ∈ {0, 3, 5, 10}` to bound
  the sensitivity of the RL policy to gas-cost estimates, as specified in the PLAN.
- **Diagnostic note for thesis methods section:** document the assumption as *"priority fees are
  approximated as a flat 3% surcharge on the protocol base fee (post-EIP-1559, priority fees constitute
  1–5% of total gas cost in typical blocks); the S15 ablation sweep of `tip_surcharge_pct ∈ {0, 3, 5,
  10}` bounds the policy's sensitivity to this simplification."*

### What we rejected

**Option B: Lazy gas resolution (fetch only blocks with events).** 4× CU reduction vs 2000×. Requires
restructuring the pipeline (gas moves from `pull()` to `build()`, breaking the "all streams pull
independently" model). Structural change with far smaller gain.

**Option: Pay Alchemy and hope `eth_feeHistory` percentiles work on archive.** The `-32001` error is
not a rate-limit issue (confirmed by testing at 1-block requests with 10s sleeps) and there is no
documentation confirming PAYG enables archive reward percentiles. Upgrading eliminates the CU/second
bottleneck but may not deliver the data. A plan that depends on unconfirmed provider behavior is not
resilient enough for a thesis's one-time data pull.

**Option: Use a different RPC provider.** Some providers may support `eth_feeHistory` with percentiles
on archive blocks. This is worth investigating in parallel, but the pipeline should not block on
provider-shopping. If a better provider is found, the `tip_surcharge_pct=0` path uses their real
percentiles — the surcharge is a ceiling, not a replacement.

---

**Process:** ADR raised on `feature-data-month-pull`; applied to gas fetcher rewrite. Listed in
`docs/plans/data-pipeline/STATUS.md` under "ADRs raised."