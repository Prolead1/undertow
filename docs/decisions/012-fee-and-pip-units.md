# ADR 012 — Fee tier units: `fee_tier_bps` holds Uniswap pips (1e-6), fee denominator is 1e6

**Status:** accepted
**Affects:** sim `CONTRACTS.md` §2/§6/§8 · tasks S07, S08, S09, S10, S15
**Raised by:** S07 (`feature-sim-pool-engine`)

## Context

`CONTRACTS.md` §6 specifies the pool engine's fee field as `fee_tier_bps: int  # e.g. 3000 for
0.30%`, and §8 specifies the slippage model's pool-fee term inside a single basis-point denominator:

```
Slippage = notional * (pool_fee_tier_bps + fixed_impact_bps) / 10000.      # §8 as written
```

The data module and the protocol disagree with that denominator. Uniswap V3's on-chain `fee` field is
denominated in **hundredths of a bip** (pips, `1e-6`): the 0.30% WETH/USDC tier is `3000` pips =
30 bps = `0.003`. The sim's `EpisodeConfig.fee_tier_bps` default of `3000`, `PoolState.fee_tier_bps`,
`Observation.pool_fee_tier_bps` and `data.FeeTier.BPS_30 = 3000` all carry this pip value; the `_bps`
name is a legacy misnomer preserved for interface stability.

The data module's exact reference path is explicit. `undertow.data.transforms.feegrowth` computes
`fee_amount = amount_in * fee_pips // 1_000_000` (`FeeTier.BPS_30 = 3000`), and `FeeTier` values are
the raw pip integers. The data `CONTRACTS.md` §3.0 / `TICK_SPACING` table uses the same convention.

## Problem

Treating `fee_tier_bps = 3000` as true basis points and dividing by `10_000` charges 3000 bps = **30%**
on notional, a **100× over-charge** versus the intended 0.30%. Two concrete symptoms:

1. §6's fee accrual — `pool_fee = swap_volume * fee_tier_bps / 10000` — would make the S07 fee-growth
   engine 100× heavier than the `undertow.data` engine that the backtester (S11) and S14 parity check
   use, so the two numeric worlds would never agree and S14's drift metric would be meaningless.
2. §8's slippage — `notional * (pool_fee_tier_bps + fixed_impact_bps) / 10000` — mixes units: the pool
   fee term is pips (`1e-6`) while `fixed_impact_bps` is true basis points (`1e-4`). Dividing both by
   `10000` over-states the fee term 100× and under-states nothing on the impact term.

## Decision

**The fee tier is pips; the pool-fee denominator is `1_000_000`; the impact term keeps `10_000`.**

- `EpisodeConfig.fee_tier_bps` (and every `fee_tier_bps` / `pool_fee_tier_bps` field) holds the
  Uniswap pip value: `3000` = 0.30%, `500` = 0.05%, `10000` = 1.00%.
- **Pool fee** on a swap of `amount_in`:
  ```
  fee = amount_in * fee_tier_pips / 1_000_000
  ```
  This matches `undertow.data.transforms.feegrowth` (`fee_seg = gross_seg * fee_pips // 1_000_000`) and
  `FeeTier`. §6's `/10000` would over-charge 100×.
- **Slippage** for a rebalancing trade:
  ```
  slippage = notional * (pool_fee_tier_pips / 1_000_000 + fixed_impact_bps / 10_000)
  ```
  The pool-fee term is pips / `1e6`; the fixed-impact term is true bps / `1e4`. §8's single `/10000` is
  wrong for the fee term.
- The `_bps` field names stay as-is (renaming would break frozen interfaces across S07–S10); this ADR
  records the real unit. `CONTRACTS.md` §2 gets a one-line note next to the field, and §6/§8 are
  corrected as above.

## Consequences

- `src/undertow/sim/core/pool.py` already uses `_FEE_DENOMINATOR = 1_000_000.0`; its unit tests pin the
  fee formula and the tick-lattice behaviour. No further code change is needed in S07.
- **S08 (frictions)** must implement `slippage_cost_usdc` with the split denominator above, not §8's
  original `/10000`. S08's tests should assert `notional=100_000`, `fee_tier=3000`, `impact=5` gives
  `100_000 * (0.003 + 0.0005) = 350` USDC.
- **S09 (reward)** and **S10 (baselines)** carry `pool_fee_tier_bps` through observations and any
  fee-income estimate; the denominator is `1e6` in both. **S15 (evaluation runner)** inherits the same
  numbers in its backtest metrics.
- **S14** can compare the S07 float path against `undertow.data`'s exact `feegrowth` on a common fee
  base; that comparison is only meaningful under this ADR.
- `CONTRACTS.md` §2/§6/§8 are edited to the corrected units. The vault thesis text in
  `~/Documents/fyp/lesson_plan/` is **not** touched from this code task; if it quotes the `/10000`
  formula it should be flagged for the human to correct.
- Rejected: renaming `fee_tier_bps` → `fee_tier_pips`. The field appears in `EpisodeConfig`,
  `PoolState`, `Observation` (S12) and several type hints, and is frozen across tasks already in
  flight; a rename would silently break S08–S12 for no functional gain. The unit note plus this ADR is
  the lower-risk correction, consistent with the project's "add, don't rename" rule.
- Rejected: keeping `/10000` and redefining the field as true bps (i.e. `30` for 0.30%). That would
  break parity with the on-chain `fee` value, with `data.FeeTier`, and with the tape's recorded fee
  tier, forcing a conversion at every data boundary.
