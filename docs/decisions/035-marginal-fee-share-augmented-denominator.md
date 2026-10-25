# ADR 035 — Per-segment marginal fee share on the augmented liquidity denominator

**Status:** accepted (implementation)
**Affects:** `src/undertow/sim/core/pool.py` · `src/undertow/sim/backtest/engine.py` · `src/undertow/data/transforms/feegrowth.py` · `src/undertow/data/feegrowth_api.py` · sim `CONTRACTS.md` §6 · ADR-013, ADR-019, ADR-032, ADR-034

**Raised by:** RL collapse diagnostics — the fee ledger paid the agent more than the pool took

## Context

Both fee engines distributed fees through the protocol growth mechanism:
`fee_growth_global += fee_segment / L_external_segment`, and a position's
entitlement was `L_pos × Δfee_growth_inside`. The implied share is therefore
`L_pos / L_external_segment`, which has **no upper bound**: when a thin tick
segment's external liquidity is smaller than the agent's `L`, the entitlement
can exceed the fee the pool actually took. The documented marginal model
(`PLAN.md` §7) is the bounded share `L_agent / (L_agent + L_pool_active)`.

Measured on the 0.05 % run `results/eth_usdc_500_full` (seed 0, episode 1, the
2022-11-08 swap): the pool's step fee was `$10,124` (token0) plus `6.56 WETH`
(`≈ $18,852` total), while the agent was credited **`$2,138,947` — 113× the
whole pool take** on a single step. The outlier was one step per episode, but
it dominated the reward: it produced the `value_loss` spikes (4.3e4, 2.7e5),
the `explained_variance` swings, and the degenerate churn the RL learned. The
same formula runs in the backtester's exact replay, so the ledger was affected
too.

## Decision

Accrue each marginal position per **swap segment** on the **augmented**
denominator:

```
fee_pos = lp_fee_segment * L_pos / (L_pos + L_external_segment)
```

integer-floored per segment in the exact tracker, float64 in the simulator.
Implemented in both engines:

- `PoolEngine.step` writes a per-position `position_accrued` ledger during the
  segment walk; `position_fees`/`close_position` return it (raw token units).
- `FeeGrowthTracker` gains a `marginal_position` ledger and
  `register_marginal_position`; `apply_swap` credits it per segment and
  `accrue` returns the ledger delta for the registered bounds (still returning
  the inside snapshots for callers that persist them).
  `FeeGrowthReplay.register_position(..., liquidity=…)` wires it, and the
  backtester passes each deployed/rebalanced position's liquidity.
- `state.fee_growth_global_*` is **unchanged** (the external-`L` protocol
  accumulator), so T10 reconciliation against on-chain `feeGrowthGlobal`
  remains exact.

## Consequences

- Fee numbers change wherever a position's `L` is not negligible against the
  segment's external `L`; every backtest and thesis result must be regenerated.
- The env/tracker parity bound is now one integer floor per accrued segment,
  not one per step; the parity tests allow two raw units for the one-tick
  crossing swaps.
- The S20 golden guard and the exact-replay goldens were re-captured; the
  observation goldens are unchanged where no fee accrues.
- `Position.uncollected_fees` / `fee_growth_inside_last` remain the pool-level
  growth accessor but are no longer the engine's fee source.
- Regression tests: the tracker's augmented share when the agent dominates a
  thin segment (`tests/data/test_feegrowth.py`), the env/tracker parity suite,
  and the updated engine unit tests.

## Rejected alternatives

- **End-of-step share cap.** Bounded, but it loses segment attribution and can
  credit (or deny) fees for segments the position was never in; it also leaves
  the cumulative level inconsistent.
- **Including `L_agent` in `fee_growth_global`.** It would fix the share by
  construction but breaks the tracker's exact mirror of on-chain
  `feeGrowthGlobal` and its reconciliation.
- **Clamping only in `position_fees`.** The level, not just the delta, feeds
  equity and PnL; a clamped delta on top of a polluted level would still
  distort the ledger.
