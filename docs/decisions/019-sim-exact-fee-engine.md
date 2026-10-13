# ADR 019 — The simulator's `PoolEngine` is the float64 twin of the exact external-pool fee engine

**Status:** accepted (implementation)
**Affects:** `src/undertow/sim/core/pool.py` · `src/undertow/sim/env/lp_env.py` · `src/undertow/sim/evaluate/protocol.py` · `src/undertow/sim/backtest/engine.py` · `src/undertow/data/__init__.py` · `tests/sim/test_pool.py` · `tests/sim/test_pool_exact_parity.py` · `tests/sim/test_backtester.py` · `tests/sim/test_env.py` · `tests/sim/test_validation.py` · sim `CONTRACTS.md` §6/§7/§12 · S07, S12, S14
**Raised by:** S14 real-data parity follow-up

## Context

S07 (`CONTRACTS.md` §6) specified a *deliberate* discrete simulator model:

> "The step model is discrete: ticks are crossed first and swap fees are then accrued over the
> resulting active liquidity, so a position crossed out during a step earns nothing on that step's
> volume (matching the minute-bar simulator, not the backtester's per-segment exact path; S14
> quantifies the drift)."

`state.liquidity` was the pool's `base + Σ agent L`, and a swap charged `volume * fee_tier_bps / 1e6`
over that single post-crossing liquidity, on **both** raw amount legs.  S12 wired the training
environment around that model, and S14 was created precisely to measure the drift between the float64
simulator and the backtester's exact integer `FeeGrowthTracker` (ADR-013).  The stated S14 goal is
that, fed identical inputs, the two agree to a stated tolerance.

The real-data S14 parity check **fails**: the mean per-step fee drift is ~`0.16` USDC/step (tolerance
`1e-3`).  The divergence is structural, not a rounding residual.

## Problem

The four causes, in order of size:

1. **Both legs were charged.**  `LpEnvironment._event_volumes` returned `abs(amount0)` **and**
   `abs(amount1)`, so a swap was charged on its input *and* output leg.  Uniswap and the exact tracker
   charge only the positive (input) leg (`CONTRACTS.md` §4.1), roughly doubling every fee.
2. **The agent's `L` inflated the fee denominator.**  `state.liquidity` included the agent's own `L`
   (base + agent) while the backtester's `FeeGrowthReplay.register_position` never mints the agent, so
   the tracker distributed fee growth over **external liquidity only**.  Measured first-order effect
   ~2.3%.
3. **Single-segment accrual.**  The sim accrued the whole swap fee over the post-crossing liquidity;
   the tracker splits the gross input across every crossed-tick segment, allocating it in proportion
   to the per-segment net inputs (last segment absorbing the remainder) and crossing each boundary
   *after* its segment's fee (`FeeGrowthTracker.apply_swap`).
4. **No external lattice.**  The external tick lattice was never seeded, so `state.liquidity` started
   from one tape value and the tick boundaries a swap could cross were absent.

There is also a fifth, latent cause that only appears once 3 and 4 are fixed: the environment's
zero-volume **alignment** step moved the pool onto the reference bar's price, which can land exactly
on an agent position's boundary and flip `fee_growth_outside` in a way the tracker (which is driven by
the tape alone) never does.  On the pinned eval window this produced spurious
`~7e5` USDC fee jumps.  And a swap's recorded `tick` can lag its `sqrt_price_x96` by one at an exact
boundary, so a tick-based crossing predicate disagrees with the tracker's price-based one.

## Decision

Make `PoolEngine` a faithful float64 replica of `FeeGrowthTracker`, and wire `LpEnvironment` to seed
it from the on-chain boundary snapshot (ADR-018).

1. **Input-leg fees.**  `_event_volumes` returns only the positive leg; the output leg is `0.0`.
2. **Marginal agent positions.**  `open_position` seeds each boundary tick (only when absent) with the
   protocol seeding convention and the sentinel `liquidity_gross=1, liquidity_net=0` of
   `FeeGrowthReplay.register_position`, and does **not** add the agent's `L` to `state.liquidity`.
   `close_position` neither touches `state.liquidity` nor deletes the sentinel ticks (the backtester
   leaves them too).  `position_fees` is unchanged: `L_agent · Δfee_growth_inside`.
3. **Piecewise `step`.**  Crossed ticks are selected and the swap is split into segments exactly like
   `apply_swap`: proportional gross allocation by per-segment net input
   (`token0: L·(1/a − 1/b)`, `token1: L·(b − a)`), `fee_seg = gross_seg · fee_pips / 1e6`,
   `fee_growth_global += fee_seg / L_seg`, then the boundary is crossed (update-then-cross).  The
   input leg's token receives the growth; the returned `pool_fee_earned_*` is the gross fee per token.
   Only segments with `L > 0` and `fee_seg > 0` accrue (mirroring `apply_swap`'s
   `if L > 0 and fee_seg > 0` guard), so a large swap that exhausts liquidity past the last
   initialized tick leaves a zero-liquidity final segment without raising `ZeroDivisionError`.
   **Protocol fees are split before distribution**: `PoolState.fee_protocol` carries the packed
   uint8 `(fp0 | fp1 << 4)` and the step applies `lp_fee = fee_seg - fee_seg // fp_per_token`, exactly
   as the tracker does (the repo dataset has `fee_protocol == 0` today, so this is future-proofing).
   `step` with no positive leg is a zero-volume alignment step and still crosses ticks (the price move
   itself changes active liquidity).
4. **Price-derived crossing direction.**  `upward` is computed from the raw prices, never the recorded
   tick: `int(post_x96) > int(pre_x96)` when exact Q64.96 is available, else `post_price > pre_price`,
   mirroring `apply_swap` (`feegrowth.py:337`).  The recorded `tick` can equal `pre_tick` while the
   price crosses a boundary, so a tick-derived direction would apply `liquidity_net` with the wrong
   sign and silently corrupt `state.liquidity`.
5. **Exact crossing predicate.**  When the caller supplies the row's raw `sqrt_price_x96`
   (`step(..., new_sqrt_price_x96=...)`), the crossed set is compared on exact Q64.96 integers
   (`_crossed_ticks_exact`); otherwise it falls back to the float `calc_sqrt_price_a` comparison used
   by fixtures.  A per-tick `tick_to_sqrt_price_x96` cache keeps resets ~O(1) per crossing.
6. **External seeding + warmup.**  `PoolEngine.from_snapshot(state, rows)` builds the lattice and
   `PoolState` from a `FEE_GROWTH_SCHEMA` snapshot (globals/cas in float, Q128 divided by `Q128`,
   `fee_protocol` read from the global row).
   `LpEnvironment.reset` slices the episode window, re-seeds from `sub.pool_state`, replays every
   `sub.warmup` row (`swap` via `step`, `mint`/`burn` via `apply_liquidity_event`), and captures the
   result as the per-episode external base.  `mint`/`burn` rows in the window are applied through
   `apply_liquidity_event`, mirroring the tracker's tick update + active-liquidity delta.
7. **No reference alignment on tape-driven views.**  When the view is backed by a boundary snapshot,
   the pool's fee-accrual state is driven by the tape alone (like the tracker); the reference bar's
   `sqrt_price`/`tick` are used for valuation only.  Hand-built fixture views (no snapshot) keep the
   old zero-volume alignment, because the shared `tiny_market_view` holds a constant tape price while
   its reference feed moves.
8. **Valuation price.**  For a seeded view the position is marked at the **last tape event's**
   `(raw sqrt_price, as-of reference close)` pair — exactly the pair the backtester's
   `_resolve_market` returns for that event.  Marking instead at the decision bar's close while
   using the last swap's raw sqrt pairs two different timestamps, which injects a one-bar
   mark-to-market error into IL/PnL (and into the WETH fee leg); on the pinned eval window that
   residual had a per-step std of ~`118` USDC.  Fixture views (no snapshot) keep the bar-close
   marking, because their engine is aligned onto the reference bar.

`PoolState` gains two additive fields, `sqrt_price_x96: int | None = None` (exact price of the last
swap) and `fee_protocol: int = 0`, and `PoolEngine.step` gains the keyword-only
`new_sqrt_price_x96: int | None = None`.

## Consequences

- **Parity.**  With the exact engine, the external snapshot/warmup, the last-event valuation
  above, **and the backtester seeding the synthetic position's inside snapshots at deploy time
  (before its first event)**, the pinned `eth_usdc_3000` eval split agrees per step: IL drift is
  exactly zero on every seed (`il_drift_mean == il_drift_std == 0.0`) and the PnL drift equals the
  fee drift.  The residual is the tracker's integer raw-unit truncation
  (`(L · ΔfeeGrowthInside) >> 128` versus the engine's float64 accumulation), now measured at
  `~1e-7` USDC/step.  It is no longer `~2e-6`: that earlier figure was dominated by the dropped
  first tape event, not by truncation.  `run_parity_check` over the default 5 seeds reports, per
  seed (fee = IL = PnL mean): `0 → 8.87e-8`, `1 → 1.43e-7`, `2 → 1.01e-7`, `3 → 1.08e-7`,
  `4 → 2.42e-8`; pooled fee mean `9.30e-8` (std `5.58e-7`).  **Every seed is individually inside
  the `1e-3` tolerance**, so the pass is robust rather than a pooled-averaging artifact.  Before
  the fix the per-seed fee means were `[2.13e-6, 9.10e-7, 1.14e-3, 2.86e-3, 9.01e-5]` — seeds 2
  and 3 exceeded tolerance, while the pooled `8.18e-4` masked it.
- **`CONTRACTS.md` §6 and the S07/S12 test contract must be updated**: the documented
  "single-segment, base+agent" model is superseded.  The S07 tests in `tests/sim/test_pool.py` were
  rewritten for marginal positions and price-based crossing; `tests/sim/test_pool_exact_parity.py` is
  new and pins `PoolEngine` against `FeeGrowthReplay` on a synthetic tape to < 1 raw token unit.
- **Fixture caveat.**  `tiny_market_view` holds a constant pool `sqrt_price_x96`, so under the exact
  engine it accrues no fee; the `tiny_env` fee-mechanics tests were updated accordingly and point at
  `test_pool_exact_parity.py` for genuine accrual.  A fixture with a *moving* price must use direction-
  consistent amounts (`token0` in ⇒ tick down), which the tiny tape does not.
- **Performance.**  Resets replay up to a lane of warmup rows; each swap scans the lattice.  The
  `tick_to_sqrt_price_x96` memoisation brought a real reset from ~109 s back to ~7 s.  Training still
  pays this per episode window (ADR-018 already flagged warmup cost).
- **The thesis text in `~/Documents/fyp/lesson_plan/` describes the S07/S12 discrete model and needs
  correcting** — flagged to the human, never edited from a code task.
- **Rejected alternatives.**
  - *Keep the discrete model and raise the S14 tolerance.*  The drift is structural (2–3× the fee,
    not a rounding residual); tolerating it would make `Δ = sim_PnL − onchain_PnL` meaningless.
  - *Keep tick-based crossing.*  It disagrees with the tracker's price-based rule at the exact
    boundary whose `tick` lags its `sqrt_price_x96`, and silently mis-crosses liquidity.
  - *Keep the reference alignment for seeded views.*  It corrupts `fee_growth_outside` (the tracker
    never aligns); measured as `~7e5` USDC fee jumps.
  - *Add the exact-Q128 tracker as the simulator's engine.*  Forbidden by the data/sim package
    boundary (ADR-008/013) and defeats the float64 fast path; the float64 twin is fast enough once
    the boundary prices are memoised.
