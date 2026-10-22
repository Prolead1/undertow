# S18 — Rebalance cost = swapped imbalance, not position value

**Wave 6 · size S · depends on: S04, S08, S11, S12 · blocks: S20, S22, S23**
**Branch:** `bug-sim-rebalance-swap-cost`

## Why this task exists

Both rebalance call sites charge the pool fee + impact on the **whole position
value**:

- `src/undertow/sim/env/lp_env.py::_rebalance` (~line 1006):
  `notional = self._position.value(sqrt_price, price)` then
  `self.slippage_model.slippage_cost_usdc(notional, …)`.
- `src/undertow/sim/backtest/engine.py::_apply_rebalance` (~line 609):
  `notional = max(0.0, position_value)` then the same call.

A Uniswap V3 rebalance only swaps the **inventory imbalance** between the old
position's composition and the new range's target composition. A range centred on
the current price is ~50/50 by value (verified with the sim's own
`initial_deposit`: width 1 → 49 285/50 715, width 10 → 49 929/50 071), so the
swap is ≤ ~50 % of notional and → 0 for a balanced in-range recentre. The current
code overcharges ~2× in the worst case and arbitrarily more for small recentres,
which makes rebalancing look economically hopeless and directly distorts the
thesis's C1/C3 results.

The slippage model itself is correct (`CONTRACTS.md` §8, ADR-012) and **must not
change** — the defect is at the call sites. `PLAN.md` §7 already says
`S = notional × …` for the *rebalance trade*; "notional" was never meant to be
the position value.

## Files you own

```
src/undertow/sim/env/lp_env.py              (_rebalance only)
src/undertow/sim/backtest/engine.py         (_apply_rebalance only)
tests/sim/test_env.py                       (rebalance-cost tests)
tests/sim/test_backtester.py                (rebalance-cost tests)
docs/plans/sim/CONTRACTS.md                 (§8 wording only)
docs/plans/sim/PLAN.md                      (§7 slippage row wording only)
```

Do not touch `frictions/slippage.py`, the reward, or `reset()` (S17 owns that).

## What to build

### 1. The corrected notional

In both call sites, compute the token0-value imbalance between the pre-rebalance
position and the **target** range:

```python
spacing = TickSpacing(int(self.config.episode.tick_spacing))  # env
lower = action.lower_tick(current_tick, spacing)
upper = action.upper_tick(current_tick, spacing)
notional = old_position.value(sqrt_price, price)          # capital for the target
target = initial_deposit(sqrt_price, price, lower, upper, notional, spacing)
swap_notional = abs(old_position.amount0(sqrt_price) - target.amount0(sqrt_price))
slippage_cost = self.slippage_model.slippage_cost_usdc(
    swap_notional, fee_tier_bps, fixed_impact_bps
)
```

Notes:

- `token0` is USDC (the numeraire, ADR-009), so `amount0` is already in USDC and
  `swap_notional` needs no price conversion.
- Compute `lower`/`upper` **before** the slippage (the current backtester computes
  them after; the target range does not depend on `new_capital`, so reordering is
  safe). Then compute `new_capital = notional + collected − gas_cost −
  slippage_cost` and `initial_deposit(..., new_capital, ...)` as today.
- Do not recompute the target a second time for the final deposit if that would
  change rounding; a second `initial_deposit` with the post-cost capital is
  required (the deployed position must match today's behaviour).

### 2. Contract wording

- `CONTRACTS.md` §8, `slippage_cost_usdc` docstring: `notional_usdc` is the
  **swapped imbalance** (token0-value difference between the pre-rebalance
  position and the target range), not the position value.
- `PLAN.md` §7 "Slippage" row: same clarification.

### 3. Tests

Add tests that assert the *new* behaviour numerically (not smoke tests):

1. **Balanced in-range recentre → ~0 slippage.** Build a position whose current
   composition already matches a range centred on the current price (use
   `initial_deposit` with a centred range), rebalance to that range, assert the
   reported `slippage_cost` is `< 1e-6 × notional` (or exactly 0 within float
   tolerance). This is the case the old code got most wrong.
2. **Fully out-of-range → ~half-notional.** Position fully below/above its range
   (all token0 or all token1), recentre to a centred band; assert
   `swap_notional ≈ notional/2` within 2 % and the slippage is ~half the old
   full-notional value `notional × (3000/1e6 + 5/1e4)`.
3. **Env and backtester agree.** The same rebalance at the same price/state
   produces the same `slippage_cost` in `LpEnvironment._rebalance` and
   `BacktestEngine._apply_rebalance` (both should call the same formula). If a
   full state match is impractical in a unit test, assert the formula on both
   call sites with the same constructed `Position` pair.
4. Update any existing test that pinned the old full-notional slippage, with a
   comment explaining the change. `tests/sim/test_frictions.py::test_slippage_exact_formula`
   must **not** change (the model is unchanged).

## Acceptance criteria

- `uv run pytest tests/sim/test_env.py tests/sim/test_backtester.py tests/sim/test_frictions.py -q` green.
- `uv run pytest -q` green (full suite).
- No change to `frictions/slippage.py` or the reward.
- The two call sites use the identical formula (a shared helper is welcome if it
  does not create a new cross-module dependency; `initial_deposit` is already
  public in `undertow.sim.core.position`).

## Out of scope

- Routing the rebalance swap through `PoolEngine` (a faithful alternative
  rejected for now in ADR-030 — revisit if S22 parity shows the proportional
  model is insufficient).
- Window resampling (S17), the hedge leg (S20).
