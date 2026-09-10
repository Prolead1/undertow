# S04 — Position & valuation math

**Wave 2 · size M · depends on: S01, S02 · blocks: S07, S10, S11, S12**  
**Branch:** `feature-sim-position-math`

## Why this task exists

Every component that values a position — the pool engine, baselines, backtester, and environment —
needs the concentrated-liquidity position math from lesson plan §5.3–§5.6. This task implements
equations (8)–(11) (token amounts, position value), impermanent loss vs HODL (§5.6), fee accrual,
and the `initial_deposit` factory. Getting these formulas right at the leaf level prevents cascading
errors through the whole sim.

The golden values from the worked examples in CONTRACTS.md §19 are your correctness proof.

## Files you own

```
src/undertow/sim/core/__init__.py         (empty, re-exports from position)
src/undertow/sim/core/position.py          (Position, calc_sqrt_price, initial_deposit,
                                            position_from_amounts)
tests/sim/test_position.py
tests/sim/conftest.py                      (appends entry_position fixture)
```

## What to build

### 1. `core/position.py` — CONTRACTS.md §5

Implement everything from CONTRACTS.md §5:

- `calc_sqrt_price_a(tick)` and `calc_sqrt_price_b(tick)` — both compute `sqrt(1.0001^tick)`. Two
  names for readability at call sites (lower vs upper tick boundary)

- `Position` dataclass (frozen, slots):
  - Fields: `tick_lower: Tick`, `tick_upper: Tick`, `liquidity: float`, `tick_spacing: TickSpacing`,
    `fee_growth_inside_last_0: float = 0.0`, `fee_growth_inside_last_1: float = 0.0`
  - `__post_init__`: validate `tick_lower < tick_upper`, `liquidity > 0`
  - `amount0(sqrt_price)` → `float`: token0 holdings. Returns 0 if the position has no token0
    at this price (out of range on the token0 side — i.e., current tick >= tick_upper)
  - `amount1(sqrt_price)` → `float`: token1 holdings. Returns 0 if current tick <= tick_lower
  - `value(sqrt_price, price)` → `Wealth`: `amount0 + amount1 * price`
  - `hodl_value(entry_sqrt_price, entry_price, current_price)` → `Wealth`: what the initial deposit
    would be worth if held
  - `il_vs_hodl(entry_sqrt_price, entry_price, current_sqrt_price, current_price)` → `float`:
    `V(current) − HODL(current)`. Negative → loss
  - `il_fraction(entry_sqrt_price, entry_price, current_sqrt_price, current_price)` → `float`:
    `(V − HODL) / HODL`
  - `uncollected_fees(fee_growth_inside_0, fee_growth_inside_1)` → `tuple[float, float]`:
    `L * (current_fee_growth_inside − last_snapshot)`
  - `snapshot_fees(fee_growth_inside_0, fee_growth_inside_1)`: update the snapshot to current
  - `in_range(sqrt_price)`, `below_range(sqrt_price)`, `above_range(sqrt_price)` → `bool`

- `initial_deposit(sqrt_price, price, tick_lower, tick_upper, capital, tick_spacing)` → `Position`:
  Solve for L such that `V(sqrt_price, price) == capital`. This is equation (11) inverted.
  The position starts with fee-growth snapshots at 0.

- `position_from_amounts(tick_lower, tick_upper, amount0, amount1, sqrt_price, tick_spacing)` →
  `Position`: Given raw token amounts, compute L (equations 8–9 inverted). Useful for
  reconstructing positions from on-chain data.

### 2. Math details

The token amount formulas (eqs 8–9 of §5.3):

```
amount0 = L * (1/sqrt_price - 1/sqrt_price_upper)    when tick_lower <= current_tick < tick_upper
amount1 = L * (sqrt_price - sqrt_price_lower)

amount0 = L * (1/sqrt_price_lower - 1/sqrt_price_upper)   when current_tick < tick_lower
amount1 = 0                                                (all in token0)

amount0 = 0                                                when current_tick >= tick_upper
amount1 = L * (sqrt_price_upper - sqrt_price_lower)        (all in token1)
```

For `initial_deposit`, solve for L from the value equation. When in-range:
```
V = L * (1/sqrt_price - 1/sqrt_price_upper) + L * (sqrt_price - sqrt_price_lower) * price
capital = V
→ L = capital / [(1/sqrt_price - 1/sqrt_price_upper) + (sqrt_price - sqrt_price_lower) * price]
```

### 3. `tests/sim/conftest.py` — `entry_position` fixture

Append an `entry_position` fixture:
- A `Position` at `tick_lower = -120`, `tick_upper = +120` (≈ ±1.2% around entry)
- Entry price = $3000 (tick ≈ 196242)
- `tick_spacing = 60`
- Capital = 100,000 USDC
- Built via `initial_deposit()` so L is computed correctly

## Tests you must write

1. **Golden: V2 IL at r=1.2** → −0.00454 (−0.45%). Use full-range position (MIN_TICK, MAX_TICK)
2. **Golden: V2 IL at r=0.5** → −0.0572 (−5.7%)
3. **Golden: Concentrated ±10% IL at r=1.2** → −0.066 (−6.6%)
4. **Golden: Concentrated ±10% IL at r=0.5** → −0.325 (−32.5%)
5. **Golden: Tick 196242 → price** ≈ ~3004 USDC/WETH (verify `calc_sqrt_price` round-trips with
   the data module's `tick_to_price`)
6. `amount0` returns 0 when current tick >= tick_upper
7. `amount1` returns 0 when current tick <= tick_lower
8. `amount0 + amount1 * price` equals `value()` when in-range (round-trip)
9. `initial_deposit` creates a position whose `value()` at the entry price equals the capital
   (within float64 tolerance)
10. `hodl_value` at entry = `value` at entry = capital
11. `il_vs_hodl` at entry = 0
12. `uncollected_fees` returns (0, 0) when fee growth hasn't changed
13. `snapshot_fees` followed by unchanged fee growth → `uncollected_fees` = (0, 0)
14. `uncollected_fees` after simulated fee growth matches `L * delta_fg`
15. `Position(tick_lower=100, tick_upper=50)` raises `PositionError`
16. `Position(..., liquidity=0)` raises `PositionError`
17. `position_from_amounts` round-trips with `amount0`/`amount1` at the same price
18. `entry_position` fixture is importable and valued at ~100,000 USDC

## Definition of done

- All files exist, typed, no TODO stubs
- All golden-value tests pass
- `uv run pytest tests/sim/test_position.py -q` green
- `code-reviewer` APPROVE
- PR open against `main` from `feature-sim-position-math`
- `STATUS.md` updated