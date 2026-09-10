# S07 — Pool engine (tick lattice, agent fee accrual)

**Wave 3 · size L · depends on: S02, S04 · blocks: S12, S14**  
**Branch:** `feature-sim-pool-engine`

## Why this task exists

The pool engine is the core of the AMM simulation. It models the Uniswap V3 tick-lattice fee accrual
mechanism (§5.4 of the lesson plan, §10.2.4 of the roadmap): a sparse tick lattice where each
initialized tick stores `fee_growth_outside` snapshots, global fee growth accumulates from swap
volume, and per-position fees are computed via the `fee_growth_inside` formula. This is the
simulator's float64 fast path — the backtester (S11) uses the data module's exact-integer
`FeeGrowthTracker` instead, and S14 quantifies the drift between the two.

This is a large task because the tick-crossing logic is subtle and must match Uniswap V3's
specification.

## Files you own

```
src/undertow/sim/core/pool.py              (TickState, PoolState, PoolEngine)
tests/sim/test_pool.py
tests/sim/conftest.py                      (appends tiny_pool_engine fixture)
```

**Note:** `core/__init__.py` was created by S04. Do not overwrite it; add `PoolEngine` to its
exports line.

## What to build

### 1. `core/pool.py` — CONTRACTS.md §6

**`TickState`** dataclass (mutable, slots):
- `fee_growth_outside_0: float` — Q128 as float64
- `fee_growth_outside_1: float`
- `liquidity_gross: float` — total L locked at this tick
- `liquidity_net: float` — signed: + for lower ticks, − for upper ticks
- `initialized: bool`

**`PoolState`** dataclass (mutable, slots):
- `sqrt_price: SqrtPrice`
- `tick: Tick`
- `liquidity: float` — active L at current tick
- `fee_growth_global_0: float`
- `fee_growth_global_1: float`
- `fee_tier_bps: int`
- `tick_spacing: int`

Methods:
- `fee_growth_below(tick_boundary, fee_growth_outside)` → `float`: fee growth below a tick boundary.
  If `current_tick >= tick_boundary`, `fee_growth_below = fee_growth_outside`;
  else `fee_growth_below = fee_growth_global − fee_growth_outside`
- `fee_growth_above(tick_boundary, fee_growth_outside)` → `float`: if `current_tick < tick_boundary`,
  `fee_growth_above = fee_growth_outside`; else
  `fee_growth_above = fee_growth_global − fee_growth_outside`
- `fee_growth_inside(tick_lower, tick_upper, fg_outside_lower_0, fg_outside_lower_1,
  fg_outside_upper_0, fg_outside_upper_1)` → `tuple[float, float]`:
  `fg_inside = fg_global − fg_below(lower) − fg_above(upper)`
- `tick_cross(new_tick)`: cross ticks from current to new_tick, updating `liquidity` by
  adding/subtracting `liquidity_net` at each crossed tick, and flipping
  `fee_growth_outside = fee_growth_global − fee_growth_outside` at each crossed tick boundary.
  This is the expensive operation — only called when the price actually crosses a tick.

**`PoolEngine`** dataclass:
- `state: PoolState`
- `ticks: dict[Tick, TickState]` — sparse, only initialized ticks
- `positions: dict[int, Position]` — registered positions keyed by position_id
- `position_fee_snapshots: dict[int, tuple[float, float]]` — per-position `(fg_inside_last_0, fg_inside_last_1)`

Methods:
- `open_position(tick_lower, tick_upper, liquidity)` → `int`:
  1. Snap `tick_lower` and `tick_upper` to the nearest tick-spacing multiple
  2. Validate: `tick_lower < tick_upper`, `liquidity > 0`, `tick_lower % tick_spacing == 0`,
     `tick_upper % tick_spacing == 0`
  3. Initialize the lower and upper ticks in the lattice if they don't exist:
     - Create `TickState` for each, setting `liquidity_gross += L`,
       `liquidity_net += L` (lower) / `liquidity_net −= L` (upper)
     - Initialize `fee_growth_outside` for new ticks: if the tick is at or below current tick
       for a lower tick, `fg_outside = fg_global`; else 0. For upper ticks, if current tick is
       at or above, `fg_outside = fg_global`; else 0.
  4. If the new position straddles the current tick, add L to `state.liquidity`
  5. Snapshot the current `fee_growth_inside` for the position
  6. Return a new unique `position_id`

- `close_position(position_id)` → `tuple[float, float]`:
  1. Compute final fees via `position_fees(position_id)`
  2. Remove L from the tick lattice: decrement `liquidity_gross` at both ticks,
     adjust `liquidity_net` (+L at lower becomes −L, −L at upper becomes +L)
  3. If the position straddled the current tick, subtract L from `state.liquidity`
  4. Clean up uninitialized ticks (set `initialized = False` if `liquidity_gross` becomes 0)
  5. Remove the position and its fee snapshot
  6. Return the final accrued fees

- `position_fees(position_id)` → `tuple[float, float]`:
  1. Get the position's tick bounds and fee snapshot
  2. Compute current `fee_growth_inside` from the lattice
  3. `fees_0 = L * (fg_inside_0_current − fg_inside_0_snapshot)`
  4. `fees_1 = L * (fg_inside_1_current − fg_inside_1_snapshot)` (token1 fees in token1 units)
  5. Return `(fees_0, fees_1)`

- `step(new_sqrt_price, new_tick, swap_volume_0=0.0, swap_volume_1=0.0)` → `dict`:
  1. If `new_tick != state.tick`, call `state.tick_cross(new_tick)` and update
     `state.sqrt_price = new_sqrt_price`, `state.tick = new_tick`
  2. Compute fee from swap volume: `fee_0 = swap_volume_0 * fee_tier_bps / 10000`,
     `fee_1 = swap_volume_1 * fee_tier_bps / 10000`
  3. If `state.liquidity > 0`, accumulate global fee growth:
     `fg_global_0 += fee_0 / state.liquidity`, `fg_global_1 += fee_1 / state.liquidity`
  4. Compute per-position accrued fees for all positions
  5. Return dict with `fees_accrued`, `ticks_crossed`, `active_liquidity`,
     `pool_fee_earned_0`, `pool_fee_earned_1`

### 2. Fee-growth formula (exact specification)

The Uniswap V3 fee-growth-inside formula (CONTRACTS.md §6):

```
fg_below(tick_i) = fg_outside_i     if current_tick >= tick_i
                 = fg_global − fg_outside_i  otherwise

fg_above(tick_i) = fg_outside_i     if current_tick < tick_i
                 = fg_global − fg_outside_i  otherwise

fg_inside(tick_lower, tick_upper) = fg_global − fg_below(tick_lower) − fg_above(tick_upper)
```

When crossing a tick boundary TO the right (price rising):
- At the crossed tick (which is a lower tick for some position): `liquidity += liquidity_net` at
  that tick
- `fg_outside = fg_global − fg_outside` (flip the outside to reflect the new side)

When crossing TO the left (price falling): same flip, liquidity adjustment in the opposite
direction.

### 3. `tests/sim/conftest.py` — `tiny_pool_engine` fixture

Append a `tiny_pool_engine` fixture:
- A `PoolEngine` initialized at `sqrt_price = sqrt(3000)`, tick ≈ 196242
- ~20 ticks initialized spanning tick −600 to +600 (≈ ±6% around entry)
- One position open at tick_lower = −120, tick_upper = +120 with L from `entry_position`
- Fee tier 3000 (30 bps), tick spacing 60

## Tests you must write

1. `TickState` default values: `fee_growth_outside_0 = 0.0`, `liquidity_gross = 0.0`,
   `initialized = False`
2. Opening a position initializes its lower and upper ticks in the lattice
3. Opening an in-range position adds liquidity to `state.liquidity`
4. Opening an out-of-range position (both ticks above current) does NOT add to active liquidity
5. `PoolState.fee_growth_below` returns `fg_outside` when `current_tick >= tick_boundary`
6. `PoolState.fee_growth_below` returns `fg_global - fg_outside` when `current_tick < tick_boundary`
7. `PoolState.fee_growth_above` returns `fg_outside` when `current_tick < tick_boundary`
8. `PoolState.fee_growth_inside` round-trips: a position's fees computed via `fg_inside` match
   manual computation from swap volume * fee tier * (L_active / L_total)
9. `PoolEngine.step()` with no price change and zero volume → no fee accrual, no ticks crossed
10. `PoolEngine.step()` with swap volume but no price change → global fee growth increases,
    in-range positions accrue fees
11. `PoolEngine.step()` crossing one tick boundary → tick crossed appears in result,
    liquidity changes correctly
12. `PoolEngine.step()` crossing multiple ticks (large price move) → all intermediate ticks crossed,
    fees accrue correctly at EACH crossed tick (test with 2 positions at different ranges)
13. `close_position()` returns correct accrued fees
14. `close_position()` removes the position and cleans up the lattice
15. `close_position()` on a position that accrued fees returns non-zero fees
16. `position_fees()` returns (0, 0) immediately after opening (no fee growth change)
17. Opening a position on top of an existing position's ticks increments `liquidity_gross` correctly
18. Closing all positions that share a tick leaves the tick uninitialized (`initialized = False`,
    `liquidity_gross = 0`)
19. **Golden: sqrt_price_x96 at ETH=$3000** → `1446501726624926496477173928747177` (CONTRACTS.md
    §19). Test the conversion but note the pool engine works in float64, not Q128 int — record
    the float64 value and its difference from the exact int as a comment (this becomes input to S14)
20. `tiny_pool_engine` fixture is importable and has `len(pool.ticks) >= 20`

## Definition of done

- All files exist, typed, no TODO stubs
- `uv run pytest tests/sim/test_pool.py -q` green
- `code-reviewer` APPROVE
- PR open against `main` from `feature-sim-pool-engine`
- `STATUS.md` updated