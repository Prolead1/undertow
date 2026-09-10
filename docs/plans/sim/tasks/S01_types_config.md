# S01 — `types.py` + `config.py`

**Wave 1 · size M · depends on: S00 · blocks: S03–S16**
**Branch:** `feature-sim-types-config`

## Why this task exists

Every other task needs the shared type system and configuration tree. The enums, aliases,
exception hierarchy, and `SimConfig` dataclass tree are the vocabulary of the entire sim
package. Getting them right — and matching `CONTRACTS.md` §§1–2 exactly — means downstream
tasks can import and type-check against them from day one.

## Files you own

```
src/undertow/sim/types.py                  (enums, aliases, Action, exception hierarchy)
src/undertow/sim/config.py                 (SimConfig tree + load_sim_config)
tests/sim/test_types.py
tests/sim/test_config.py
configs/sim_default.toml                   (fills the S00 skeleton with real default values)
```

Also **edit** `configs/sim_default.toml` (created by S00 as a skeleton). You replace the
placeholder comments with actual TOML key-value pairs matching the `SimConfig` defaults.

## What to build

### 1. `types.py` — CONTRACTS.md §1

Implement exactly the types defined in CONTRACTS.md §1:

- Domain aliases: `Tick`, `TickSpacing`, `SqrtPrice`, `Price`, `Wealth`
- `Regime` enum (mirrors data module's labels but is a local copy — the sim must not depend on the
  data module's enum values changing)
- `PriceMode` enum (`replay`, `calibrated`)
- `Action` dataclass: `action_type: Literal["hold", "rebalance"]`, `center_offset: int`, `width: int`,
  with methods `is_hold()`, `lower_tick(current_tick, tick_spacing)`, `upper_tick(current_tick, tick_spacing)`
- Exception hierarchy: `UndertowSimError` → `SimConfigError`, `MarketViewError` (→ `LookAheadError`),
  `PositionError`, `EnvError`, `BacktestError`, `ParityError`

### 2. `config.py` — CONTRACTS.md §2

Implement the full `SimConfig` tree:

- `EpisodeConfig`: `duration_days=30`, `step_minutes=10`, `agent_capital_usdc=100_000.0`,
  `marginal_agent=True`, `fee_tier_bps=3000`, `tick_spacing=60`
- `ActionGridConfig`: `center_offsets=(-4,-3,-2,-1,0,1,2,3,4)`, `widths=(1,2,5,10,25,50)`,
  `include_hold=True`
- `GasConfig`: constants from the pinned values in CONTRACTS.md §2
- `SlippageConfig`: `fixed_impact_bps=5.0`
- `RewardConfig`: all six enable flags + `risk_penalty_lambda=0.0`, `risk_rolling_window_steps=144`,
  `normalize_by_capital=True`
- `SplitConfig`: pinned train/eval boundaries from PLAN.md §7
- `TrainingConfig`: PPO defaults from PLAN.md §7
- `BacktestConfig`
- Container `SimConfig` with `rng()` method
- `load_sim_config(path)` — TOML reader that constructs a `SimConfig`, overriding defaults

All dataclasses are `frozen=True, slots=True`.

### 3. `configs/sim_default.toml` — fill the skeleton

Replace the S00 placeholder comments with real TOML values. The `SimConfig` constructor with no
arguments must produce the same object as `load_sim_config("configs/sim_default.toml")`.

### 4. Constants

Gas constants live at module level in `config.py` (not inside a class):

```python
GAS_MINT: int = 460_000
GAS_BURN: int = 215_000
GAS_COLLECT: int = 130_000
GAS_REBALANCE_SWAP: int = 150_000
FIXED_IMPACT_BPS: float = 5.0
```

## Tests you must write

1. `Action.is_hold()` returns True for hold, False for rebalance
2. `Action.lower_tick()` / `upper_tick()` compute correct ticks for a given current_tick and
   tick_spacing (test at least 3 combinations)
3. `SimConfig()` with defaults validates (no exceptions)
4. `SimConfig.rng()` returns a `numpy.random.Generator`, and two calls with the same seed produce
   identical sequences
5. `load_sim_config` on `configs/sim_default.toml` produces a `SimConfig` identical to the default
   constructor
6. `load_sim_config` on a file with a non-existent key raises a clear error
7. Every exception class is raiseable and catchable as its parent (e.g. `LookAheadError` is caught
   by `except MarketViewError`)
8. Gas unit constants are positive integers
9. `SplitConfig` validates that `train_end_utc <= eval_start_utc` (via `__post_init__`)
10. `ActionGridConfig` validates no duplicate center_offsets or widths

## Definition of done

- All files exist, typed, no TODO stubs
- `uv run pytest tests/sim/ -q` green
- `code-reviewer` APPROVE
- PR open against `main` from `feature-sim-types-config`
- `STATUS.md` updated