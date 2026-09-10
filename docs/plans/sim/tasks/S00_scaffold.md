# S00 — sim scaffold

**Wave 0 · size L · depends on: nothing · blocks: S01–S04**
**Branch:** `feature-sim-scaffold`

## Why this task exists

The sim scaffold is the foundation for all subsequent sim tasks. Every later task — PPO runner,
backtester, baselines, analysis — needs the pool, price processes, env, and metrics to exist first.
This task delivers all six contracted modules in one pass, keeping them internally consistent.

## Files you own

```
src/undertow/sim/
├── __init__.py          public API surface with __all__
├── math.py              self-contained tick/liquidity/fixed-point math
├── config.py            frozen SimConfig dataclass tree + load_sim_config
├── pool.py              ConcentratedLiquidityPool state machine
├── price.py             GBMPrice, RegimeSwitchingPrice, ReplayPrice
├── env.py               UndertowEnv (gymnasium.Env)
└── metrics.py           Sharpe, Sortino, MDD, VaR/CVaR, annualized return/vol

tests/sim/
├── conftest.py           shared fixtures (rng)
├── test_math.py          tick round-trips, boundaries, alignment, liquidity math
├── test_config.py        defaults, validation, TOML load round-trip
├── test_pool.py          mint, swap, burn, collect, fee growth invariants
├── test_price.py         GBM, regime changes, jumps, replay end-of-series
├── test_env.py           reset, step, hold, termination, reproducibility, check_env
└── test_metrics.py       Sharpe, Sortino, MDD, VaR/CVaR

configs/
└── sim_default.toml      filled-in default SimConfig

docs/plans/sim/
├── README.md             plan overview
├── PLAN.md               this file
├── CONTRACTS.md          frozen cross-task interfaces (update to match as-built)
├── STATUS.md             mark S00 as done
└── tasks/S00_scaffold.md this file

pyproject.toml             [project.optional-dependencies].sim section
uv.lock                    regenerated
```

Do **not** import `undertow.data`. The scaffold test `test_sim_does_not_import_data` enforces this.

## What to build

### 1. `sim/math.py` — Fixed-point & tick math

Self-contained duplicate of `undertow.data.fixedpoint` functions the sim needs:
- Price/tick conversions: `sqrt_price_x96_to_price`, `price_to_sqrt_price_x96`, `tick_to_sqrt_price_x96`, `sqrt_price_x96_to_tick`, `tick_to_price`, `price_to_tick`
- Wrapping arithmetic: `wrapping_sub_256`, `wrapping_add_256`
- Q128 decimal: `q128_to_decimal`
- Liquidity ↔ amounts: `get_amount0_delta`, `get_amount1_delta`, `liquidity_for_amounts`, `amounts_for_liquidity`
- Tick alignment: `align_tick_down`, `align_tick_up`
- Constants: `Q96`, `Q128`, `MIN_TICK`, `MAX_TICK`, `MIN_SQRT_RATIO`, `MAX_SQRT_RATIO`

### 2. `sim/config.py` — SimConfig tree

Frozen dataclass hierarchy: `SimConfig` → `EpisodeConfig`, `ActionGrid`, `GasConfig`, `SlippageConfig`, `RewardConfig`, `TrainingConfig`, `PriceConfig`. Plus `load_sim_config(path: str) -> SimConfig` for TOML loading with list→tuple auto-conversion.

### 3. `sim/pool.py` — ConcentratedLiquidityPool

Exact-integer pool state machine: `swap`, `mint`, `burn`, `collect` with per-position fee-growth tracking via the Uniswap V3 fee-growth-inside/last mechanism. Positions are frozen dataclasses; fee uncollected tracking uses `object.__setattr__` (acceptable for scaffold; later tasks may extract mutable tracking).

### 4. `sim/price.py` — Price processes

Three pluggable processes via `PriceProcess` Protocol:
- `GBMPrice` — geometric Brownian motion
- `RegimeSwitchingPrice` — Markov-regime-switching jump-diffusion
- `ReplayPrice` — replay from a real price series

### 5. `sim/env.py` — UndertowEnv

Gymnasium `Env` wrapping pool + price process + friction. Discrete action space (action grid). Observation vector: `[log_price, portfolio_value_pct, in_range_flag, fee_rate, gas_price, step_frac]`. Reward: `fees − gas − IL − risk_penalty` with per-component ablation flags.

### 6. `sim/metrics.py` — Performance metrics

Pure numpy: `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `var_cvar`, `decomposition`, `annualized_return`, `annualized_volatility`.

## Critical design decisions

- **Tick ordering:** Higher price → lower tick (Uniswap V3 convention). `reset()` and `_rebalance()` must swap `price_low`/`price_high` when computing ticks.
- **Marginal agent:** Default is `True` — the agent's liquidity does not move the pool price. The pool sqrt price is set directly from the external price process.
- **Single-tick-range swap:** The pool's `swap()` does not cross tick boundaries. Acceptable for the marginal-agent scaffold because swap sizes are small.
- **Fallback range:** When tick alignment collapses a range (e.g. `tick_lower >= tick_upper`), `_rebalance` falls back to a minimum-width range around the current pool tick.

## Tests you must write

1. **Math:** tick round-trips, MIN/MAX boundaries, tick alignment (including negatives), wrapping arithmetic, liquidity ↔ amounts round-trips
2. **Config:** `EpisodeConfig` defaults and validation, `ActionGrid.decode`, `SimConfig` construction, `load_sim_config` from TOML (minimal, override, full round-trip)
3. **Pool:** mint with tick alignment, mint invalid ranges, swap zero-for-one and one-for-zero, fee growth accumulation after swap, collect fees, collect reset, burn returns principal, fee growth inside full-range position
4. **Price:** GBM basic/reset, regime switching (basic + regime changes + jumps), ReplayPrice (basic + end-of-series + invalid shape)
5. **Env:** reset, step, multiple steps, hold action, termination, observation space, reproducibility (same seed = same trajectory), `gymnasium.utils.env_checker.check_env`
6. **Metrics:** Sharpe (positive trend + flat + short series), Sortino, max drawdown (basic + no drawdown + empty), VaR/CVaR, annualized return (zero + positive), annualized volatility

## Acceptance criteria

- `uv sync` succeeds with `[project.optional-dependencies].sim` installed.
- `uv run pytest tests/sim/` green, all tests have meaningful assertions.
- `uv run pytest tests/test_scaffold.py::test_sim_does_not_import_data` green.
- `uv run ruff check src/undertow/sim/ tests/sim/` clean.
- `CONTRACTS.md` signatures match the as-built implementation.
- `configs/sim_default.toml` filled (not skeleton).
- PR open against `main`; `STATUS.md` updated.
- All reviewer findings addressed.

## Process (mandatory)

`git checkout -b feature-sim-scaffold` off `main` → implement → `uv run pytest` → launch
`code-reviewer` subagent → fix findings → commit → push → open PR against `main` → update `STATUS.md`.
Do not merge.