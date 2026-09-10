# S08 — Friction models (gas + slippage)

**Wave 3 · size M · depends on: S01 (code) + S03 (data for gas stream) · blocks: S11, S12**  
**Branch:** `feature-sim-frictions`

## Why this task exists

The thesis's core claim is that friction matters. G1 in the problem statement (§9.2) is "existing
work uses zero or flat costs." This task implements realistic gas and slippage models so every
action in the sim and backtester pays an honest cost. The gas model consumes the tape's per-block
base fee + priority fee, converted at that block's ETH price. The slippage model approximates the
cost of rebalancing through the pool.

Ablation flags in `RewardConfig` (S01) control which cost terms are active — this task just
computes the costs correctly; S09 decides whether they appear in the reward.

## Files you own

```
src/undertow/sim/frictions/__init__.py     (re-exports GasModel, SlippageModel, concrete impls)
src/undertow/sim/frictions/gas.py          (GasModel protocol, GasCostCalculator, ReplayGasModel,
                                            FlatGasModel, SpikeGasModel)
src/undertow/sim/frictions/slippage.py     (SlippageModel protocol, ProportionalSlippageModel)
tests/sim/test_frictions.py
tests/sim/conftest.py                      (appends mock_gas_model, mock_slippage_model fixtures)
```

## What to build

### 1. `frictions/gas.py` — CONTRACTS.md §8

`GasModel` protocol (`runtime_checkable`):
- `gas_cost_usdc(action_type, block_number, base_fee_per_gas_wei, priority_fee_p50_wei,
  eth_usd_price)` → `float`: cost in USDC (always ≥ 0, returned as positive but subtracted from PnL)

`GasCostCalculator` (shared logic, not a model itself):
- Maps `action_type` string to gas units using `GasConfig` constants:
  - `"mint"` → `GAS_MINT`
  - `"burn"` → `GAS_BURN`
  - `"collect"` → `GAS_COLLECT`
  - `"rebalance_swap"` → `GAS_REBALANCE_SWAP`
  - `"rebalance"` → `GAS_BURN + GAS_MINT + GAS_REBALANCE_SWAP`
- Computes: `cost = (base_fee + priority_fee) * gas_units * eth_usd_price / 1e18`
- Raises `ValueError` for unknown action_type

**Concrete models:**

- `ReplayGasModel`: Uses the tape's actual per-block gas data. Wraps a `MarketView` for the gas
  feed lookup. `gas_cost_usdc` calls `market_view.gas_at_block(block_number)`, extracts
  `base_fee_per_gas` and `priority_fee_p50`, and computes cost.

- `FlatGasModel`: Returns a fixed cost per action type. Useful for controlled experiments.
  `__init__(self, gas_config: GasConfig, eth_usd_price: float = 3000.0, base_fee_gwei: float = 30.0,
  priority_fee_gwei: float = 1.0)`.

- `SpikeGasModel`: Returns the flat cost most of the time, but with a configurable probability of
  a gas spike (10× the base cost). Useful for stress-testing. `__init__(self, base_model: GasModel,
  spike_multiplier: float = 10.0, spike_probability: float = 0.05)`.

### 2. `frictions/slippage.py` — CONTRACTS.md §8

`SlippageModel` protocol (`runtime_checkable`):
- `slippage_cost_usdc(notional_usdc, pool_fee_tier_bps, fixed_impact_bps)` → `float`:
  cost in USDC (≥ 0, returned as positive)

`ProportionalSlippageModel`:
- `slippage_cost_usdc(notional_usdc, pool_fee_tier_bps, fixed_impact_bps)`:
  `return notional * (pool_fee_tier_bps / 10000 + fixed_impact_bps / 10000)`
  This is `pool_fee_tier_bps` (the 30 bps Uniswap fee on the swap) plus `fixed_impact_bps`
  (a small additional impact approximation). Both are basis points.

### 3. `tests/sim/conftest.py` — mock fixtures

Append:
- `mock_gas_model`: A `FlatGasModel` returning ~$5 per action type at $3000 ETH, 30 gwei base + 1
  gwei priority. Include explicit expected values for `"mint"`, `"rebalance"`, `"hold"` (hold costs
  nothing — it's not in the action_type map, return 0).
- `mock_slippage_model`: A `ProportionalSlippageModel` that applies
  `notional * (0.003 + 0.0005)` per the defaults.

## Tests you must write

1. `GasCostCalculator` for `"mint"` returns correct cost for known gas units and ETH price
2. `GasCostCalculator` for `"rebalance"` = burn + mint + swap cost
3. `GasCostCalculator` for unknown action_type raises `ValueError`
4. `ReplayGasModel` retrieves the correct gas data from `tiny_market_view` for a known block
5. `ReplayGasModel` returns `None` gas cost → 0.0 for a block without gas data (don't crash)
6. `FlatGasModel` returns the same cost for all blocks (deterministic)
7. `SpikeGasModel` with `spike_probability=1.0` and seed always returns the spike cost
8. `SpikeGasModel` with `spike_probability=0.0` returns the base cost
9. `ProportionalSlippageModel` returns `notional * (fee_tier_bps + impact_bps) / 10000`
10. `ProportionalSlippageModel` returns 0 for zero notional
11. Both protocols pass `isinstance(obj, GasModel)` / `isinstance(obj, SlippageModel)`
12. `mock_gas_model` and `mock_slippage_model` fixtures are importable
13. Gas cost is always ≥ 0 (never negative)
14. Slippage cost is always ≥ 0

## Definition of done

- All files exist, typed, no TODO stubs
- `uv run pytest tests/sim/test_frictions.py -q` green
- `code-reviewer` APPROVE
- PR open against `main` from `feature-sim-frictions`
- `STATUS.md` updated