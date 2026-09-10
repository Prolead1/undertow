# S12 — Gymnasium environment

**Wave 4 · size L · depends on: S03, S04, S06, S07, S08, S09, S10 · blocks: S13, S14**  
**Branch:** `feature-sim-gym-env`

## Why this task exists

The Gymnasium environment is the training loop's world. It wires together every wave-2/3 component —
prices, pool, frictions, reward — into the MDP defined in §6.2 of the lesson plan. The agent
observes the state, chooses an action from the discrete action space, and receives a reward.
This is the interface that PPO (S13) trains against.

This task also owns the `Observation` dataclass and the observation builder, which translates
raw pool/market state into the feature vector the policy sees.

## Files you own

```
src/undertow/sim/env/observations.py       (Observation dataclass + ObservationBuilder)
src/undertow/sim/env/lp_env.py             (LpEnvironment gym.Env)
tests/sim/test_env.py
tests/sim/conftest.py                      (appends tiny_env fixture)
```

**Note:** `env/__init__.py` was created by S09. Add `Observation` and `LpEnvironment` to its
exports. Do not overwrite.

## What to build

### 1. `env/observations.py` — CONTRACTS.md §11

**`Observation`** dataclass (frozen, slots) with exactly these fields:

- `price: Price`, `sqrt_price: SqrtPrice`, `tick: Tick`
- `price_returns_1h: float` (log return over last 6 bars at 10-min cadence)
- `price_returns_24h: float` (log return over last 144 bars)
- `realized_vol_24h: float` (annualized σ over last 144 bars)
- `position_in_range: bool`
- `position_value: Wealth`
- `uncollected_fees_token0: float`, `uncollected_fees_token1: float`
- `il_vs_hodl: float` (as fraction of initial capital, e.g. -0.05 = -5%)
- `pool_active_liquidity: float`
- `pool_fee_tier_bps: int`
- `gas_price_recent_wei: float` (base_fee + priority_fee from most recent gas row)
- `eth_usd_price_recent: float`
- `regime: str`
- `step_in_episode: int`, `steps_remaining: int`

**`to_vector()` → `np.ndarray`**: Flatten to a 1-D float64 array. Order must be stable.
Convert bool to 0/1 float. Convert `regime` string to a one-hot or ordinal encoding —
use ordinal mapping: `{"bull": 0, "bear": 1, "sideways": 2, "high_vol": 3, "unknown": 4}`.
Returns a vector of length ~20 (count the fields: 3 price + 3 returns + 1 bool + 1 value +
2 fees + 1 IL + 1 liq + 1 fee_tier + 1 gas + 1 eth + 1 regime (becomes 1 ordinal) +
2 time = ~19).

**`ObservationBuilder`** class:
- `__init__(self, market_view: MarketView, lookback_bars: int = 144)`:
  `lookback_bars` is the maximum lookback for 24h returns (144 bars at 10-min cadence).
- `build(step: int, position: Position, pool_engine: PoolEngine, gas_model, entry_price,
  entry_sqrt_price, initial_capital) → Observation`:
  Computes all observation fields from current state. All features are backward-looking
  (closed on the right at `step`). No data from `step + 1` or beyond.

### 2. `env/lp_env.py` — CONTRACTS.md §12

**`LpEnvironment(gym.Env)`**:

```python
class LpEnvironment(gym.Env):
    def __init__(self, market_view, pool_engine, price_process,
                 gas_model, slippage_model, config):
        # Define observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(observation_vector_length,), dtype=np.float64
        )
        n_actions = 1 + len(config.action_grid.center_offsets) * len(config.action_grid.widths)
        self.action_space = gym.spaces.Discrete(n_actions)
        # Action mapping: index 0 = hold, indices 1+ = rebalance with specific
        # (center_offset, width) combo. Build a lookup table.

    def reset(self, *, seed=None, options=None):
        # 1. Set RNG from seed
        # 2. Sample episode window via market_view.build_episode(seed, max_steps)
        # 3. Reset price_process
        # 4. Initialize pool_engine with fresh state at entry price
        # 5. Deploy initial position (the env does this, not the policy)
        # 6. Build initial observation
        # 7. Return (observation_vector, info_dict)
        ...

    def step(self, action_idx):
        # 1. Map action_idx to Action (hold or rebalance)
        # 2. Advance price_process by one step
        # 3. Advance pool_engine with new sqrt_price and swap volume (0 for now —
        #    swap volume comes from a config or is derived from the tape in replay mode)
        # 4. If action is rebalance:
        #    a. Compute gas cost
        #    b. Compute slippage cost
        #    c. Close current position (collect fees)
        #    d. Open new position
        # 5. Compute IL change since last step
        # 6. Compute reward via compute_reward(...)
        # 7. Compute trailing PnL volatility for risk penalty
        # 8. Build observation for next step
        # 9. Check termination: step count exceeds episode length OR
        #    price process exhausted (replay mode)
        # 10. Return (obs_vector, reward, terminated, truncated, info)
        ...
```

**Swap volume handling:** In replay mode, the swap volume comes from the tape events between the
current and next decision boundary. The pool engine's `step()` is called for each tape event
individually (not once per decision), and the env accumulates the per-block fees. This is a
critical detail: the env steps the pool engine at the tape-event granularity, but makes decisions
at the decision cadence.

**Replay mode vs calibrated mode:**
- **Replay**: `price_process` is `ReplayPriceProcess`. Swap volume is read from the tape.
  The env is deterministic (given seed for episode window selection).
- **Calibrated**: `price_process` is `CalibratedPriceProcess`. Swap volume is synthetic
  (e.g., a fixed fraction of pool liquidity per step, or a config-driven constant).
  The env is stochastic.

**`LpEnvironment` supports both modes — it switches behavior based on
`config.price_mode` and the type of `price_process` passed.**

### 3. `tests/sim/conftest.py` — `tiny_env` fixture

Append a `tiny_env` fixture: an `LpEnvironment` with all mock components (mock gas, mock slippage,
replay prices from `tiny_market_view`), 100-step episodes. Deterministic.

## Tests you must write

1. `LpEnvironment` observation space has the correct shape
2. `LpEnvironment` action space has `1 + |offsets| * |widths|` actions
3. `reset()` returns an observation vector and info dict
4. `reset(seed=42)` followed by `reset(seed=42)` → same initial observation (determinism)
5. `step(0)` (hold) advances the price, does not change the position, accrues fees
6. `step()` with a rebalance action changes the position's range
7. Running until `truncated` returns True (episode ends)
8. `step()` returns 5-tuple: (obs, reward, terminated, truncated, info)
9. `info` dict contains the full `RewardBreakdown` as a dict
10. Reward is computed from fees − gas − slippage − IL (validate a known case:
    no rebalance → gas=0, slippage=0, but fees and IL contribute)
11. Episode length matches config: `duration_days / step_minutes` steps
12. **Look-ahead probe**: build observation at step `t` and verify all rolling features
    (returns, vol) use only data at steps ≤ `t`. Feed an observation builder data with a known
    jump at step `t+1` and verify it's NOT reflected in the step `t` observation.
13. Replay mode is deterministic: same seed → same observation sequence
14. Calibrated mode with known seed produces a sequence
15. `tiny_env` fixture is importable and can be `reset()` and `step()`'d
16. Initial equity ≈ `config.episode.agent_capital_usdc` (after deploying the initial position,
    the position value should be ~capital)

## Definition of done

- All files exist, typed, no TODO stubs
- `uv run pytest tests/sim/test_env.py -q` green
- `code-reviewer` APPROVE
- PR open against `main` from `feature-sim-gym-env`
- `STATUS.md` updated