# S10 — Policy protocol + baseline ladder

**Wave 3 · size M · depends on: S01, S04 · blocks: S11, S13, S15**  
**Branch:** `feature-sim-baselines`

## Why this task exists

The thesis needs a set of simple, deterministic baseline policies to compare against the RL agent
(roadmap §10.5). These are the "baseline ladder": HODL, passive narrow/wide ranges, full-range V2
equivalent, τ-reset, and cost-aware rebalance. All policies implement the same `Policy` protocol,
which the backtester calls via `act()` and the training harness calls during evaluation.

The `Policy` protocol is deliberately frozen: `act()` takes an `Observation` and returns an
`Action`. There is no `update()` method — the backtester cannot accidentally train a policy.

## Files you own

```
src/undertow/sim/policies/__init__.py      (re-exports Policy, all baselines)
src/undertow/sim/policies/base.py          (Policy protocol)
src/undertow/sim/policies/baselines.py     (all 6 baseline policies)
tests/sim/test_policies.py
tests/sim/conftest.py                      (appends dummy_policy fixture)
```

## What to build

### 1. `policies/base.py` — CONTRACTS.md §10

`Policy` protocol (`runtime_checkable`):
- `act(self, observation: Observation, rng: np.random.Generator | None = None) -> Action`
- `name: str` (property)
- `reset(self) -> None` (default no-op)

Also define a simple `HOLDAction` constant: `Action(action_type="hold", center_offset=0, width=0)`.
Baselines that always hold can return this directly.

### 2. `policies/baselines.py` — CONTRACTS.md §10

All six baselines. Each implements the `Policy` protocol.

**`HODLPolicy`**:
- On the first call to `act()`, returns `hold`. Always returns `hold`. Never deploys liquidity.
  (The agent's initial capital stays as the token mix from initialization; the backtester/env
  handle the initial deposit for other policies.)
- Actually, for consistency: the env/backtester deploys the initial position before the first
  `act()` call for all policies. HODL should just never rebalance — always return `hold`.
- `name` → `"hodl"`

**`PassiveNarrowPolicy`**:
- On first `act()`, returns a `rebalance` action with the narrow range (±5% around entry price):
  `center_offset = 0`, `width` corresponding to ±5% in tick-spacings.
  Approximate: ±5% price range → ~50 tick-spacings at tick spacing 60 (each tick spacing is
  ~0.6% at $3000). Compute: `width = int(0.05 / (tick_spacing * 0.0001))` ≈ 8 for tick_spacing=60
  and 1bp per tick. Actually, compute the tick width directly:
  `tick_lower = price_to_tick(entry_price * 0.95)`, `tick_upper = price_to_tick(entry_price * 1.05)`,
  `width = (tick_upper - tick_lower) // (2 * tick_spacing)`.
- After the first call, always returns `hold`.
- `name` → `"passive_narrow"`

**`PassiveWidePolicy`**:
- Same as narrow but ±20% around entry.
- `name` → `"passive_wide"`

**`FullRangeV2Policy`**:
- On first `act()`, returns a rebalance with `tick_lower = MIN_TICK`, `tick_upper = MAX_TICK`
  (the widest possible range). After deployment, always `hold`.
- `name` → `"full_range_v2"`

**`TauResetPolicy`**:
- `__init__(self, half_width_ticks: int)`: `half_width_ticks` is the half-width in number of ticks
  (not tick-spacings). Store entry tick from `reset()`.
- `reset()`: called at episode start. Reset `current_lower` and `current_upper` to the initial
  range (will be set on first `act()`).
- `act(obs, rng)`:
  1. On first call: deploy at `[entry_tick - half_width_ticks, entry_tick + half_width_ticks]`,
     snapped to tick spacing. Store the current range.
  2. On subsequent calls: if `obs.tick < current_lower` or `obs.tick > current_upper` (price
     has exited the range), recenter: `current_lower = obs.tick - half_width_ticks`,
     `current_upper = obs.tick + half_width_ticks`, return `rebalance` with appropriate
     `center_offset` and `width`.
  3. Otherwise, return `hold`.
- `name` → `"tau_reset"`

**`CostAwareRebalancePolicy`**:
- `__init__(self, half_width_ticks: int, min_interval_steps: int, gas_model: GasModel)`:
  `min_interval_steps` is the minimum number of steps between rebalances (at most daily).
- `reset()`: reset internal state (last rebalance step, accumulated fees since last rebalance).
- `act(obs, rng)`:
  1. Check if price exited the range.
  2. If exited AND `step_in_episode - last_rebalance_step >= min_interval_steps`:
     estimate gas cost of a rebalance, compare to estimated fees accumulated since last rebalance.
     Only rebalance if `estimated_fees > gas_cost`. Estimate fees roughly:
     `daily_fee_rate * L_active / L_pool * fee_tier * volume`. A simpler heuristic:
     use `obs.uncollected_fees_token0` as the fee estimate and compare against the gas model's
     cost for `"rebalance"`.
  3. If conditions are met, recenter and return `rebalance`.
  4. Otherwise, return `hold`.
- `name` → `"cost_aware"`

### 3. Observation dependency note

The `Policy.act()` takes an `Observation` (from S12/CONTRACTS.md §11). Since S12 won't exist
yet when S10 is written, you have two options:

1. **Import `Observation` from S12's future location** (`from undertow.sim.env.observations import Observation`).
   Create a minimal stub `Observation` dataclass in `policies/base.py` that S12 will replace, with
   a comment: `# Replaced by S12 — keep field names in sync with CONTRACTS.md §11`.

2. **Use structural typing**: the `Policy` protocol doesn't type the observation parameter beyond
   `object`, and each baseline accesses only the fields it needs from the observation dict/object.

**Choose option 1**: Create a minimal `Observation` stub in `policies/base.py` with the fields
that baselines actually use (`tick`, `position_in_range`, `uncollected_fees_token0`,
`step_in_episode`, `price`). S12 will later move this to `env/observations.py` and add the
remaining fields. Mark it clearly with a comment referencing S12.

### 4. `tests/sim/conftest.py` — `dummy_policy` fixture

Append a `dummy_policy` fixture: a `HODLPolicy` instance. Simple and deterministic.

## Tests you must write

1. `HODLPolicy.act()` always returns a hold action (test 3 calls)
2. `HODLPolicy.name == "hodl"`
3. `PassiveNarrowPolicy` first call returns a rebalance action
4. `PassiveNarrowPolicy` second call returns hold
5. `PassiveWidePolicy` deploys a wider range than `PassiveNarrowPolicy` (compare widths)
6. `FullRangeV2Policy` deploys at `MIN_TICK` / `MAX_TICK`
7. `TauResetPolicy` deploys on first call
8. `TauResetPolicy` holds when price is still in range
9. `TauResetPolicy` recenters when price exits the range (feed an observation with tick
   outside the current bounds)
10. `CostAwareRebalancePolicy` does NOT rebalance when `min_interval_steps` hasn't elapsed
11. `CostAwareRebalancePolicy` does NOT rebalance when estimated fees < gas cost
12. `CostAwareRebalancePolicy` DOES rebalance when both conditions are met
13. `CostAwareRebalancePolicy.reset()` resets internal state
14. All six policies pass `isinstance(policy, Policy)` (protocol check)
15. `dummy_policy` fixture is importable and its `act()` returns hold
16. `Policy` protocol rejects a class without `act()` and `name`

## Definition of done

- All files exist, typed, no TODO stubs
- `uv run pytest tests/sim/test_policies.py -q` green
- `code-reviewer` APPROVE
- PR open against `main` from `feature-sim-baselines`
- `STATUS.md` updated