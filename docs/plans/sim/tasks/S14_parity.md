# S14 — Parity & look-ahead validation

**Wave 5 · size M · depends on: S11, S12 · blocks: S15 (gate)**  
**Branch:** `feature-sim-parity`

## Why this task exists

The simulator and backtester run in two different numeric worlds: float64 and exact integer. The
simulator has injectable models; the backtester replays the real tape. The thesis's credibility
hinges on proving they agree when fed identical inputs. This task quantifies the drift between
simulator and backtester, states (and justifies) a tolerance, and either passes or fails parity.

Additionally, this task runs look-ahead probes on the composed system: verifying that neither the
environment nor the backtester can see future data.

**S15 may not run until parity passes.** This is a blocking gate.

## Files you own

```
src/undertow/sim/validation/__init__.py    (re-exports parity and lookahead functions)
src/undertow/sim/validation/parity.py      (run_parity_check, ParityReport)
src/undertow/sim/validation/lookahead.py   (run_lookahead_probes, LookAheadReport)
tests/sim/test_validation.py
```

## What to build

### 1. `validation/parity.py` — CONTRACTS.md §16

**`ParityReport`** dataclass (frozen, slots):
- `fee_drift_mean/std: float` — mean and std of per-step fee difference
  (sim_fees − backtest_fees), in USDC
- `il_drift_mean/std: float` — per-step IL difference
- `pnl_drift_mean/std: float` — per-step PnL difference
- `tolerance: float` — stated tolerance (USDC per step)
- `passed: bool` — all mean drifts within tolerance
- `justification: str` — prose justification of the tolerance

**`run_parity_check(sim_env, backtest_ledger, num_episodes=10, seeds=(0,1,2,3,4))` → `ParityReport`**:

The parity check works as follows:

1. For each seed, create a replay-mode simulation episode using the exact same settings as the
   backtester: same episode window (start_seq, end_seq), same initial position, same price path
   (the reference feed), same gas data, and the dummy policy (always hold, so no rebalance
   costs).
2. Run the simulator episode → per-step fees, IL, PnL.
3. Run the backtester over the same window with the same config → per-step fees, IL, PnL.
4. For each step, compute the difference:
   - `fee_drift = sim_fees - backtest_fees`
   - `il_drift = sim_il - backtest_il`
   - `pnl_drift = sim_pnl - backtest_pnl`
5. Aggregate across all episodes: mean and std of each drift term.
6. Compare against tolerance. If all mean drifts are within tolerance, `passed = True`.

**Tolerance justification** (write this in the docstring and the report):

The two numeric worlds differ by design: the backtester uses the data module's exact-integer
`FeeGrowthTracker` (Q128 math), while the simulator uses float64. The expected per-step drift
from float64 round-trip of Q128 values is on the order of `1e-12` in fee-growth units,
which for a position with `L = 10^12` (representative of $100k in a WETH/USDC pool) translates
to fee drift < `10^-6` USDC per step. This is the tolerance floor.

However, if the backtester is also using float64 (because `FeeGrowthTracker` wasn't exported
by S02), the tolerance should be effectively zero (both use identical float64 math).
State which path is being used, and set the tolerance accordingly:

- Both float64: tolerance = `1e-9` USDC per step (numerical noise from different code paths)
- Backtester exact-int, simulator float64: tolerance = `1e-3` USDC per step (Q128 → float64
  conversion drift, dominated by the sqrt_price conversion)

### 2. `validation/lookahead.py` — CONTRACTS.md §16

**`LookAheadReport`** dataclass (frozen, slots):
- `probes_run: int`
- `violations: int`
- `details: list[str]`

**`run_lookahead_probes(market_view, env, backtest_fn)` → `LookAheadReport`**:

Probe suite (at minimum):

1. **Split wall integrity**: Try `market_view.slice()` with `end_seq` beyond the active split's
   boundary → expect `LookAheadError`. No violation means it was raised; violation means it
   silently succeeded.

2. **Observation window**: Build an observation at step `t`. Verify that the price returns
   and volatility in the observation use only bars with `close_time <= t`, not `t+1`.

3. **Backtester decision**: At decision boundary step `t`, verify that the backtester's
   observation does not include information from the tape event at `t+1`.

4. **MarketView train/eval isolation**: Create a `MarketView` with `split="train"`. Verify that
   `reference_close()` at a time in the eval window returns a value clamped to the train window's
   end (or raises `LookAheadError`).

5. **Episode boundary**: `build_episode()` returns `(start_seq, end_seq)` where both are within
   the active split's range, and `end_seq` never enters the eval split.

6. **Regime label as-of**: `regime_at(t)` returns a label whose timestamp ≤ `t`, not a future
   label.

7. **Env step count**: After `reset()`, the env's `step_in_episode` is 0. After `n` calls to
   `step()`, `step_in_episode == n` and `steps_remaining == episode_length - n`.

Each probe that passes records a detail line. Each violation increments `violations` and records
a detailed message (what was attempted, what was expected, what actually happened).

## Tests you must write

1. `ParityReport` with all drifts zero → `passed = True`
2. `ParityReport` with a drift exceeding tolerance → `passed = False`
3. `run_parity_check` on the `tiny_env` + `tiny_ledger` (both built from same synthetic data)
   passes (drifts are zero or within tolerance) — test with 2 episodes
4. Parity check is deterministic: same inputs → same report
5. `LookAheadReport` with zero violations has `passed = True` (or `violations == 0`)
6. Split wall probe catches an out-of-bounds slice (the probe itself tests the mechanism;
   test that the probe reports the violation)
7. Observation window probe: construct a scenario where `t+1` data differs from `t` data,
   verify the probe detects if `t+1` leaks into the `t` observation
8. `run_lookahead_probes` returns `probes_run >= 7`
9. Integration: parity between `tiny_env` and `tiny_ledger` on a hold-only policy passes
   — **mark `@pytest.mark.slow`**

## Definition of done

- All files exist, typed, no TODO stubs
- `uv run pytest tests/sim/test_validation.py -q` green
- Parity passes on the `tiny_env` vs `tiny_ledger` test
- Look-ahead probes report zero violations
- `code-reviewer` APPROVE
- PR open against `main` from `feature-sim-parity`
- `STATUS.md` updated