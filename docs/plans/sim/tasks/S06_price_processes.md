# S06 — Price processes (replay + calibrated MRSJD)

**Wave 2 · size L · depends on: S01 (code) + S03 (data for calibration) · blocks: S12**  
**Branch:** `feature-sim-price-processes`

## Why this task exists

The training simulator needs two price modes (roadmap §10.2.2): deterministic **replay** of the
reference feed (so the environment can be compared head-to-head with the backtester in S14), and a
**calibrated Markov-regime-switching jump-diffusion** (MRSJD) that fixes the "i.i.d. returns" gap
(G2) identified in the problem statement. The MRSJD is the generative model that lets us train
policies with realistic regime persistence and tail events, then test on real data.

This is the largest wave-2 task because the MRSJD fitting is non-trivial.

## Files you own

```
src/undertow/sim/prices/__init__.py        (re-exports PriceProcess, ReplayPriceProcess,
                                            CalibratedPriceProcess, MRSJDParams)
src/undertow/sim/prices/base.py            (PriceProcess protocol)
src/undertow/sim/prices/replay.py          (ReplayPriceProcess)
src/undertow/sim/prices/regime_jump.py     (CalibratedPriceProcess, MRSJDParams, fit)
tests/sim/test_price_processes.py
```

## What to build

### 1. `prices/base.py` — CONTRACTS.md §7

`PriceProcess` protocol (`runtime_checkable`):
- `reset(rng: np.random.Generator) -> None`
- `step(rng: np.random.Generator) -> tuple[Price, SqrtPrice, Tick]`
- `mode: str` (property)

### 2. `prices/replay.py` — CONTRACTS.md §7

`ReplayPriceProcess`:
- `__init__(self, market_view: MarketView, start_seq: int, end_seq: int)`: extracts the reference
  bars covering the episode window from the market view. Stores them internally.
- `reset(rng)`: reset the bar index to 0
- `step(rng)`: return the next bar's (price, sqrt_price, tick). All three must be consistent:
  `price = reference_close`, `sqrt_price = sqrt(price)`, `tick = floor(log(price)/log(1.0001))`.
  Raise `StopIteration` if past the last bar (the env wraps this as `truncated=True`).
- `mode` → `"replay"`

**Look-ahead test: check that the replay process's bar pointer never advances beyond `end_seq`.**

### 3. `prices/regime_jump.py` — CONTRACTS.md §7

**`MRSJDParams`** dataclass (frozen, slots):
- `state_names: tuple[str, ...]` — e.g. `("bull", "bear", "sideways", "high_vol")`
- `transition_matrix: np.ndarray` — 4×4, row-stochastic (each row sums to 1)
- `drift: tuple[float, ...]` — per-state μ (annualized)
- `diffusion: tuple[float, ...]` — per-state σ (annualized)
- `jump_intensity: tuple[float, ...]` — per-state λ_J (jumps per year)
- `jump_loc/scale/dof: tuple[float, ...]` — per-state Student-t jump parameters

Validate in `__post_init__`: transition matrix is square, rows sum to 1 (within tolerance),
all tuples have the same length, σ > 0 for all states, df > 2 for finite variance.

**`CalibratedPriceProcess`**:
- `__init__(self, params: MRSJDParams, dt: float = 1 / (6 * 365.25))`: `dt` defaults to 10 minutes
  in annualized units. Initialize the hidden regime state (sample from the stationary distribution
  of the transition matrix).
- `reset(rng)`: reset price to an arbitrary starting value (e.g., log-price = 0 for simplicity,
  or take a starting price parameter). Resample the initial regime from the stationary distribution.
  Store `rng` (or use the one passed to `step`).
- `step(rng)` → `tuple[Price, SqrtPrice, Tick]`:
  1. Sample regime transition: use the current state's row of `transition_matrix`
  2. Sample whether a jump occurs: `Bernoulli(1 - exp(-λ_J * dt))` approximately (or
     `Poisson(λ_J * dt)` for the number of jumps)
  3. Compute log-return for this step:
     ```
     dlogP = μ * dt + σ * sqrt(dt) * ε  +  sum(jump_sizes)
     ```
     where ε ~ N(0,1) and jump sizes ~ StudentT(df, loc, scale) if a jump occurs
  4. Update log-price, convert to price, sqrt_price, tick (consistent)
  5. Return (price, sqrt_price, tick)
- `mode` → `"calibrated"`

**`CalibratedPriceProcess.fit(market_view)`** → `MRSJDParams` (static method):
- `market_view.is_train` must be True (raise `ValueError` if not — fitting on eval data is
  look-ahead)
- Extract the reference close prices from the market view's train split
- Compute log-returns at 10-minute cadence
- Fit a 4-state hidden Markov model with Gaussian emissions using `hmmlearn` (if available) or
  a manual EM implementation. The 4 states correspond to the regime labels, but the HMM
  discovers them unsupervised — label them post-hoc by matching the fitted μ/σ to the regime
  definitions (high μ → bull, low μ → bear, low σ → sideways, high σ → high_vol)
- Estimate jump parameters from the residuals (returns minus the per-state mean return):
  fit a Student-t to the tail residuals (e.g., |residual| > 3σ per state)
- If `hmmlearn` is not installable on Python 3.14, implement a minimal Baum-Welch EM for
  Gaussian-emission HMM (~150 lines in pure numpy). This is acceptable — it's a standard
  algorithm and the plan explicitly prefers clarity over SOTA

### 4. Look-ahead design note

The MRSJD is fit once on the train split and its parameters are frozen. At simulation time,
`step()` generates prices forward from the fitted model — it never reads from any split. This means
the calibrated process has **no look-ahead risk at runtime** (it's a generative model). The
look-ahead risk is only in the fitting step, which S06 guards with `is_train` check.

## Tests you must write

1. `ReplayPriceProcess` replays the exact reference prices in order
2. `ReplayPriceProcess.reset()` restarts from bar 0
3. `ReplayPriceProcess` raises `StopIteration` past the last bar
4. `ReplayPriceProcess` price, sqrt_price, tick are mutually consistent
5. `CalibratedPriceProcess` with trivial params (σ=0, no jumps) returns a constant price
6. `CalibratedPriceProcess` with known drift (μ=0.1/year, σ=0, no jumps) produces prices that
   trend upward over many steps
7. `CalibratedPriceProcess` with known volatility (μ=0, σ=0.5/year, no jumps) produces returns
   with the correct standard deviation (test over 10k steps)
8. `CalibratedPriceProcess.reset()` produces a different initial regime with different seeds
9. `CalibratedPriceProcess.fit()` raises `ValueError` when `market_view.is_train` is False
10. `MRSJDParams.__post_init__` rejects a transition matrix with rows not summing to 1
11. `MRSJDParams.__post_init__` rejects σ ≤ 0
12. `MRSJDParams.__post_init__` rejects df ≤ 2
13. `ReplayPriceProcess` + `MRSJDParams` both satisfy `isinstance(x, PriceProcess)` (protocol check)
14. Both price processes are deterministic given RNG: same seed → same price sequence
15. **Look-ahead probe**: replay process built from train data can only replay bars within the
    train split's time range

## Definition of done

- All files exist, typed, no TODO stubs
- `uv run pytest tests/sim/test_price_processes.py -q` green
- `code-reviewer` APPROVE
- PR open against `main` from `feature-sim-price-processes`
- `STATUS.md` updated