# S15 — Evaluation runner (RQ1 ablation, RQ2 regime matrix, RQ3 gap)

**Wave 6 · size L · depends on: S05, S10, S11, S13, S14 · blocks: S16**  
**Branch:** `feature-sim-evaluation`

## Why this task exists

The evaluation runner produces the thesis's three primary research deliverables:

- **RQ1**: "What is the contribution of each friction term to RL performance?" → the ablation table:
  {full, no_gas, static_fee, no_il} × {PnL, Sharpe, MaxDD, turnover, decomposition}
- **RQ2**: "Is the policy robust to distinct market regimes?" → the per-regime results matrix:
  agent + all baselines × four regimes × metrics, walk-forward, mean ± std over seeds
- **RQ3**: "How well does the simulator predict on-chain performance?" → the sim-to-reality gap:
  `Δ = sim_PnL − onchain_PnL`, decomposed by cost term

This task also owns writing the results as markdown/CSV artifacts, ready for inclusion in the
thesis.

**Prerequisite:** S14 parity must have passed. The evaluation runner may assume the simulator and
backtester agree within the stated tolerance.

## Files you own

```
src/undertow/sim/evaluate/__init__.py     (re-exports all public functions)
src/undertow/sim/evaluate/protocol.py     (walk-forward, multi-seed evaluation runner)
src/undertow/sim/evaluate/ablation.py     (RQ1 ablation matrix)
src/undertow/sim/evaluate/gap.py          (RQ3 sim-to-reality gap)
src/undertow/sim/evaluate/report.py       (results tables: markdown/CSV artifacts)
tests/sim/test_evaluation.py
```

## What to build

### 1. `evaluate/protocol.py` — CONTRACTS.md §17

**`evaluate_policy(policy, market_view, config, num_seeds=5)` → `dict`**:

Core evaluation protocol used by both ablation and regime evaluation:

1. Walk-forward loop over the evaluation window (eval split):
   - Split the eval window into non-overlapping 30-day episodes
   - For each episode: run the policy (frozen — no learning)
   - For each seed in `config.training.seeds`: run the same evaluation with a fresh env
     (seeded for episode window sampling; policies are deterministic so seed affects only
     which episodes are sampled in calibrated mode)
2. Collect per-episode metrics (Sharpe, Sortino, MaxDD, annualized return, total PnL,
   fee income, gas paid, slippage paid, IL realized)
3. Aggregate across episodes and seeds: mean and std for each metric
4. Return results dict: `{"policy": name, "metrics": {...}, "per_episode": [...], ...}`

For backtester-based evaluation (the ground truth), use `run_backtest()` on each eval episode
instead of the simulator.

### 2. `evaluate/ablation.py` — CONTRACTS.md §17

**`run_ablations(config, policy, market_view)` → `list[AblationResult]`**:

**`AblationResult`** dataclass (frozen, slots):
- `ablation_name: str` — `"full"`, `"no_gas"`, `"no_slippage"`, `"no_il"`, `"static_fee"`
- `flags: dict[str, bool]` — the ablation flag settings
- `metrics: dict[str, float]` — per-metric values (mean over seeds)
- `per_regime: pl.DataFrame | None` — optional per-regime breakdown

Ablation variants (matching §9.3.3 of the roadmap):

| Ablation | Fees | Gas | Slippage | IL | Description |
|---|---|---|---|---|---|
| `full` | True | True | True | True | All costs active (the honest baseline) |
| `no_gas` | True | False | True | True | Gas disabled — agent pays no gas |
| `no_slippage` | True | True | False | True | Rebalancing is free |
| `no_il` | True | True | True | False | IL not penalized — agent only sees fee income |
| `static_fee` | True | True | True | True | Same as full but with a flat gas model instead of replay — tests sensitivity to gas volatility |

For each ablation variant:
1. Create a modified `SimConfig` with the corresponding `RewardConfig` flag settings
2. Train a new policy (or reload from checkpoint — for RL policies, each ablation variant needs
   its own training run, since the reward function changes)
3. Evaluate on the eval window
4. Compute all metrics
5. Package into `AblationResult`

For baseline policies (deterministic), only evaluate — no training needed. The ablation
just changes which costs are subtracted from the PnL ledger.

**Output:** `write_results()` produces a markdown table:

```
| Ablation | PnL (USDC) | Sharpe | MaxDD | Fee Income | Gas Paid | IL Realized |
|---|---|---|---|---|---|---|
| full | ... | ... | ... | ... | ... | ... |
| no_gas | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... |
```

### 3. `evaluate/ablation.py` — RQ2 regime matrix

**`run_regime_evaluation(config, policies, market_view)` → `list[RegimeMatrix]`**:

**`RegimeMatrix`** dataclass (frozen, slots):
- `policy_name: str`
- `regimes: list[str]` — `["bull", "bear", "sideways", "high_vol"]`
- `metrics_per_regime: pl.DataFrame` — rows = regimes, columns = metrics + mean + std

For each policy (agent + all 6 baselines):
1. Run the evaluation protocol over the eval window
2. Tag each episode by its dominant regime (the regime label that appears most frequently
   during that episode; use `market_view.regime_at()` for each step)
3. Group episodes by regime
4. Compute per-regime metrics (mean and std over episodes + seeds)
5. Package into `RegimeMatrix`

**Output:** `write_results()` produces a markdown table per policy:

```
Policy: ppo_full

| Regime | PnL | Sharpe | MaxDD | Fee Income | Gas Paid | IL |
|---|---|---|---|---|---|---|
| bull | 1200 ± 200 | 1.5 ± 0.3 | ... | ... | ... | ... |
| bear | -300 ± 150 | -0.8 ± 0.4 | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... |
```

And a comparison matrix (policies × regimes × key metric, e.g., PnL).

### 4. `evaluate/gap.py` — CONTRACTS.md §17

**`run_gap_analysis(config, policy, market_view)` → `GapReport`**:

**`GapReport`** dataclass (frozen, slots):
- `policy_name: str`
- `sim_pnl: float` — PnL from simulator (mean over eval episodes)
- `onchain_pnl: float` — PnL from backtester (ground truth)
- `gap: float` — `sim_pnl - onchain_pnl`
- `gap_by_cost_type: dict[str, float]` — decomposed by `fees`, `gas`, `slippage`, `il`

1. Run the policy on the backtester over the eval window → on-chain PnL and decomposition
2. Run the same policy on the simulator (replay mode) over the same episodes → sim PnL and
   decomposition
3. Compute gap = sim − onchain for each component
4. Report the total gap and the per-component gaps

This is the RQ3 deliverable. The gap should be small (S14 quantified the per-step drift;
this aggregates over full episodes). If the gap is large, that's a finding — not a failure.
Document it honestly.

### 5. `evaluate/report.py` — CONTRACTS.md §17

**`write_results(ablations, regime_matrices, gap_report, output_dir)`**:

Produces:
1. `output_dir/ablation_table.md` + `output_dir/ablation_table.csv`
2. `output_dir/regime_matrix_<policy>.md` per policy + `output_dir/regime_comparison.md`
3. `output_dir/gap_report.md` + `output_dir/gap_report.csv`
4. `output_dir/results_summary.md` — a single-page summary with the key numbers

All markdown tables are GitHub-flavored (pipe tables). All CSV files have headers.

## Tests you must write

1. `AblationResult` correctly stores all fields
2. `RegimeMatrix` correctly stores all fields
3. `GapReport` correctly stores all fields
4. `run_ablations` on dummy policy + `tiny_market_view` (train split only, for speed) returns
   one `AblationResult` per ablation variant — **mark `@pytest.mark.slow`**
5. Each ablation result has the expected `ablation_name`
6. Each ablation result's `flags` dict matches the expected settings
7. `run_regime_evaluation` on dummy policy + tiny data returns a `RegimeMatrix` with all four
   regime labels — **mark `@pytest.mark.slow`**
8. `run_gap_analysis` on dummy policy + tiny data returns a `GapReport` with non-None fields
   — **mark `@pytest.mark.slow`**
9. `write_results` creates all expected files in the output directory
10. `write_results` markdown files contain valid pipe tables (at minimum: headers, separator
    row, at least one data row)
11. `write_results` CSV files can be parsed by `polars.read_csv()`
12. Gap report's `gap` = `sim_pnl - onchain_pnl` matches manual computation

## Definition of done

- All files exist, typed, no TODO stubs
- `uv run pytest tests/sim/test_evaluation.py -q` green (slow tests pass when run explicitly)
- `code-reviewer` APPROVE
- PR open against `main` from `feature-sim-evaluation`
- `STATUS.md` updated