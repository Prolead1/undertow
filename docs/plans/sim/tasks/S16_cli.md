# S16 — CLI + public API + docs

**Wave 7 · size M · depends on: S11, S13, S15 · blocks: nothing**
**Branch:** `feature-sim-cli`

## Why this task exists

The sim package needs a command-line entry point so the user can run `undertow-sim train`,
`undertow-sim backtest`, `undertow-sim evaluate`, `undertow-sim ablate`, and `undertow-sim info`
without writing Python. It also needs a clean public API surface (`undertow.sim.__init__.py`) and
user-facing documentation (`docs/running_experiments.md`) that walks a fresh clone from
`undertow-data snapshot` to a filled ablation table.

This is the final task in the sim plan. It ties the bow.

## Files you own

```
src/undertow/sim/__init__.py              (replaces stub with real public API surface)
src/undertow/sim/cli.py                   (click or argparse CLI with 5 subcommands)
pyproject.toml                             (adds [project.scripts] entry for undertow-sim)
docs/running_experiments.md               (user-facing quickstart + experiment guide)
tests/sim/test_cli.py
```

**Also edit:**
- `pyproject.toml`: add `[project.scripts]` section with `undertow-sim = "undertow.sim.cli:main"`
  (if S00 didn't already create the section). Do NOT touch existing `[project].dependencies` or
  `[project.optional-dependencies]`.
- `src/undertow/sim/__init__.py`: replace the stub with the real public API surface.
- `README.md` (repo root): add Status checklist and sim Quickstart section.

## What to build

### 1. `src/undertow/sim/__init__.py` — public API surface

The public API of `undertow.sim` exports:

```python
# Config
from undertow.sim.config import SimConfig, load_sim_config

# Types
from undertow.sim.types import (
    Action, Tick, TickSpacing, SqrtPrice, Price, Wealth,
    Regime, PriceMode, UndertowSimError,
)

# Core
from undertow.sim.core.position import Position, initial_deposit
from undertow.sim.core.pool import PoolEngine, PoolState, TickState

# Market data
from undertow.sim.marketview import MarketView, build_market_view

# Prices
from undertow.sim.prices import PriceProcess, ReplayPriceProcess, CalibratedPriceProcess, MRSJDParams

# Frictions
from undertow.sim.frictions import GasModel, SlippageModel

# Policies
from undertow.sim.policies import Policy, HODLPolicy, PassiveNarrowPolicy, PassiveWidePolicy, \
    FullRangeV2Policy, TauResetPolicy, CostAwareRebalancePolicy

# Environment
from undertow.sim.env import LpEnvironment, Observation, compute_reward, RewardBreakdown

# Backtester
from undertow.sim.backtest import run_backtest, summarize_backtest, BacktestLedger, BacktestResult

# Metrics
from undertow.sim.metrics import sharpe_ratio, sortino_ratio, max_drawdown, \
    annualized_return, all_metrics, pnl_decomposition, per_regime_metrics

# Training
from undertow.sim.train import train_ppo, train_single_seed, TrainingResult, RunManifest

# Validation
from undertow.sim.validation import run_parity_check, run_lookahead_probes

# Evaluation
from undertow.sim.evaluate import run_ablations, run_regime_evaluation, run_gap_analysis, write_results
```

Use `__all__` to make the surface explicit.

### 2. `src/undertow/sim/cli.py` — command-line interface

Use `click` if it's in the venv (add it as a dependency to the `sim` optional group if not),
otherwise use `argparse`. Five subcommands:

```
undertow-sim train --config configs/sim_default.toml [--output runs/]
    → trains PPO, writes checkpoints and run manifest to output dir

undertow-sim backtest --config configs/sim_default.toml --policy passive_narrow [--output results/]
    → runs backtester with the named policy, writes ledger and summary to output dir

undertow-sim evaluate --config configs/sim_default.toml --checkpoint runs/best.zip [--output results/]
    → runs evaluation protocol on the trained policy + all baselines, writes results

undertow-sim ablate --config configs/sim_default.toml [--output results/]
    → runs the full ablation matrix, writes tables

undertow-sim info [--config configs/sim_default.toml]
    → prints config summary, data split boundaries, available policies,
      action space size, and parity status (if available)
```

Policy selection for `backtest`: accept `--policy` as a string and map it:
- `"hodl"` → `HODLPolicy()`
- `"passive_narrow"` → `PassiveNarrowPolicy()`
- `"passive_wide"` → `PassiveWidePolicy()`
- `"full_range_v2"` → `FullRangeV2Policy()`
- `"tau_reset"` → `TauResetPolicy(half_width_ticks=...)` (read half_width from config or
  use a sensible default)
- `"cost_aware"` → `CostAwareRebalancePolicy(...)` (read params from config)

The `--config` flag loads a TOML file via `load_sim_config()`. All paths are resolved relative
to the current working directory.

The `--output` flag specifies the directory for results. Default: `./results/` (create if it
doesn't exist).

### 3. `docs/running_experiments.md` — user guide

Write a self-contained walkthrough that covers:

```markdown
# Running Experiments with Undertow Sim

## Prerequisites
- Python 3.14
- `uv` installed
- Data snapshot available (see data pipeline quickstart)

## Setup
```bash
git clone <repo>
cd undertow
uv sync --group sim
```

## 1. Check your setup
```bash
uv run undertow-sim info --config configs/sim_default.toml
```

## 2. Run a backtest
```bash
uv run undertow-sim backtest --config configs/sim_default.toml --policy hodl
uv run undertow-sim backtest --policy passive_narrow
uv run undertow-sim backtest --policy full_range_v2
```

## 3. Train an RL agent
```bash
uv run undertow-sim train --config configs/sim_default.toml --output runs/experiment_1/
```
This will take ~X hours on a modern CPU (Y timesteps × Z envs).

## 4. Evaluate
```bash
uv run undertow-sim evaluate --config configs/sim_default.toml \
    --checkpoint runs/experiment_1/best.zip --output results/
```

## 5. Run the ablation (RQ1)
```bash
uv run undertow-sim ablate --config configs/sim_default.toml --output results/
```

## 6. View results
Results are in `results/` as markdown tables and CSV files:
- `ablation_table.md` — RQ1: friction ablation
- `regime_matrix_*.md` — RQ2: per-regime results
- `gap_report.md` — RQ3: sim-to-reality gap
- `results_summary.md` — one-page summary

## Reproducing exact thesis results
The pinned config (`configs/sim_default.toml`) and pinned seeds (0–4) reproduce the
exact numbers in the thesis. Run all commands above in order with the default config.

## Troubleshooting
- "No data found for split window": run `undertow-data snapshot` first
- "Parity check failed": run `undertow-sim info` to check parity status
- Training too slow: reduce `training.total_timesteps` or increase `training.parallel_envs`
```

### 4. `README.md` (repo root) — add sim section

After the existing data pipeline sections, add:

```markdown
## `undertow.sim` — Simulator, Backtester & RL Environment

| Artifact | Status |
|---|---|
| Simulator (Gymnasium env) | ✅ |
| Backtester (on-chain replay) | ✅ |
| PPO training harness | ✅ |
| Baseline ladder (6 policies) | ✅ |
| Parity validation | ✅ |
| RQ1 ablation runner | ✅ |
| RQ2 regime evaluation | ✅ |
| RQ3 gap analysis | ✅ |

**Quickstart:** `uv sync --group sim && uv run undertow-sim info`
**Full guide:** `docs/running_experiments.md`
```

## Tests you must write

1. `undertow-sim info` runs without error and prints config summary (test via `subprocess.run`)
2. `undertow-sim backtest --policy hodl` runs and creates output files
3. `undertow-sim backtest` with invalid --policy name exits with non-zero and prints available
   policies
4. `undertow-sim info --config nonexistent.toml` exits with non-zero
5. CLI help text (`--help`) includes all five subcommands
6. `from undertow.sim import SimConfig, MarketView, LpEnvironment, run_backtest, train_ppo`
   works (the public API surface is importable)
7. `undertow.sim.__all__` contains all expected names
8. `docs/running_experiments.md` exists and is non-empty
9. `README.md` contains the sim status section

## Definition of done

- All files exist, typed, no TODO stubs
- `uv run undertow-sim info` works
- `uv run undertow-sim backtest --policy hodl` runs end-to-end (may use synthetic/fixture data
  if real data isn't available yet; add a note to the PR)
- `uv run pytest tests/sim/test_cli.py -q` green
- Public API surface is self-consistent (all exported names resolve)
- `docs/running_experiments.md` is complete and accurate
- `README.md` has the sim section
- `code-reviewer` APPROVE
- PR open against `main` from `feature-sim-cli`
- `STATUS.md` updated