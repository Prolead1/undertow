# S00 — Sim scaffold: deps, test markers, config skeleton, RL-library ADR

**Wave 0 · size S · depends on: nothing · blocks: all tasks**
**Branch:** `feature-sim-scaffold`

## Why this task exists

The sim package currently has a stub `__init__.py`. Before any task can write code, we need the
project scaffolding: dev dependencies, test infrastructure, a config-file skeleton, and — critically
— a decision on which RL library we use. The latter gates S13's design; getting it wrong or late
wastes multiple tasks' time.

## Files you own

```
pyproject.toml                             (adds sim deps ONLY)
configs/sim_default.toml                   (empty section skeleton — S01 fills the real fields)
tests/sim/__init__.py                      (empty)
tests/sim/conftest.py                      (seeded-rng fixtures + marker registration)
docs/decisions/004-rl-library.md           (the ADR)
```

## What to build

### 1. `pyproject.toml` — sim dependencies

Add to `[project.optional-dependencies]` a group called `sim`:

```toml
[project.optional-dependencies]
sim = [
    "gymnasium>=1.0",
    "numpy>=2.0",
    "polars>=1.0",
    "pyarrow>=19",
]
```

If `stable-baselines3` installs cleanly on Python 3.14 (test it: `uv add stable-baselines3` in the
project venv), add it here. If not, note this in the ADR — we vendored PPO instead.

Do NOT touch existing `[project].dependencies` or `[dependency-groups].dev`. The sim deps live
in a separate optional group so the data pipeline stays lightweight.

### 2. `tests/sim/conftest.py` — test infrastructure

```python
import pytest
import numpy as np

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: > 5s (default-on, budgeted)")
    # The `network` marker is already registered in tests/conftest.py (data plan S00).
    # Re-use it; do not re-register.

@pytest.fixture
def rng():
    """Seeded Generator for deterministic tests."""
    return np.random.default_rng(42)
```

### 3. `configs/sim_default.toml` — skeleton

Create sections with a comment each, referencing the `SimConfig` sections from `CONTRACTS.md` §2.
S01 fills the actual field set. Example:

```toml
# Default config for undertow.sim — filled by S01.
[episode]
# duration_days, step_minutes, agent_capital_usdc, ...

[action_grid]
# center_offsets, widths, include_hold

[gas]
# mint_units, burn_units, collect_units, rebalance_swap_units

[slippage]
# fixed_impact_bps

[reward]
# risk_penalty_lambda, normalize_by_capital, ablation flags

[split]
# train_start_utc, train_end_utc, eval_start_utc, eval_end_utc

[training]
# algorithm, seeds, discount_gamma, gae_lambda, ...

[backtest]
# compute_decomposition, log_decision_every_step

[price]
# mode: "replay" | "calibrated"
```

### 4. `docs/decisions/004-rl-library.md` — the ADR

```markdown
# ADR-004: RL library for sim training

**Status:** accepted
**Date:** [today]
**Task:** S00

## Context

The training harness (S13) needs PPO. We have two options: (a) `stable-baselines3` (the community
standard, SB3-compatible with Gymnasium, used by the anchor paper's community), or (b) a vendored
single-file PPO in `train/ppo_runner.py`.

## Decision

[IF SB3 installs on Python 3.14]: Use `stable-baselines3` as a dependency. It installs cleanly,
provides battle-tested PPO with GAE, and its MultiVectorEnv interface means we don't need to
write our own parallel rollout loop.

[IF SB3 does NOT install]: Vendored PPO in `train/ppo_runner.py`. We implement PPO-clip with GAE
directly against Gymnasium's `step()`/`reset()` API, using numpy + a small MLP in pure numpy or
torch (whichever is already in the venv or adds the smallest dep). ~200 lines, single file,
tested against the CartPole-v1 equivalent. The plan's pinned hyperparameters (§7 of PLAN.md)
become the vendored module's defaults.

## Consequences

- [SB3 path]: S13 imports `stable_baselines3.PPO`; config maps directly; tensorboard logging is
  free.
- [Vendored path]: S13 imports `undertow.sim.train.ppo_runner`; we own the training loop and
  must test it ourselves; no tensorboard (CSV logs instead).
- Either way, the `Policy` protocol (§10 of CONTRACTS.md) is unchanged — the backtester and eval
  runner do not care about the RL library.
```

Verify the installability claim by actually running `uv add stable-baselines3` in the project
venv. Record the exact result (success + version, or failure + error) in the ADR.

## Tests you must write

1. `tests/sim/conftest.py` has the `rng` fixture and it returns a `numpy.random.Generator`.
2. `configs/sim_default.toml` parses as valid TOML: `tomllib.load` it.
3. The added `sim` optional-dependency group installs without conflicts: `uv sync --group sim`.

## Definition of done

- `uv sync --group sim` succeeds
- `uv run pytest tests/sim/ -q` passes (even if only the conftest fixtures exist)
- `code-reviewer` APPROVE
- PR open against `main` from `feature-sim-scaffold`
- `STATUS.md` updated