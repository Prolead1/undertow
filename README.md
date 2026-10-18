# Undertow

*Friction-aware reinforcement learning for concentrated liquidity provision.*

<!--
Scaffolding placeholder. README will be filled in as the data pipeline
(`undertow.data`) and AMM simulator (`undertow.sim`) take shape.
-->

## Layout

```
undertow/
├── data/   # on-chain Uniswap V3 ingestion & processing pipeline
└── sim/    # AMM / concentrated-liquidity simulator, RL environment
```

## Setup

```bash
uv sync
```

## Status

- [x] Data pipeline scaffolding (T00)
- [x] AMM simulator scaffolding

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

**Quickstart:** `uv sync && uv run undertow-sim info`
**Full guide:** [`docs/running_experiments.md`](docs/running_experiments.md)

## Running network-flavoured tests

Most tests run entirely against fixtures and never touch the network. The handful
marked `@pytest.mark.network` hit live endpoints and are **skipped by default**.
To run them, pass the `--run-network` flag:

```bash
uv run pytest            # fixtures only; network-marked tests skipped
uv run pytest --run-network   # also runs the @pytest.mark.network tests
```

This flag is wired up in `tests/conftest.py` (`pytest_addoption` +
`pytest_collection_modifyitems`); it is not part of pytest's `addopts`, so it
never collides with a later `-m` selector. Later pipeline tasks use it to run
their live fetcher checks.

## Test speed and parallelism

The suite runs under `pytest-xdist` by default (`addopts = "-q --strict-markers
-n auto"`). `tests/conftest.py` pins BLAS/OpenMP/NumExpr to **one thread per
worker**: the training and backtest tests are small enough that thread-pool
synchronisation costs more than the work, and without the pin each worker would
spawn one thread per core and oversubscribe the machine (making `-n auto`
slower than serial). Set an explicit value (e.g. `OMP_NUM_THREADS=8`) to
override the pin for a run.

Run serially when you need deterministic ordering or `--pdb`:

```bash
uv run pytest -n0        # single process, no xdist
```