# Undertow

[![quality](https://github.com/Prolead1/undertow/actions/workflows/quality.yml/badge.svg)](https://github.com/Prolead1/undertow/actions/workflows/quality.yml)

**Friction-aware reinforcement learning for concentrated liquidity provision.**

Undertow studies when concentrated-liquidity (Uniswap V3) market making actually
pays after real-world frictions. It pairs an **exact, integer-arithmetic replay
backtester** over on-chain events with a **fast Gymnasium simulator**, so the same
policies can be trained in simulation and measured against ground truth — and the
gap between the two can be quantified rather than assumed.

## What's inside

| Package | Role |
|---|---|
| `undertow.data` | On-chain ingestion and processing: swaps, mints, burns and collects; reference prices, gas, funding and regime labels; a look-ahead-safe, block-ordered event tape. |
| `undertow.sim` | Gymnasium environment, exact Q128 fee-growth backtester, six-policy baseline ladder, PPO training harness, and the RQ1–RQ3 evaluation runners. |
| `undertow.ui` | Read-only cockpit: live learning curves, policy replay against price, and run comparison. |

**Research questions.** RQ1: which frictions (gas, slippage, impermanent loss,
fees) drive net LP performance? RQ2: how do policies perform across bull, bear,
sideways and high-volatility regimes? RQ3: how large is the sim-to-reality gap
between the float simulator and the exact replay engine?

## Quickstart

Requires Python 3.14 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run undertow-sim info          # configuration summary; needs no dataset
```

Fetch the validated datasets (checksum-pinned release assets, not stored in git):

```bash
uv run undertow-data fetch-data   # download + SHA-256 verify + extract to data/datasets/
```

Then run the primary 0.05 % USDC/WETH study-pool backtest, train PPO, and
evaluate the result:

```bash
uv run undertow-sim backtest --policy passive_narrow
uv run undertow-sim train --output runs/experiment_1/
uv run undertow-sim evaluate --checkpoint runs/experiment_1/seed_0/ppo_seed_0.zip
```

The defaults target the primary pool (`configs/eth_usdc_500.toml`, 0.05 %,
spacing 10). For the RQ1 fee-tier contrast use the matching pair,
`configs/sim_eth_usdc_3000.toml` + `configs/eth_usdc_3000.toml` (0.30 %). `uv run undertow-sim --help` lists every subcommand; the full workflow is in
[`docs/running_experiments.md`](docs/running_experiments.md).

## Dashboard

The read-only dashboard serves the artifacts a run already writes — no training,
backtesting or mutation:

```bash
uv run undertow-ui serve --runs runs/ --results results/   # http://127.0.0.1:8050
```

**Live Cockpit** streams PPO diagnostics and the action heatmap while a policy
trains; **Policy Replay** steps through a checkpoint's position path against
price; **Leaderboard / Compare** ranks runs by held-out Sharpe. See
[`docs/running_the_dashboard.md`](docs/running_the_dashboard.md).

## Outputs

`undertow-sim` writes its artifacts under `--output` (default `results/`; the
`train` examples above use `runs/experiment_1/`):

- `ablation_table.md` / `.csv` — RQ1 friction ablation
- `regime_matrix_*.md`, `regime_comparison.md` — RQ2 per-regime results
- `gap_report.md` / `.csv` — RQ3 simulator-vs-backtester gap
- `summary.json`, `equity_curve.csv`, `decision_log.csv`, `cost_ledger.csv` — metrics and per-step ledgers
- `runs/<experiment>/seed_<n>/` — checkpoints, manifests, telemetry and rollouts

## Datasets

The pinned window is **2022-01-01 → 2024-12-31 UTC**, split walk-forward into
2022–2023 train and 2024 eval. Two USDC/WETH pools are processed:

| Pool | Fee | Spacing | Role |
|---|---|---|---|
| `0x88e6a0…f5640` | 0.05 % | 10 | primary study pool |
| `0x8ad599…e6d8` | 0.30 % | 60 | fee-tier contrast (RQ1) |

`fetch-data` skips datasets already present, accepts `--dataset eth_usdc_500` for
a single set and `--json` for scripts. To rebuild a snapshot from scratch instead
(the only step that touches the network, requiring `GRAPH_API_KEY` and
`ETH_RPC_URL`):

```bash
uv run undertow-data snapshot --config configs/eth_usdc_500.toml
```

## Development

```bash
uv run pytest         # full suite; coverage is gated at 90 %
uv run ruff check .   # lint
uv run mypy src/undertow
```

Every logic change ships with tests, and `main` only receives reviewed PRs.
Network-marked tests (live endpoints) are skipped by default; opt in with
`uv run pytest --run-network`. The suite parallelises across CPUs via
`pytest-xdist` and pins BLAS/OpenMP to one thread per worker.

**Design principles.** Fee accrual stays in exact Q128 integers — no float on the
ground-truth path; `MarketView` is the only door to history and makes look-ahead
unrepresentable; policies are frozen at evaluation time; and every ledger records
the config hash and git commit that produced it.

## Layout

```
src/undertow/
├── data/   # ingestion, transforms, validation, parquet storage
├── sim/    # core AMM math, env, backtester, policies, training, evaluation
└── ui/     # FastAPI backend for the dashboard (React frontend under web/)
configs/    # data-pipeline and simulator TOML configs
docs/       # experiment guides, plans and ADRs
tests/      # pytest suite mirroring the package layout
```
