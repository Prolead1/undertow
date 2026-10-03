# Running Experiments with Undertow Sim

This guide walks a fresh clone from a data snapshot to a filled RQ1–RQ3 results
directory using the `undertow-sim` command-line interface. It is self-contained:
every command is copy-pasteable, and the only prerequisite is a dataset produced
by the `undertow.data` pipeline.

The simulator package (`undertow.sim`) and the data package (`undertow.data`) are
strictly separate. `undertow-sim` **never fetches data** — it reads a snapshot
that `undertow-data snapshot` already built.

---

## Prerequisites

- Python 3.11+ (the project targets 3.11; the pinned venv is 3.14)
- [`uv`](https://docs.astral.sh/uv/) installed
- A data snapshot on disk (see below)

## Setup

```bash
git clone <repo>
cd undertow
uv sync
```

**Why plain `uv sync`?** `gymnasium` (the environment API) is a *core*
dependency, and `stable-baselines3` (the PPO trainer) lives in the `dev`
dependency group, which `uv sync` installs by default. There is no `sim`
dependency group — `sim` is an optional-extra name (`uv sync --extra sim`) and
is not required for the documented workflow.

The `undertow-sim` console script is registered by the project, so `uv run
undertow-sim …` works from the repo root.

## Build a data snapshot

`undertow-sim backtest`, `train`, `evaluate` and `ablate` all need a dataset.
Build it once with the data pipeline (this is the only step that touches the
network; it requires the endpoints' environment variables):

```bash
export GRAPH_API_KEY=...      # required by configs/eth_usdc_3000.toml
export ETH_RPC_URL=...        # archive node URL
uv run undertow-data snapshot --config configs/eth_usdc_3000.toml
```

The snapshot lands in the `[paths].output_dir` recorded in that config
(`data/datasets/eth_usdc_3000` by default). The sim CLI reads it via
`--data-config` (default: `configs/eth_usdc_3000.toml`).

---

## 1. Check your setup

`info` needs no dataset and prints the effective configuration, the pinned split
boundaries, the policy menu, the discrete action-space size, and the parity
status if a `parity_report.json` has been recorded under `--output`:

```bash
uv run undertow-sim info
uv run undertow-sim info --config configs/sim_default.toml
```

## 2. Run a backtest

The backtester replays one **frozen** baseline policy over the real event tape
with exact integer fee accrual. Choose the split with `--split` (default `eval`):

```bash
uv run undertow-sim backtest --config configs/sim_default.toml --policy hodl
uv run undertow-sim backtest --policy passive_narrow
uv run undertow-sim backtest --policy passive_wide
uv run undertow-sim backtest --policy full_range_v2
uv run undertow-sim backtest --policy tau_reset
uv run undertow-sim backtest --policy cost_aware
```

Artifacts are written to `results/` (override with `--output DIR`):

- `equity_curve.csv` — per-step equity, cumulative PnL, per-step fees/gas/slippage/IL
- `decision_log.csv` — every decision, its resolved range, and its costs
- `cost_ledger.csv` — the cost terms by type
- `summary.json` — headline metrics, PnL decomposition and run provenance

## 3. Train an RL agent

```bash
uv run undertow-sim train --config configs/sim_default.toml --output runs/experiment_1/
```

`train` runs PPO once per configured seed (pinned to `0,1,2,3,4`), writing
checkpoints and a `manifest.json` (config hash, git commit, library versions,
learning curve) under `runs/experiment_1/seed_<n>/`. A `training_summary.json`
records the best checkpoint. `--split` defaults to `train`; the calibrated price
mode (`[price] mode = "calibrated"` in the config) fits its MRSJD on the train
split only.

Training time depends on `training.total_timesteps` and `training.parallel_envs`
in the config (defaults: 1,000,000 timesteps, 4 envs per seed, 5 seeds).

## 4. Evaluate

Evaluate a trained checkpoint together with every baseline over the walk-forward
episodes of the split:

```bash
uv run undertow-sim evaluate --config configs/sim_default.toml \
    --checkpoint runs/experiment_1/seed_0/ppo_seed_0.zip --output results/
```

This writes the RQ2 per-regime matrices (`regime_matrix_*.md`,
`regime_comparison.md`) and the RQ3 gap report (`gap_report.md`,
`gap_report.csv`, `results_summary.md`).

## 5. Run the ablation (RQ1)

```bash
uv run undertow-sim ablate --config configs/sim_default.toml --output results/
uv run undertow-sim ablate --policy full_range_v2 --output results/
```

`ablate` runs the `{full, no_gas, no_slippage, no_il, static_fee}` friction
variants for the chosen baseline and writes `ablation_table.md` /
`ablation_table.csv`. (The RQ3 gap report is produced by `evaluate`; a baseline
band is not representable in the agent's discrete action grid, so `ablate`
deliberately does not fabricate one.)

## 6. View results

Everything is under the `--output` directory (default `results/`):

- `ablation_table.md` / `ablation_table.csv` — RQ1: friction ablation
- `regime_matrix_*.md`, `regime_comparison.md` — RQ2: per-regime results
- `gap_report.md` / `gap_report.csv` — RQ3: sim-to-reality gap
- `results_summary.md` — one-page summary

## Reproducing exact thesis results

The pinned config (`configs/sim_default.toml`), the pinned split
(`2022-01-01 → 2023-12-31` train, `2024-01-01 → 2024-12-31` eval, UTC) and the
pinned seeds (`0–4`) make every artifact reproducible: run the commands above in
order with the default config. Each backtest ledger and training manifest records
its `config_hash` and git commit so a result can always be traced to the code and
configuration that produced it.

## Troubleshooting

- **`data config not found` / `could not load the … split`** — build the
  snapshot first: `uv run undertow-data snapshot --config <data-config>`.
- **`config file not found`** — check the `--config` path; the default is
  `configs/sim_default.toml`.
- **`unknown policy '…'`** — pick one of the six listed in the error / `info`:
  `hodl`, `passive_narrow`, `passive_wide`, `full_range_v2`, `tau_reset`,
  `cost_aware`.
- **`Environment variable … is not set`** — the data config contains a
  `${ENV_VAR}` placeholder; export it before running `undertow-data snapshot`.
- **`Parity check failed` / parity `FAIL`** — run `uv run undertow-sim info` to
  inspect the recorded parity status, and re-run the S14 parity suite.
- **Training too slow** — reduce `training.total_timesteps` or increase
  `training.parallel_envs` in the config.
