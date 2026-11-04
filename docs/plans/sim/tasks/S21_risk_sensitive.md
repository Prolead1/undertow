# S21 — Risk-sensitive objective (CVaR) and validation-based checkpoint selection

**Wave 8 · size M · depends on: S20b · blocks: S23**
**Branch:** `feature-sim-risk-sensitive` (stacked on `feature-sim-hedge-ratio-head`)

## Why this task exists

ADR-039 promoted the hedge ratio to a learned head, but the first 5-seed × 1M
run exposed two measurement problems the thesis cannot ship with:

1. **The objective ignores the tail.** The reward penalizes only the trailing
   *standard deviation* of per-step equity changes (`risk_penalty_lambda = 0`,
   `risk_penalty_enabled = false` in every shipped config). Under λ = 0 the
   agent has no reason to prefer a smaller drawdown, and the joint-head run
   collapsed to a seed-specific constant short.
2. **Train-reward selection over-hedges.** The cross-seed winner was chosen by
   `max(best_reward)` on the *training* curve. It picked seed 3 (constant ratio
   0.75), which on the 2024 eval split returned `-1313 USDC`, Sharpe `-1.294`,
   versus `hodl` `+735 / 0.654` and `passive_wide` `+1623 / 1.526`. A
   seed-specific constant that merely *fits the train up-market* won; the
   over-hedged model was selected because training reward had not plateaued and
   noise dominated the ranking.

ADR-030's measurement protocol is binding: the primary metric is
risk-adjusted — annualised Sharpe and CVaR at 5 % — with the risk penalty an
ablatable part of the objective. This task supplies both halves: the CVaR
reward term, the CVaR evaluation metric, and a validation-window checkpoint
rule that cannot see the eval split.

## Deliverable A — risk-sensitive objective + CVaR evaluation

1. `RewardConfig` gains `risk_penalty_mode: {"volatility","cvar"} = "volatility"`
   and `risk_cvar_alpha: float = 0.95` (validated to `(0, 1)`). The default mode
   keeps today's `-λ·σ_PnL` reward exactly.
2. `compute_reward` gains a keyword-only `pnl_cvar: float = 0.0` — the trailing
   expected shortfall of per-step equity **losses** as a positive USDC
   magnitude, so the penalty is `-λ·pnl_cvar`. `RewardBreakdown` is unchanged.
   With the default mode, or with the flag off, every returned value is
   byte-identical for the same inputs.
3. `LpEnvironment._pnl_cvar()` computes the statistic over the trailing
   `risk_rolling_window_steps` per-step equity changes
   (`del = diff(history)`, losses `-del`) at `risk_cvar_alpha`, as a positive
   magnitude (a gains-only tail clamps to 0). It reuses
   `metrics.performance.cvar_of_equity_changes` (which delegates to `cvar`) and
   is computed only when `risk_penalty_enabled and risk_penalty_mode == "cvar"`.
4. `cvar_95` is appended to `METRIC_NAMES` and to `episode_metrics`, computed on
   the same **joint** equity series the Sharpe/Sortino/maxDD already use
   (negative for losses, matching `cvar`'s convention). Both evaluation
   backends report it; the regime tables gain a `CVaR` column.

## Deliverable B — validation-based checkpoint selection

1. `TrainingConfig` gains `selection_metric: {"reward","val_sharpe","val_cvar"}
   = "reward"` (today's behaviour) and `validation_days: int = 30`.
2. `undertow-sim train` carves the **last `validation_days` of the train split**
   when the metric is not `reward`: the train view ends at the first active row
   at/after `train_end_utc - validation_days`, and the validation view covers
   the remainder. The subprocess `EnvFactory` carries the slice bounds so every
   worker — including one that rebuilt the full split from disk — trains only on
   the train lane. Calibrated MRSJD parameters are fitted on the train lane
   only. `validation_days >=` the train span is a clear error, as is an
   eval-split run.
3. After each seed's training, every periodic checkpoint
   (`checkpoints/*.zip`) plus the final `ppo_seed_<seed>.zip` is scored
   deterministically on the validation factory: `val_sharpe` is the annualised
   Sharpe and `val_cvar` the `cvar_95` statistic of the joint equity path. The
   winner is copied over the canonical `ppo_seed_<seed>.zip`; the manifest
   records `selection_metric` and `selection_value`. Higher is better for both
   (val_cvar is negative for losses, so the largest value is the smallest tail
   loss). Without a validation factory nothing changes.
4. `RunManifest` gains `selection_metric: str = "reward"` and
   `selection_value: float | None = None`; the cross-seed `best_checkpoint`
   ranks on `selection_value` when the metric is not `reward`.

## Files

```
src/undertow/sim/config.py                   (RewardConfig + TrainingConfig fields)
src/undertow/sim/env/reward.py               (pnl_cvar keyword arg)
src/undertow/sim/env/lp_env.py               (_pnl_cvar + step wiring)
src/undertow/sim/metrics/performance.py      (cvar_of_equity_changes helper)
src/undertow/sim/metrics/__init__.py         (export)
src/undertow/sim/evaluate/protocol.py        (cvar_95 metric)
src/undertow/sim/evaluate/report.py          (CVaR label)
src/undertow/sim/train/runs.py               (manifest fields + cross-seed key)
src/undertow/sim/train/envs.py               (EnvFactory slice bounds)
src/undertow/sim/train/ppo_runner.py         (checkpoint scoring + promotion)
src/undertow/sim/cli.py                      (validation-window carving)
configs/sim_eth_usdc_500_hedged_risk.toml    (new: cvar + val_sharpe)
docs/decisions/040-risk-sensitive-objective-and-validation-selection.md
docs/plans/sim/CONTRACTS.md                  (§2, §9, §15, §17 additive)
docs/plans/sim/STATUS.md                     (row + ADR)
docs/running_experiments.md                  (how to run)
```

## Tests

- reward: mode switch, enabled/disabled gating, `pnl_cvar` sign, default path
  byte-identical (field-for-field).
- env: `_pnl_cvar` exact on a hand-computed tiny history, the rolling-window
  slice, the gains-only clamp, and the `compute_reward` integration (including
  that `volatility` mode never computes the CVaR).
- metrics: `cvar_of_equity_changes` exact hand-computed value and edge cases.
- evaluation: `cvar_95` exact; present in both backends; LP-only metrics
  unchanged.
- selection: pure selector ties/errors, `val_sharpe`/`val_cvar` exact on a known
  equity path, hard error without a validation factory, a known-best synthetic
  checkpoint promoted (stale checkpoints excluded), and a tiny end-to-end
  `train_single_seed` run on the real fixture.
- CLI: carving is disjoint and exhaustive, boundary seqs, span error,
  eval-split error, and the train factory slices both backends.

## Acceptance

- Full suite `uv run pytest -q -n 4` green with the 90 % coverage gate.
- `hedge.enabled=false` / `selection_metric="reward"` remains byte-identical;
  `TestDisabledGoldenGuard` passes untouched.
- The risk config trains the joint head with a CVaR penalty and selects
  checkpoints on the carved validation window.

## Out of scope

- Tuning λ or the validation length (S23's sweep runner).
- Selecting on the eval split (forbidden; selection on it is look-ahead).
- Per-step rollout schema changes.
