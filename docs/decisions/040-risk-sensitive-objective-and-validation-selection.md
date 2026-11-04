# ADR 040 — Risk-sensitive objective (CVaR) and validation-based checkpoint selection

**Status:** accepted (implementation)
**Affects:** `src/undertow/sim/config.py` · `env/reward.py` · `env/lp_env.py` · `metrics/performance.py` · `evaluate/protocol.py` · `evaluate/report.py` · `train/runs.py` · `train/envs.py` · `train/ppo_runner.py` · `cli.py` · `CONTRACTS.md` §2/§9/§15/§17 · `configs/sim_eth_usdc_500_hedged_risk.toml`
**Raised by:** ADR-030's measurement protocol (Sharpe / CVaR primary) and the ADR-039 joint-head experiment evidence

## Context

ADR-039 gave the agent a joint `(LP, hedge-ratio)` head and measured the first
5-seed × 1M-step run on `configs/sim_eth_usdc_500_hedged.toml`:

- Every seed held the LP 100 % and chose a seed-specific, near-constant hedge
  ratio (0.25–0.75); no seed had plateaued at 1M steps.
- The cross-seed winner by the manifest's `best_reward` (the maximum mean
  episode return on the **training** curve) was seed 3 — the constant ratio 0.75
  seed. On the 2024 eval split that policy returned `-1312.99 USDC`, Sharpe
  `-1.294`, maxDD `-0.0416`, while `hodl` returned `+735.21 / 0.654 / -0.1221`
  and `passive_wide` `+1622.95 / 1.526 / -0.1012`. The selected seed is the
  worst plausible constant ratio for that market.
- Hedging was monotone in the eval tail (maxDD `-0.340` unhedged → `-0.0145`
  fully hedged in the high-vol regime) at a large return cost: in the 2024
  up-market the funding carry does not cover the short's mark loss.

Two causes are structural, not seed luck:

1. **The objective never asked for tail protection.** The reward's risk term is
   `-λ · std(delta equity)` and every shipped config sets
   `risk_penalty_enabled = false` / `λ = 0` (S09, ADR-030 §4). Under λ = 0 the
   agent is indifferent to drawdown, so the hedge head can only chase expected
   excess-vs-HODL, which 2024 punishes.
2. **Model selection used the wrong signal.** `train_ppo` picked the seed with
   `max(best_reward)` measured on the training learning curve. A per-seed
   constant that fits the train up-market can win that ranking while being the
   over-hedged model out of sample; there was no held-out (but still not eval)
   window to choose on.

ADR-030 declares Sharpe and CVaR at 5 % the primary statistics, with the risk
penalty an ablatable part of the objective. The evaluation already had the
machinery for a CVaR metric (`metrics.performance.cvar`), but it was never
wired into the reward, the per-episode metric set, or checkpoint selection.

## Decision

### 1. The risk penalty gains a CVaR mode (A)

`RewardConfig` appends two additively-defaulted fields:

```python
risk_penalty_mode: Literal["volatility", "cvar"] = "volatility"
risk_cvar_alpha: float = 0.95                       # in (0, 1)
```

`compute_reward` appends a keyword-only `pnl_cvar: float = 0.0`, the trailing
empirical expected shortfall of per-step equity **losses** as a **positive
magnitude**:

```
risk_penalty = -lambda * (pnl_cvar if mode == "cvar" else pnl_volatility)
```

so a larger tail loss produces a more negative reward, exactly like the
historical `-λ·σ_PnL` term. The default mode, and `risk_penalty_enabled=false`,
produce byte-identical breakdowns for the same inputs; `RewardBreakdown` is
unchanged.

`LpEnvironment._pnl_cvar()` computes the statistic only when
`risk_penalty_enabled and risk_penalty_mode == "cvar"` (zero cost otherwise).
It takes `del = diff(history)` over the trailing
`risk_rolling_window_steps`, treats `-del` as the loss, and returns the mean of
the worst `1 - alpha` fraction as a non-negative magnitude (a gains-only tail
has no loss to penalize and clamps to 0). It reuses
`metrics.performance.cvar_of_equity_changes`, which delegates to the existing
`cvar` order statistic, so the reward and the evaluation cannot drift apart.
The history it diffs is the **joint (LP + hedge) equity** path
(`_joint_equity_history`, appended after the hedge is marked), so the penalty
varies with the hedge ratio and the S21 experiment can actually test it; the
historical `volatility` mode keeps its LP-only `_pnl_history` semantics
byte-identical (ADR-036/ADR-039).

### 2. The evaluation reports `cvar_95` (A)

`cvar_95` is appended to `METRIC_NAMES` and computed in `episode_metrics` on
the same **joint** equity series (LP + hedge) the other risk metrics use:
`cvar_of_equity_changes(equity)` is the mean of the worst 5 % of per-step
equity changes, negative for losses. It is deliberately the *change*
distribution, not `pct_change` returns: that is the quantity the reward's
`pnl_cvar` term controls, so the primary metric and the objective report the
same tail. Both evaluation backends (exact backtester and replay simulator)
already produce the joint series from S20b, so both records carry the metric.
The regime tables gain a `CVaR` column; no existing metric is renamed or
renumbered.

### 3. Checkpoints are selected on a carved validation window (B)

`TrainingConfig` appends:

```python
selection_metric: Literal["reward", "val_sharpe", "val_cvar"] = "reward"
validation_days: int = 30
```

With the default `"reward"` nothing changes. Otherwise the CLI carves the
**last `validation_days` of the train split**:

- the validation start is the first active row at or after
  `active_end_utc - validation_days`;
- `train_view` is `[min_seq, validation_start_seq)` and `validation_view` is
  `[validation_start_seq, max_seq + 1)`, disjoint and exhaustive;
- the subprocess `EnvFactory` carries the slice bounds, so a worker that
  rebuilds the full split from disk still trains only on the train lane (the
  published IPC frames stay unbounded; the carve's prefix slice takes
  `MarketView.slice`'s cheap `head` path instead of a full-lane filter);
  calibrated MRSJD parameters are fitted on the train lane only and reused for
  the validation environment, so checkpoints are scored on the training-time
  price process rather than one re-fitted on the validation rows;
- `validation_days` at least the train span, or a run on the eval split, is a
  clear `CliError`.

After each seed, every periodic checkpoint written by that run plus the final
`ppo_seed_<seed>.zip` is scored deterministically (greedy, fixed reset seed) on
a small in-process validation environment over the joint equity path:
`val_sharpe` = annualised Sharpe, `val_cvar` = `cvar_95`. Both rank
higher-is-better because `val_cvar` is negative for losses — the largest value
is the smallest tail loss. The winner is copied over the canonical
`ppo_seed_<seed>.zip`, and the manifest records `selection_metric` /
`selection_value`. `train_ppo`'s cross-seed `best_checkpoint` ranks on
`selection_value` when the metric is not `reward`. The recorded rollout pass
uses the selected model. A `selection_metric` other than `reward` without a
validation factory is a hard `SimConfigError` rather than a silent fallback.

### 4. The risk config (B)

`configs/sim_eth_usdc_500_hedged_risk.toml` is the S20b hedged config with
`risk_penalty_enabled=true`, `risk_penalty_mode="cvar"`,
`risk_penalty_lambda=0.05` and `training.selection_metric="val_sharpe"`. The
shipped LP-only configs are untouched. λ = 0.05 is a stated calibration, not an
optimum: with a 100k book the trailing 95 % CVaR is O(100–1000 USDC), so the
tail-step penalty is O(5–50 USDC) — comparable to a step's fee/gas terms and
smaller than a full tail loss, enough to tilt the hedge without swamping the
excess-vs-HODL signal. It must be swept (S23) before any claim.

## Consequences

- **The default paths are byte-identical.** `risk_penalty_mode="volatility"`
  plus `pnl_cvar=0.0` reproduces every pre-S21 reward value;
  `selection_metric="reward"` and no validation factory reproduce the training
  loop, the manifest (additive fields default to `"reward"`/`None`) and the
  cross-seed ranking exactly. `TestDisabledGoldenGuard` passes untouched.
- **The evaluation's metric set grew by one appended key.** Consumers that
  iterate `METRIC_NAMES` (regime tables, ablation CSVs, `summary.json`'s
  `_SUMMARY_METRICS` subset is untouched) see the new column; existing keys keep
  their order.
- **Selection is legal.** The validation window comes from the train split; the
  eval split is read only by the final walk-forward evaluation. This is the
  honest way to fix the ADR-039 selection failure.
- **The cost of selection is bounded.** One deterministic episode per checkpoint
  (with the default 30-day window and 30-day episode, the whole validation
  slice) per seed; with the pinned 10 checkpoints that is ~10 × 4320 extra
  single-process decisions per seed, small next to the 1M-step training run.
- **The CVaR term is a preference, not an economic flow.** It is excluded from
  `pnl_net` and from the ungated economics payload, so the S14 parity harness
  and the RQ1 ablations are unaffected.
- **A gain-only tail yields `cvar_95 > 0` and `_pnl_cvar == 0`.** The
  evaluation metric stays signed (it describes the distribution); the reward
  penalty clamps, because a negative loss magnitude would *reward* risk.

## Rejected alternatives

- **Keep reward-based checkpoint selection.** ADR-039 is the counter-example:
  the train-reward winner was the over-hedged seed and the worst plausible
  constant on 2024. The learning curve is also still improving at 1M, so its
  maximum is a noise statistic, not a converged performance estimate.
- **Select on the eval split.** Forbidden: it is the reported out-of-sample
  window, and selecting on it converts the headline number into a training
  result. The validation window must be carved from train.
- **Use a fraction of the train split instead of `validation_days`.** Days are
  the thesis's wall-clock unit, match `EpisodeConfig.duration_days`, and make
  the carve reproducible from the config alone.
- **Rank checkpoints by the training `best_reward` but penalize CVaR in the
  reward only.** The reward term changes what the policy learns, not how the
  seed/checkpoint is chosen; ADR-039's failure is a selection failure, so both
  fixes are required.
- **Use returns (`pct_change`) for `cvar_95`.** The reward's `pnl_cvar` is an
  absolute per-step equity change, and the existing volatility term is too; a
  return-based evaluation CVaR would report a different quantity from the one
  the objective controls. The sign convention (`cvar`, negative for losses) is
  unchanged.
- **Evaluate every checkpoint on multiple random validation seeds.** Fairness
  across seeds and reproducibility matter more than coverage here; one fixed
  deterministic window per checkpoint keeps the comparison paired and the cost
  bounded. More coverage is an S23 sweep question.
- **Let a missing validation factory silently fall back to reward selection.**
  A config that asks for validation selection and silently gets reward
  selection is a measurement bug; it raises instead.
