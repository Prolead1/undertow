# ADR 036 — Stabilize PPO updates: configurable optimizer knobs, a hold prior, and a reduced primary action grid

**Status:** accepted (implementation)
**Affects:** `src/undertow/sim/config.py` · `src/undertow/sim/train/ppo_runner.py` · `configs/sim_eth_usdc_500.toml` · sim `CONTRACTS.md` §15 · ADR-011, ADR-022, ADR-034
**Raised by:** primary 0.05 % PPO thrashing diagnostics

## Context

The primary 0.05 % pool (`configs/sim_eth_usdc_500.toml`) still thrashed after
ADR-034's observation compression and the `hold_bias` prior: the hold fraction
flipped between ~0.51 and ~0.01 update-to-update. Capturing the rollout buffer
during real training settled the diagnosis: the *signal* was correct — the
captured batches had `adv_hold > 0` and `adv_reb < 0` — but the updates were
oscillating. A scripted hold-only policy on the same sampled train episode
returns **+0.0885**, the reference a stable policy must reproduce.

The update-size knobs were invisible to the config. `ppo_runner` hard-coded
`learning_rate=3e-4`, `batch_size=64` and `n_epochs=10`, left `ent_coef` at
SB3 2.9's PPO default of `0.0` (the kwarg was never passed), and never set a
KL stop, so a run could not trade update size for stability without editing
code. Ten epochs over 64-sample mini-batches on
the ~2048-step rollout of a 91-action problem can move the policy much further
per rollout than the behaviour policy that collected it — exactly the
oscillation observed.

Independently, ADR-011's full grid (`{hold} ∪ 9 offsets × 10 widths`) makes
`hold` a priori ~1 % likely. Even with a logit prior, the consensus action is
sampled too rarely for the advantage estimate to consolidate it before entropy
collapses.

## Decision

1. **Promote the PPO optimisation knobs to `TrainingConfig` fields** with
   validation: `learning_rate: float = 3e-4`, `batch_size: int = 64`,
   `n_epochs: int = 10`, `ent_coef: float = 0.0`,
   `target_kl: float | None = None`. The `learning_rate`/`batch_size`/
   `n_epochs` defaults are the PLAN.md §7 anchor values and `ent_coef` is
   SB3 2.9's PPO default, so together they preserve the historical behaviour
   exactly. Validation additionally requires `batch_size`/`n_epochs` to be
   finite integers. `_derive_rollout` now
   takes `batch_size` (snapped to a divisor of the rollout) and `n_epochs` from
   the config; `_train_single_seed` passes `learning_rate`, `ent_coef` and
   `target_kl` through to SB3. `PPO_N_STEPS` stays the rollout anchor.
2. **Set the stabilized recipe on the primary pool only**:
   `learning_rate=1e-4`, `batch_size=512`, `n_epochs=4`, `ent_coef=0.001`,
   `target_kl=0.01`.
3. **Keep the hold prior**: `hold_bias=4.0` on the reduced grid, so `hold` is
   explorable from the first update while the advantage signal is consolidated.
4. **Reduce the primary action grid to 28 actions**: `center_offsets=[-1, 0, 1]`
   × `widths=[2, 5, 10, 25, 50, 100, 250, 500, 1000]` plus `hold`. ADR-011's
   factored encoding is unchanged; this is a config-level reduction for this
   pool: keeping the ±1-spacing recenter and a 2–1000 spacing width ladder,
   and dropping the large-offset recenters and the 1-spacing width.

## Consequences

- **The recipe works on the real train env.** Two independent stabilizations
  converged to the same 100 % hold policy; with `batch_size=512`, `n_epochs=4`,
  `learning_rate=1e-4`, `target_kl=0.01`, `ent_coef=0.001` and `hold_bias=4`,
  the greedy policy held all 4320 steps and matched the scripted hold-only
  return (+0.0885) exactly.
- **Evidence on the held-out 2024 walk-forward split.** The stabilized
  200k-step checkpoint scored **+735.2 USDC** with 0 rebalances and 0 gas,
  identical to the `hodl` baseline, versus **-80,642 USDC** for the
  reduced-grid churning pilot (`pilot_holdbias` checkpoint, same split,
  28-action grid; `logs/eth_usdc_500/pilot-eval-*.log`) and **-86,889 USDC**
  for the original 91-action run (`results/eth_usdc_500_main`). The full
  5-seed × 1M-step budget (`results/eth_usdc_500_rl`, commit `fb740ac`;
  manifest HEAD `afcb68f` is identical `src/` training code; config hash
  `78db78f032d34abc`) reproduced the outcome: every seed's 5 deterministic
  rollouts (25/25 episodes) held 100 % of steps, final `ep_rew` ranged -0.17
  to -0.29 across seeds (previous churning run: ≈ -0.95 to -0.97), with KL ~0.001–0.002
  and clip fraction ~0.02; `best_reward` selected seed 4 (-0.1669). On the
  same held-out 2024 split `ppo_seed_4` scored **+735.2 USDC** (Sharpe 0.654,
  maxDD -0.122, 0 rebalances, 0 gas) — identical to `hodl`; baselines for
  context: `full_range_v2` +2,183 (0.639), `passive_wide` +1,623 (1.526),
  `passive_narrow` +1,082 (1.296), `cost_aware` -1,992, `tau_reset` -17,680.
  Sim-to-backtest gap: 742.96 sim vs 735.21 on-chain = 7.75 USDC
  (fees +0.61, gas/slippage 0.00, IL -5.78). Honest limitation: the learned
  policy is a pass-hold policy — positive, stable and non-churning — but it
  ties `hodl` and does not beat `passive_wide`/`full_range_v2`; band
  selection (learning to widen) is the next iteration.
- **Old 91-action checkpoints are incompatible** with the 28-action grid: SB3's
  action-head shape guard raises on load. Pre-change primary runs must be
  retrained and their results are not comparable; the manifest's `git_commit`
  remains the audit trail.
- **Baselines and walk-forward windows are unaffected.** The baselines are
  scripted evaluation references and never consume `ActionGridConfig`
  (ADR-011); the `[split]` train/eval windows are untouched, so the evaluation
  set is identical.
- **PLAN.md §7 anchor values remain the code defaults.** The primary config is
  a documented, config-level override, so `configs/sim_default.toml` and the
  0.30 % contrast config keep the anchor recipe (`ent_coef` stays at SB3's
  `0.0` default), and the default-action-space CLI/`info` behaviour is
  unchanged.
- `TrainingConfig` validation now rejects non-finite/non-integral
  `batch_size`/`n_epochs` (and `batch_size < 2`/`n_epochs < 1`), a non-positive
  or non-finite `learning_rate`, a negative or non-finite `ent_coef` (so the
  SB3 default `0.0` stays accepted), and a non-positive or non-finite
  `target_kl` (`None` disables the stop); `CONTRACTS.md` §15 lists the new
  fields.
- The primary config's hash changes, so fresh run ids are expected.

## Rejected alternatives

- **A larger hold prior instead of stabilizing the update.** The captured
  advantages were already correct; a bigger prior would hide the oscillation
  and bias the policy away from the evidence rather than fix it.
- **Lowering `n_steps` rather than the batch/epoch/learning rate.** It reduces
  samples per update but leaves the effective step size and sample re-use
  unchanged; the smaller step and KL stop attack the actual instability.
- **Changing the shared code defaults to the stabilized recipe.** That would
  silently change every config and contradict PLAN §7's citable anchor; the
  override belongs in the primary config.
