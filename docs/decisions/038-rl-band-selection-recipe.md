# ADR 038 — RL band-selection recipe: one-shot band choice under a rebalance interval and a near-undiscounted horizon

**Status:** accepted (pilot-validated; full 5-seed × 1M evaluation complete)
**Affects:** `configs/sim_eth_usdc_500_rl.toml` · `tests/sim/test_config.py` · ADR-011, ADR-036, ADR-037
**Raised by:** RL band-selection pilot on the primary 0.05 % pool

## Context

After ADR-037 added `episode.min_rebalance_interval_steps`, no config set it,
and the strongest primary-pool learned policy was still degenerate: the
ADR-036 stabilized run escaped the churn attractor by holding 100 % of steps
and merely tied `hodl` (+735.2 USDC on the held-out 2024 split). The original
91-action run and the reduced-grid churning pilot lost **-86,889** and
**-80,642 USDC** respectively on the same split because they moved the band
every few steps and paid gas/slippage for it. Band *selection* — the policy
actually choosing a width — had never been demonstrated: the reward gradient
was dominated by the rebalance decision, not the band choice.

The interval constraint removes the rebalance decision entirely: with a
positive interval the environment coerces early rebalance attempts to `hold`,
and the checkpoint wrapper mirrors the gate so simulator and backtester agree
(ADR-037). That leaves a clean, low-dimensional question for PPO: *which band
should the step-0 deployment pick?*

## Problem

The per-step learning reward is far too noisy to teach a band choice. Fee
income, IL and slippage over a single 10-minute step are small and dominated
by the price path, while the band-width choice only pays off in aggregate
over the multi-day episode. With the default `discount_gamma = 0.99` the
effective horizon is `1 / (1 - 0.99) = 100` steps (≈ 17 hours), and the
step-0 choice is discounted to `0.99 ** 4320 ≈ 1e-19` by episode end, so it
is effectively invisible to the value targets. On top of that, the
`hold_bias = 4.0` prior from ADR-036 keeps biasing the policy toward `hold`
even after the interval gate has made churn impossible, and the 28-action
grid spends almost all of its probability mass on widen/recentre arms the
one-shot setting can never use.

So the previous recipe can only converge to the trivially safe action: hold.
The band width is never compared.

## Decision

1. **Ship a dedicated config, `configs/sim_eth_usdc_500_rl.toml`, derived
   from `configs/sim_eth_usdc_500.toml`.** It changes exactly the fields
   below; gas, slippage, reward, split windows, backtest and price mode are
   identical, so the baselines and the walk-forward evaluation are unaffected.
2. **One rebalance per episode.** `episode.min_rebalance_interval_steps =
   1000000000`; `reset()` arms the gate at `-min_interval`, so the first
   rebalance attempt is always allowed and every later attempt is coerced to
   `hold`. The policy deploys once (at step 0) and holds.
3. **A 4-action one-shot band grid.** `action_grid.center_offsets = [0]` and
   `action_grid.widths = [50, 250, 1000]` with `include_hold = true`, i.e.
   hold + 1 offset × 3 widths = **4 actions**. The agent picks a band at
   step 0; keeping a single offset isolates the width choice, and recentring
   is deferred to the future active-timing workstream.
4. **Near-undiscounted gamma.** `training.discount_gamma = 0.9999`: the
   effective horizon becomes ~10,000 steps (≈ 69 days at the 10-minute
   cadence), longer than the 4320-step episode, so the step-0 choice stays
   in the value targets without truncation bias.
5. **Keep ADR-036's stabilized update recipe** (`learning_rate = 0.0001`,
   `batch_size = 512`, `n_epochs = 4`, `ent_coef = 0.001`, `target_kl =
   0.01`, `checkpoint_freq_steps = 100000`), and set `hold_bias = 0.0` — the
   interval gate already prevents churn, so no hold prior is needed and it
   would only mask the band comparison. Seeds, budget and vectorisation stay
   at the primary values (`seeds = [0, 1, 2, 3, 4]`, `total_timesteps =
   1000000`, `log_freq_steps = 10000`, `parallel_envs = 4`, `vec_env =
   "subproc"`).
6. **No code change is needed.** ADR-037's interval gate and the existing
   config loader carry the whole recipe; the new file is a config-level
   override, exactly as ADR-036 did for the update knobs.

## Consequences

- **Pilot evidence (seed 0, 1M steps, held-out 2024 walk-forward split).**
  The greedy rollouts rebalance exactly once — at step 0, to width 1000 — and
  then hold for all 4320 steps. The recorded rollout parquet under
  `results/pilot_gated/seed_0/rollouts/` labels that deployment decision
  `step=1`; its `step=0` row is the reset-time pre-action placeholder, so the
  single `rebalance` row at `step=1` is the step-0 deployment claimed here.
  The checkpoint scores **+2,146.6 USDC**
  (Sharpe 0.942, maxDD -0.097, 1 rebalance/episode, gas -$36.4), versus
  `hodl` +735.2, `passive_narrow` +1,082, `passive_wide` +1,623 and
  `full_range_v2` +2,183; the previous churning run was **-86,889 USDC** on
  the same split. This is the first learned primary-pool policy that beats
  `hodl` without collapsing to it.
- **Full 5-seed × 1M evaluation.** `results/eth_usdc_500_full_rl` ran the
  shipped `configs/sim_eth_usdc_500_rl.toml` (hash `7b007d2cf07abf27`) for
  seeds 0–4 at 1M steps. Every seed learned the one-shot structure: at most
  one executed rebalance per episode, then `hold` for the rest. Seeds 0, 1,
  2 and 4 deploy at step 0; seed 3 instead holds its reset-time initial band
  and makes its single rebalance around step 3440–3525 (one of its 13
  evaluation windows never rebalances). The bandit settles on different
  widths across seeds: seeds 0 and 2 → width 1000, seed 4 → 250, seeds 1 and
  3 → 50.
- **Per-seed 2024 walk-forward (backtester, one seed, 13 windows).** Seed 0
  +2,146.6 (Sharpe 0.942, maxDD -0.097), seed 1 +1,082.3 (1.296, -0.113),
  seed 2 +2,146.6 (0.942, -0.097), seed 3 +805.5 (0.591, -0.121), seed 4
  +1,731.7 (1.591, -0.101). All five beat `hodl` (+735.2); mean +1,582.5 and
  the best risk-adjusted seed is 4 (Sharpe 1.591).
- **Official evaluation of the `best_reward` checkpoint (seed 0).** Total PnL
  **+2,146.6 USDC** (Sharpe 0.942, maxDD -0.097, 1 rebalance/episode, gas
  -$36.4), versus `full_range_v2` +2,183 (Sharpe 0.639), `passive_wide`
  +1,623 (1.526), `passive_narrow` +1,082 (1.296), `hodl` +735 (0.654),
  `cost_aware` -1,992 and `tau_reset` -17,680. The sim-to-backtest gap is
  9.85 USDC (sim 2,156.42 vs on-chain 2,146.56; fees -0.08, gas 0.00,
  slippage -0.36, IL -2.60) and the best regime is bull +6,948.3.
- **Honest limitation.** The learned policy is a one-shot band deploy under
  the once-per-episode constraint: the 1000-wide seeds beat `hodl` by ~2.9×
  and match `full_range_v2` to within $36 with a higher Sharpe (0.942 vs
  0.639), but they do not learn active timing or width switching beyond the
  initial choice. The bandit's exploration noise leaves two of five seeds on
  the narrower 50-tick band; widening the width ladder and extending the
  budget are the next steps.
- **Pilot provenance vs. shipped config.** The pilot ran a temporary
  `/tmp/pilot_gated.toml` (`seeds = [0]`, `checkpoint_freq_steps = 200000`,
  config hash `faa6ce551b573ea7`); the shipped config trains seeds 0–4 with
  `checkpoint_freq_steps = 100000`. Checkpoint cadence only controls snapshot
  frequency, not the optimisation trajectory, so it does not change training
  dynamics and the pilot's result carries over to the shipped recipe.
- **The shipped config is pinned by a unit test.**
  `tests/sim/test_config.py::test_shipped_rl_config_carries_the_band_selection_recipe`
  loads `configs/sim_eth_usdc_500_rl.toml` and asserts the action-space size
  (`1 + 1 × 3 = 4`), the 1e9 interval, `discount_gamma = 0.9999`,
  `hold_bias = 0.0`, the stabilized update knobs and the five seeds, so the
  recipe cannot drift silently.
- **Other configs keep their behaviour.** `min_rebalance_interval_steps`
  defaults to `0` and no other TOML sets it, so baselines, parity runs, the
  0.30 % contrast pool and `configs/sim_eth_usdc_500.toml` are unchanged
  (ADR-037). The new config's `SimConfig` hash is fresh, as expected.
- **The interval is for learned-policy runs only.** The exact backtester has
  no gate, so baseline comparisons must keep using their own interval-0
  configs; the checkpoint evaluator's wrapper mirrors the env gate, which is
  what keeps the pilot numbers sim/backtest-faithful (ADR-031, ADR-037).

## Rejected alternatives

- **Reward-shaping a band-choice signal instead of fixing the horizon.**
  Makes the learned objective differ from the measured PnL and adds a second
  policy-visible target; the near-undiscounted gamma addresses the credit
  assignment without touching the reward the evaluation measures.
- **A shorter interval (e.g. daily, 144 steps) instead of one-shot.**
  Still lets the policy churn daily and re-ignites the attractor; band width
  and rebalance timing would remain entangled.
- **Keeping the 28-action grid and `hold_bias=4.0`.** Both keep pushing
  probability mass to `hold` and to unusable recentre arms, so the width
  comparison stays under-sampled even with the interval gate.
- **Including centre offsets in the one-shot grid.** Multiplies the arms and
  adds a second, currently meaningless choice dimension; offsets matter only
  once active timing exists.
- **Keeping `discount_gamma = 0.99`.** The step-0 choice is invisible to the
  value targets; the pilot's only lever on the band signal is the horizon.
