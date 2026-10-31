# ADR 037 — Minimum rebalance interval: coerce learned-policy churn in the env and the checkpoint wrapper

**Status:** accepted (implementation)
**Affects:** `src/undertow/sim/config.py` · `src/undertow/sim/env/lp_env.py` · `src/undertow/sim/cli.py` · sim `CONTRACTS.md` §2 · ADR-011, ADR-031, ADR-035, ADR-036
**Raised by:** churn-attractor diagnostics on the primary 0.05 % PPO run

## Context

Transaction costs (`gas + slippage` on every rebalance) make high-frequency
rebalancing uneconomic, but nothing in the environment prevented the learned
policy from attempting it on every decision step. On the primary 0.05 % pool
the churning policies lost heavily: the original 91-action PPO run scored
**-86,889 USDC** on the held-out 2024 split and the reduced-grid churning pilot
(scored **-80,642 USDC**) moved the band constantly instead of learning band
selection. ADR-036's diagnostics showed the correct advantage signal existed
(`adv_hold > 0`, `adv_reb < 0`) but the policy was stuck in a *churn
attractor*: once high-frequency rebalancing dominated the rollout, its gas and
slippage sunk costs poisoned the value baseline and PPO could not consolidate a
hold policy, let alone compare bands. The stabilized recipe (ADR-036) escaped
the attractor only by collapsing to 100 % hold — it ties `hodl` and does not
demonstrate band selection.

`CostAwareRebalancePolicy` already encodes a scripted "at most daily" rule
(`DEFAULT_MIN_INTERVAL_STEPS = 144` at the 10-minute cadence) for exactly this
cost reason, but no equivalent lever existed for the RL policy, and none that
the checkpoint evaluator understood.

## Problem

Without a rate limit, any learned policy that samples rebalances at a
meaningful rate pays repeated full rebalance costs, so the environment's
reward makes wide, rarely-touched bands look worse than they are and the
gradient pushes toward hold/churn extremes rather than band width. Worse, a
constraint applied only inside `LpEnvironment` would not survive evaluation:
the backtester (`run_backtest`) drives the checkpoint through
`_CheckpointPolicy`, which maps model indices to `Action`s *before* the
backtester's own engine sees anything. If the env gated but the wrapper did
not, the same checkpoint would churn in the backtest and hold in the
simulator, breaking the sim/backtest parity harness (ADR-031 / S22).

## Decision

1. **Add `EpisodeConfig.min_rebalance_interval_steps: int = 0`.** `0` disables
   the constraint and preserves the historical behaviour byte-for-byte. A
   positive value is the minimum number of decision steps between rebalance
   *attempts* for the learned policy. `__post_init__` requires a finite,
   non-negative integer and raises `SimConfigError` naming
   `episode.min_rebalance_interval_steps` otherwise. TOML loading picks the
   field up automatically through `_populate`.
2. **`LpEnvironment` enforces it.** It tracks `_last_rebalance_attempt_step`,
   re-armed in `reset()` to `-min_interval` so the first action of an episode
   is always allowed. In `step()`, immediately after
   `action_for_index(action_idx)`, a rebalance attempt while
   `_step_count - _last_rebalance_attempt_step < min_interval` is replaced by
   the hold action and flagged as `rebalance_gated`. Every attempt the gate
   allows records `_last_rebalance_attempt_step = _step_count`, **whether or
   not the trade executes** (an unaffordable rebalance still consumed the
   attempt) — the wrapper can mirror attempts, not affordability. `info`
   surfaces `"rebalance_gated": True`; `info["action_type"]` describes the
   executed hold while `info["action_index"]` keeps the model's requested
   index, matching the existing `rebalance_rejected` convention.
3. **`_CheckpointPolicy` mirrors it exactly.** It reads
   `config.episode.min_rebalance_interval_steps`, resets `_step = 0` and
   `_last_rebalance_attempt_step = -min_interval` in `reset()`, and in `act()`
   returns `self._actions[0]` for an early rebalance attempt, recording an
   allowed attempt otherwise; `_step` advances on every `act()`. Both
   `run_backtest` and the walk-forward simulator call `policy.reset()` exactly
   once per episode, so the wrapper and the environment see the same decision
   count.
4. **The knob is for learned-policy runs; keep it at `0` for baselines.**
   The scripted baselines (`HODLPolicy`, `PassiveNarrowPolicy`,
   `PassiveWidePolicy`, `FullRangeV2Policy`, `TauResetPolicy`,
   `CostAwareRebalancePolicy`) never read
   `episode.min_rebalance_interval_steps`, but `LpEnvironment.step` is
   policy-agnostic: it coerces early rebalance attempts for *any* policy
   stepped in a positive-interval env, including a scripted baseline under the
   replay simulator.  The exact backtester has no gate, so baseline evaluations
   and parity runs must keep the interval at `0` for like-for-like
   comparisons.  For the learned policy, `_CheckpointPolicy` mirrors the env
   gate exactly, which is what keeps the checkpoint's simulator and backtester
   behaviour aligned at a positive interval.
   `CostAwareRebalancePolicy`'s own internal 144-step interval is part of its
   scripted cost-aware rule and is unchanged.
5. **No config file sets the field yet.** The default `0` keeps every existing
   recipe identical, so this ADR adds the lever without changing current
   results.

## Consequences

- `LpEnvironment` gains `_last_rebalance_attempt_step` and a gate in `step()`;
  `info` gains `rebalance_gated` (only on gated steps, so
  `min_rebalance_interval_steps == 0` output is unchanged); `_CheckpointPolicy`
  becomes stateful across `reset()`/`act()`.
- A run that sets a positive interval deliberately changes the action
  distribution and therefore the numbers; `0` reproduces all pre-ADR-037
  results. Because `config_hash` serialises the dataclass tree, every
  `SimConfig` hash changes once the field exists, so fresh run ids are
  expected even for default-`0` runs (the same effect ADR-036 had).
- Sim/backtest parity holds by construction: the env and the wrapper apply the
  same state machine, and the parity harness (`_run_sim`) also resets the
  policy once per episode. The text-based rollouts pass in `ppo_runner` calls
  `model.predict` and steps `LpEnvironment` directly, so it inherits the env
  gate without needing a second adapter; `_CheckpointPolicy` remains the only
  model→`Action` adapter.
- The RQ1 story is cleaner: the constraint is a documented transaction-cost
  realism rule (a mandatory cooldown, like on-chain position management
  practice), not a reward-shaping hack, so a policy that learns within it is
  still comparable against the unconstrained scripted baselines.
- The next iteration should sweep a non-zero interval on the primary config
  now that band selection is reachable; the field exists for that sweep.

## Rejected alternatives

- **A reward-shaping penalty on rebalances.** Adds a second, policy-visible
  objective and makes the learned objective differ from the measured PnL; the
  cost of churn is already in the reward. A hard cooldown is the physical
  rule the agent should plan around.
- **Gating only in the environment.** Breaks backtester/simulator parity for
  the same checkpoint, which S22's harness explicitly guards.
- **Reusing `CostAwareRebalancePolicy`'s interval constant.** It is a baseline
  construction parameter, not an environment rule; coupling the learned policy
  to a baseline's scripted rule would hide the knob from configs and from the
  checkpoint evaluator.
- **A non-zero shared default.** Would silently invalidate every historical
  result and contradict the default-`0` historical-behaviour guarantee.
