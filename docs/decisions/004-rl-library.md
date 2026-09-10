# ADR-004: RL library for sim training

**Status:** accepted
**Date:** 2026-09-09
**Task:** S00

## Context

The training harness (S13) needs PPO with clipping and GAE. Two options:

1. **`stable-baselines3`** — the community-standard RL library, Gymnasium-compatible,
   used by the anchor paper's community. Provides battle-tested PPO, GAE, and
   `SubprocVecEnv` for parallel rollouts. Adds `torch` as a transitive dependency.

2. **Vendored single-file PPO** in `train/ppo_runner.py` — a ~200-line pure-numpy or
   torch implementation, tested against a CartPole-v1 equivalent.

## Decision

**Use `stable-baselines3 >= 2.9`.** Verified on Python 3.14:

```
$ uv add stable-baselines3 gymnasium
Resolved 74 packages in 22ms
Installed 33 packages in 612ms
 + gymnasium==1.3.0
 + stable-baselines3==2.9.0
 + torch==2.14.0
```

The package installs cleanly on the project's Python 3.14 venv with no conflicts.
The torch transitive dependency is acceptable — it is the de-facto standard for RL
and the anchor paper's community uses it.

## Consequences

- **S13** imports `stable_baselines3.PPO` and `stable_baselines3.common.env_checker`.
  The config maps directly to SB3 keyword arguments.
- Tensorboard logging is available via SB3's callback system.
- The `Policy` protocol (§10 of CONTRACTS.md) is unchanged — backtester and eval
  runner do not depend on the RL library.
- Sim optional-deps become `gymnasium` and `stable-baselines3` in
  `[project.optional-dependencies].sim`, keeping the data pipeline lightweight.
- The pinned PPO hyperparameters in `TrainingConfig` (§2 of CONTRACTS.md) map
  directly: `gamma`, `gae_lambda`, `clip_range`, `policy_kwargs` with `net_arch`.
- If SB3 is later found incompatible with a future Python version, the vendored
  PPO option remains a viable fallback — the env interface is Gymnasium-native and
  does not depend on SB3 internals.