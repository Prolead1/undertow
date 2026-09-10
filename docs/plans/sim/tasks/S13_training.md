# S13 — Training harness (PPO, multi-seed, checkpoints)

**Wave 5 · size L · depends on: S05, S10, S12 · blocks: S15**  
**Branch:** `feature-sim-training`

## Why this task exists

The RL agent is trained against the Gymnasium environment using PPO with clipping, the pinned
hyperparameters from PLAN.md §7, and ≥5 seeds. The training harness orchestrates multi-seed runs,
writes checkpoints, produces learning curves, and records a run manifest with full provenance
(config hash, git commit, library versions). This is the thesis's primary result-producing loop.

The RL library decision was made by S00's ADR-004. This task follows that decision:
use `stable-baselines3` if it was installable, or the vendored PPO otherwise.

## Files you own

```
src/undertow/sim/train/__init__.py         (re-exports train_ppo, train_single_seed,
                                            TrainingResult, RunManifest)
src/undertow/sim/train/ppo_runner.py       (PPO training loop — either SB3 wrapper or vendored)
src/undertow/sim/train/runs.py             (multi-seed orchestration, checkpoints, run manifest)
tests/sim/test_training.py
```

## What to build

### 1. S00 ADR check

Before writing any code, read `docs/decisions/004-rl-library.md` to determine which path to take.

### 2. Path A: stable-baselines3 (if ADR-004 chose it)

**`train/ppo_runner.py`**:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def train_single_seed(env_fn, config, seed, log_dir=None):
    """Train PPO for one seed. env_fn() creates a fresh env."""
    def make_env():
        env = env_fn()
        env.reset(seed=seed)
        return env

    vec_env = DummyVecEnv([make_env])

    # Optional: normalize observations and rewards
    # vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_reward=10.0)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,  # standard PPO default
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=config.training.discount_gamma,
        gae_lambda=config.training.gae_lambda,
        clip_range=config.training.clip_epsilon,
        policy_kwargs=dict(
            net_arch=[config.training.hidden_size] * config.training.n_hidden,
        ),
        seed=seed,
        verbose=1,
        tensorboard_log=str(log_dir) if log_dir else None,
    )

    model.learn(total_timesteps=config.training.total_timesteps)

    # Save checkpoint
    checkpoint_path = log_dir / f"ppo_seed_{seed}.zip" if log_dir else None
    if checkpoint_path:
        model.save(str(checkpoint_path))

    # Collect best reward from training logs
    return RunManifest(
        run_id=f"ppo_s{seed}_{config_hash}",
        seed=seed,
        config_hash=config_hash,
        git_commit=git_commit,
        algorithm="ppo",
        total_timesteps=config.training.total_timesteps,
        best_reward=best_reward,
        checkpoint_path=checkpoint_path,
    )
```

### 3. Path B: Vendored PPO (if ADR-004 chose it)

**`train/ppo_runner.py`** (~200 lines, single file):

Implement PPO-clip with GAE directly against the Gymnasium API:

```python
class PPOBuffer:
    """Rollout buffer for on-policy PPO."""
    def __init__(self, obs_dim, act_dim, capacity, gamma, gae_lambda): ...
    def store(self, obs, act, rew, val, logp, done): ...
    def finish_path(self, last_val): ...  # compute GAE
    def get(self): ...  # return (obs, act, adv, ret, logp) as arrays

class MLPPolicy:
    """2-hidden-layer MLP with separate actor and critic heads.
    Implement in numpy (or torch if available)."""
    def __init__(self, obs_dim, act_dim, hidden_size, n_hidden): ...
    def forward(self, obs): ...  # returns (action_logits, value)
    def act(self, obs, rng): ...  # returns (action_idx, log_prob, value)
    def update(self, batch): ...  # PPO-clip update
    def save(self, path): ...
    def load(self, path): ...

def train_single_seed(env_fn, config, seed, log_dir=None):
    """Train PPO for one seed. Returns RunManifest."""
    # 1. Create env, reset with seed
    # 2. Initialize policy and buffer
    # 3. For each iteration:
    #    a. Collect rollout (config.parallel_envs * steps_per_env steps)
    #    b. Compute GAE
    #    c. PPO-clip update for n_epochs
    #    d. Evaluate (optional) and log
    #    e. Save checkpoint at checkpoint_freq_steps
    # 4. Return RunManifest
    ...
```

The vendored PPO must be tested end-to-end on a trivial environment (e.g., a 2-armed bandit
or CartPole-like with known optimal policy) before being used with the LP environment.

### 4. `train/runs.py` — CONTRACTS.md §15

**`RunManifest`** dataclass (frozen, slots):
- `run_id: str`, `seed: int`, `config_hash: str`, `git_commit: str`
- `algorithm: str`, `total_timesteps: int`, `best_reward: float`
- `checkpoint_path: Path`

**`TrainingResult`** dataclass (frozen, slots):
- `seeds: Sequence[int]`, `manifests: Sequence[RunManifest]`
- `learning_curves: dict[str, list[float]]` — `metric_name` → per-eval-step values
- `best_checkpoint: Path`

**`train_ppo(env_fn, config, log_dir=None)` → `TrainingResult`**:
- Loop over `config.training.seeds`
- For each seed, call `train_single_seed(env_fn, config, seed, seed_log_dir)`
- Collect all `RunManifest`s
- Aggregate learning curves across seeds (mean over seeds at each eval step)
- Determine best checkpoint (highest mean reward across seeds)
- Return `TrainingResult`

Learning curve tracking: during training, evaluate the policy every `log_freq_steps` steps for
a fixed number of evaluation episodes (e.g., 5). Record the mean reward. The learning curve is
the list of these mean rewards over time.

### 5. Parallel environments

For SB3 path: `SubprocVecEnv` or `DummyVecEnv` with `config.training.parallel_envs`.
For vendored path: step `config.training.parallel_envs` environments in a loop (sequential is
fine for initial implementation; vectorized = future optimization).

## Tests you must write

1. `train_single_seed` with a trivial env (2-armed bandit, known optimal arm) completes and
   returns a `RunManifest` — **mark this test `@pytest.mark.slow`**
2. The trained policy on the bandit selects the optimal arm > 90% of the time
3. `train_ppo` with 2 seeds returns a `TrainingResult` with `len(manifests) == 2`
4. Learning curves have the expected length: `total_timesteps / log_freq_steps` entries per seed
5. Two runs with the same seed produce identical `RunManifest.best_reward` (determinism)
6. Checkpoint is written to the expected path and can be loaded
7. `RunManifest` fields are all populated (no None)
8. `TrainingResult.best_checkpoint` points to the seed with the highest `best_reward`
9. Policy trained on the bandit for 0 extra steps after convergence stays converged (no
   catastrophic forgetting on the trivial case)
10. **Smoke test with tiny_env**: `train_single_seed(lambda: tiny_env, tiny_config, 0)` completes
    within 30 seconds — **mark `@pytest.mark.slow`**

## Definition of done

- All files exist, typed, no TODO stubs
- `uv run pytest tests/sim/test_training.py -q` green (slow tests pass when run explicitly)
- `code-reviewer` APPROVE
- PR open against `main` from `feature-sim-training`
- `STATUS.md` updated