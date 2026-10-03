"""PPO training loop for a single seed (S13, CONTRACTS.md §15).

Uses **stable-baselines3** exactly as decided by `docs/decisions/004-rl-library.md`.
The pinned hyperparameters are the PLAN.md §7 anchor setup: an MLP policy of
``hidden_size`` × ``n_hidden``, ``gamma=discount_gamma``, ``gae_lambda`` and
``clip_range=clip_epsilon`` (γ=0.99, λ=0.95, clip=0.15 for the default config).

Rollout sizing
--------------
The anchor's SB3 defaults (``n_steps=2048``, ``batch_size=64``, ``n_epochs=10``,
``learning_rate=3e-4``) are *not* fields of :class:`TrainingConfig`, so the
rollout is derived from the config in a way that reproduces those values for the
production budget (1e6 timesteps, 4 envs → 2048/64) and shrinks gracefully for
the tiny CPU configurations used by the test-suite. The derivation is purely a
function of the config, so it is deterministic and reviewable.

Learning curve
--------------
A :class:`_LearningCurveCallback` appends the mean recent episode return to the
curve every ``log_freq_steps`` timesteps, which yields exactly
``total_timesteps // log_freq_steps`` points for a config whose rollout divides
the budget. No separate evaluation environment is created: ``env_fn`` is called
only for the training vector envs, keeping the "fresh env + RNG per seed"
contract intact.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from undertow.sim.train.runs import (
    RunManifest,
    compute_config_hash,
    current_git_commit,
    library_versions,
    write_run_manifest,
)

if TYPE_CHECKING:
    import gymnasium as gym

    from undertow.sim.config import SimConfig, TrainingConfig

__all__ = ["train_single_seed"]

logger = logging.getLogger("undertow.sim.train.ppo_runner")

#: Anchor PPO rollout/optimisation defaults (PLAN.md §7 anchor setup, §8.4.4).
PPO_N_STEPS: int = 2048
PPO_BATCH_SIZE: int = 64
PPO_N_EPOCHS: int = 10
PPO_LEARNING_RATE: float = 3e-4
PPO_DEVICE: str = "cpu"  # CPU keeps runs bit-reproducible and GPU-free in tests


def _snap_batch_size(rollout: int, cap: int) -> int:
    """Pick a valid SB3 mini-batch size for ``rollout`` samples.

    Targets a quarter of the rollout, capped at ``cap`` (the anchor's
    ``batch_size=64`` comes from the cap, not the quarter), then snaps to a
    divisor of ``rollout``. The result always satisfies
    ``1 < batch_size <= rollout`` and ``rollout % batch_size == 0``. When the
    target quarter has no divisor, the smallest divisor above it is used (which
    is ``rollout`` itself for a prime rollout), i.e. full-batch gradient steps.
    """
    target = min(cap, max(2, rollout // 4))
    for divisor in range(target, 1, -1):
        if rollout % divisor == 0:
            return divisor
    for divisor in range(target + 1, rollout + 1):
        if rollout % divisor == 0:
            return divisor
    return rollout


def _derive_rollout(training: TrainingConfig) -> tuple[int, int, int]:
    """Derive ``(n_steps, batch_size, n_epochs)`` from the training config.

    For the production budget (1e6 timesteps / 4 envs) this reproduces the
    anchor values 2048 / 64 / 10. Smaller budgets get proportionally smaller
    rollouts so a test never trains much more than ``total_timesteps``. The
    returned ``batch_size`` always satisfies SB3's invariants
    ``1 < batch_size <= rollout`` and ``rollout % batch_size == 0``.
    """
    parallel = max(1, int(training.parallel_envs))
    total = max(2, int(training.total_timesteps))
    per_env_budget = max(2, total // parallel)
    n_steps = int(min(PPO_N_STEPS, max(per_env_budget // 4, 1)))
    rollout = n_steps * parallel
    if rollout < 2:  # SB3 requires batch_size > 1 and batch_size <= rollout
        n_steps = 2
        rollout = n_steps * parallel
    batch_size = _snap_batch_size(rollout, PPO_BATCH_SIZE)
    return n_steps, batch_size, PPO_N_EPOCHS


class _LearningCurveCallback(BaseCallback):
    """Record the mean recent episode return every ``log_freq_steps`` timesteps."""

    def __init__(self, log_freq_steps: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_freq_steps = max(1, int(log_freq_steps))
        self.curve: list[float] = []
        self._next_eval = self.log_freq_steps

    def mean_reward(self) -> float:
        """Mean reward of the episodes currently in the model's info buffer."""
        buffer = getattr(self.model, "ep_info_buffer", None)
        if not buffer:
            return 0.0
        return float(np.mean([ep["r"] for ep in buffer]))

    def _on_step(self) -> bool:
        while self.num_timesteps >= self._next_eval:
            self.curve.append(self.mean_reward())
            self._next_eval += self.log_freq_steps
        return True


def _resolve_log_dir(seed: int, log_dir: Path | None) -> Path:
    if log_dir is None:
        return Path(tempfile.mkdtemp(prefix=f"undertow-ppo-seed{seed}-"))
    return Path(log_dir)


def _make_vec_env(
    env_fn: Callable[[], gym.Env], parallel_envs: int, seed: int
) -> VecMonitor:
    """Build a seeded single- or multi-env vector env.

    ``env_fn`` is invoked once per parallel environment; no env is shared
    between seeds or between the parallel slots.
    """
    vec_env = DummyVecEnv([env_fn for _ in range(max(1, parallel_envs))])
    vec_env.seed(seed)  # per-env seeds become seed, seed+1, ...
    return VecMonitor(vec_env)


def _tensorboard_log(seed_dir: Path) -> str | None:
    """Return a tensorboard log dir when the optional writer is installed.

    SB3 only needs the writer present to emit tensorboard-compatible events;
    when it is absent we skip the hook rather than emit a warning, since the
    JSON manifest and checkpoints are the primary artifacts.
    """
    try:
        import tensorboard  # noqa: F401  (presence check only)
    except ImportError:
        return None
    return str(seed_dir / "tensorboard")


def _train_single_seed(
    env_fn: Callable[[], gym.Env],
    config: SimConfig,
    seed: int,
    log_dir: Path | None = None,
) -> tuple[RunManifest, list[float]]:
    """Internal single-seed trainer returning ``(manifest, learning_curve)``."""
    seed = int(seed)
    training = config.training
    seed_dir = _resolve_log_dir(seed, log_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)

    parallel = max(1, int(training.parallel_envs))
    n_steps, batch_size, n_epochs = _derive_rollout(training)

    vec_env = _make_vec_env(env_fn, parallel, seed)
    curve_cb = _LearningCurveCallback(training.log_freq_steps)
    callbacks: list[BaseCallback] = [curve_cb]

    checkpoint_freq = int(training.checkpoint_freq_steps)
    if checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(checkpoint_freq // parallel, 1),
                save_path=str(seed_dir / "checkpoints"),
                name_prefix=f"ppo_seed_{seed}",
                verbose=0,
            )
        )

    try:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=PPO_LEARNING_RATE,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=training.discount_gamma,
            gae_lambda=training.gae_lambda,
            clip_range=training.clip_epsilon,
            policy_kwargs={"net_arch": [training.hidden_size] * training.n_hidden},
            seed=seed,
            device=PPO_DEVICE,
            verbose=0,
            tensorboard_log=_tensorboard_log(seed_dir),
        )
        model.learn(total_timesteps=int(training.total_timesteps), callback=callbacks)
        checkpoint_path = seed_dir / f"ppo_seed_{seed}.zip"
        model.save(str(checkpoint_path))
    finally:
        vec_env.close()

    curve = list(curve_cb.curve)
    best_reward = float(max(curve)) if curve else float(curve_cb.mean_reward())
    config_hash = compute_config_hash(config)

    manifest = RunManifest(
        run_id=f"ppo_s{seed}_{config_hash}",
        seed=seed,
        config_hash=config_hash,
        git_commit=current_git_commit(),
        algorithm=str(training.algorithm),
        total_timesteps=int(training.total_timesteps),
        best_reward=best_reward,
        checkpoint_path=checkpoint_path,
        library_versions=tuple(sorted(library_versions().items())),
    )
    write_run_manifest(seed_dir / "manifest.json", manifest, curve)
    logger.info(
        "seed %d finished: %d timesteps, best_reward=%.4f, checkpoint=%s",
        seed,
        manifest.total_timesteps,
        best_reward,
        checkpoint_path,
    )
    return manifest, curve


def train_single_seed(
    env_fn: Callable[[], gym.Env],
    config: SimConfig,
    seed: int,
    log_dir: Path | None = None,
) -> RunManifest:
    """Train PPO for one seed (CONTRACTS.md §15).

    ``env_fn`` creates a fresh environment; it is called once per parallel
    environment (``config.training.parallel_envs``). Checkpoints are written at
    ``config.training.checkpoint_freq_steps`` and a final checkpoint plus
    ``manifest.json`` is always written under ``log_dir`` (a temp directory when
    ``log_dir`` is ``None``).
    """
    manifest, _ = _train_single_seed(env_fn, config, seed, log_dir)
    return manifest