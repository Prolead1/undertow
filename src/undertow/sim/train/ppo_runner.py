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
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor

from undertow.sim.train.rollouts import RolloutRecorder, rows_from_env
from undertow.sim.train.runs import (
    RunManifest,
    compute_config_hash,
    current_git_commit,
    library_versions,
    write_run_manifest,
)
from undertow.sim.train.telemetry import (
    STATUS_FILENAME,
    TELEMETRY_FILENAME,
    TelemetryCallback,
)
from undertow.sim.types import SimConfigError

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

#: Accepted ``training.vec_env`` backends (validated again by ``TrainingConfig``).
_VEC_ENV_BACKENDS: tuple[str, ...] = ("dummy", "subproc")

#: Hard cap on decisions in one recorded evaluation episode.  The trainers'
#: environments terminate after a bounded episode, so this is a defensive guard
#: against a pathological env hanging the post-training pass, not a budget.
_MAX_ROLLOUT_STEPS: int = 1_000_000

#: Thread-pool variables pinned before starting worker processes.  Polars
#: (rayon) reads ``POLARS_MAX_THREADS`` at import, so each worker's one-off
#: index build must not open a per-core pool either.
_THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "POLARS_MAX_THREADS",
)


def _pin_thread_env() -> None:
    """Pin BLAS/OpenMP to one thread per process.

    With ``SubprocVecEnv`` every worker would otherwise spawn a thread per core,
    and N workers x N threads on N CPUs spends more time synchronising than
    computing (the same oversubscription ``tests/conftest.py`` pins away for the
    suite).  Set before the workers are forked/spawned so they inherit it; an
    explicit non-empty value is honoured.
    """
    for var in _THREAD_ENV_VARS:
        if not os.environ.get(var):
            os.environ[var] = "1"


def _pin_torch_threads() -> None:
    """One torch thread in the parent.

    The PPO net is tiny, so thread-pool synchronisation dominates: a sweep on
    8 CPUs showed 4 threads only trading update time for slower action sampling
    (same wall, 3x the CPU) and 8 threads ~5x slower overall.  One thread is also
    what keeps the suite fast under xdist.
    """
    torch.set_num_threads(1)


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
    """Record the mean recent episode return every ``log_freq_steps`` timesteps.

    The curve keeps the fixed ``log_freq_steps`` grid, so an early boundary that
    is reached before any episode has completed contributes a ``0.0``
    placeholder: there is no reward to average yet.  :meth:`best` deliberately
    excludes those placeholders -- on a task whose rewards are all negative the
    placeholder would otherwise dominate ``max(curve)`` and every manifest would
    report ``best_reward = 0.0``.
    """

    def __init__(self, log_freq_steps: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_freq_steps = max(1, int(log_freq_steps))
        self.curve: list[float] = []
        #: Rewards measured on completed episodes only (grid placeholders excluded).
        self._measured: list[float] = []
        self._next_eval = self.log_freq_steps

    def _has_episodes(self) -> bool:
        """True when at least one episode has completed in the info buffer."""
        return bool(getattr(self.model, "ep_info_buffer", None))

    def mean_reward(self) -> float:
        """Mean reward of the episodes currently in the model's info buffer."""
        buffer = getattr(self.model, "ep_info_buffer", None)
        if not buffer:
            return 0.0
        return float(np.mean([ep["r"] for ep in buffer]))

    def best(self) -> float:
        """Best measured mean episode return.

        Falls back to the current info-buffer mean -- ``0.0`` when the buffer is
        empty -- if no measurement has been recorded yet.
        """
        if self._measured:
            return float(max(self._measured))
        return float(self.mean_reward())

    def _on_step(self) -> bool:
        while self.num_timesteps >= self._next_eval:
            value = self.mean_reward()
            self.curve.append(value)
            # SB3 refreshes ``ep_info_buffer`` *after* ``on_step``, so an
            # episode ending exactly on a grid boundary is not visible yet: its
            # point stays a placeholder and the next boundary captures it.
            if self._has_episodes():
                self._measured.append(value)
            self._next_eval += self.log_freq_steps
        return True


def _record_rollouts_pass(
    env_fn: Callable[[], gym.Env],
    model: PPO,
    training: TrainingConfig,
    run_id: str,
    seed: int,
    seed_dir: Path,
) -> None:
    """Run deterministic evaluation episodes and persist their trajectories (U06).

    A *fresh* environment is built from ``env_fn``; the training vector env is
    never stepped here, so the pass cannot perturb the training RNG or the
    saved model/manifest.  ``model.predict`` is called with
    ``deterministic=True`` (argmax, no sampling), and each episode is reset with
    a distinct derived seed (``seed + episode``) so the sampled market windows
    differ; resetting every episode with the same seed would write identical
    parquets.  The derived seed keeps the pass reproducible.  Failures
    propagate to the caller, which logs and continues.
    """
    max_episodes = max(0, int(training.rollout_max_episodes))
    if max_episodes == 0:
        return
    recorder = RolloutRecorder(
        run_id=run_id,
        seed=seed,
        out_dir=seed_dir,
        max_episodes=max_episodes,
    )
    env = env_fn()
    try:
        for episode in range(max_episodes):
            observation, info = env.reset(seed=seed + episode)
            infos: list[dict[str, object]] = [info]
            done = False
            for _ in range(_MAX_ROLLOUT_STEPS):
                action, _state = model.predict(observation, deterministic=True)
                action_index = int(np.asarray(action).reshape(-1)[0])
                observation, _reward, terminated, truncated, info = env.step(
                    action_index
                )
                infos.append(info)
                done = bool(terminated or truncated)
                if done:
                    break
            if not done:
                raise RuntimeError(
                    f"rollout episode {episode} did not terminate within "
                    f"{_MAX_ROLLOUT_STEPS} decisions"
                )
            recorder.record_episode(
                episode,
                rows_from_env(infos, run_id=run_id, seed=seed, episode=episode),
            )
    finally:
        recorder.close()
        env.close()


def _resolve_log_dir(seed: int, log_dir: Path | None) -> Path:
    if log_dir is None:
        return Path(tempfile.mkdtemp(prefix=f"undertow-ppo-seed{seed}-"))
    return Path(log_dir)


def _make_vec_env(
    env_fn: Callable[[], gym.Env],
    parallel_envs: int,
    seed: int,
    *,
    backend: str = "dummy",
) -> VecMonitor:
    """Build a seeded single- or multi-env vector env.

    ``env_fn`` is invoked once per parallel environment; no env is shared
    between seeds or between the parallel slots.  ``backend="subproc"`` runs
    each slot in its own process (``SubprocVecEnv``); ``"dummy"`` steps them
    sequentially in this process.  A single slot always uses ``DummyVecEnv``:
    process overhead cannot pay for itself there.
    """
    if backend not in _VEC_ENV_BACKENDS:
        raise SimConfigError(
            f"training.vec_env must be one of {_VEC_ENV_BACKENDS}, got {backend!r}"
        )
    parallel = max(1, int(parallel_envs))
    vec_env: VecEnv
    if backend == "subproc" and parallel > 1:
        # SB3 defaults to forkserver where available and spawn otherwise; both
        # are thread-safe (plain ``fork`` after torch import can deadlock).
        vec_env = SubprocVecEnv([env_fn for _ in range(parallel)])
    else:
        vec_env = DummyVecEnv([env_fn for _ in range(parallel)])
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


def _apply_hold_bias(model: PPO, bias: float) -> None:
    """Add ``bias`` to the hold action's logit before training starts.

    ``hold`` is index 0 of the discrete action grid (ADR-011). The flat
    softmax otherwise samples it ~1/N of the time, which is not enough for
    PPO to discover that avoiding the rebalance gas dominates; a positive
    prior lets the advantage estimate see it. 0.0 is a no-op.
    """
    bias = float(bias)
    if bias == 0.0:
        return
    action_net = getattr(model.policy, "action_net", None)
    bias_term = getattr(action_net, "bias", None)
    if bias_term is None or bias_term.numel() == 0:
        raise SimConfigError(
            "training.hold_bias requires a discrete action head with a bias term, "
            f"got action_net={type(action_net).__name__} without one"
        )
    with torch.no_grad():
        bias_term[0] += bias
    logger.info(
        "applied hold_bias=%.4f to the hold logit (now %.4f)",
        bias,
        float(bias_term[0].detach()),
    )


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

    config_hash = compute_config_hash(config)
    run_id = f"ppo_s{seed}_{config_hash}"

    parallel = max(1, int(training.parallel_envs))
    _pin_thread_env()  # inherited by SubprocVecEnv workers
    _pin_torch_threads()
    n_steps, batch_size, n_epochs = _derive_rollout(training)

    vec_env = _make_vec_env(env_fn, parallel, seed, backend=training.vec_env)
    curve_cb = _LearningCurveCallback(training.log_freq_steps)
    callbacks: list[BaseCallback] = [curve_cb]

    # Telemetry sidecars (U00). The callback is optional so
    # ``telemetry=False`` is byte-for-byte today's behaviour. The
    # ``_LearningCurveCallback`` is always registered: it feeds the manifest.
    telemetry_cb: TelemetryCallback | None = None
    if training.telemetry:
        telemetry_cb = TelemetryCallback(
            run_id=run_id,
            seed=seed,
            total_timesteps=int(training.total_timesteps),
            log_freq_steps=int(training.log_freq_steps),
            status_path=seed_dir / STATUS_FILENAME,
            telemetry_path=seed_dir / TELEMETRY_FILENAME,
        )
        callbacks.append(telemetry_cb)
        # Initial heartbeat before training starts, so a dashboard can discover
        # the run even if it never reaches the first log-frequency boundary.
        telemetry_cb.write_running_status()

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
        _apply_hold_bias(model, training.hold_bias)
        model.learn(total_timesteps=int(training.total_timesteps), callback=callbacks)
        checkpoint_path = seed_dir / f"ppo_seed_{seed}.zip"
        model.save(str(checkpoint_path))

        # U06: a deterministic, offline evaluation pass over a fresh env. The
        # pass is best-effort: a recording failure is logged and training
        # continues with the checkpoint/manifest untouched.
        if training.record_rollouts:
            try:
                _record_rollouts_pass(
                    env_fn, model, training, run_id, seed, seed_dir
                )
            except Exception:
                logger.exception(
                    "rollout recording failed for seed %d; training result "
                    "is unaffected",
                    seed,
                )

        curve = list(curve_cb.curve)
        best_reward = float(curve_cb.best())
        manifest = RunManifest(
            run_id=run_id,
            seed=seed,
            config_hash=config_hash,
            git_commit=current_git_commit(),
            algorithm=str(training.algorithm),
            total_timesteps=int(training.total_timesteps),
            best_reward=best_reward,
            checkpoint_path=checkpoint_path,
            library_versions=tuple(sorted(library_versions().items())),
        )
        # The manifest/checkpoint bookkeeping stays inside the ``try`` so that a
        # failure here flips the status to ``failed``: the invariant
        # "``done`` => artifacts exist" must hold even if the writes fail.
        write_run_manifest(seed_dir / "manifest.json", manifest, curve)
        # Re-write the terminal status *after* the manifest/checkpoint exists so
        # ``state="done"`` guarantees the run's artifacts are on disk (U00 §4).
        # ``_on_training_end`` already wrote ``done``; re-writing here is a
        # deliberate, idempotent redundancy that tightens the ordering.
        if telemetry_cb is not None:
            telemetry_cb.write_done_status()
    except BaseException as exc:
        # A crash must never leave a stale ``running`` status on disk. Guard the
        # fallback write itself so it cannot mask the original exception.
        if telemetry_cb is not None:
            try:
                telemetry_cb.write_failed_status(str(exc))
            except Exception:
                logger.exception("failed to write 'failed' telemetry status")
        raise
    finally:
        vec_env.close()

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