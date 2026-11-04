"""Multi-seed PPO orchestration, run manifests and provenance (S13).

Implements ``docs/plans/sim/CONTRACTS.md`` §15. The training harness is the
thesis's result-producing loop: :func:`train_ppo` runs PPO once per configured
seed, each seed against its own freshly-constructed environment, and aggregates
the per-seed learning curves.

Provenance
----------
A :class:`RunManifest` records everything needed to reproduce a run: the
``config_hash`` and ``git_commit`` are computed with the canonical helpers
already used by the backtester (``undertow.sim.backtest.ledger``) so a
backtest and a training run of the same :class:`~undertow.sim.config.SimConfig`
agree on provenance. ``library_versions`` is a small frozen view of the
training stack (SB3 / torch / gymnasium / numpy) recorded at run time. The full
manifest artifact (including the learning curve) is also written to
``<log_dir>/manifest.json``.

Determinism
-----------
Each seed is passed to ``PPO(seed=...)`` and to ``vec_env.seed(seed)`` before
training, so the policy RNG and every environment RNG are explicitly seeded.
``env_fn`` is invoked once per parallel environment per seed, and never shared
across seeds (PLAN.md §4).

Extending CONTRACTS
-------------------
``RunManifest`` adds one field with a default (``library_versions``) under the
extension rule of PLAN.md §0.2; all contract fields keep their names, types and
order.

"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from undertow.sim.backtest.ledger import config_hash as _canonical_config_hash
from undertow.sim.backtest.ledger import git_commit as _canonical_git_commit
from undertow.sim.types import SimConfigError

if TYPE_CHECKING:
    import gymnasium as gym

    from undertow.sim.config import SimConfig

__all__ = [
    "RunManifest",
    "TrainingResult",
    "compute_config_hash",
    "current_git_commit",
    "library_versions",
    "train_ppo",
    "write_run_manifest",
]

#: Training-stack distributions recorded in every run manifest.
_TRACKED_LIBRARIES: tuple[str, ...] = (
    "stable_baselines3",
    "torch",
    "gymnasium",
    "numpy",
)


def compute_config_hash(config: SimConfig) -> str:
    """Return the canonical 16-hex-char SHA256 of the config (backtester parity)."""
    return _canonical_config_hash(config)


def current_git_commit() -> str:
    """Return the current git ``HEAD`` SHA (or ``"unknown"`` outside a checkout)."""
    return _canonical_git_commit()


def library_versions() -> dict[str, str]:
    """Return installed versions of the training stack.

    A missing distribution is reported as ``"unknown"`` rather than raising, so
    a manifest can always be written.
    """
    versions: dict[str, str] = {}
    for name in _TRACKED_LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:  # pragma: no cover - env dependent
            versions[name] = "unknown"
    return versions


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Metadata for a single training run (CONTRACTS.md §15)."""

    run_id: str
    seed: int
    config_hash: str
    git_commit: str
    algorithm: str
    total_timesteps: int
    best_reward: float
    checkpoint_path: Path
    # CONTRACTS §15 extension (PLAN.md §0.2): frozen name/version pairs so the
    # manifest stays hashable. The on-disk JSON renders this as an object.
    library_versions: tuple[tuple[str, str], ...] = ()
    # -- S21 additive checkpoint-selection fields (ADR-040) --
    #: Metric that ranked this seed's checkpoints.  ``"reward"`` is the
    #: historical learning-curve selection (``best_reward``); ``"val_sharpe"``
    #: and ``"val_cvar"`` are validation-window statistics.
    selection_metric: str = "reward"
    #: Value of ``selection_metric`` on the selected checkpoint (validation
    #: metrics only; ``None`` for the default reward path).  Higher is always
    #: better: Sharpe is maximised and ``val_cvar`` is negative-for-loss, so
    #: the largest value is the smallest tail loss.
    selection_value: float | None = None


def _manifest_rank_key(manifest: RunManifest) -> float:
    """Cross-seed ranking key used by :func:`train_ppo`.

    A validation-selected manifest ranks on ``selection_value`` (higher is
    better for both ``val_sharpe`` and ``val_cvar``: the latter is negative for
    losses, so the largest value is the smallest tail loss).  The default
    reward path keeps today's ``best_reward`` ordering exactly.
    """
    if manifest.selection_metric != "reward":
        if manifest.selection_value is None:
            raise SimConfigError(
                f"run manifest for seed {manifest.seed} records "
                f"selection_metric={manifest.selection_metric!r} without a "
                "selection_value; the validation selection step did not run"
            )
        return float(manifest.selection_value)
    return float(manifest.best_reward)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Aggregated results across all seeds (CONTRACTS.md §15)."""

    seeds: Sequence[int]
    manifests: Sequence[RunManifest]
    learning_curves: dict[str, list[float]]
    best_checkpoint: Path


def write_run_manifest(
    path: Path,
    manifest: RunManifest,
    learning_curve: Sequence[float] = (),
) -> None:
    """Write the full run manifest (including learning curve) as JSON.

    The dataclass only carries the CONTRACTS §15 fields; the sidecar adds the
    learning curve and rendered library versions used by reporting (S15/S16).
    """
    payload = {
        "run_id": manifest.run_id,
        "seed": manifest.seed,
        "config_hash": manifest.config_hash,
        "git_commit": manifest.git_commit,
        "algorithm": manifest.algorithm,
        "total_timesteps": manifest.total_timesteps,
        "best_reward": manifest.best_reward,
        "checkpoint_path": str(manifest.checkpoint_path),
        "library_versions": dict(manifest.library_versions),
        "selection_metric": manifest.selection_metric,
        "selection_value": manifest.selection_value,
        "learning_curve": [float(v) for v in learning_curve],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _aggregate_learning_curves(
    seeds: Sequence[int], curves: Sequence[Sequence[float]]
) -> dict[str, list[float]]:
    """Mean/std across seeds at each evaluation step, plus each seed's curve.

    Curves are truncated to the shortest seed curve (they should be equal length
    for a fixed config) so aggregation is well-defined.
    """
    non_empty = [list(c) for c in curves if len(c) > 0]
    length = min((len(c) for c in non_empty), default=0)
    aggregated: dict[str, list[float]] = {}
    if length:
        stacked = np.asarray([list(c)[:length] for c in non_empty], dtype=np.float64)
        aggregated["mean_reward"] = [float(v) for v in stacked.mean(axis=0)]
        aggregated["std_reward"] = [float(v) for v in stacked.std(axis=0)]
    for seed, curve in zip(seeds, curves, strict=True):
        aggregated[f"seed_{seed}_reward"] = [float(v) for v in curve]
    return aggregated


def train_ppo(
    env_fn: Callable[[], gym.Env],
    config: SimConfig,
    log_dir: Path | None = None,
    *,
    validation_env_fn: Callable[[], gym.Env] | None = None,
) -> TrainingResult:
    """Train PPO across ``config.training.seeds`` (CONTRACTS.md §15).

    ``env_fn`` is a zero-argument factory; each seed gets its own freshly-built
    environment (and its own RNG). Logs and checkpoints land under ``log_dir``;
    when ``log_dir`` is ``None`` an isolated temporary directory per seed is
    used. Returns the per-seed manifests, the aggregated learning curves and the
    checkpoint of the seed with the highest best reward.

    ``validation_env_fn`` (S21, ADR-040) is an optional zero-argument factory
    over the carved validation window.  When supplied alongside
    ``config.training.selection_metric != "reward"``, each seed's saved
    checkpoints are scored on it and the canonical checkpoint is replaced by
    the best one; the cross-seed winner then ranks on ``selection_value``
    instead of ``best_reward``.  Omitting it preserves today's behaviour.
    """
    seeds = tuple(int(s) for s in config.training.seeds)
    if not seeds:
        raise ValueError("config.training.seeds must contain at least one seed")

    # Imported lazily to keep the module dependency one-directional
    # (ppo_runner -> runs) and avoid a circular import.
    from undertow.sim.train.ppo_runner import _train_single_seed

    root = Path(log_dir) if log_dir is not None else None
    manifests: list[RunManifest] = []
    curves: list[list[float]] = []

    for seed in seeds:
        seed_dir = (root / f"seed_{seed}") if root is not None else None
        manifest, curve = _train_single_seed(
            env_fn, config, seed, seed_dir, validation_env_fn=validation_env_fn
        )
        manifests.append(manifest)
        curves.append(list(curve))

    best = max(manifests, key=_manifest_rank_key)
    return TrainingResult(
        seeds=seeds,
        manifests=tuple(manifests),
        learning_curves=_aggregate_learning_curves(seeds, curves),
        best_checkpoint=best.checkpoint_path,
    )