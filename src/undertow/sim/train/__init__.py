"""``undertow.sim.train`` — PPO training harness (S13, CONTRACTS.md §15).

Public surface:

* :func:`~undertow.sim.train.ppo_runner.train_single_seed` — train PPO for one
  seed and return its :class:`RunManifest`.
* :func:`~undertow.sim.train.runs.train_ppo` — multi-seed orchestration that
  aggregates the per-seed learning curves.
* :class:`~undertow.sim.train.runs.RunManifest` /
  :class:`~undertow.sim.train.runs.TrainingResult` — the run provenance and
  aggregated-result dataclasses.

The library is stable-baselines3 per ``docs/decisions/004-rl-library.md``.
"""

from __future__ import annotations

from undertow.sim.train.ppo_runner import train_single_seed
from undertow.sim.train.runs import RunManifest, TrainingResult, train_ppo

__all__ = [
    "RunManifest",
    "TrainingResult",
    "train_ppo",
    "train_single_seed",
]