"""``undertow.sim.env`` — reward decomposition and the Gymnasium environment.

S09 owns this package and exports the reward function and its per-step
decomposition (`CONTRACTS.md` §9). S12 later appends the observation builder
and :class:`LpEnvironment` to this surface.
"""

from __future__ import annotations

from undertow.sim.env.reward import (
    RewardBreakdown,
    compute_reward,
    normalize_reward,
)

__all__ = [
    "RewardBreakdown",
    "compute_reward",
    "normalize_reward",
]