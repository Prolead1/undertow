"""``undertow.sim.env`` — reward decomposition and the Gymnasium environment.

S09 owns this package and exports the reward function and its per-step
decomposition (`CONTRACTS.md` §9). S12 appends the observation builder and
:class:`LpEnvironment` to this surface (`CONTRACTS.md` §11/§12).
"""

from __future__ import annotations

from undertow.sim.env.lp_env import LpEnvironment
from undertow.sim.env.observations import (
    OBSERVATION_VECTOR_LENGTH,
    Observation,
    ObservationBuilder,
)
from undertow.sim.env.reward import (
    RewardBreakdown,
    compute_reward,
    normalize_reward,
)

__all__ = [
    "OBSERVATION_VECTOR_LENGTH",
    "LpEnvironment",
    "Observation",
    "ObservationBuilder",
    "RewardBreakdown",
    "compute_reward",
    "normalize_reward",
]
