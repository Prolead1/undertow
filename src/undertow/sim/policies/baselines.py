"""The baseline ladder for ``undertow.sim`` (CONTRACTS.md §10).

Six deterministic, frozen baselines used to evaluate the RL agent against
sensible reference strategies (roadmap §10.5 items 1–3 + 5; PLAN.md §7
"Baseline ladder"):

* :class:`HODLPolicy` — buy and hold the initial token mix.
* :class:`PassiveNarrowPolicy` — deploy a ±5 % band once, then never rebalance.
* :class:`PassiveWidePolicy` — deploy a ±20 % band once, then never rebalance.
* :class:`FullRangeV2Policy` — the Uniswap-V2-equivalent full range.
* :class:`TauResetPolicy` — recenter whenever the price exits the current band.
* :class:`CostAwareRebalancePolicy` — recenter at most once per interval and only
  when the fee estimate exceeds the gas cost of the rebalance.

All prices follow ADR-009: ``sqrt_price``/``tick``/``liquidity`` are raw protocol
quantities and the human ``price`` (USDC per WETH) is *decreasing* in tick, so the
lower tick carries the **higher** price bound.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from undertow.data import MAX_TICK, MIN_TICK, price_to_tick
from undertow.sim.core.position import DEFAULT_DEC0, DEFAULT_DEC1
from undertow.sim.policies.base import HOLDAction
from undertow.sim.types import Action, Tick

if TYPE_CHECKING:
    import numpy as np

    from undertow.sim.env.observations import Observation
    from undertow.sim.frictions.gas import GasModel

# Default tick spacing for the pinned USDC/WETH 0.30 % pool.  Mirrors
# ``undertow.sim.config.EpisodeConfig.tick_spacing``; policies that must convert a
# tick width into the action grid's tick-spacing units take this as a keyword-only
# override (PLAN.md §0.2 allows additive keyword-only arguments).
DEFAULT_TICK_SPACING: int = 60

# Half-width, in tick-spacings, that spans the widest *snapped* range that stays
# inside ``MIN_TICK..MAX_TICK`` at the default spacing.  The factored ``Action``
# (CONTRACTS.md §1) can only encode a symmetric band around the current tick, so it
# cannot hit the absolute ``MIN_TICK``/``MAX_TICK`` exactly (neither bound is a
# multiple of the tick spacing).  ``FullRangeV2Policy`` therefore returns a
# ``rebalance`` action whose *resolved* ``lower_tick``/``upper_tick`` are the
# snapped full range (ADR-011), and reports this half-width for the default spacing.
FULL_RANGE_WIDTH: int = int(MAX_TICK) // DEFAULT_TICK_SPACING - 1


def _price_to_tick(price: float) -> int:
    """Human price → raw tick using data's floor semantics (ADR-009)."""
    return int(price_to_tick(Decimal(str(price)), DEFAULT_DEC0, DEFAULT_DEC1))


def _band_width_spacings(
    entry_price: float, half_fraction: float, tick_spacing: int
) -> int:
    """Half-width in tick-spacings for a ``±half_fraction`` human-price band.

    ``price_to_tick`` is used for both bounds rather than deriving from the current
    tick so the width tracks the requested price band.  Human price decreases in
    tick, hence the tick swap.  Floors at 1 spacing so the result is a valid band.
    """
    tick_a = _price_to_tick(entry_price * (1.0 - half_fraction))
    tick_b = _price_to_tick(entry_price * (1.0 + half_fraction))
    tick_lower, tick_upper = (tick_a, tick_b) if tick_a < tick_b else (tick_b, tick_a)
    return max(1, (tick_upper - tick_lower) // (2 * tick_spacing))


def _band_action(half_width_ticks: int, tick_spacing: int) -> Action:
    """A centred ``rebalance`` action for a ``±half_width_ticks`` band."""
    width = max(1, half_width_ticks // tick_spacing)
    return Action(action_type="rebalance", center_offset=0, width=width)


class HODLPolicy:
    """Buy-and-hold the initial token mix. Always returns ``hold`` (CONTRACTS §10)."""

    @property
    def name(self) -> str:
        return "hodl"

    def reset(self) -> None:
        """No episode-local state to reset."""

    def act(
        self,
        observation: Observation,
        rng: np.random.Generator | None = None,
    ) -> Action:
        """Always hold; HODL never deploys or rebalances liquidity."""
        return HOLDAction


class _PassiveBandPolicy:
    """Shared behaviour for the one-shot passive band policies.

    Deploys once on the first ``act`` (anchored on the observation's entry price)
    and returns ``hold`` forever after, until :meth:`reset`.
    """

    _half_fraction: float
    _name: str

    def __init__(self, *, tick_spacing: int = DEFAULT_TICK_SPACING) -> None:
        if tick_spacing <= 0:
            raise ValueError("tick_spacing must be > 0")
        self._tick_spacing = int(tick_spacing)
        self._deployed = False

    @property
    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        """Forget deployment so a new episode deploys again."""
        self._deployed = False

    def act(
        self,
        observation: Observation,
        rng: np.random.Generator | None = None,
    ) -> Action:
        """Deploy the configured band once, then always hold."""
        if self._deployed:
            return HOLDAction
        width = _band_width_spacings(
            float(observation.price), self._half_fraction, self._tick_spacing
        )
        action = Action(action_type="rebalance", center_offset=0, width=width)
        self._deployed = True
        return action


class PassiveNarrowPolicy(_PassiveBandPolicy):
    """Deploy at ±5 % around the entry price, then never rebalance (CONTRACTS §10)."""

    _half_fraction = 0.05
    _name = "passive_narrow"


class PassiveWidePolicy(_PassiveBandPolicy):
    """Deploy at ±20 % around the entry price, then never rebalance (CONTRACTS §10)."""

    _half_fraction = 0.20
    _name = "passive_wide"


class FullRangeV2Policy:
    """Uniswap-V2-equivalent full range: ``MIN_TICK .. MAX_TICK`` (CONTRACTS §10).

    The factored ``Action`` can only encode a symmetric band around the current
    tick, so an exact ``MIN_TICK``/``MAX_TICK`` band is not representable (neither
    bound is a multiple of the tick spacing).  ``act`` therefore returns a
    ``rebalance`` action whose resolved ``lower_tick``/``upper_tick`` are the
    widest snapped band that stays inside the protocol range, centred near tick 0
    (see ADR-011).  This is the V2-equivalent position to within one tick spacing.
    """

    @property
    def name(self) -> str:
        return "full_range_v2"

    def __init__(self, *, tick_spacing: int = DEFAULT_TICK_SPACING) -> None:
        if tick_spacing <= 0:
            raise ValueError("tick_spacing must be > 0")
        self._tick_spacing = int(tick_spacing)
        self._deployed = False

    @property
    def tick_spacing(self) -> int:
        """Tick spacing this policy snaps to."""
        return self._tick_spacing

    def reset(self) -> None:
        """Forget deployment so a new episode deploys again."""
        self._deployed = False

    def _full_range_action(self, current_tick: int) -> Action:
        """Centred action whose resolved bounds are the snapped full range.

        Choosing the widest snapped half-width and a centre offset that puts the
        band centre near tick 0 guarantees the resolved bounds stay inside
        ``[MIN_TICK, MAX_TICK]`` for every current tick (ADR-011).
        """
        half_width = int(MAX_TICK) // self._tick_spacing - 1
        if half_width <= 0:
            raise ValueError(
                f"tick_spacing={self._tick_spacing} is too large to form a full range"
            )
        center_offset = int(round(-current_tick / self._tick_spacing))
        action = Action(
            action_type="rebalance", center_offset=center_offset, width=half_width
        )
        lower = action.lower_tick(Tick(current_tick), self._tick_spacing)
        upper = action.upper_tick(Tick(current_tick), self._tick_spacing)
        assert MIN_TICK <= lower < upper <= MAX_TICK
        return action

    def act(
        self,
        observation: Observation,
        rng: np.random.Generator | None = None,
    ) -> Action:
        """Deploy the full range once, then always hold."""
        if self._deployed:
            return HOLDAction
        action = self._full_range_action(int(observation.tick))
        self._deployed = True
        return action


class TauResetPolicy:
    """Recenter when the price exits the current band (CONTRACTS §10).

    ``half_width_ticks`` is the half-width in **ticks** (not tick-spacings), as
    clarified by the S10 brief; the band half-width is converted to the action's
    tick-spacing units internally.  The band width is fixed for the episode
    ("τ = entry width"); only its centre moves when ``obs.tick`` leaves
    ``[current_lower, current_upper]``.
    """

    def __init__(self, half_width_ticks: int, *, tick_spacing: int = DEFAULT_TICK_SPACING) -> None:
        if half_width_ticks <= 0:
            raise ValueError("half_width_ticks must be > 0")
        if tick_spacing <= 0:
            raise ValueError("tick_spacing must be > 0")
        self._half_width_ticks = int(half_width_ticks)
        self._tick_spacing = int(tick_spacing)
        self._deployed = False
        self._current_lower: Tick | None = None
        self._current_upper: Tick | None = None

    @property
    def name(self) -> str:
        return "tau_reset"

    def reset(self) -> None:
        """Reset the stored band; it is (re)established on the first ``act``."""
        self._deployed = False
        self._current_lower = None
        self._current_upper = None

    def _deploy_action(self, current_tick: int) -> Action:
        action = _band_action(self._half_width_ticks, self._tick_spacing)
        self._current_lower = action.lower_tick(Tick(current_tick), self._tick_spacing)
        self._current_upper = action.upper_tick(Tick(current_tick), self._tick_spacing)
        self._deployed = True
        return action

    def act(
        self,
        observation: Observation,
        rng: np.random.Generator | None = None,
    ) -> Action:
        """Deploy on the first call, recenter on an exit, otherwise hold."""
        current_tick = int(observation.tick)
        if not self._deployed:
            return self._deploy_action(current_tick)
        assert self._current_lower is not None and self._current_upper is not None
        if self._current_lower <= current_tick <= self._current_upper:
            return HOLDAction
        return self._deploy_action(current_tick)


class CostAwareRebalancePolicy:
    """Recenter at most once per interval, only if fee gain beats gas (CONTRACTS §10).

    Guards each rebalance with two conditions:

    1. ``step_in_episode - last_rebalance_step >= min_interval_steps`` (at most
       daily when constructed with the 10-minute cadence's 144 steps), and
    2. the position's ``uncollected_fees_token0`` exceeds the gas model's
       ``rebalance`` cost estimate.

    The initial deployment on the first ``act`` is unconditional — there is no
    position to rebalance otherwise.  The gas estimate is read from the
    observation's ``gas_price_recent_wei`` (base + priority fee, in wei) and
    ``eth_usd_price_recent``; ``block_number`` is passed as ``0`` because the
    observation contract carries no block index.
    """

    def __init__(
        self,
        half_width_ticks: int,
        min_interval_steps: int,
        gas_model: GasModel,
        *,
        tick_spacing: int = DEFAULT_TICK_SPACING,
    ) -> None:
        if half_width_ticks <= 0:
            raise ValueError("half_width_ticks must be > 0")
        if min_interval_steps < 0:
            raise ValueError("min_interval_steps must be >= 0")
        if tick_spacing <= 0:
            raise ValueError("tick_spacing must be > 0")
        self._half_width_ticks = int(half_width_ticks)
        self._min_interval_steps = int(min_interval_steps)
        self._gas_model = gas_model
        self._tick_spacing = int(tick_spacing)
        self._deployed = False
        self._current_lower: Tick | None = None
        self._current_upper: Tick | None = None
        self._last_rebalance_step = -self._min_interval_steps

    @property
    def name(self) -> str:
        return "cost_aware"

    def reset(self) -> None:
        """Forget the deployed band and re-arm the interval; next ``act`` deploys."""
        self._deployed = False
        self._current_lower = None
        self._current_upper = None
        self._last_rebalance_step = -self._min_interval_steps

    def _deploy_action(self, current_tick: int) -> Action:
        action = _band_action(self._half_width_ticks, self._tick_spacing)
        self._current_lower = action.lower_tick(Tick(current_tick), self._tick_spacing)
        self._current_upper = action.upper_tick(Tick(current_tick), self._tick_spacing)
        self._deployed = True
        return action

    def _estimated_gas_cost(self, observation: Observation) -> float:
        """Gas estimate for a ``rebalance`` from the observation's recent gas fields."""
        return float(
            self._gas_model.gas_cost_usdc(
                "rebalance",
                0,
                int(observation.gas_price_recent_wei),
                0,
                float(observation.eth_usd_price_recent),
            )
        )

    def act(
        self,
        observation: Observation,
        rng: np.random.Generator | None = None,
    ) -> Action:
        """Deploy, or recenter only when interval and fee-vs-gas both allow it."""
        current_tick = int(observation.tick)
        step = int(observation.step_in_episode)

        if not self._deployed:
            action = self._deploy_action(current_tick)
            self._last_rebalance_step = step
            return action

        assert self._current_lower is not None and self._current_upper is not None
        in_range = self._current_lower <= current_tick <= self._current_upper
        if in_range:
            return HOLDAction
        if step - self._last_rebalance_step < self._min_interval_steps:
            return HOLDAction

        estimated_fees = float(observation.uncollected_fees_token0)
        # token0 is the numeraire USDC (CONTRACTS.md §0), so token0 fees are
        # already in USDC and compare directly against the USDC gas estimate.
        if estimated_fees <= self._estimated_gas_cost(observation):
            return HOLDAction

        action = self._deploy_action(current_tick)
        self._last_rebalance_step = step
        return action
