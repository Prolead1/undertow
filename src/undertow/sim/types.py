"""Shared type system for ``undertow.sim``.

Domain aliases, enums, the Action space, and the exception hierarchy.
Implements `CONTRACTS.md` §1 exactly — no extensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, NewType

# -- Domain aliases -----------------------------------------------------------
Tick = NewType("Tick", int)
TickSpacing = NewType("TickSpacing", int)
SqrtPrice = float  # sqrt(P) in human units; P = s²
Price = float  # token1 per token0 (USDC per WETH)
Wealth = float  # USDC


# -- Regime labels (consumed from data module; never recomputed in sim) --------
class Regime(StrEnum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"
    UNKNOWN = "unknown"


# -- Price process mode -------------------------------------------------------
class PriceMode(StrEnum):
    REPLAY = "replay"
    CALIBRATED = "calibrated"


# -- Action space -------------------------------------------------------------
# The factored discrete action: {hold} ∪ {rebalance(center_offset, width)}.
# center_offset: signed tick-spacings around current tick, ∈ {−4,…,+4}
# width: tick-spacings per side, ∈ {1, 2, 5, 10, 25, 50}
# S10 defines the policy protocol; S01 owns the action types.


@dataclass(frozen=True, slots=True)
class Action:
    action_type: Literal["hold", "rebalance"]
    center_offset: int = 0  # tick-spacings from current tick; ignored for hold
    width: int = 0  # tick-spacings per side; ignored for hold

    def is_hold(self) -> bool:
        return self.action_type == "hold"

    def lower_tick(self, current_tick: Tick, tick_spacing: TickSpacing) -> Tick:
        """Tick for the lower bound of the range.

        For a rebalance action:
        lower = floor((current + center_offset − width) / spacing) * spacing.
        Raises PositionError for hold actions.
        """
        if self.is_hold():
            raise PositionError("hold actions have no lower_tick")
        raw = (
            int(current_tick)
            + self.center_offset * int(tick_spacing)
            - self.width * int(tick_spacing)
        )
        return Tick((raw // int(tick_spacing)) * int(tick_spacing))

    def upper_tick(self, current_tick: Tick, tick_spacing: TickSpacing) -> Tick:
        """Tick for the upper bound of the range.

        For a rebalance action: upper = ceil((current + center_offset + width) / spacing) * spacing.
        Raises PositionError for hold actions.
        """
        if self.is_hold():
            raise PositionError("hold actions have no upper_tick")
        raw = (
            int(current_tick)
            + self.center_offset * int(tick_spacing)
            + self.width * int(tick_spacing)
        )
        # ceil division for potentially negative raw
        ts = int(tick_spacing)
        return Tick(-((-raw) // ts) * ts)


# -- Exception hierarchy ------------------------------------------------------
class UndertowSimError(Exception):
    """Base for all sim exceptions."""


class SimConfigError(UndertowSimError):
    """Configuration validation error."""


class MarketViewError(UndertowSimError):
    """Error accessing market data."""


class LookAheadError(MarketViewError):
    """Raised when a consumer tries to read beyond the wall."""


class PositionError(UndertowSimError):
    """Error in position construction or access."""


class EnvError(UndertowSimError):
    """Error in the Gymnasium environment."""


class BacktestError(UndertowSimError):
    """Error during backtest execution."""


class ParityError(UndertowSimError):
    """Sim-backtester divergence exceeds tolerance."""