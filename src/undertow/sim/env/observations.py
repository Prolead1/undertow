"""Observation dataclass and backward-looking observation builder (S12).

Implements ``docs/plans/sim/CONTRACTS.md`` §11. The observation is the state
vector the RL agent sees at every decision step; :class:`ObservationBuilder`
assembles it from ``MarketView`` (history + regime + gas) and the live
``PoolEngine`` state.

Look-ahead wall
---------------
Every feature is **backward-looking and closed on the right at** the current
timestamp: the current reference price is the last reference close whose
``close_time`` is ``<=`` that timestamp, and the rolling return/volatility
windows only use reference bars at or before it. ``MarketView`` is the only
door to history and it enforces the train/eval wall. The builder never reads a
bar from a later step.

Orientation (ADR-009)
---------------------
``price`` is the **human** USDC-per-WETH reference price; ``sqrt_price`` and
``tick`` are the **raw** protocol quantities carried by the ``PoolEngine``;
``pool_active_liquidity`` is raw ``L``. ``position_value`` is USDC and
``il_vs_hodl`` is expressed as a fraction of the initial capital.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from undertow.sim.core.pool import PoolEngine
from undertow.sim.core.position import DEFAULT_DEC0, DEFAULT_DEC1, Position
from undertow.sim.marketview import MarketView
from undertow.sim.types import Price, SqrtPrice, Tick, Wealth

__all__ = ["OBSERVATION_VECTOR_LENGTH", "Observation", "ObservationBuilder"]

#: Ordinal encoding of the regime labels (CONTRACTS.md §11).
REGIME_ORDINAL: dict[str, int] = {
    "bull": 0,
    "bear": 1,
    "sideways": 2,
    "high_vol": 3,
    "unknown": 4,
}

#: Length of :meth:`Observation.to_vector` — one scalar per field.
OBSERVATION_VECTOR_LENGTH: int = 18

#: 10-minute reference bars per year (6 per hour x 24 h x 365.25 d).
_STEPS_PER_YEAR: float = 6.0 * 24.0 * 365.25


@dataclass(frozen=True, slots=True)
class Observation:
    """The state vector the agent sees at each decision step.

    All fields are backward-looking, closed on the right (CONTRACTS.md §11).
    """

    # -- Market --
    price: Price  # current human reference price (USDC per WETH)
    sqrt_price: SqrtPrice  # raw Uniswap sqrt price (ADR-009)
    tick: Tick  # raw protocol tick
    price_returns_1h: float  # log return over the last 6 bars (10-min cadence)
    price_returns_24h: float  # log return over the last 144 bars
    realized_vol_24h: float  # annualized sigma over the last 144 bars

    # -- Position --
    position_in_range: bool
    position_value: Wealth
    uncollected_fees_token0: float
    uncollected_fees_token1: float
    il_vs_hodl: float  # as a fraction of initial capital (e.g. -0.05 = -5%)

    # -- Pool --
    pool_active_liquidity: float
    pool_fee_tier_bps: int  # Uniswap pips (3000 = 0.30%; ADR-012)

    # -- Costs --
    gas_price_recent_wei: float  # recent base_fee + priority_fee
    eth_usd_price_recent: float

    # -- Regime --
    regime: str  # one of the Regime enum values

    # -- Time --
    step_in_episode: int
    steps_remaining: int

    def to_vector(self) -> np.ndarray:
        """Flatten to a stable-order 1-D ``float64`` array for the policy network.

        ``bool`` fields become ``0.0``/``1.0``; the regime string is mapped to
        its ordinal in :data:`REGIME_ORDINAL` (``unknown`` = 4 for any
        unrecognised label). The order follows the field declaration order.
        """
        return np.array(
            [
                float(self.price),
                float(self.sqrt_price),
                float(self.tick),
                float(self.price_returns_1h),
                float(self.price_returns_24h),
                float(self.realized_vol_24h),
                1.0 if self.position_in_range else 0.0,
                float(self.position_value),
                float(self.uncollected_fees_token0),
                float(self.uncollected_fees_token1),
                float(self.il_vs_hodl),
                float(self.pool_active_liquidity),
                float(self.pool_fee_tier_bps),
                float(self.gas_price_recent_wei),
                float(self.eth_usd_price_recent),
                float(REGIME_ORDINAL.get(self.regime, REGIME_ORDINAL["unknown"])),
                float(self.step_in_episode),
                float(self.steps_remaining),
            ],
            dtype=np.float64,
        )


def _human_price_from_sqrt(
    sqrt_price: SqrtPrice,
    dec0: int = DEFAULT_DEC0,
    dec1: int = DEFAULT_DEC1,
) -> float:
    """Convert a raw sqrt price to the human price (ADR-009), as a fallback."""
    if sqrt_price <= 0.0:
        return 0.0
    return 10.0 ** (dec1 - dec0) / (sqrt_price * sqrt_price)


def _log_return(closes: np.ndarray, bars: int) -> float:
    """Log return over the last ``bars`` reference bars.

    Returns ``0.0`` when fewer than ``bars + 1`` closes are available, so the
    feature is defined (zero) rather than ``NaN`` at the start of the feed.
    """
    if closes.size <= bars:
        return 0.0
    return float(math.log(closes[-1]) - math.log(closes[-1 - bars]))


class ObservationBuilder:
    """Builds :class:`Observation` instances from ``MarketView`` + ``PoolEngine``.

    ``lookback_bars`` is the maximum lookback used by the 24h return and
    realized-volatility features (144 bars at the pinned 10-minute cadence).
    ``episode_start`` / ``episode_length`` are additive keyword-only context
    (PLAN.md §0.2) used for the ``step_in_episode`` / ``steps_remaining``
    features; the Gymnasium environment sets them on every :meth:`reset` via
    :meth:`configure_episode`.
    """

    #: Bar counts for the rolling features (10-minute cadence).
    RETURN_1H_BARS: int = 6
    RETURN_24H_BARS: int = 144

    def __init__(
        self,
        market_view: MarketView,
        lookback_bars: int = 144,
        *,
        episode_start: int = 0,
        episode_length: int = 0,
    ) -> None:
        if lookback_bars <= 0:
            raise ValueError(f"lookback_bars must be > 0, got {lookback_bars!r}")
        self.market_view = market_view
        self.lookback_bars = int(lookback_bars)
        self.episode_start = int(episode_start)
        self.episode_length = int(episode_length)

    def configure_episode(self, start_seq: int, length: int) -> None:
        """Set the episode context used for the time features.

        ``start_seq`` is the inclusive first tape seq of the episode and
        ``length`` the number of decision steps.
        """
        self.episode_start = int(start_seq)
        self.episode_length = int(length)

    # -- History helpers (backward-looking) ------------------------------
    def _reference_closes_through(self, at_time: object) -> np.ndarray:
        """Reference closes with ``close_time <= at_time``, ascending.

        This is the look-ahead wall in code: a bar whose close is strictly
        after ``at_time`` is excluded, so a jump in the next bar cannot appear.
        """
        reference = self.market_view.reference
        if reference.height == 0:
            return np.empty(0, dtype=float)
        sub = reference.filter(pl.col("close_time") <= at_time).sort("close_time")
        if sub.height == 0:
            return np.empty(0, dtype=float)
        return sub["close"].to_numpy().astype(float)

    def _gas_price_recent(
        self, tape_row: dict[str, object]
    ) -> tuple[float, int, int]:
        """Return ``(base + priority, base, priority)`` in wei for the step.

        Prefers the joined gas feed row for the tape's block; falls back to the
        tape row's own joined gas columns when the feed has no row (the tiny
        fixture's gas feed is shorter than its tape).
        """
        block_number = int(tape_row.get("block_number", 0) or 0)
        row = self.market_view.gas_at_block(block_number)
        if row is None:
            row = tape_row
        base = int(row.get("base_fee_per_gas") or 0)
        priority = int(row.get("priority_fee_p50_wei") or 0)
        return float(base + priority), base, priority

    # -- Build -----------------------------------------------------------
    def build(
        self,
        step: int,
        position: Position,
        pool_engine: PoolEngine,
        gas_model: object,
        entry_price: Price,
        entry_sqrt_price: SqrtPrice,
        initial_capital: Wealth,
    ) -> Observation:
        """Assemble the observation at tape seq ``step`` (CONTRACTS §11).

        Time features are closed on the right at the tape row's timestamp. The
        ``step_in_episode`` / ``steps_remaining`` counters are derived from the
        episode context set by :meth:`configure_episode`. ``gas_model`` is
        accepted for interface completeness (the gas price itself comes from
        the market view). Raises ``ValueError`` for non-positive capital.
        """
        tape_row = self.market_view.tape_at(int(step))
        step_in_episode = int(step) - self.episode_start
        return self.build_at(
            at_time=tape_row["block_timestamp"],
            tape_row=tape_row,
            step_in_episode=step_in_episode,
            steps_remaining=max(0, self.episode_length - 1 - step_in_episode),
            position=position,
            pool_engine=pool_engine,
            gas_model=gas_model,
            entry_price=entry_price,
            entry_sqrt_price=entry_sqrt_price,
            initial_capital=initial_capital,
        )

    def build_at(
        self,
        *,
        at_time: datetime,
        tape_row: dict[str, object],
        step_in_episode: int,
        steps_remaining: int,
        position: Position,
        pool_engine: PoolEngine,
        gas_model: object,
        entry_price: Price,
        entry_sqrt_price: SqrtPrice,
        initial_capital: Wealth,
    ) -> Observation:
        """Assemble an observation at an explicit timestamp.

        Additive extension (PLAN §0.2) used by the environment when decisions
        are made at reference-bar boundaries rather than tape rows: ``at_time``
        is the decision timestamp and ``tape_row`` the latest tape event at or
        before it (for gas / regime context). The look-ahead wall is unchanged —
        only reference bars with ``close_time <= at_time`` are read. Raises
        ``ValueError`` for non-positive ``initial_capital``.
        """
        del gas_model  # part of the frozen signature; nothing to read from it here
        if initial_capital <= 0.0:
            raise ValueError("initial_capital must be > 0 to express IL as a fraction")

        closes = self._reference_closes_through(at_time)
        if closes.size > 0:
            price = float(closes[-1])
        else:
            price = _human_price_from_sqrt(pool_engine.state.sqrt_price)

        price_returns_1h = _log_return(closes, self.RETURN_1H_BARS)
        price_returns_24h = _log_return(closes, self.RETURN_24H_BARS)

        window = closes[-(self.lookback_bars + 1) :]
        log_returns = (
            np.diff(np.log(window))
            if window.size >= 2
            else np.empty(0, dtype=float)
        )
        if log_returns.size >= 2:
            realized_vol_24h = float(np.std(log_returns, ddof=1)) * math.sqrt(
                _STEPS_PER_YEAR
            )
        else:
            realized_vol_24h = 0.0

        sqrt_price = pool_engine.state.sqrt_price
        tick = pool_engine.state.tick

        # Uncollected fees for the position, in human token units. The engine's
        # lattices hold the fee_growth_outside snapshots; Position.uncollected_fees
        # subtracts its own (open-time) snapshot and humanises by 10**dec (ADR-009).
        lower_state = pool_engine.ticks.get(position.tick_lower)
        upper_state = pool_engine.ticks.get(position.tick_upper)
        outside_lower_0 = lower_state.fee_growth_outside_0 if lower_state else 0.0
        outside_lower_1 = lower_state.fee_growth_outside_1 if lower_state else 0.0
        outside_upper_0 = upper_state.fee_growth_outside_0 if upper_state else 0.0
        outside_upper_1 = upper_state.fee_growth_outside_1 if upper_state else 0.0
        inside_0, inside_1 = pool_engine.state.fee_growth_inside(
            position.tick_lower,
            position.tick_upper,
            outside_lower_0,
            outside_lower_1,
            outside_upper_0,
            outside_upper_1,
        )
        fees_token0, fees_token1 = position.uncollected_fees(inside_0, inside_1)

        il_usdc = position.il_vs_hodl(
            entry_sqrt_price, entry_price, sqrt_price, price
        )
        il_fraction = il_usdc / initial_capital

        gas_price_recent, _, _ = self._gas_price_recent(tape_row)

        return Observation(
            price=price,
            sqrt_price=sqrt_price,
            tick=tick,
            price_returns_1h=price_returns_1h,
            price_returns_24h=price_returns_24h,
            realized_vol_24h=realized_vol_24h,
            position_in_range=position.in_range(sqrt_price),
            position_value=position.value(sqrt_price, price),
            uncollected_fees_token0=fees_token0,
            uncollected_fees_token1=fees_token1,
            il_vs_hodl=il_fraction,
            pool_active_liquidity=pool_engine.state.liquidity,
            pool_fee_tier_bps=int(pool_engine.state.fee_tier_bps),
            gas_price_recent_wei=gas_price_recent,
            eth_usd_price_recent=price,
            regime=self.market_view.regime_at(at_time),
            step_in_episode=int(step_in_episode),
            steps_remaining=int(steps_remaining),
        )
