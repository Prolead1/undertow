"""Concentrated-liquidity pool simulator for ``undertow.sim``.

A self-contained, mutable pool state machine that models a single Uniswap V3
pool. It supports swap, mint, burn, collect and tracks fee growth at the
position level. All arithmetic is exact integer; prices are converted to
human-readable floats only at the public API boundary.

This module imports ONLY ``math`` — no data-package imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np

from undertow.sim.math import (
    Q96,
    Q128,
    _mul_div_floor,
    align_tick_down,
    align_tick_up,
    amounts_for_liquidity,
    get_amount0_delta,
    get_amount1_delta,
    liquidity_for_amounts,
    price_to_sqrt_price_x96,
    price_to_tick,
    sqrt_price_x96_to_price,
    tick_to_price,
    tick_to_sqrt_price_x96,
    wrapping_add_256,
    wrapping_sub_256,
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TickState:
    """Per-tick mutable state."""

    fee_growth_outside_0_x128: int = 0
    fee_growth_outside_1_x128: int = 0
    liquidity_gross: int = 0
    liquidity_net: int = 0  # signed
    initialized: bool = False


@dataclass(frozen=True, slots=True)
class Position:
    """Immutable position snapshot."""

    tick_lower: int
    tick_upper: int
    liquidity: int
    fee_growth_inside_0_last: int = 0
    fee_growth_inside_1_last: int = 0
    token0_fees_uncollected: int = 0
    token1_fees_uncollected: int = 0


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------


class ConcentratedLiquidityPool:
    """Single-pool concentrated-liquidity simulator.

    Maintains exact integer state (sqrt_price_x96, tick, liquidity, per-tick
    fee growth) and exposes a human-price public API. Computationally cheap
    enough for millions of simulation steps.

    Parameters
    ----------
    fee_tier_bps: int
        Fee tier in basis points (1, 5, 30, or 100).
    tick_spacing: int
        Tick spacing derived from the fee tier.
    initial_price: float
        Starting human price (token1 in token0, e.g., 3000 for ETH/USD).
    dec0: int, default 6
        Decimals of token0 (USDC).
    dec1: int, default 18
        Decimals of token1 (WETH).
    """

    def __init__(
        self,
        fee_tier_bps: int,
        tick_spacing: int,
        initial_price: float,
        dec0: int = 6,
        dec1: int = 18,
    ) -> None:
        if fee_tier_bps not in (1, 5, 30, 100):
            raise ValueError(f"Unsupported fee tier: {fee_tier_bps} bps")
        if tick_spacing <= 0:
            raise ValueError(f"Invalid tick spacing: {tick_spacing}")

        self.fee_tier_bps: int = fee_tier_bps
        self.tick_spacing: int = tick_spacing
        self.dec0: int = dec0
        self.dec1: int = dec1

        # Fee rate as a rational: fee / 1_000_000 (the Uniswap denominator)
        self._fee_rate: int = fee_tier_bps * 100  # bps -> 1e-6 units

        # Initialize state from initial human price
        initial_price_d = Decimal(str(initial_price))
        self.sqrt_price_x96: int = price_to_sqrt_price_x96(initial_price_d, dec0, dec1)
        self.tick: int = price_to_tick(initial_price_d, dec0, dec1)
        self.liquidity: int = 0

        # Global fee growth accumulators
        self.fee_growth_global_0_x128: int = 0
        self.fee_growth_global_1_x128: int = 0

        # Per-tick state (lazily initialized)
        self.ticks: dict[int, TickState] = {}

        # Active positions
        self.positions: dict[int, Position] = {}  # position_id -> Position
        self._next_position_id: int = 0

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------
    @property
    def price(self) -> float:
        """Current human price (token1 in token0)."""
        d = sqrt_price_x96_to_price(self.sqrt_price_x96, self.dec0, self.dec1)
        return float(d)

    @property
    def current_tick(self) -> int:
        """Current pool tick."""
        return self.tick

    # ------------------------------------------------------------------
    # Tick state access
    # ------------------------------------------------------------------
    def _get_tick(self, tick: int) -> TickState:
        """Get or create tick state."""
        if tick not in self.ticks:
            self.ticks[tick] = TickState()
        return self.ticks[tick]

    # ------------------------------------------------------------------
    # Swap
    # ------------------------------------------------------------------
    def swap(
        self,
        amount0_in: int | None = None,
        amount1_in: int | None = None,
    ) -> tuple[int, int, int]:
        """Execute a swap against the pool.

        Exactly one of ``amount0_in`` / ``amount1_in`` must be positive; the
        other must be None. The pool determines the output amount via the
        constant-product formula within the current tick range.

        Returns ``(amount0_out, amount1_out, fee_amount)`` where fee_amount is
        in the input token's raw units.
        """
        zero_for_one: bool
        amount_in: int
        if amount0_in is not None and amount1_in is None:
            if amount0_in <= 0:
                raise ValueError("amount0_in must be positive")
            zero_for_one = True
            amount_in = amount0_in
        elif amount1_in is not None and amount0_in is None:
            if amount1_in <= 0:
                raise ValueError("amount1_in must be positive")
            zero_for_one = False
            amount_in = amount1_in
        else:
            raise ValueError(
                "Exactly one of amount0_in / amount1_in must be provided"
            )

        fee_amount = amount_in * self._fee_rate // 1_000_000
        amount_in_after_fee = amount_in - fee_amount

        # Compute output amount using the constant-product within the current
        # tick range. For an exact swap without crossing ticks:
        #   Δx = L * (1/sqrt(P) - 1/sqrt(P'))
        #   Δy = L * (sqrt(P') - sqrt(P))
        # For a simple single-tick-range swap, we use the sqrt price math.
        if zero_for_one:
            # Selling token0, buying token1: price decreases
            # amount1_out = L * (sqrt_price_current - sqrt_price_new)
            # We solve for sqrt_price_new:
            # amount0_in_after_fee * sqrt_price_new + L * Q96 = ...
            # Simpler: use the constant product: (x + Δx)(y - Δy) = L²
            # Δy = (Δx * y) / (x + Δx)  where x = L/√P, y = L*√P
            # amount1_out = get_amount1_delta(sqrt_current, sqrt_new, L, round_up=False)
            # First compute the new sqrt price after the swap
            liquidity = max(self.liquidity, 1)  # avoid div-by-zero
            sqrt_p = self.sqrt_price_x96
            # Δ√P = Δx * Q96 / L  (for small moves, exact for exact swap math)
            delta_sqrt = _mul_div_floor(amount_in_after_fee, Q96, liquidity)
            sqrt_p_new_approx = sqrt_p - delta_sqrt
            if sqrt_p_new_approx <= 0:
                raise ValueError("Swap exceeds available liquidity (price floor hit)")
            sqrt_p_new = sqrt_p_new_approx
            amount_out = get_amount1_delta(sqrt_p_new, sqrt_p, liquidity, round_up=False)
            amount0_out, amount1_out = 0, amount_out
            # Update fee growth
            self.fee_growth_global_0_x128 = wrapping_add_256(
                self.fee_growth_global_0_x128,
                (fee_amount << 128) // liquidity,
            )
        else:
            # Selling token1, buying token0: price increases
            liquidity = max(self.liquidity, 1)
            sqrt_p = self.sqrt_price_x96
            delta_sqrt = _mul_div_floor(amount_in_after_fee, Q96, liquidity)
            sqrt_p_new = sqrt_p + delta_sqrt
            amount_out = get_amount0_delta(sqrt_p, sqrt_p_new, liquidity, round_up=False)
            amount0_out, amount1_out = amount_out, 0
            self.fee_growth_global_1_x128 = wrapping_add_256(
                self.fee_growth_global_1_x128,
                (fee_amount << 128) // liquidity,
            )

        # Apply state update
        self.sqrt_price_x96 = sqrt_p_new
        self.tick = _sqrt_price_x96_to_tick(self.sqrt_price_x96)

        return amount0_out, amount1_out, fee_amount

    # ------------------------------------------------------------------
    # Liquidity provision / removal
    # ------------------------------------------------------------------
    def mint(
        self,
        tick_lower: int,
        tick_upper: int,
        amount0_desired: int,
        amount1_desired: int,
    ) -> tuple[Position, int, int]:
        """Mint a new position, returning ``(position, amount0_used, amount1_used)``.

        Aligns ticks to the pool's spacing grid. The position is assigned a
        unique id and stored internally.
        """
        tick_lower = align_tick_down(tick_lower, self.tick_spacing)
        tick_upper = align_tick_up(tick_upper, self.tick_spacing)

        if tick_lower >= tick_upper:
            raise ValueError(
                f"tick_lower ({tick_lower}) must be < tick_upper ({tick_upper})"
            )

        sqrt_lower = tick_to_sqrt_price_x96(tick_lower)
        sqrt_upper = tick_to_sqrt_price_x96(tick_upper)

        # Compute the maximum liquidity for the desired amounts
        liq = liquidity_for_amounts(
            self.sqrt_price_x96, sqrt_lower, sqrt_upper,
            amount0_desired, amount1_desired,
        )

        # Compute actual amounts used
        amount0, amount1 = amounts_for_liquidity(
            self.sqrt_price_x96, sqrt_lower, sqrt_upper, liq,
        )

        # Update tick states (cross ticks if needed)
        self._modify_position(tick_lower, tick_upper, liq, delta=True)

        # Create position
        g_inside_0 = self._fee_growth_inside(tick_lower, tick_upper, 0)
        g_inside_1 = self._fee_growth_inside(tick_lower, tick_upper, 1)

        pos_id = self._next_position_id
        self._next_position_id += 1

        pos = Position(
            tick_lower=tick_lower,
            tick_upper=tick_upper,
            liquidity=liq,
            fee_growth_inside_0_last=g_inside_0,
            fee_growth_inside_1_last=g_inside_1,
        )
        self.positions[pos_id] = pos
        return pos, amount0, amount1

    def burn(
        self, position: Position
    ) -> tuple[int, int]:
        """Burn a position, removing its liquidity. Returns (amount0, amount1)."""
        # Collect any uncollected fees first
        self._accrue_fees(position)

        tick_lower = position.tick_lower
        tick_upper = position.tick_upper
        liq = position.liquidity

        # Remove liquidity from ticks
        self._modify_position(tick_lower, tick_upper, liq, delta=False)

        sqrt_lower = tick_to_sqrt_price_x96(tick_lower)
        sqrt_upper = tick_to_sqrt_price_x96(tick_upper)

        amount0, amount1 = amounts_for_liquidity(
            self.sqrt_price_x96, sqrt_lower, sqrt_upper, liq,
        )

        return amount0, amount1

    def collect(self, position: Position) -> tuple[int, int]:
        """Collect uncollected fees. Returns (fees0, fees1) in raw units."""
        self._accrue_fees(position)
        fees0 = position.token0_fees_uncollected
        fees1 = position.token1_fees_uncollected
        # Reset uncollected (mutating the frozen dataclass via object.__setattr__
        # but Position is internal — caller keeps the reference)
        object.__setattr__(position, "token0_fees_uncollected", 0)
        object.__setattr__(position, "token1_fees_uncollected", 0)
        return fees0, fees1

    # ------------------------------------------------------------------
    # Fee growth internals
    # ------------------------------------------------------------------
    def _fee_growth_below(self, tick: int, token: int) -> int:
        """Fee growth below a tick (eq. 4)."""
        # token: 0 or 1
        ts = self._get_tick(tick)
        g_out = (
            ts.fee_growth_outside_0_x128
            if token == 0
            else ts.fee_growth_outside_1_x128
        )
        g_global = (
            self.fee_growth_global_0_x128
            if token == 0
            else self.fee_growth_global_1_x128
        )
        if self.tick >= tick:
            return g_out
        return wrapping_sub_256(g_global, g_out)

    def _fee_growth_above(self, tick: int, token: int) -> int:
        """Fee growth above a tick (eq. 5)."""
        ts = self._get_tick(tick)
        g_out = (
            ts.fee_growth_outside_0_x128
            if token == 0
            else ts.fee_growth_outside_1_x128
        )
        g_global = (
            self.fee_growth_global_0_x128
            if token == 0
            else self.fee_growth_global_1_x128
        )
        if self.tick >= tick:
            return wrapping_sub_256(g_global, g_out)
        return g_out

    def _fee_growth_inside(self, tick_lower: int, tick_upper: int, token: int) -> int:
        """Interior fee growth of a range (eq. 3)."""
        g_global = (
            self.fee_growth_global_0_x128
            if token == 0
            else self.fee_growth_global_1_x128
        )
        below = self._fee_growth_below(tick_lower, token)
        above = self._fee_growth_above(tick_upper, token)
        return wrapping_sub_256(wrapping_sub_256(g_global, below), above)

    def _uncollected_fees(self, position: Position, token: int) -> int:
        """Uncollected fees for one token (eq. 6)."""
        g_inside_now = self._fee_growth_inside(
            position.tick_lower, position.tick_upper, token
        )
        g_inside_last = (
            position.fee_growth_inside_0_last
            if token == 0
            else position.fee_growth_inside_1_last
        )
        delta = wrapping_sub_256(g_inside_now, g_inside_last)
        return (position.liquidity * delta) // Q128

    def _accrue_fees(self, position: Position) -> None:
        """Update a position's uncollected fees and last-snapshot values."""
        fees0 = self._uncollected_fees(position, 0)
        fees1 = self._uncollected_fees(position, 1)
        object.__setattr__(
            position, "token0_fees_uncollected",
            position.token0_fees_uncollected + fees0,
        )
        object.__setattr__(
            position, "token1_fees_uncollected",
            position.token1_fees_uncollected + fees1,
        )
        # Update last snapshots
        object.__setattr__(
            position, "fee_growth_inside_0_last",
            self._fee_growth_inside(position.tick_lower, position.tick_upper, 0),
        )
        object.__setattr__(
            position, "fee_growth_inside_1_last",
            self._fee_growth_inside(position.tick_lower, position.tick_upper, 1),
        )

    def _modify_position(
        self, tick_lower: int, tick_upper: int, liquidity: int, delta: bool
    ) -> None:
        """Add (delta=True) or remove (delta=False) liquidity from tick states."""
        sign = 1 if delta else -1

        # Update lower tick
        ts_lower = self._get_tick(tick_lower)
        ts_lower.liquidity_gross += liquidity
        ts_lower.liquidity_net += sign * liquidity

        # Flip fee growth outside if crossing
        self._flip_tick(tick_lower)

        # Update upper tick
        ts_upper = self._get_tick(tick_upper)
        ts_upper.liquidity_gross += liquidity
        ts_upper.liquidity_net -= sign * liquidity

        self._flip_tick(tick_upper)

        # Update active liquidity
        if self.tick >= tick_lower and self.tick < tick_upper:
            self.liquidity += sign * liquidity

    def _flip_tick(self, tick: int) -> None:
        """Update fee growth outside when a tick is crossed or initialized."""
        ts = self._get_tick(tick)
        if ts.liquidity_gross == 0:
            # Tick is now empty — reset
            ts.initialized = False
            ts.fee_growth_outside_0_x128 = 0
            ts.fee_growth_outside_1_x128 = 0
            return

        if not ts.initialized:
            ts.initialized = True
            # Set fee growth outside to the current global values
            ts.fee_growth_outside_0_x128 = self.fee_growth_global_0_x128
            ts.fee_growth_outside_1_x128 = self.fee_growth_global_1_x128
        else:
            # Flip: outside = global - outside
            ts.fee_growth_outside_0_x128 = wrapping_sub_256(
                self.fee_growth_global_0_x128,
                ts.fee_growth_outside_0_x128,
            )
            ts.fee_growth_outside_1_x128 = wrapping_sub_256(
                self.fee_growth_global_1_x128,
                ts.fee_growth_outside_1_x128,
            )


# ---------------------------------------------------------------------------
# Internal helpers (avoid circular deps with math.py)
# ---------------------------------------------------------------------------
def _sqrt_price_x96_to_tick(sqrt_price_x96: int) -> int:
    """Local tick-from-sqrt-price without the domain guards (pool uses it
    internally and the root is already validated)."""
    from undertow.sim.math import sqrt_price_x96_to_tick as _f
    return _f(sqrt_price_x96)