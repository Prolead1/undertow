"""Concentrated-liquidity position and valuation math (`CONTRACTS.md` §5).

This module implements the Uniswap V3 position math of the lesson plan
§5.3–§5.6: token amounts (eqs 8–9), position value (eq 11), impermanent loss
against a HODL benchmark (§5.6), fee accrual, and the two position factories.

Orientation (ADR-009)
---------------------
The protocol quantities are kept **raw** and only the money is humanised:

* ``sqrt_price`` is the raw Uniswap sqrt price ``sqrt(1.0001**tick)`` — the
  quantity produced by :func:`calc_sqrt_price_a` / :func:`calc_sqrt_price_b`
  and by ``undertow.data.tick_to_sqrt_price_x96(tick) / Q96``.  It is
  monotonically increasing in tick, so the standard V3 amount formulas
  (eqs 8–9) produce non-negative amounts for ``tick_lower < tick_upper``.
  ``tick`` and ``liquidity`` (``L``) are likewise raw protocol quantities;
  ``Position.liquidity`` is directly comparable to the tape's active
  liquidity.
* ``price`` is the **human** price ``10**(dec1 - dec0) / sqrt_price**2`` for
  the pinned USDC/WETH pools (USDC per WETH, decreasing in tick), i.e. the
  value returned by ``undertow.data.tick_to_price(tick, dec0, dec1)``.
* ``Position.amount0`` / ``Position.amount1`` return **human** token units:
  the raw amount divided by ``10**dec0`` / ``10**dec1``.
  ``value = amount0 + amount1 * price`` is therefore USDC (eq 11).
* ``Position.uncollected_fees`` computes the raw ``L * Δfee_growth`` and then
  divides by ``10**dec0`` / ``10**dec1`` to report human token fees.

The ``10**dec`` factors cancel in the IL helpers, so ``il_vs_hodl`` /
``il_fraction`` are unchanged in sign and magnitude from the raw picture.

Conversion helpers (ADR-009; used by S06/S07/S11/S12):

    sqrt_price_raw = 10**((dec1 - dec0) / 2) / sqrt(price)
    price_human    = 10**(dec1 - dec0) / sqrt_price_raw**2
    tick           = price_to_tick(price, dec0, dec1)   # floor semantics

Equations in comments refer to lesson plan §5.3–§5.6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from undertow.sim.types import (
    PositionError,
    Price,
    SqrtPrice,
    Tick,
    TickSpacing,
    Wealth,
)

# Default token decimals for the pinned USDC/WETH (token0/token1) pools.
DEFAULT_DEC0: int = 6
DEFAULT_DEC1: int = 18

# The tick price base.  This is the mathematical constant of the Uniswap V3 tick
# definition (1.0001), not a data-pipeline primitive: S04's math is pure float64
# and the exact integer form is cross-checked against
# ``undertow.data.tick_to_sqrt_price_x96`` in ``tests/sim/test_position.py``.
_TICK_BASE: float = 1.0001

__all__ = [
    "DEFAULT_DEC0",
    "DEFAULT_DEC1",
    "Position",
    "calc_sqrt_price_a",
    "calc_sqrt_price_b",
    "initial_deposit",
    "position_from_amounts",
]


def calc_sqrt_price_a(tick: Tick) -> SqrtPrice:
    """``sqrt(1.0001 ** tick)`` — the **raw** sqrt price at ``tick``.

    Named ``_a`` for readability at call sites where ``tick`` is the lower
    bound; identical to :func:`calc_sqrt_price_b`.  This is the Uniswap
    ``sqrt_price_x96 / Q96`` quantity (ADR-009), not ``sqrt(price_human)``.
    """
    return _TICK_BASE ** (int(tick) / 2.0)


def calc_sqrt_price_b(tick: Tick) -> SqrtPrice:
    """``sqrt(1.0001 ** tick)`` — raw alias for :func:`calc_sqrt_price_a`."""
    return calc_sqrt_price_a(tick)


@dataclass(frozen=True, slots=True)
class Position:
    """A concentrated-liquidity position on ``[tick_lower, tick_upper]``.

    Matches the Uniswap V3 NFT position model: **raw** liquidity ``L``, raw
    tick bounds, fee tier.  ``fee_growth_inside_last_0/1`` are the per-position
    snapshots used to compute uncollected fees since the last update/collect.
    ``dec0`` / ``dec1`` are the token decimals (default 6/18 for USDC/WETH)
    used to humanise the returned token amounts and fees (ADR-009).
    """

    tick_lower: Tick
    tick_upper: Tick
    liquidity: float  # L — the raw invariant quantity
    tick_spacing: TickSpacing
    fee_growth_inside_last_0: float = 0.0  # snapshot for token0 fee accrual
    fee_growth_inside_last_1: float = 0.0  # snapshot for token1 fee accrual
    dec0: int = DEFAULT_DEC0  # token0 (USDC) decimals
    dec1: int = DEFAULT_DEC1  # token1 (WETH) decimals

    def __post_init__(self) -> None:
        if self.tick_lower >= self.tick_upper:
            raise PositionError("tick_lower must be < tick_upper")
        if self.liquidity <= 0:
            raise PositionError("liquidity must be > 0")
        if int(self.tick_spacing) <= 0:
            raise PositionError("tick_spacing must be > 0")
        if int(self.dec0) < 0 or int(self.dec1) < 0:
            raise PositionError("token decimals must be >= 0")

    # -- Bounds -----------------------------------------------------------
    @property
    def sqrt_price_lower(self) -> SqrtPrice:
        """Raw ``sqrt(1.0001**tick_lower)``."""
        return calc_sqrt_price_a(self.tick_lower)

    @property
    def sqrt_price_upper(self) -> SqrtPrice:
        """Raw ``sqrt(1.0001**tick_upper)``."""
        return calc_sqrt_price_b(self.tick_upper)

    # -- Token amounts (eqs 8–9 of §5.3) ----------------------------------
    def amount0(self, sqrt_price: SqrtPrice) -> float:
        """Human token0 (USDC) holdings at raw ``sqrt_price``.

        The raw amount ``L·(1/s − 1/s_u)`` is divided by ``10**dec0`` per
        ADR-009.  Returns 0 when the position has no token0 at this price (at
        or above the upper bound).
        """
        sqrt_lower = self.sqrt_price_lower
        sqrt_upper = self.sqrt_price_upper
        if sqrt_price >= sqrt_upper:
            raw = 0.0
        elif sqrt_price <= sqrt_lower:
            raw = self.liquidity * (1.0 / sqrt_lower - 1.0 / sqrt_upper)
        else:
            raw = self.liquidity * (1.0 / sqrt_price - 1.0 / sqrt_upper)
        return raw / 10.0**self.dec0

    def amount1(self, sqrt_price: SqrtPrice) -> float:
        """Human token1 (WETH) holdings at raw ``sqrt_price``.

        The raw amount ``L·(s − s_l)`` is divided by ``10**dec1`` per ADR-009.
        Returns 0 when the position has no token1 at this price (at or below
        the lower bound).
        """
        sqrt_lower = self.sqrt_price_lower
        sqrt_upper = self.sqrt_price_upper
        if sqrt_price <= sqrt_lower:
            raw = 0.0
        elif sqrt_price >= sqrt_upper:
            raw = self.liquidity * (sqrt_upper - sqrt_lower)
        else:
            raw = self.liquidity * (sqrt_price - sqrt_lower)
        return raw / 10.0**self.dec1

    # -- Value (eq 11 of §5.3.4) ------------------------------------------
    def value(self, sqrt_price: SqrtPrice, price: Price) -> Wealth:
        """Position value in USDC: ``amount0 + amount1 * price``.

        ``sqrt_price`` is raw and ``price`` is the human USDC-per-WETH price
        paired with it (ADR-009).
        """
        return self.amount0(sqrt_price) + self.amount1(sqrt_price) * price

    # -- IL vs HODL (§5.6) ------------------------------------------------
    def hodl_value(
        self,
        entry_sqrt_price: SqrtPrice,
        entry_price: Price,
        current_price: Price,
    ) -> Wealth:
        """Value of the initial deposit if simply held (no LPing).

        The initial human token amounts are those of this position at
        ``entry_sqrt_price``; they are marked at ``current_price``.
        ``entry_price`` is accepted for interface symmetry and is not needed
        once the entry token amounts are known.
        """
        del entry_price  # retained for the frozen signature; amounts fix the HODL basket
        initial_token0 = self.amount0(entry_sqrt_price)
        initial_token1 = self.amount1(entry_sqrt_price)
        return initial_token0 + initial_token1 * current_price

    def il_vs_hodl(
        self,
        entry_sqrt_price: SqrtPrice,
        entry_price: Price,
        current_sqrt_price: SqrtPrice,
        current_price: Price,
    ) -> float:
        """Impermanent loss: ``V(current) - HODL(current)``. Negative = loss."""
        return self.value(current_sqrt_price, current_price) - self.hodl_value(
            entry_sqrt_price, entry_price, current_price
        )

    def il_fraction(
        self,
        entry_sqrt_price: SqrtPrice,
        entry_price: Price,
        current_sqrt_price: SqrtPrice,
        current_price: Price,
    ) -> float:
        """IL as a fraction of HODL value: ``(V - HODL) / HODL``.

        The ``10**dec`` token scalings cancel, so this equals the raw-position
        IL fraction (ADR-009).
        """
        hodl = self.hodl_value(entry_sqrt_price, entry_price, current_price)
        if hodl == 0.0:
            raise PositionError("HODL value is zero; IL fraction is undefined")
        loss = self.value(current_sqrt_price, current_price) - hodl
        return loss / hodl

    # -- Fee accrual (§10.2.4 of roadmap) ---------------------------------
    def uncollected_fees(
        self,
        fee_growth_inside_0: float,
        fee_growth_inside_1: float,
    ) -> tuple[float, float]:
        """``L * (current_fee_growth_inside - last_snapshot)`` per token.

        The raw fee, computed on the raw ``L`` and raw fee-growth
        accumulators, is divided by ``10**dec0`` / ``10**dec1`` so the result
        is in human token units (ADR-009).  Returns
        ``(fees_token0, fees_token1)``.
        """
        raw_0 = self.liquidity * (
            fee_growth_inside_0 - self.fee_growth_inside_last_0
        )
        raw_1 = self.liquidity * (
            fee_growth_inside_1 - self.fee_growth_inside_last_1
        )
        return (raw_0 / 10.0**self.dec0, raw_1 / 10.0**self.dec1)

    def snapshot_fees(
        self,
        fee_growth_inside_0: float,
        fee_growth_inside_1: float,
    ) -> None:
        """Update the fee-growth snapshots to current (a Collect without withdrawal).

        ``Position`` is frozen/slots, so the update goes through
        ``object.__setattr__``.  Note this also changes the dataclass-generated
        ``__hash__``; positions are used as values, not dict/set keys, so the
        post-init mutation is intentional but should not be relied on for
        hashing after a snapshot.
        """
        object.__setattr__(self, "fee_growth_inside_last_0", fee_growth_inside_0)
        object.__setattr__(self, "fee_growth_inside_last_1", fee_growth_inside_1)

    # -- Range predicates -------------------------------------------------
    def in_range(self, sqrt_price: SqrtPrice) -> bool:
        """True when ``s_lower < sqrt_price < s_upper`` (raw sqrt/tick)."""
        return self.sqrt_price_lower < sqrt_price < self.sqrt_price_upper

    def below_range(self, sqrt_price: SqrtPrice) -> bool:
        """True at or below the raw lower bound (position holds only token0)."""
        return sqrt_price <= self.sqrt_price_lower

    def above_range(self, sqrt_price: SqrtPrice) -> bool:
        """True at or above the raw upper bound (position holds only token1)."""
        return sqrt_price >= self.sqrt_price_upper


def initial_deposit(
    sqrt_price: SqrtPrice,
    price: Price,
    tick_lower: Tick,
    tick_upper: Tick,
    capital: Wealth,
    tick_spacing: TickSpacing,
    *,
    dec0: int = DEFAULT_DEC0,
    dec1: int = DEFAULT_DEC1,
) -> Position:
    """Solve for raw ``L`` such that ``V(sqrt_price, price) == capital``.

    Equation (11) inverted, with human token amounts and human ``price``
    (ADR-009).  Works in range and in both out-of-range branches (token0-only
    below the raw lower bound, token1-only above).  Fee-growth snapshots start
    at 0.
    """
    if capital <= 0:
        raise PositionError("capital must be > 0")
    if sqrt_price <= 0:
        raise PositionError("sqrt_price must be > 0")
    if price <= 0:
        raise PositionError("price must be > 0")
    if dec0 < 0 or dec1 < 0:
        raise PositionError("token decimals must be >= 0")

    scale0 = 10.0**dec0
    scale1 = 10.0**dec1

    probe = Position(tick_lower, tick_upper, 1.0, tick_spacing)
    lower = probe.sqrt_price_lower
    upper = probe.sqrt_price_upper

    if sqrt_price >= upper:
        # All token1: raw amount1 = L*(upper - lower), priced at human `price`.
        value_per_liquidity = (upper - lower) * price / scale1
    elif sqrt_price <= lower:
        # All token0: raw amount0 = L*(1/lower - 1/upper).
        value_per_liquidity = (1.0 / lower - 1.0 / upper) / scale0
    else:
        raw0 = 1.0 / sqrt_price - 1.0 / upper
        raw1 = sqrt_price - lower
        value_per_liquidity = raw0 / scale0 + raw1 * price / scale1

    if value_per_liquidity <= 0:
        raise PositionError("capital cannot be represented by this position range")
    liquidity = capital / value_per_liquidity
    return Position(
        tick_lower, tick_upper, liquidity, tick_spacing, dec0=dec0, dec1=dec1
    )


def position_from_amounts(
    tick_lower: Tick,
    tick_upper: Tick,
    amount0: float,
    amount1: float,
    sqrt_price: SqrtPrice,
    tick_spacing: TickSpacing,
    *,
    dec0: int = DEFAULT_DEC0,
    dec1: int = DEFAULT_DEC1,
) -> Position:
    """Recover raw ``L`` from **human** token amounts (eqs 8–9 inverted).

    The amounts are scaled back to raw units (``* 10**dec0`` / ``* 10**dec1``,
    ADR-009) before inverting.  In range the two legs must agree on ``L``; the
    smaller (binding) value is used, matching Uniswap's
    ``LiquidityAmounts.getLiquidityForAmounts``.  Below the raw lower bound
    only ``amount0`` is used; above the raw upper bound only ``amount1``.
    """
    if amount0 < 0 or amount1 < 0:
        raise PositionError("amounts must be >= 0")
    if dec0 < 0 or dec1 < 0:
        raise PositionError("token decimals must be >= 0")

    raw_amount0 = amount0 * 10.0**dec0
    raw_amount1 = amount1 * 10.0**dec1

    probe = Position(tick_lower, tick_upper, 1.0, tick_spacing)
    lower = probe.sqrt_price_lower
    upper = probe.sqrt_price_upper

    if sqrt_price <= lower:
        delta0 = 1.0 / lower - 1.0 / upper
        if delta0 <= 0:  # defensive: __post_init__ already forbids empty ranges
            raise PositionError("degenerate range")
        liquidity = raw_amount0 / delta0
    elif sqrt_price >= upper:
        delta1 = upper - lower
        if delta1 <= 0:  # defensive: __post_init__ already forbids empty ranges
            raise PositionError("degenerate range")
        liquidity = raw_amount1 / delta1
    else:
        delta0 = 1.0 / sqrt_price - 1.0 / upper
        delta1 = sqrt_price - lower
        l0 = raw_amount0 / delta0 if delta0 > 0 else math.inf
        l1 = raw_amount1 / delta1 if delta1 > 0 else math.inf
        liquidity = min(l0, l1)

    return Position(
        tick_lower, tick_upper, liquidity, tick_spacing, dec0=dec0, dec1=dec1
    )
