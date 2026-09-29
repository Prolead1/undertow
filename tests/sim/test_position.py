"""S04 — position & valuation math (`CONTRACTS.md` §5, golden values §19).

ADR-009 orientation: ``sqrt_price`` is the **raw** Uniswap sqrt price
(``sqrt(1.0001**tick)``, ≈18244.326 at tick 196242), ``liquidity`` is raw, and
``price`` is the **human** USDC-per-WETH price (``10**(dec1-dec0)/sqrt²``,
≈3004.307 at tick 196242) returned by ``undertow.data.tick_to_price(t, 6, 18)``.
``Position.amount0``/``amount1`` return human token units, so
``value = amount0 + amount1*price`` is USDC.

The golden IL numbers are the lesson-plan §5.6.1 table transcribed into
``CONTRACTS.md`` §19.  The V2 ``r=1.2`` row carries the known arithmetic slip
(``-0.00454``); the analytic value ``2√r/(1+r) − 1 = -0.004141`` is asserted
and the correction is documented at the test.
"""

from __future__ import annotations

import math
from decimal import Decimal

import pytest

from undertow.data import (
    MAX_TICK,
    MIN_TICK,
    Q96,
    price_to_tick,
    tick_to_price,
    tick_to_sqrt_price_x96,
)
from undertow.sim.core.position import (
    DEFAULT_DEC0,
    DEFAULT_DEC1,
    Position,
    calc_sqrt_price_a,
    calc_sqrt_price_b,
    initial_deposit,
    position_from_amounts,
)
from undertow.sim.types import PositionError, Tick, TickSpacing

# Tick 196 242 is the data-module reference tick for ~$3004 USDC/WETH
# (CONTRACTS §19 / data CONTRACTS §3.0).  It is our canonical entry point.
ENTRY_TICK = Tick(196242)
TICK_SPACING = TickSpacing(60)
CAPITAL = 100_000.0


def _entry_sqrt() -> float:
    """Raw sqrt price at the entry tick (~18244.326)."""
    return calc_sqrt_price_a(ENTRY_TICK)


def _entry_price() -> float:
    """Human USDC-per-WETH price at the entry tick (~3004.307)."""
    return float(tick_to_price(ENTRY_TICK, DEFAULT_DEC0, DEFAULT_DEC1))


def _sqrt_for_price(price: float, dec0: int = DEFAULT_DEC0, dec1: int = DEFAULT_DEC1) -> float:
    """ADR-009 conversion: ``sqrt_price_raw = 10**((dec1-dec0)/2)/sqrt(price)``."""
    return math.sqrt(10.0 ** (dec1 - dec0) / price)


def _near_entry_position(liquidity: float = 1.0) -> Position:
    return Position(
        Tick(int(ENTRY_TICK) - 120),
        Tick(int(ENTRY_TICK) + 120),
        liquidity,
        TICK_SPACING,
    )


def _concentrated_position(half_width: float) -> tuple[Position, float, float]:
    """A position whose human-price range is ~``[(1-w)p0, (1+w)p0]``, tick-quantized.

    Human price is *decreasing* in tick (ADR-009), so the **lower tick** carries
    the **higher** price bound and vice versa.  Returns
    ``(position, effective_lower_price, effective_upper_price)`` where the two
    prices are the actual tick-realized bounds
    (``tick_to_price(tick_upper)`` / ``tick_to_price(tick_lower)``) used by the
    independent closed form.
    """
    entry_price = _entry_price()
    lower_price = Decimal(str((1.0 - half_width) * entry_price))
    upper_price = Decimal(str((1.0 + half_width) * entry_price))
    tick_lower = Tick(price_to_tick(upper_price, DEFAULT_DEC0, DEFAULT_DEC1))
    tick_upper = Tick(price_to_tick(lower_price, DEFAULT_DEC0, DEFAULT_DEC1))
    assert tick_lower < tick_upper
    position = Position(tick_lower, tick_upper, 1.0, TICK_SPACING)
    price_lower = float(tick_to_price(int(tick_upper), DEFAULT_DEC0, DEFAULT_DEC1))
    price_upper = float(tick_to_price(int(tick_lower), DEFAULT_DEC0, DEFAULT_DEC1))
    return position, price_lower, price_upper


def _concentrated_il_closed_form(
    entry_price: float, price_lower: float, price_upper: float, price: float
) -> float:
    """Independent closed form for the concentrated IL fraction (derivation below).

    With ``K = L·10**(-(dec0+dec1)/2)`` the amount legs are
    ``amount0 = K(√P − √Pa)`` and ``amount1 = K(1/√P − 1/√Pb)``, so the
    ``10**dec`` scalings cancel and the fraction depends only on human prices::

        IL/K      = V − HODL  (piecewise on P vs [Pa, Pb])
        HODL/K    = √P0 − √Pa + P/√P0 − P/√Pb
    """
    sq0, sqa, sqb = math.sqrt(entry_price), math.sqrt(price_lower), math.sqrt(price_upper)
    sqp = math.sqrt(price)
    if price > price_upper:
        il = sqb - sq0 + price * (1.0 / sqb - 1.0 / sq0)
    elif price < price_lower:
        il = price / sqa - sq0 + sqa - price / sq0
    else:
        il = 2.0 * sqp - sq0 - price / sq0
    hodl = sq0 - sqa + price / sq0 - price / sqb
    return il / hodl


# ---------------------------------------------------------------------------
# Golden conversions (§19)
# ---------------------------------------------------------------------------
def test_calc_sqrt_price_golden() -> None:
    sqrt_price = calc_sqrt_price_a(ENTRY_TICK)
    assert sqrt_price == pytest.approx(18244.326, rel=1e-6)
    assert calc_sqrt_price_b(ENTRY_TICK) == sqrt_price
    # Raw sqrt price is the exact integer Q64.96 form / 2**96.
    assert sqrt_price == pytest.approx(tick_to_sqrt_price_x96(ENTRY_TICK) / Q96, rel=1e-9)
    # Monotonically increasing in tick.
    assert calc_sqrt_price_a(Tick(int(ENTRY_TICK) + 1)) > sqrt_price


def test_tick_to_price_golden() -> None:
    entry_price = _entry_price()
    assert entry_price == pytest.approx(3004.307, rel=1e-6)
    # Human price = 10**(dec1-dec0) / sqrt_price_raw**2 (ADR-009).
    assert entry_price == pytest.approx(
        10.0 ** (DEFAULT_DEC1 - DEFAULT_DEC0) / (_entry_sqrt() ** 2), rel=1e-9
    )
    # And it is decreasing in tick.
    p_hi = float(tick_to_price(int(ENTRY_TICK) + 1, DEFAULT_DEC0, DEFAULT_DEC1))
    p_lo = float(tick_to_price(int(ENTRY_TICK) - 1, DEFAULT_DEC0, DEFAULT_DEC1))
    assert p_hi < entry_price < p_lo


def test_price_to_tick_round_trip() -> None:
    entry_price = _entry_price()
    # `price_to_tick` takes a Decimal; the float tick_to_price value stringifies
    # to the lower IEEE-754 neighbour, so this round-trips exactly.
    assert price_to_tick(Decimal(str(entry_price)), DEFAULT_DEC0, DEFAULT_DEC1) == ENTRY_TICK
    # The exact Decimal from tick_to_price sits a hair above the boundary, so
    # floor semantics land on the neighbour tick; assert the ±1 contract.
    floor_tick = price_to_tick(
        tick_to_price(ENTRY_TICK, DEFAULT_DEC0, DEFAULT_DEC1), DEFAULT_DEC0, DEFAULT_DEC1
    )
    assert abs(floor_tick - int(ENTRY_TICK)) <= 1


# ---------------------------------------------------------------------------
# Token amounts and range behaviour (eqs 8–9)
# ---------------------------------------------------------------------------
def test_entry_position_fixture_amounts_are_human() -> None:
    """The fixture's value is the capital and the legs are sane human sizes."""
    entry_sqrt = _entry_sqrt()
    entry_price = _entry_price()
    position = initial_deposit(
        entry_sqrt,
        entry_price,
        Tick(int(ENTRY_TICK) - 120),
        Tick(int(ENTRY_TICK) + 120),
        CAPITAL,
        TICK_SPACING,
    )
    amount0 = position.amount0(entry_sqrt)
    amount1 = position.amount1(entry_sqrt)

    assert position.value(entry_sqrt, entry_price) == pytest.approx(CAPITAL, rel=1e-9)
    # USDC leg ~50k, WETH leg ~16.6 (NOT ~1e13 raw units).
    assert amount0 == pytest.approx(50_000.0, rel=1e-3)
    assert amount1 == pytest.approx(16.64, rel=1e-2)
    assert 1_000.0 < amount0 < CAPITAL
    assert 1.0 < amount1 < 100.0


def test_amount0_zero_above_range() -> None:
    position = _near_entry_position(liquidity=10.0)
    sqrt_above = position.sqrt_price_upper * 1.0001
    assert position.above_range(sqrt_above)
    assert position.amount0(sqrt_above) == 0.0
    assert position.amount1(sqrt_above) == pytest.approx(
        10.0 * (position.sqrt_price_upper - position.sqrt_price_lower) / 10.0**18
    )


def test_amount1_zero_below_range() -> None:
    position = _near_entry_position(liquidity=10.0)
    sqrt_below = position.sqrt_price_lower / 1.0001
    assert position.below_range(sqrt_below)
    assert position.amount1(sqrt_below) == 0.0
    assert position.amount0(sqrt_below) == pytest.approx(
        10.0 * (1.0 / position.sqrt_price_lower - 1.0 / position.sqrt_price_upper)
        / 10.0**6
    )


def test_amounts_in_range_and_value_identity() -> None:
    position = _near_entry_position(liquidity=10.0)
    entry_sqrt = _entry_sqrt()
    entry_price = _entry_price()
    assert position.in_range(entry_sqrt)
    assert not position.below_range(entry_sqrt)
    assert not position.above_range(entry_sqrt)

    expected = position.amount0(entry_sqrt) + position.amount1(entry_sqrt) * entry_price
    assert position.value(entry_sqrt, entry_price) == pytest.approx(expected, rel=1e-15)

    # Round-trip: human token amounts invert to the same raw L.
    rebuilt = position_from_amounts(
        position.tick_lower,
        position.tick_upper,
        position.amount0(entry_sqrt),
        position.amount1(entry_sqrt),
        entry_sqrt,
        TICK_SPACING,
    )
    assert rebuilt.liquidity == pytest.approx(position.liquidity, rel=1e-12)


def test_amounts_scale_with_decimals() -> None:
    """dec0/dec1 only scale the human output; raw L is untouched (ADR-009)."""
    sqrt_price = _entry_sqrt()
    human = _near_entry_position(liquidity=1.0)
    raw = Position(
        human.tick_lower,
        human.tick_upper,
        1.0,
        TICK_SPACING,
        dec0=0,
        dec1=0,
    )
    assert human.amount0(sqrt_price) == pytest.approx(raw.amount0(sqrt_price) / 10.0**6, rel=1e-12)
    assert human.amount1(sqrt_price) == pytest.approx(raw.amount1(sqrt_price) / 10.0**18, rel=1e-12)


def test_full_range_amounts_are_non_negative() -> None:
    """Full-range regression: raw sqrt ordering keeps the V3 amount legs ≥ 0.

    Under the rejected human ``sqrt_price = sqrt(price)`` orientation the range
    bounds invert (``s_lower > s_upper``) and these amounts go negative (ADR-009
    Problem 3).  With the raw orientation every leg is non-negative, and the
    in-range legs are strictly positive.
    """
    entry_sqrt = _entry_sqrt()
    entry_price = _entry_price()
    position = initial_deposit(
        entry_sqrt, entry_price, MIN_TICK, MAX_TICK, CAPITAL, TICK_SPACING
    )
    assert position.sqrt_price_lower < position.sqrt_price_upper

    for multiplier in (0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0):
        price = multiplier * entry_price
        sqrt_price = _sqrt_for_price(price)
        amount0 = position.amount0(sqrt_price)
        amount1 = position.amount1(sqrt_price)
        assert amount0 >= 0.0, (multiplier, amount0)
        assert amount1 >= 0.0, (multiplier, amount1)
        # Entry price is far inside MIN_TICK..MAX_TICK, so both legs are live.
        assert amount0 > 0.0
        assert amount1 > 0.0
        # Value identity holds on the whole domain.
        assert position.value(sqrt_price, price) == pytest.approx(
            amount0 + amount1 * price, rel=1e-15
        )


# ---------------------------------------------------------------------------
# Golden IL values (§19)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        # CONTRACTS §19 historically tabulated -0.00454 for r=1.2.  The
        # analytic V2 value 2*sqrt(r)/(1+r) - 1 is -0.004141 (ADR-009 corrects
        # the slip; the r=0.5 row is exact).  Assert the real value.
        (1.2, -0.004141),
        (0.5, -0.057191),
    ],
)
def test_v2_il_golden(ratio: float, expected: float) -> None:
    entry_sqrt = _entry_sqrt()
    entry_price = _entry_price()
    position = initial_deposit(
        entry_sqrt, entry_price, MIN_TICK, MAX_TICK, CAPITAL, TICK_SPACING
    )

    current_price = ratio * entry_price
    current_sqrt = _sqrt_for_price(current_price)

    il = position.il_fraction(entry_sqrt, entry_price, current_sqrt, current_price)
    assert il == pytest.approx(expected, abs=1e-5)
    # The analytic closed form for a full-range (V2) position.
    assert il == pytest.approx(2.0 * math.sqrt(ratio) / (1.0 + ratio) - 1.0, rel=1e-9)
    assert il < 0.0


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (1.2, -0.066),
        (0.5, -0.325),
    ],
)
def test_concentrated_il_golden(ratio: float, expected: float) -> None:
    """Concentrated ±10% IL, range defined as ``[0.9·p0, 1.1·p0]`` in human price.

    Ticks quantize the bounds, so the independent closed form is evaluated at the
    tick-realized bounds; it matches the code to float64 precision.  Tick-realized
    bounds give -0.065685 / -0.325266; the exact ±10% bounds give
    -0.065658 / -0.325404 — both within the 2e-3 tolerance of the tabulated
    -0.066 / -0.325.
    """
    position, price_lower, price_upper = _concentrated_position(half_width=0.10)
    entry_sqrt = _entry_sqrt()
    entry_price = _entry_price()

    current_price = ratio * entry_price
    current_sqrt = _sqrt_for_price(current_price)

    il = position.il_fraction(entry_sqrt, entry_price, current_sqrt, current_price)
    closed_form = _concentrated_il_closed_form(
        entry_price, price_lower, price_upper, current_price
    )
    assert il == pytest.approx(closed_form, rel=1e-9)
    assert il == pytest.approx(expected, abs=2e-3)
    assert il < 0.0


def test_concentrated_il_in_range() -> None:
    """A genuinely in-range concentrated move (r=1.05) exercises the in-range
    branch of the independent closed form and the production amount path."""
    position, price_lower, price_upper = _concentrated_position(half_width=0.10)
    entry_sqrt = _entry_sqrt()
    entry_price = _entry_price()
    current_price = 1.05 * entry_price
    current_sqrt = _sqrt_for_price(current_price)

    il = position.il_fraction(entry_sqrt, entry_price, current_sqrt, current_price)
    closed_form = _concentrated_il_closed_form(
        entry_price, price_lower, price_upper, current_price
    )
    assert il == pytest.approx(closed_form, rel=1e-9)
    assert il < 0.0


# ---------------------------------------------------------------------------
# initial_deposit (eq 11 inverted) — capital conservation in all 3 branches
# ---------------------------------------------------------------------------
def test_initial_deposit_conserves_capital_in_range() -> None:
    entry_sqrt = _entry_sqrt()
    entry_price = _entry_price()
    position = initial_deposit(
        entry_sqrt,
        entry_price,
        Tick(int(ENTRY_TICK) - 120),
        Tick(int(ENTRY_TICK) + 120),
        CAPITAL,
        TICK_SPACING,
    )
    assert position.liquidity > 0.0
    assert position.in_range(entry_sqrt)
    assert position.value(entry_sqrt, entry_price) == pytest.approx(CAPITAL, rel=1e-9)
    assert position.fee_growth_inside_last_0 == 0.0
    assert position.fee_growth_inside_last_1 == 0.0

    # Doubling capital doubles raw L and leaves value proportional.
    doubled = initial_deposit(
        entry_sqrt,
        entry_price,
        Tick(int(ENTRY_TICK) - 120),
        Tick(int(ENTRY_TICK) + 120),
        2.0 * CAPITAL,
        TICK_SPACING,
    )
    assert doubled.liquidity == pytest.approx(2.0 * position.liquidity, rel=1e-12)


def test_initial_deposit_conserves_capital_below_range() -> None:
    """Raw ``sqrt_price <= s_lower`` branch (all token0; high human price)."""
    reference = _near_entry_position()
    sqrt_below = reference.sqrt_price_lower / 1.01
    price = 10.0 ** (DEFAULT_DEC1 - DEFAULT_DEC0) / (sqrt_below**2)
    position = initial_deposit(
        sqrt_below,
        price,
        reference.tick_lower,
        reference.tick_upper,
        CAPITAL,
        TICK_SPACING,
    )
    assert position.below_range(sqrt_below)
    assert position.amount1(sqrt_below) == 0.0
    assert position.amount0(sqrt_below) > 0.0
    assert position.value(sqrt_below, price) == pytest.approx(CAPITAL, rel=1e-9)


def test_initial_deposit_conserves_capital_above_range() -> None:
    """Raw ``sqrt_price >= s_upper`` branch (all token1; low human price)."""
    reference = _near_entry_position()
    sqrt_above = reference.sqrt_price_upper * 1.01
    price = 10.0 ** (DEFAULT_DEC1 - DEFAULT_DEC0) / (sqrt_above**2)
    position = initial_deposit(
        sqrt_above,
        price,
        reference.tick_lower,
        reference.tick_upper,
        CAPITAL,
        TICK_SPACING,
    )
    assert position.above_range(sqrt_above)
    assert position.amount0(sqrt_above) == 0.0
    assert position.amount1(sqrt_above) > 0.0
    assert position.value(sqrt_above, price) == pytest.approx(CAPITAL, rel=1e-9)


def test_initial_deposit_honours_custom_decimals() -> None:
    entry_sqrt = _entry_sqrt()
    # 18/18-decimal pool: the same human capital needs a different raw L.
    position_6_18 = initial_deposit(
        entry_sqrt, _entry_price(), Tick(196122), Tick(196362), CAPITAL, TICK_SPACING
    )
    price_18_18 = 10.0 ** 0 / (entry_sqrt**2)
    position_18_18 = initial_deposit(
        entry_sqrt,
        price_18_18,
        Tick(196122),
        Tick(196362),
        CAPITAL,
        TICK_SPACING,
        dec0=18,
        dec1=18,
    )
    assert position_18_18.liquidity != pytest.approx(position_6_18.liquidity, rel=1e-6)
    assert position_18_18.value(entry_sqrt, price_18_18) == pytest.approx(CAPITAL, rel=1e-9)


# ---------------------------------------------------------------------------
# HODL / IL invariants
# ---------------------------------------------------------------------------
def test_hodl_and_il_at_entry_are_neutral() -> None:
    entry_sqrt = _entry_sqrt()
    entry_price = _entry_price()
    position = initial_deposit(
        entry_sqrt,
        entry_price,
        Tick(int(ENTRY_TICK) - 120),
        Tick(int(ENTRY_TICK) + 120),
        CAPITAL,
        TICK_SPACING,
    )
    assert position.hodl_value(entry_sqrt, entry_price, entry_price) == pytest.approx(
        CAPITAL, rel=1e-9
    )
    assert position.value(entry_sqrt, entry_price) == pytest.approx(CAPITAL, rel=1e-9)
    assert position.il_vs_hodl(entry_sqrt, entry_price, entry_sqrt, entry_price) == pytest.approx(
        0.0, abs=1e-9
    )
    assert position.il_fraction(
        entry_sqrt, entry_price, entry_sqrt, entry_price
    ) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Fee accrual and snapshots (dec-scaled to human fees)
# ---------------------------------------------------------------------------
def test_uncollected_fees_unchanged_is_zero() -> None:
    position = _near_entry_position(liquidity=5.0)
    assert position.uncollected_fees(0.0, 0.0) == (0.0, 0.0)


def test_snapshot_fees_zeroes_uncollected() -> None:
    position = Position(
        Tick(int(ENTRY_TICK) - 120),
        Tick(int(ENTRY_TICK) + 120),
        1_000.0,
        TICK_SPACING,
        fee_growth_inside_last_0=1.0,
        fee_growth_inside_last_1=2.0,
    )
    position.snapshot_fees(5.0, 7.0)
    assert position.fee_growth_inside_last_0 == 5.0
    assert position.fee_growth_inside_last_1 == 7.0
    assert position.uncollected_fees(5.0, 7.0) == (0.0, 0.0)


def test_uncollected_fees_are_dec_scaled() -> None:
    """Raw ``L·Δfee_growth`` is divided by ``10**dec0`` / ``10**dec1``."""
    position = Position(
        Tick(int(ENTRY_TICK) - 120),
        Tick(int(ENTRY_TICK) + 120),
        1_000.0,
        TICK_SPACING,
        fee_growth_inside_last_0=1.0,
        fee_growth_inside_last_1=2.0,
    )
    fees_0, fees_1 = position.uncollected_fees(3.0, 5.0)
    assert fees_0 == pytest.approx(2_000.0 / 10.0**6)
    assert fees_1 == pytest.approx(3_000.0 / 10.0**18)

    # With zero decimals the fees are the raw accumulator product.
    raw = Position(
        position.tick_lower,
        position.tick_upper,
        1_000.0,
        TICK_SPACING,
        fee_growth_inside_last_0=1.0,
        fee_growth_inside_last_1=2.0,
        dec0=0,
        dec1=0,
    )
    assert raw.uncollected_fees(3.0, 5.0) == (pytest.approx(2_000.0), pytest.approx(3_000.0))

    position.snapshot_fees(3.0, 5.0)
    assert position.uncollected_fees(3.0, 5.0) == (0.0, 0.0)
    assert position.uncollected_fees(4.0, 5.0) == (pytest.approx(1_000.0 / 10.0**6), 0.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"tick_lower": Tick(100), "tick_upper": Tick(50)},
        {"tick_lower": Tick(50), "tick_upper": Tick(100), "liquidity": 0.0},
        {"tick_lower": Tick(50), "tick_upper": Tick(100), "liquidity": -1.0},
        {"tick_lower": Tick(50), "tick_upper": Tick(100), "tick_spacing": TickSpacing(0)},
        {"tick_lower": Tick(50), "tick_upper": Tick(100), "dec0": -1},
        {"tick_lower": Tick(50), "tick_upper": Tick(100), "dec1": -1},
    ],
)
def test_invalid_positions_raise(kwargs: dict) -> None:
    kwargs.setdefault("tick_lower", Tick(50))
    kwargs.setdefault("tick_upper", Tick(100))
    kwargs.setdefault("liquidity", 1.0)
    kwargs.setdefault("tick_spacing", TICK_SPACING)
    with pytest.raises(PositionError):
        Position(**kwargs)


def test_initial_deposit_rejects_bad_inputs() -> None:
    entry_sqrt = _entry_sqrt()
    entry_price = _entry_price()
    tick_lower = Tick(int(ENTRY_TICK) - 120)
    tick_upper = Tick(int(ENTRY_TICK) + 120)
    with pytest.raises(PositionError):
        initial_deposit(entry_sqrt, entry_price, tick_lower, tick_upper, 0.0, TICK_SPACING)
    with pytest.raises(PositionError):
        initial_deposit(entry_sqrt, entry_price, tick_lower, tick_upper, -1.0, TICK_SPACING)
    with pytest.raises(PositionError):
        initial_deposit(0.0, entry_price, tick_lower, tick_upper, CAPITAL, TICK_SPACING)
    with pytest.raises(PositionError):
        initial_deposit(entry_sqrt, 0.0, tick_lower, tick_upper, CAPITAL, TICK_SPACING)
    with pytest.raises(PositionError):
        initial_deposit(
            entry_sqrt, entry_price, tick_lower, tick_upper, CAPITAL, TICK_SPACING, dec0=-1
        )


def test_il_fraction_rejects_zero_hodl() -> None:
    position, _, _ = _concentrated_position(half_width=0.10)
    sqrt_above = position.sqrt_price_upper * 1.01
    with pytest.raises(PositionError):
        position.il_fraction(sqrt_above, 1.0, sqrt_above, 0.0)


def test_position_from_amounts_rejects_zero_and_negative_amounts() -> None:
    tick_lower = Tick(int(ENTRY_TICK) - 120)
    tick_upper = Tick(int(ENTRY_TICK) + 120)
    entry_sqrt = _entry_sqrt()
    with pytest.raises(PositionError):
        position_from_amounts(tick_lower, tick_upper, -1.0, 0.0, entry_sqrt, TICK_SPACING)
    with pytest.raises(PositionError):
        position_from_amounts(tick_lower, tick_upper, 0.0, -1.0, entry_sqrt, TICK_SPACING)
    # Zero amounts imply zero liquidity, which `Position.__post_init__` rejects.
    with pytest.raises(PositionError):
        position_from_amounts(tick_lower, tick_upper, 0.0, 0.0, entry_sqrt, TICK_SPACING)
    with pytest.raises(PositionError):
        position_from_amounts(
            tick_lower, tick_upper, 0.0, 0.0, entry_sqrt, TICK_SPACING, dec0=-1
        )


# ---------------------------------------------------------------------------
# position_from_amounts in the single-sided branches
# ---------------------------------------------------------------------------
def test_position_from_amounts_below_range() -> None:
    reference = _near_entry_position()
    sqrt_below = reference.sqrt_price_lower / 1.01
    price = 10.0 ** (DEFAULT_DEC1 - DEFAULT_DEC0) / (sqrt_below**2)
    position = initial_deposit(
        sqrt_below,
        price,
        reference.tick_lower,
        reference.tick_upper,
        CAPITAL,
        TICK_SPACING,
    )
    assert position.amount1(sqrt_below) == 0.0
    rebuilt = position_from_amounts(
        position.tick_lower,
        position.tick_upper,
        position.amount0(sqrt_below),
        position.amount1(sqrt_below),
        sqrt_below,
        TICK_SPACING,
    )
    assert rebuilt.liquidity == pytest.approx(position.liquidity, rel=1e-9)


def test_position_from_amounts_above_range() -> None:
    reference = _near_entry_position()
    sqrt_above = reference.sqrt_price_upper * 1.01
    price = 10.0 ** (DEFAULT_DEC1 - DEFAULT_DEC0) / (sqrt_above**2)
    position = initial_deposit(
        sqrt_above,
        price,
        reference.tick_lower,
        reference.tick_upper,
        CAPITAL,
        TICK_SPACING,
    )
    assert position.amount0(sqrt_above) == 0.0
    rebuilt = position_from_amounts(
        position.tick_lower,
        position.tick_upper,
        position.amount0(sqrt_above),
        position.amount1(sqrt_above),
        sqrt_above,
        TICK_SPACING,
    )
    assert rebuilt.liquidity == pytest.approx(position.liquidity, rel=1e-9)


def test_position_from_amounts_round_trips_custom_decimals() -> None:
    """Human amounts at custom decimals must invert to the same raw L."""
    sqrt_price = _entry_sqrt()
    position = Position(
        Tick(int(ENTRY_TICK) - 120),
        Tick(int(ENTRY_TICK) + 120),
        1e17,
        TICK_SPACING,
        dec0=0,
        dec1=0,
    )
    rebuilt = position_from_amounts(
        position.tick_lower,
        position.tick_upper,
        position.amount0(sqrt_price),
        position.amount1(sqrt_price),
        sqrt_price,
        TICK_SPACING,
        dec0=0,
        dec1=0,
    )
    assert rebuilt.liquidity == pytest.approx(position.liquidity, rel=1e-12)


# ---------------------------------------------------------------------------
# Shared fixture (owned by S04, appended to tests/sim/conftest.py)
# ---------------------------------------------------------------------------
def test_entry_position_fixture(entry_position: Position) -> None:
    assert isinstance(entry_position, Position)
    assert entry_position.tick_spacing == TICK_SPACING
    assert entry_position.tick_lower == Tick(int(ENTRY_TICK) - 120)
    assert entry_position.tick_upper == Tick(int(ENTRY_TICK) + 120)
    # Default decimals come from ADR-009.
    assert entry_position.dec0 == DEFAULT_DEC0
    assert entry_position.dec1 == DEFAULT_DEC1

    entry_sqrt = _entry_sqrt()
    entry_price = _entry_price()
    assert entry_position.value(entry_sqrt, entry_price) == pytest.approx(CAPITAL, rel=1e-9)
    assert entry_position.hodl_value(entry_sqrt, entry_price, entry_price) == pytest.approx(
        CAPITAL, rel=1e-9
    )
