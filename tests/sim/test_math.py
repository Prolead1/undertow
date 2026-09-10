"""Tests for ``undertow.sim.math`` — fixed-point and tick math."""

from __future__ import annotations

from decimal import Decimal

import pytest

from undertow.sim.math import (
    MAX_TICK,
    MIN_SQRT_RATIO,
    MIN_TICK,
    Q96,
    Q128,
    MAX_SQRT_RATIO,
    align_tick_down,
    align_tick_up,
    amounts_for_liquidity,
    get_amount0_delta,
    get_amount1_delta,
    liquidity_for_amounts,
    price_to_sqrt_price_x96,
    price_to_tick,
    sqrt_price_x96_to_price,
    sqrt_price_x96_to_tick,
    tick_to_price,
    tick_to_sqrt_price_x96,
    wrapping_add_256,
    wrapping_sub_256,
)


# -------------------------------------------------------------------
# Price conversion
# -------------------------------------------------------------------
class TestPriceConversion:
    def test_sqrt_price_x96_to_price_worked_example(self) -> None:
        """CONTRACTS §3.0 worked check: ETH=$3000 exactly."""
        sqrt_price_x96 = 1446501726624926496477173928747177
        result = sqrt_price_x96_to_price(sqrt_price_x96, dec0=6, dec1=18)
        assert abs(float(result) - 3000.0) < 0.01

    def test_sqrt_price_x96_to_price_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            sqrt_price_x96_to_price(0, 6, 18)

    def test_price_round_trip(self) -> None:
        """price_to_sqrt_price_x96 inverts sqrt_price_x96_to_price within 1 unit."""
        for price in (100.0, 1000.0, 3000.0, 10000.0):
            p = Decimal(str(price))
            s = price_to_sqrt_price_x96(p, 6, 18)
            recovered = sqrt_price_x96_to_price(s, 6, 18)
            assert abs(float(recovered) - price) / price < 1e-15

    def test_tick_to_price_direction(self) -> None:
        """Higher tick → LOWER USDC-per-WETH price."""
        p_low = tick_to_price(200000, 6, 18)
        p_high = tick_to_price(190000, 6, 18)
        assert float(p_low) < float(p_high)


# -------------------------------------------------------------------
# Tick ↔ sqrt price
# -------------------------------------------------------------------
class TestTickSqrtPrice:
    def test_tick_round_trip(self) -> None:
        """sqrt_price_x96_to_tick(tick_to_sqrt_price_x96(t)) == t."""
        for t in (
            MIN_TICK,
            MIN_TICK + 1,
            -887000,
            -100000,
            -60,
            -1,
            0,
            1,
            60,
            100000,
            887000,
            MAX_TICK - 1,  # MAX_TICK itself: sqrt=MAX_SQRT_RATIO, not round-trippable
        ):
            s = tick_to_sqrt_price_x96(t)
            recovered = sqrt_price_x96_to_tick(s)
            assert recovered == t, f"round-trip failed for tick {t}"

    def test_min_max_sqrt_ratios(self) -> None:
        """MIN_TICK maps to MIN_SQRT_RATIO, MAX_TICK to MAX_SQRT_RATIO."""
        assert tick_to_sqrt_price_x96(MIN_TICK) == MIN_SQRT_RATIO
        assert tick_to_sqrt_price_x96(MAX_TICK) == MAX_SQRT_RATIO
        assert sqrt_price_x96_to_tick(MIN_SQRT_RATIO) == MIN_TICK
        # MAX_SQRT_RATIO maps to MAX_TICK
        assert sqrt_price_x96_to_tick(MAX_SQRT_RATIO) == MAX_TICK

    def test_tick_out_of_bounds_raises(self) -> None:
        with pytest.raises(ValueError):
            tick_to_sqrt_price_x96(MIN_TICK - 1)
        with pytest.raises(ValueError):
            tick_to_sqrt_price_x96(MAX_TICK + 1)

    def test_sqrt_price_out_of_bounds_raises(self) -> None:
        with pytest.raises(ValueError):
            sqrt_price_x96_to_tick(MIN_SQRT_RATIO - 1)
        with pytest.raises(ValueError):
            sqrt_price_x96_to_tick(MAX_SQRT_RATIO + 1)


# -------------------------------------------------------------------
# Tick alignment
# -------------------------------------------------------------------
class TestTickAlignment:
    def test_align_tick_down(self) -> None:
        assert align_tick_down(0, 60) == 0
        assert align_tick_down(61, 60) == 60
        assert align_tick_down(-61, 60) == -120
        assert align_tick_down(-1, 60) == -60
        assert align_tick_down(887000, 60) == 886980

    def test_align_tick_up(self) -> None:
        assert align_tick_up(0, 60) == 0
        assert align_tick_up(61, 60) == 120
        assert align_tick_up(-61, 60) == -60
        assert align_tick_up(-1, 60) == 0

    def test_aligned_is_idempotent(self) -> None:
        for tick in (-887040, -180, -120, -60, 0, 60, 120, 886980):
            assert align_tick_down(tick, 60) == tick
            assert align_tick_up(tick, 60) == tick


# -------------------------------------------------------------------
# Wrapping arithmetic
# -------------------------------------------------------------------
class TestWrapping:
    def test_wrapping_sub_basic(self) -> None:
        assert wrapping_sub_256(10, 3) == 7
        assert wrapping_sub_256(0, 1) == 2**256 - 1
        assert wrapping_sub_256(0, 2) == 2**256 - 2

    def test_wrapping_add_basic(self) -> None:
        assert wrapping_add_256(10, 3) == 13
        assert wrapping_add_256(2**256 - 1, 1) == 0
        assert wrapping_add_256(2**256 - 1, 2) == 1


# -------------------------------------------------------------------
# Liquidity ↔ amounts
# -------------------------------------------------------------------
class TestLiquidityAmounts:
    def test_get_amount0_delta_round_down(self) -> None:
        """Basic sanity: delta between two prices."""
        sa = tick_to_sqrt_price_x96(0)
        sb = tick_to_sqrt_price_x96(100)
        liq = 10**18
        a0 = get_amount0_delta(sa, sb, liq, round_up=False)
        assert a0 > 0

    def test_get_amount1_delta_round_down(self) -> None:
        sa = tick_to_sqrt_price_x96(0)
        sb = tick_to_sqrt_price_x96(100)
        liq = 10**18
        a1 = get_amount1_delta(sa, sb, liq, round_up=False)
        assert a1 > 0

    def test_amounts_for_liquidity_price_below_lower(self) -> None:
        """Price below lower bound → only amount0."""
        s_low = tick_to_sqrt_price_x96(-100)
        s_high = tick_to_sqrt_price_x96(100)
        s_price = tick_to_sqrt_price_x96(-200)  # below
        a0, a1 = amounts_for_liquidity(s_price, s_low, s_high, 10**18)
        assert a0 > 0
        assert a1 == 0

    def test_amounts_for_liquidity_price_above_upper(self) -> None:
        """Price above upper bound → only amount1."""
        s_low = tick_to_sqrt_price_x96(-100)
        s_high = tick_to_sqrt_price_x96(100)
        s_price = tick_to_sqrt_price_x96(200)  # above
        a0, a1 = amounts_for_liquidity(s_price, s_low, s_high, 10**18)
        assert a0 == 0
        assert a1 > 0

    def test_amounts_for_liquidity_price_inside(self) -> None:
        s_low = tick_to_sqrt_price_x96(-100)
        s_high = tick_to_sqrt_price_x96(100)
        s_price = tick_to_sqrt_price_x96(0)
        a0, a1 = amounts_for_liquidity(s_price, s_low, s_high, 10**18)
        assert a0 > 0
        assert a1 > 0

    def test_liquidity_for_amounts_round_trip(self) -> None:
        """liquidity_for_amounts then amounts_for_liquidity recovers amounts."""
        s_low = tick_to_sqrt_price_x96(-200)
        s_high = tick_to_sqrt_price_x96(200)
        s_price = tick_to_sqrt_price_x96(0)
        amt0 = 10**12
        amt1 = 10**18
        liq = liquidity_for_amounts(s_price, s_low, s_high, amt0, amt1)
        a0, a1 = amounts_for_liquidity(s_price, s_low, s_high, liq)
        assert a0 <= amt0
        assert a1 <= amt1
        assert a0 > 0
        assert a1 > 0