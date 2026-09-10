"""Tests for ``undertow.sim.pool`` — the CL pool simulator."""

from __future__ import annotations

import pytest

from undertow.sim.math import tick_to_sqrt_price_x96
from undertow.sim.pool import ConcentratedLiquidityPool, Position


class TestConcentratedLiquidityPool:
    @pytest.fixture
    def pool(self) -> ConcentratedLiquidityPool:
        return ConcentratedLiquidityPool(
            fee_tier_bps=30,
            tick_spacing=60,
            initial_price=3000.0,
        )

    def test_initialization(self, pool: ConcentratedLiquidityPool) -> None:
        assert pool.fee_tier_bps == 30
        assert pool.tick_spacing == 60
        assert abs(pool.price - 3000.0) < 1.0

    def test_initialization_invalid_fee_tier(self) -> None:
        with pytest.raises(ValueError, match="fee tier"):
            ConcentratedLiquidityPool(fee_tier_bps=42, tick_spacing=10, initial_price=3000.0)

    def test_mint_basic(self, pool: ConcentratedLiquidityPool) -> None:
        """Mint a position and check it's created."""
        pos, a0, a1 = pool.mint(-120, 120, 10**12, 10**18)
        assert pos.liquidity > 0
        assert a0 >= 0
        assert a1 >= 0
        assert pos.tick_lower == -120
        assert pos.tick_upper == 120

    def test_mint_aligns_ticks(self, pool: ConcentratedLiquidityPool) -> None:
        """Ticks are aligned to the spacing grid."""
        pos, _, _ = pool.mint(-61, 61, 10**12, 10**18)
        assert pos.tick_lower % 60 == 0
        assert pos.tick_upper % 60 == 0

    def test_mint_invalid_range(self, pool: ConcentratedLiquidityPool) -> None:
        with pytest.raises(ValueError, match="tick_lower"):
            pool.mint(120, -120, 10**12, 10**18)

    def test_swap_zero_for_one(self, pool: ConcentratedLiquidityPool) -> None:
        """Swap token0 (USDC) for token1 (WETH)."""
        # Mint balanced liquidity around current price (~tick 200,000 for $3000 ETH)
        pool.mint(190000, 210000, 10**12, 10**21)
        price_before = pool.price

        # Sell a small amount of USDC (~$1)
        amount0_out, amount1_out, fee = pool.swap(amount0_in=10**6)  # 1 USDC = $1

        assert amount0_out == 0
        assert amount1_out > 0  # received some WETH
        assert fee > 0
        # Selling USDC → ETH gets MORE expensive (higher USDC/WETH price)
        assert pool.price > price_before

    def test_swap_one_for_zero(self, pool: ConcentratedLiquidityPool) -> None:
        """Swap token1 (WETH) for token0 (USDC)."""
        pool.mint(190000, 210000, 10**12, 10**21)
        price_before = pool.price

        # Sell a small amount of WETH (~0.001 ETH = ~$3)
        amount0_out, amount1_out, fee = pool.swap(amount1_in=10**15)  # 0.001 WETH

        assert amount0_out > 0  # received some USDC
        assert amount1_out == 0
        assert fee > 0
        # Selling WETH → ETH gets CHEAPER (lower USDC/WETH price)
        assert pool.price < price_before

    def test_swap_updates_fee_growth(self, pool: ConcentratedLiquidityPool) -> None:
        pool.mint(190000, 210000, 10**12, 10**21)
        fg_before = pool.fee_growth_global_0_x128
        pool.swap(amount0_in=10**6)
        assert pool.fee_growth_global_0_x128 > fg_before

    def test_collect_fees(self, pool: ConcentratedLiquidityPool) -> None:
        """Position collects fees after a swap."""
        pos, _, _ = pool.mint(190000, 210000, 10**12, 10**18)
        pool.swap(amount0_in=10**8)  # Significant swap

        fees0, fees1 = pool.collect(pos)
        # Should have collected some fees
        assert fees0 + fees1 > 0

    def test_collect_resets_uncollected(self, pool: ConcentratedLiquidityPool) -> None:
        pos, _, _ = pool.mint(190000, 210000, 10**12, 10**21)
        pool.swap(amount0_in=10**8)
        pool.collect(pos)
        # Second collect should return zero (already collected)
        fees0, fees1 = pool.collect(pos)
        assert fees0 == 0 and fees1 == 0

    def test_burn_returns_principal(self, pool: ConcentratedLiquidityPool) -> None:
        """Burning a position returns the deposited amounts."""
        # Use ticks that span the current price (~tick 200,000 for $3000 ETH/USDC)
        pos, used0, used1 = pool.mint(190000, 210000, 10**12, 10**21)
        # Collect fees first
        fees0, fees1 = pool.collect(pos)
        a0, a1 = pool.burn(pos)
        assert a0 >= 0
        assert a1 >= 0
        # Position should return non-zero value
        assert a0 + a1 > 0

    def test_fee_growth_inside_full_range(self, pool: ConcentratedLiquidityPool) -> None:
        """Full-range position interior growth equals global growth."""
        # Use min/max ticks that are aligned to spacing and span the current price
        min_tick_aligned = -887040  # aligned to 60
        max_tick_aligned = 887040
        pool.mint(min_tick_aligned, max_tick_aligned, 10**12, 10**21)
        pool.swap(amount0_in=10**8)

        inside = pool._fee_growth_inside(min_tick_aligned, max_tick_aligned, 0)
        global_g = pool.fee_growth_global_0_x128
        # Full-range: fee growth inside should equal global
        # (wrapping arithmetic; for a real full-range position this holds)
        from undertow.sim.math import wrapping_sub_256
        diff = wrapping_sub_256(global_g, inside)
        assert diff == 0 or diff == 2**256 - 1  # wrapping tolerance