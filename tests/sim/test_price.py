"""Tests for ``undertow.sim.price``."""

from __future__ import annotations

import numpy as np
import pytest

from undertow.sim.price import GBMPrice, ReplayPrice, RegimeSwitchingPrice


class TestGBMPrice:
    def test_basic(self) -> None:
        rng = np.random.default_rng(42)
        gbm = GBMPrice(mu=0.0, sigma=0.2, dt=1.0 / 365)
        gbm.reset(rng, initial_price=100.0)
        assert gbm.current() == 100.0

        prices = [gbm.step(rng) for _ in range(100)]
        assert all(p > 0 for p in prices)

    def test_reset(self) -> None:
        rng = np.random.default_rng(42)
        gbm = GBMPrice(mu=0.0, sigma=0.2, dt=1.0 / 365)
        gbm.reset(rng, 200.0)
        assert gbm.current() == 200.0


class TestRegimeSwitchingPrice:
    def test_basic(self) -> None:
        rng = np.random.default_rng(42)
        rs = RegimeSwitchingPrice(
            dt=1.0 / 365,
            transition_scale=30.0,
            jump_intensity=0.0,
        )
        rs.reset(rng, 100.0)
        assert rs.regime in rs.REGIMES

        prices = [rs.step(rng) for _ in range(100)]
        assert all(p > 0 for p in prices)

    def test_regime_changes(self) -> None:
        rng = np.random.default_rng(42)
        rs = RegimeSwitchingPrice(
            dt=1.0 / 365,
            transition_scale=1.0,  # fast transitions
            jump_intensity=0.0,
        )
        rs.reset(rng, 100.0)
        regimes_seen = {rs.regime}
        for _ in range(1000):
            rs.step(rng)
            regimes_seen.add(rs.regime)
        # With fast transitions, should see multiple regimes
        assert len(regimes_seen) > 1

    def test_jumps(self) -> None:
        rng = np.random.default_rng(42)
        rs = RegimeSwitchingPrice(
            dt=1.0 / 365,
            jump_intensity=100.0,  # many jumps expected
            jump_mean=-0.10,
            jump_std=0.05,
        )
        rs.reset(rng, 100.0)
        prices = [rs.step(rng) for _ in range(500)]
        # With so many negative jumps, price should trend down
        # (but not guaranteed — just a sanity check on the code)
        assert all(p > 0 for p in prices)


class TestReplayPrice:
    def test_basic(self) -> None:
        rng = np.random.default_rng(42)
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        rp = ReplayPrice(prices)
        rp.reset(rng, 100.0)
        assert rp.current() == 100.0
        assert rp.step(rng) == 101.0
        assert rp.step(rng) == 102.0
        assert rp.current() == 102.0

    def test_end_of_series(self) -> None:
        rng = np.random.default_rng(42)
        prices = np.array([100.0, 101.0])
        rp = ReplayPrice(prices)
        rp.reset(rng, 100.0)
        rp.step(rng)
        with pytest.raises(StopIteration):
            rp.step(rng)

    def test_invalid_shape(self) -> None:
        with pytest.raises(ValueError):
            ReplayPrice(np.array([100.0]))  # too short