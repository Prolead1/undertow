"""Tests for ``undertow.sim.metrics``."""

from __future__ import annotations

import numpy as np

from undertow.sim.metrics import (
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    var_cvar,
)


def _make_random_walk(
    n: int = 252, seed: int = 42, mu: float = 0.0002, sigma: float = 0.01
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = rng.normal(mu, sigma, n)
    return 1000.0 * np.exp(np.cumsum(returns))


class TestSharpeRatio:
    def test_positive_trend(self) -> None:
        equity = _make_random_walk(seed=0)
        s = sharpe_ratio(equity, periods_per_year=252)
        assert s > 0.0

    def test_flat(self) -> None:
        equity = np.full(100, 100.0)
        assert sharpe_ratio(equity, periods_per_year=252) == 0.0

    def test_short_series(self) -> None:
        assert sharpe_ratio(np.array([100.0]), periods_per_year=252) == 0.0


class TestSortinoRatio:
    def test_positive_trend(self) -> None:
        equity = _make_random_walk(seed=0)
        s = sortino_ratio(equity, periods_per_year=252)
        assert isinstance(s, float)
        assert np.isfinite(s)


class TestMaxDrawdown:
    def test_basic(self) -> None:
        equity = np.array([100.0, 90.0, 95.0, 85.0, 100.0])
        mdd = max_drawdown(equity)
        assert mdd < 0.0  # negative means drawdown
        assert abs(mdd - (-0.15)) < 0.01  # 100→85 is 15% drawdown

    def test_no_drawdown(self) -> None:
        equity = np.array([100.0, 110.0, 120.0])
        assert max_drawdown(equity) == 0.0

    def test_empty(self) -> None:
        assert max_drawdown(np.array([])) == 0.0


class TestVaRCVaR:
    def test_basic(self) -> None:
        rng = np.random.default_rng(0)
        pnl = rng.normal(0.0, 100.0, 1000)
        var, cvar = var_cvar(pnl, alpha=0.05)
        assert var < 0.0  # VaR should be negative (loss)
        assert cvar <= var  # CVaR ≤ VaR (more extreme)


class TestAnnualized:
    def test_return_zero(self) -> None:
        equity = np.full(100, 100.0)
        assert annualized_return(equity) == 0.0

    def test_return_positive(self) -> None:
        equity = np.array([100.0, 110.0, 121.0])  # +10% each step
        ar = annualized_return(equity, periods_per_year=365)
        assert ar > 0.0

    def test_volatility(self) -> None:
        equity = _make_random_walk()
        av = annualized_volatility(equity, periods_per_year=252)
        assert av > 0.0