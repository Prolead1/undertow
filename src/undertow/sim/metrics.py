"""Performance metrics for ``undertow.sim``.

Sharpe, Sortino, max drawdown, VaR/CVaR, and PnL decomposition.
All functions operate on numpy arrays of equity values.
"""

from __future__ import annotations

import numpy as np


def sharpe_ratio(
    equity: np.ndarray,
    periods_per_year: int = 365,
    rf: float = 0.0,
) -> float:
    """Annualized Sharpe ratio from an equity curve.

    Parameters
    ----------
    equity: np.ndarray
        Equity curve (cumulative portfolio value at each step).
    periods_per_year: int
        Number of steps per year (e.g., 365 for daily, 525600 for minutely).
    rf: float
        Annual risk-free rate.

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    if len(equity) < 2:
        return 0.0
    r = np.diff(equity) / np.maximum(equity[:-1], 1e-10)
    excess = r - rf / periods_per_year
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def sortino_ratio(
    equity: np.ndarray,
    periods_per_year: int = 365,
    rf: float = 0.0,
) -> float:
    """Annualized Sortino ratio (downside deviation only).

    Parameters
    ----------
    equity: np.ndarray
        Equity curve.
    periods_per_year: int
        Number of steps per year.
    rf: float
        Annual risk-free rate.

    Returns
    -------
    float
        Annualized Sortino ratio.
    """
    if len(equity) < 2:
        return 0.0
    r = np.diff(equity) / np.maximum(equity[:-1], 1e-10)
    excess = r - rf / periods_per_year
    downside = np.minimum(excess, 0.0)
    sigma_down = np.sqrt((downside**2).sum() / max(len(downside) - 1, 1))
    if sigma_down == 0:
        return 0.0
    return float(excess.mean() / sigma_down * np.sqrt(periods_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    """Maximum drawdown from an equity curve.

    Returns a negative number representing the deepest peak-to-trough decline
    as a fraction (e.g., -0.25 for a 25% drawdown).
    """
    if len(equity) == 0:
        return 0.0
    running_peak = np.maximum.accumulate(equity)
    drawdowns = equity / running_peak - 1.0
    return float(drawdowns.min())


def var_cvar(
    pnl: np.ndarray, alpha: float = 0.05
) -> tuple[float, float]:
    """Value-at-Risk and Conditional VaR from a PnL distribution.

    Parameters
    ----------
    pnl: np.ndarray
        Array of PnL values (positive = profit).
    alpha: float
        Tail quantile (0.05 = 5th percentile).

    Returns
    -------
    tuple[float, float]
        (VaR, CVaR) — negative means loss.
    """
    if len(pnl) == 0:
        return 0.0, 0.0
    var = float(np.quantile(pnl, alpha))
    tail = pnl[pnl <= var]
    cvar = float(tail.mean()) if len(tail) > 0 else var
    return var, cvar


def decomposition(
    pnl_history: dict,
) -> dict:
    """PnL decomposition from tracked components.

    Parameters
    ----------
    pnl_history: dict
        Dictionary with keys: 'fees', 'gas', 'il', 'slippage', 'total_pnl'.

    Returns
    -------
    dict
        Same keys with summed values.
    """
    return {k: float(np.sum(v)) for k, v in pnl_history.items()}


def annualized_return(equity: np.ndarray, periods_per_year: int = 365) -> float:
    """Annualized return from an equity curve."""
    if len(equity) < 2:
        return 0.0
    total_return = equity[-1] / equity[0] - 1.0
    years = (len(equity) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def annualized_volatility(
    equity: np.ndarray, periods_per_year: int = 365
) -> float:
    """Annualized volatility from an equity curve."""
    if len(equity) < 2:
        return 0.0
    r = np.diff(equity) / np.maximum(equity[:-1], 1e-10)
    return float(r.std() * np.sqrt(periods_per_year))