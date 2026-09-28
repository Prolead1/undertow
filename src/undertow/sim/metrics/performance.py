"""Risk/return performance metrics for ``undertow.sim``.

Implements equations (7)–(10) of the roadmap §10.4, exposed as the frozen
signatures in ``docs/plans/sim/CONTRACTS.md`` §14.

Units and cadence
-----------------
The simulator's decision cadence is ``EpisodeConfig.step_minutes`` (10 minutes by
default), so a year of simulated time contains ``periods_per_year`` steps.  Every
annualizing metric takes ``periods_per_year`` as a keyword argument with the documented
default of :data:`DEFAULT_PERIODS_PER_YEAR` = ``365 * 24 * 6`` (i.e. ten-minute steps).
Use :func:`periods_per_year_from_step_minutes` to derive it from a config cadence rather
than writing magic numbers.  ``risk_free_rate`` is a *nominal annual* rate, de-annualized
to a per-step rate by dividing by ``periods_per_year`` (simple division, not
continuous compounding) before subtraction.

Return / loss conventions
-------------------------
* Period returns are simple returns from ``equity_curve.pct_change()``; the first
  element is null and is dropped.
* Non-finite signals (NaN/inf returns) are discarded.
* Drawdowns and VaR/CVaR are reported as *negative* numbers by convention: a value of
  ``-0.25`` means a 25% loss.
* Empty and single-observation inputs are well defined: they return ``0.0`` (there is
  no dispersion or path to annualize) instead of raising.

This module is pure numpy/polars; it has no dependency on the other sim subpackages.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl

#: Number of 10-minute decision steps in a non-leap year: 365 days × 24 h × 6.
DEFAULT_PERIODS_PER_YEAR: Final[int] = 365 * 24 * 6

#: Minutes in a non-leap year, used to derive an annualization factor from a cadence.
MINUTES_PER_YEAR: Final[int] = 365 * 24 * 60


def periods_per_year_from_step_minutes(step_minutes: int) -> int:
    """Annualization factor for a decision cadence of ``step_minutes`` minutes.

    Uses a 365-day year (the simulator calendar; leap days are ignored).  Raises
    ``ValueError`` for a non-positive cadence.

    For the pinned 10-minute cadence this returns :data:`DEFAULT_PERIODS_PER_YEAR`.
    """
    if step_minutes <= 0:
        raise ValueError("step_minutes must be positive")
    return int(round(MINUTES_PER_YEAR / step_minutes))


def _period_returns(equity_curve: pl.Series) -> np.ndarray:
    """Simple period returns as a finite float64 array; empty if fewer than 2 rows."""
    if equity_curve.len() < 2:
        return np.empty(0, dtype=np.float64)
    rets = equity_curve.pct_change().drop_nulls().to_numpy().astype(np.float64)
    return rets[np.isfinite(rets)]


def _deannualized(risk_free_rate: float, periods_per_year: int) -> float:
    return risk_free_rate / periods_per_year


def sharpe_ratio(
    equity_curve: pl.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio.

    ``mean(r − rf_step) / std(r, ddof=1) * sqrt(periods_per_year)`` where ``r`` are
    period returns and ``rf_step = risk_free_rate / periods_per_year`` (the nominal
    annual rate is de-annualized by simple division; it is not continuously compounded).

    Returns ``0.0`` when there are fewer than two returns or the standard deviation is
    zero or non-finite (a flat curve has an undefined, not infinite, Sharpe).
    """
    rets = _period_returns(equity_curve)
    if rets.size < 2:
        return 0.0
    excess = rets - _deannualized(risk_free_rate, periods_per_year)
    std = float(np.std(excess, ddof=1))
    if not np.isfinite(std) or std == 0.0:
        return 0.0
    mean = float(np.mean(excess))
    return float(mean / std * np.sqrt(periods_per_year))


def sortino_ratio(
    equity_curve: pl.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Annualized Sortino ratio.

    Same numerator as :func:`sharpe_ratio`; the denominator is the downside deviation
    about the zero target::

        dd = sqrt( mean( min(excess, 0) ** 2 ) )

    i.e. the root-mean-square of the negative excess returns over *all* periods
    (positive excesses contribute zero).  This is the standard semi-deviation
    denominator; it differs from ``np.std(negative_excess)`` by measuring dispersion
    *below the target* rather than around the tail's own mean, which is what makes the
    ratio penalize a fat left tail relative to Sharpe.

    Returns ``0.0`` when there are fewer than two returns or when no period has a
    negative excess (zero downside deviation).

    Definition reconciliation (CONTRACTS §14): the contract says Sortino "uses only
    downside deviation in denominator", which this RMS semi-deviation satisfies; the
    frozen spec text is updated in ``docs/plans/sim/tasks/S05_metrics.md`` to state the
    denominator explicitly so it no longer reads as the std of the negative subset.
    """
    rets = _period_returns(equity_curve)
    if rets.size < 2:
        return 0.0
    excess = rets - _deannualized(risk_free_rate, periods_per_year)
    downside = np.minimum(excess, 0.0)
    dd = float(np.sqrt(np.mean(downside**2)))
    if not np.isfinite(dd) or dd == 0.0:
        return 0.0
    mean = float(np.mean(excess))
    return float(mean / dd * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pl.Series) -> float:
    """Maximum peak-to-trough decline as a negative fraction.

    Tracks the running maximum and returns ``min((equity − running_max) / running_max)``.
    A monotonic-up or flat curve returns ``0.0``.  Empty input returns ``0.0``.  Steps
    where the running maximum is zero (a wiped-out or unsigned equity path) contribute
    no drawdown rather than dividing by zero.
    """
    if equity_curve.len() == 0:
        return 0.0
    arr = equity_curve.to_numpy().astype(np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(arr)
    denom = np.where(running_max > 0.0, running_max, 1.0)
    drawdowns = (arr - running_max) / denom
    return float(np.min(drawdowns))


def value_at_risk(returns: pl.Series, confidence: float = 0.95) -> float:
    """Historical Value-at-Risk at ``confidence`` as a negative loss number.

    The empirical ``1 − confidence`` quantile of the return distribution.  For a
    standard-normal sample at 95% this is approximately ``−1.645``.

    Returns ``0.0`` for an empty sample.  ``confidence`` must lie in ``(0, 1)``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    arr = _finite_returns(returns)
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, 1.0 - confidence))


def cvar(returns: pl.Series, confidence: float = 0.95) -> float:
    """Conditional VaR (expected shortfall): mean of returns at/below the VaR.

    Always ``<= value_at_risk`` for non-empty input.  Empty input returns ``0.0``.
    If the interpolated VaR falls below every observation (possible only for tiny
    samples), the VaR itself is returned so the ordering invariant still holds.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    arr = _finite_returns(returns)
    if arr.size == 0:
        return 0.0
    var = value_at_risk(returns, confidence)
    tail = arr[arr <= var]
    if tail.size == 0:
        return float(var)
    return float(np.mean(tail))


def _finite_returns(returns: pl.Series) -> np.ndarray:
    arr = returns.drop_nulls().to_numpy().astype(np.float64)
    return arr[np.isfinite(arr)]


def annualized_return(
    equity_curve: pl.Series,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Annualized simple return from the equity curve.

    ``(equity[-1] / equity[0]) ** (periods_per_year / n_periods) − 1`` where
    ``n_periods = len(equity_curve) − 1`` (an ``n``-point curve spans ``n − 1``
    returns).  Returns ``0.0`` for fewer than two points or a non-positive starting
    equity, and ``−1.0`` when the ending equity is non-positive (total loss).
    """
    if equity_curve.len() < 2:
        return 0.0
    arr = equity_curve.to_numpy().astype(np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    start = float(arr[0])
    end = float(arr[-1])
    if start <= 0.0:
        return 0.0
    if end <= 0.0:
        return -1.0
    n_periods = arr.size - 1
    try:
        growth = (end / start) ** (periods_per_year / n_periods)
    except OverflowError:
        # Annualizing a large short-horizon gain can exceed float range; report the
        # mathematically correct limit rather than raising.
        return float("inf")
    return float(growth - 1.0)


def all_metrics(
    equity_curve: pl.Series,
    returns: pl.Series,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    *,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Convenience wrapper returning all six headline metrics.

    Keys: ``sharpe``, ``sortino``, ``maxdd``, ``var_95``, ``cvar_95``,
    ``annualized_return``.  ``returns`` is the per-step return series (used for VaR and
    CVaR) and is passed through unchanged so callers can supply their own definition.
    """
    return {
        "sharpe": sharpe_ratio(equity_curve, risk_free_rate, periods_per_year),
        "sortino": sortino_ratio(equity_curve, risk_free_rate, periods_per_year),
        "maxdd": max_drawdown(equity_curve),
        "var_95": value_at_risk(returns, 0.95),
        "cvar_95": cvar(returns, 0.95),
        "annualized_return": annualized_return(equity_curve, periods_per_year),
    }


# Explicit re-export list for the subpackage ``__init__``.
__all__ = [
    "DEFAULT_PERIODS_PER_YEAR",
    "MINUTES_PER_YEAR",
    "all_metrics",
    "annualized_return",
    "cvar",
    "max_drawdown",
    "periods_per_year_from_step_minutes",
    "sharpe_ratio",
    "sortino_ratio",
    "value_at_risk",
]
