"""Performance metrics, PnL decomposition and per-regime slicing (S05).

Public surface re-exported from the three modules:

* :mod:`undertow.sim.metrics.performance` — eqs (7)–(10): Sharpe, Sortino, MaxDD, VaR,
  CVaR, annualized return, :func:`all_metrics`.
* :mod:`undertow.sim.metrics.decompose` — the cost-honest PnL decomposition.
* :mod:`undertow.sim.metrics.regimes` — per-regime metric slicing.
"""

from __future__ import annotations

from undertow.sim.metrics.decompose import (
    IL_COLUMN,
    IL_COLUMN_ALIASES,
    REQUIRED_COLUMNS,
    PnLDecomposition,
    pnl_decomposition,
)
from undertow.sim.metrics.performance import (
    DEFAULT_PERIODS_PER_YEAR,
    MINUTES_PER_YEAR,
    all_metrics,
    annualized_return,
    cvar,
    max_drawdown,
    periods_per_year_from_step_minutes,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)
from undertow.sim.metrics.regimes import (
    DEFAULT_EQUITY_COLUMN,
    MetricFn,
    per_regime_metrics,
)

__all__ = [
    "DEFAULT_EQUITY_COLUMN",
    "DEFAULT_PERIODS_PER_YEAR",
    "IL_COLUMN",
    "IL_COLUMN_ALIASES",
    "MINUTES_PER_YEAR",
    "REQUIRED_COLUMNS",
    "MetricFn",
    "PnLDecomposition",
    "all_metrics",
    "annualized_return",
    "cvar",
    "max_drawdown",
    "per_regime_metrics",
    "periods_per_year_from_step_minutes",
    "pnl_decomposition",
    "sharpe_ratio",
    "sortino_ratio",
    "value_at_risk",
]
