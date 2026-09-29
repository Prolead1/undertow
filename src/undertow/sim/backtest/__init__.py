"""Frozen-policy event-replay backtester (S11, CONTRACTS.md §13).

Public surface:

* :func:`~undertow.sim.backtest.engine.run_backtest` — replay a frozen policy over
  a :class:`~undertow.sim.marketview.MarketView` event tape.
* :func:`~undertow.sim.backtest.ledger.summarize_backtest` — headline metrics from
  a run's ledger.
* :class:`~undertow.sim.backtest.ledger.BacktestLedger` — the immutable run
  record (equity curve, decision log, cost ledger, decomposition, provenance).
* :class:`~undertow.sim.backtest.ledger.BacktestResult` — the summary result.

The package deliberately imports ``undertow.data`` only through the top-level
public API (PLAN.md §4/§0.7).
"""

from __future__ import annotations

from undertow.sim.backtest.engine import (
    DEFAULT_INITIAL_HALF_WIDTH_TICKS,
    BacktestObservation,
    run_backtest,
)
from undertow.sim.backtest.ledger import (
    COST_LEDGER_COLUMNS,
    DECISION_LOG_COLUMNS,
    EQUITY_CURVE_COLUMNS,
    BacktestLedger,
    BacktestResult,
    summarize_backtest,
)

__all__ = [
    "COST_LEDGER_COLUMNS",
    "DECISION_LOG_COLUMNS",
    "DEFAULT_INITIAL_HALF_WIDTH_TICKS",
    "EQUITY_CURVE_COLUMNS",
    "BacktestLedger",
    "BacktestObservation",
    "BacktestResult",
    "run_backtest",
    "summarize_backtest",
]
