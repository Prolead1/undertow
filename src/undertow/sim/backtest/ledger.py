"""Backtest ledger, summary result and run-metadata helpers (S11).

Implements ``docs/plans/sim/CONTRACTS.md`` §13: the immutable
:class:`BacktestLedger` record, the :class:`BacktestResult` summary and
:func:`summarize_backtest`.

The ledger is a measurement instrument's output — it is never mutated after a run
and it records the provenance (config hash, git commit, policy name, split) that
makes a result reproducible.

Column conventions
------------------
``equity_curve`` has exactly the contract columns ``step, equity, pnl, fees, gas,
slippage, il``:

* ``equity`` is the mark-to-market LP equity in USDC: the position value plus any
  uncollected fees, net of every gas/slippage cost paid so far.
* ``pnl`` is the **cumulative** absolute PnL in USDC, so ``equity == capital +
  pnl`` by construction.
* ``fees``, ``gas``, ``slippage`` and ``il`` are **per-step** terms in USDC with
  costs already signed negative (``gas``/``slippage`` ≤ 0).  ``fees`` is the
  mark-to-market **change** of the accrued fee balance (new accrual plus the
  revaluation of the previously accrued WETH leg), so the equity change
  reconciles step-by-step with ``fees + il + gas + slippage + ΔHODL``; ``il`` is
  the change in the current position's (value − HODL) mark against the **fixed
  initial basket**, so it is ≤ 0 on most steps.
  Summing these four columns is exactly S05's cost-honest decomposition.

The decomposition dict uses the CONTRACTS §13 spelling
(``total_fees/total_gas/total_slippage/total_il/net``); S05's
:func:`~undertow.sim.metrics.decompose.pnl_decomposition` spells the same terms
``total_il_change``/``net_pnl``, so the values are remapped here — the ledger
surface is the frozen one.

``net`` is the S05 cost-honest decomposition (``fees + il + gas + slippage``),
which is the strategy's **excess over the buy-and-hold benchmark** (the initial
deposit's token mix held for the window, PLAN §7).  It therefore equals
``BacktestResult.excess_vs_hodl`` by construction, while
``BacktestResult.total_pnl`` remains the absolute ``equity − capital``; the two
differ by ``hodl_return``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass

import polars as pl

from undertow.sim.metrics import (
    annualized_return,
    max_drawdown,
    pnl_decomposition,
    sharpe_ratio,
)
from undertow.sim.metrics.performance import DEFAULT_PERIODS_PER_YEAR
from undertow.sim.types import Wealth

__all__ = [
    "COST_LEDGER_COLUMNS",
    "DECISION_LOG_COLUMNS",
    "EQUITY_CURVE_COLUMNS",
    "BacktestLedger",
    "BacktestResult",
    "config_hash",
    "git_commit",
    "summarize_backtest",
]

#: Frozen column order of ``BacktestLedger.equity_curve`` (CONTRACTS §13).
EQUITY_CURVE_COLUMNS: tuple[str, ...] = (
    "step",
    "equity",
    "pnl",
    "fees",
    "gas",
    "slippage",
    "il",
)

#: Frozen column order of ``BacktestLedger.decision_log`` (S11 brief §1).
DECISION_LOG_COLUMNS: tuple[str, ...] = (
    "step",
    "action_type",
    "center_offset",
    "width",
    "tick_lower",
    "tick_upper",
    "gas_paid",
    "slippage_paid",
    "fees_collected",
)

#: Frozen column order of ``BacktestLedger.cost_ledger`` (S11 brief §1).
COST_LEDGER_COLUMNS: tuple[str, ...] = ("step", "cost_type", "amount_usdc")


@dataclass(frozen=True, slots=True)
class BacktestLedger:
    """Full immutable record of a backtest run (CONTRACTS §13).

    The ten frozen fields are reproduced exactly.  ``hodl_return``,
    ``initial_capital``, ``periods_per_year`` and ``n_steps`` are **additive,
    defaulted** extensions (PLAN.md §0.2) that let the frozen
    :func:`summarize_backtest(ledger) <summarize_backtest>` compute the HODL
    benchmark and annualize the per-step equity curve without a second argument.
    """

    equity_curve: pl.DataFrame
    decision_log: pl.DataFrame
    cost_ledger: pl.DataFrame
    pnl_decomposition: dict[str, float]
    config_hash: str
    git_commit: str
    policy_name: str
    split: str
    # --- additive extensions (defaulted; do not reorder the frozen fields) ---
    #: PnL of the buy-and-hold benchmark over the same window, in USDC.
    hodl_return: float = 0.0
    #: Capital deployed at t=0; ``equity[0] == initial_capital``.
    initial_capital: float = 0.0
    #: Annualization basis derived from the tape's event spacing.
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR
    #: Number of recorded tape events.
    n_steps: int = 0


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Summary of a backtest run (CONTRACTS §13)."""

    total_pnl: Wealth
    annualized_return: float
    sharpe: float
    max_drawdown: float
    fee_income: float
    gas_paid: float
    slippage_paid: float
    il_realized: float
    hodl_return: float
    excess_vs_hodl: float


def config_hash(config: object) -> str:
    """SHA256 of the JSON-serialised config, truncated to 16 hex chars.

    ``dataclasses.asdict`` flattens the :class:`~undertow.sim.config.SimConfig`
    tree; ``sort_keys`` makes the digest order-independent and ``default=str``
    renders the non-JSON natives (``Path``, ``datetime``, tuples) deterministically.
    """
    payload = asdict(config)  # type: ignore[call-overload]
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def git_commit() -> str:
    """Current git ``HEAD`` SHA, or ``"unknown"`` outside a git checkout.

    Never raises: a backtest run must not fail because the working tree has no
    git metadata (e.g. a wheel-installed test run).
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - env dependent
        return "unknown"
    return out.decode().strip()


def summarize_backtest(ledger: BacktestLedger) -> BacktestResult:
    """Compute the headline metrics of a run from its ledger (CONTRACTS §13).

    Reuses S05's :func:`~undertow.sim.metrics.performance.sharpe_ratio`,
    :func:`~undertow.sim.metrics.performance.annualized_return` and
    :func:`~undertow.sim.metrics.performance.max_drawdown` on the ledger's
    ``equity`` column, annualized on the ledger's own event-spacing basis.

    ``hodl_return`` is the buy-and-hold benchmark PnL recorded by
    :func:`~undertow.sim.backtest.engine.run_backtest`; ``excess_vs_hodl`` is the
    strategy's absolute PnL minus it.  Cost terms are reported with the ledger's
    sign convention (costs negative).
    """
    equity = ledger.equity_curve["equity"]
    periods = ledger.periods_per_year

    total_pnl = float(ledger.equity_curve["pnl"][-1]) if ledger.n_steps else 0.0
    decomposition = ledger.pnl_decomposition
    hodl_return = float(ledger.hodl_return)

    return BacktestResult(
        total_pnl=total_pnl,
        annualized_return=annualized_return(equity, periods),
        sharpe=sharpe_ratio(equity, periods_per_year=periods),
        max_drawdown=max_drawdown(equity),
        fee_income=float(decomposition.get("total_fees", 0.0)),
        gas_paid=float(decomposition.get("total_gas", 0.0)),
        slippage_paid=float(decomposition.get("total_slippage", 0.0)),
        il_realized=float(decomposition.get("total_il", 0.0)),
        hodl_return=hodl_return,
        excess_vs_hodl=total_pnl - hodl_return,
    )


def decompose_equity_curve(equity_curve: pl.DataFrame) -> dict[str, float]:
    """S05 decomposition remapped to the CONTRACTS §13 key spelling.

    S05 returns ``total_il_change`` / ``net_pnl``; the frozen ledger surface names
    the same terms ``total_il`` / ``net``.  Values are identical — only the keys
    are translated so downstream tasks can read either spelling.
    """
    raw = pnl_decomposition(equity_curve)
    return {
        "total_fees": float(raw["total_fees"]),
        "total_gas": float(raw["total_gas"]),
        "total_slippage": float(raw["total_slippage"]),
        "total_il": float(raw["total_il_change"]),
        "net": float(raw["net_pnl"]),
    }
