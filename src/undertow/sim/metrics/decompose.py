"""PnL decomposition for the cost-honesty rule (§10.3.4).

The reported net PnL is the sum of the fee income minus the *costs* (impermanent loss,
gas, slippage)::

    PnL = ΣF − ΣΔIL − Σ(G + S)

The ledger stores each term in USDC with loss/cost terms already signed **negative**
(the convention of ``BacktestLedger.equity_curve`` and ``RewardBreakdown``: the
impermanent-loss term as well as ``gas`` and ``slippage`` are ``≤ 0``), so the identity is
a straight sum::

    net_pnl = Σfees + Σil + Σgas + Σslippage

The impermanent-loss column is named ``il`` on ``BacktestLedger.equity_curve``
(CONTRACTS §13); the task brief's ``il_change`` spelling is accepted as an alias.

Every returned field is explicitly named and the net is included, so a caller can never
mistake gross fee income for net PnL.
"""

from __future__ import annotations

from typing import TypedDict

import polars as pl

#: Canonical impermanent-loss column on ``BacktestLedger.equity_curve`` per CONTRACTS
#: §13 (``step, equity, pnl, fees, gas, slippage, il``).
IL_COLUMN: str = "il"

#: Accepted spellings of the impermanent-loss column, contract name first. The S05 task
#: brief and ``RewardBreakdown`` spell the same term ``il_change``.
IL_COLUMN_ALIASES: tuple[str, ...] = (IL_COLUMN, "il_change")

#: Per-step ledger columns required by :func:`pnl_decomposition`, in USDC. The IL term
#: accepts either :data:`IL_COLUMN` or its ``il_change`` alias.
REQUIRED_COLUMNS: tuple[str, ...] = ("fees", IL_COLUMN, "gas", "slippage")


class PnLDecomposition(TypedDict):
    """Typed view of the decomposition dict.

    Runtime value is a plain ``dict`` (``TypedDict``), so it satisfies the frozen
    ``dict[str, float]`` contract while remaining statically checkable.
    """

    total_fees: float  # ΣF: gross fee income, ≥ 0
    total_il_change: float  # ΣΔIL: impermanent-loss mark, ≤ 0
    total_gas: float  # ΣG: gas cost, ≤ 0
    total_slippage: float  # ΣS: slippage cost, ≤ 0
    net_pnl: float  # ΣF + ΣΔIL + ΣG + ΣS — the cost-honest bottom line


def pnl_decomposition(ledger: pl.DataFrame) -> dict[str, float]:
    """Decompose a per-step ledger into the four PnL terms and their net.

    The return value is a plain ``dict[str, float]`` (frozen CONTRACTS §14) whose keys
    are exactly the fields of :class:`PnLDecomposition`, which is exported as the typed
    view for callers that want static key checking.

    Parameters
    ----------
    ledger:
        A per-step ``polars`` DataFrame containing numeric ``fees``, ``il`` (or the
        ``il_change`` alias), ``gas`` and ``slippage`` columns (USDC).  An empty frame
        yields all-zero terms.

    Raises
    ------
    KeyError
        If any required column is missing, naming the absent column(s).
    """
    missing = [col for col in ("fees", "gas", "slippage") if col not in ledger.columns]
    # Resolve the impermanent-loss column once: CONTRACTS §13 names the ledger column
    # ``il`` while the S05 brief (and ``RewardBreakdown``) spell it ``il_change``.  Prefer
    # the contract spelling and fall back to the alias, tolerating either.
    il_column = next((name for name in IL_COLUMN_ALIASES if name in ledger.columns), None)
    if il_column is None:
        missing.append(f"{IL_COLUMN} (or il_change)")
    if missing:
        raise KeyError("ledger is missing required PnL column(s): " + ", ".join(missing))

    total_fees = float(ledger["fees"].sum())
    total_il_change = float(ledger[il_column].sum())
    total_gas = float(ledger["gas"].sum())
    total_slippage = float(ledger["slippage"].sum())
    net_pnl = total_fees + total_il_change + total_gas + total_slippage

    decomposition: dict[str, float] = {
        "total_fees": total_fees,
        "total_il_change": total_il_change,
        "total_gas": total_gas,
        "total_slippage": total_slippage,
        "net_pnl": net_pnl,
    }
    return decomposition


__all__ = [
    "IL_COLUMN",
    "IL_COLUMN_ALIASES",
    "REQUIRED_COLUMNS",
    "PnLDecomposition",
    "pnl_decomposition",
]
