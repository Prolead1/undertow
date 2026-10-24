"""Short-perp hedge overlay (S20, ADR-030 C3).

The hedge instrument the RL agent controls in S20b is a **short ETH-perp
position** whose size is a fraction (the hedge ratio) of the LP position's
current ETH delta.  This module owns only the accounting: mark PnL, the perp
trade fee, funding and equity.  It is deliberately pure — no market data, no
look-ahead, no frictions beyond the configured ``fee_bps``.  Funding is supplied
by the caller from the dataset's ``stream=funding`` rates; the *mark* used for
mark-to-market is the simulator's own reference price, **not** the funding
stream's nullable ``mark_price`` (2005/3288 rows are null in 2022–23 — see
:meth:`~undertow.sim.marketview.MarketView.funding_at`).

Sign conventions (all USDC unless stated):

* A short **gains** when the mark falls: ``mark_pnl = short_size * (entry - mark)``.
* ``trade_to`` charges ``|target - short_size| * mark * fee_bps / 1e4`` and
  returns it as a **positive magnitude**; the account stores it in
  ``fees_paid`` (a cost).
* ``accrue_funding`` returns ``short_size * mark * rate`` **signed**: a short
  *receives* funding when the rate is positive (the typical case on Binance
  ETHUSDT, S19: +84 % of periods).
* ``equity(mark) = realized_pnl + mark_pnl(mark) + funding_pnl - fees_paid``.

The account is mutable (the env and backtester trade it in place); a frozen
:class:`HedgeState` snapshot is available for logging, and :class:`HedgeTrade`
records the last executed trade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from undertow.sim.types import UndertowSimError

__all__ = [
    "DEFAULT_HEDGE_FEE_BPS",
    "ETH_DELTA_EPS",
    "HedgeAccount",
    "HedgeError",
    "HedgeState",
    "HedgeTrade",
]

#: Default perp taker fee, in true basis points (S19 measured 5 bps).
DEFAULT_HEDGE_FEE_BPS: float = 5.0

#: ETH-delta floor for the hedge-ratio observation feature (a flat LP has no
#: delta to hedge against; the feature is defined as 0 rather than dividing by
#: zero).  Used by both the environment and the backtester.
ETH_DELTA_EPS: float = 1e-12


class HedgeError(UndertowSimError):
    """Invalid hedge operation or account parameter."""


def _as_finite(value: float, name: str) -> float:
    """Coerce ``value`` to ``float`` and reject NaN/Inf with a named error."""
    number = float(value)
    if not math.isfinite(number):
        raise HedgeError(f"{name} must be finite, got {value!r}")
    return number


@dataclass(frozen=True, slots=True)
class HedgeState:
    """Immutable snapshot of a :class:`HedgeAccount`'s book.

    ``short_size`` is the current short in ETH (``>= 0``); ``entry_mark`` the
    mark the short was opened/wound to (USDC/ETH, ``>= 0``); ``realized_pnl``
    and ``funding_pnl`` are signed USDC; ``fees_paid`` is a non-negative USDC
    cost magnitude.
    """

    short_size: float = 0.0
    entry_mark: float = 0.0
    realized_pnl: float = 0.0
    funding_pnl: float = 0.0
    fees_paid: float = 0.0

    def __post_init__(self) -> None:
        for name in ("short_size", "entry_mark", "realized_pnl", "funding_pnl", "fees_paid"):
            _as_finite(getattr(self, name), name)
        if self.short_size < 0.0:
            raise HedgeError(f"short_size must be >= 0, got {self.short_size!r}")
        if self.entry_mark < 0.0:
            raise HedgeError(f"entry_mark must be >= 0, got {self.entry_mark!r}")
        if self.fees_paid < 0.0:
            raise HedgeError(f"fees_paid must be >= 0, got {self.fees_paid!r}")


@dataclass(frozen=True, slots=True)
class HedgeTrade:
    """One executed perp trade (the last trade is kept on the account).

    ``traded_size`` is the signed change in short size (ETH, positive = more
    short); ``short_size`` the post-trade size; ``mark`` the execution mark;
    ``fee`` the positive USDC magnitude charged; ``realized_pnl`` the mark PnL
    the trade crystallised at that mark.
    """

    traded_size: float
    short_size: float
    mark: float
    fee: float
    realized_pnl: float


@dataclass(slots=True)
class HedgeAccount:
    """Mutable short-perp account for the deterministic delta overlay.

    Parameters
    ----------
    fee_bps:
        Perp taker fee in true basis points (S19 pins 5 bps).
    funding_enabled:
        When ``False``, :meth:`accrue_funding` is a no-op (RQ1 ablation).  The
        rate is still validated.
    """

    fee_bps: float = DEFAULT_HEDGE_FEE_BPS
    funding_enabled: bool = True
    short_size: float = 0.0
    entry_mark: float = 0.0
    realized_pnl: float = 0.0
    funding_pnl: float = 0.0
    fees_paid: float = 0.0
    #: The most recent trade executed through :meth:`trade_to` (``None`` before
    #: the first trade).  Excluded from equality so two accounts with equal
    #: books compare equal.
    last_trade: HedgeTrade | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        self.fee_bps = _as_finite(self.fee_bps, "fee_bps")
        if self.fee_bps < 0.0:
            raise HedgeError(f"fee_bps must be >= 0, got {self.fee_bps!r}")
        state = HedgeState(
            short_size=self.short_size,
            entry_mark=self.entry_mark,
            realized_pnl=self.realized_pnl,
            funding_pnl=self.funding_pnl,
            fees_paid=self.fees_paid,
        )
        self.short_size = state.short_size
        self.entry_mark = state.entry_mark
        self.realized_pnl = state.realized_pnl
        self.funding_pnl = state.funding_pnl
        self.fees_paid = state.fees_paid

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def state(self) -> HedgeState:
        """A frozen snapshot of the account's book."""
        return HedgeState(
            short_size=self.short_size,
            entry_mark=self.entry_mark,
            realized_pnl=self.realized_pnl,
            funding_pnl=self.funding_pnl,
            fees_paid=self.fees_paid,
        )

    def reset(self) -> None:
        """Flatten the account to its zero state (keeps ``fee_bps``/funding)."""
        self.short_size = 0.0
        self.entry_mark = 0.0
        self.realized_pnl = 0.0
        self.funding_pnl = 0.0
        self.fees_paid = 0.0
        self.last_trade = None

    # ------------------------------------------------------------------
    # Accounting
    # ------------------------------------------------------------------
    def mark_pnl(self, mark: float) -> float:
        """Unrealised PnL of the open short at ``mark`` (signed USDC).

        A short gains when the mark falls: ``short_size * (entry_mark - mark)``.
        """
        price = self._check_mark(mark)
        return self.short_size * (self.entry_mark - price)

    def trade_to(self, target_short: float, mark: float) -> float:
        """Re-size the short to ``target_short`` at ``mark``; return the fee.

        The fee is ``|Δ| * mark * fee_bps / 1e4`` as a **positive magnitude**.
        The trade realises the current mark PnL into ``realized_pnl``, resets
        ``entry_mark`` to ``mark`` and sets ``short_size`` to the target.  A
        zero target flattens the short (valid); a negative target raises.
        """
        price = self._check_mark(mark)
        target = _as_finite(target_short, "target_short")
        if target < 0.0:
            raise HedgeError(f"target_short must be >= 0, got {target_short!r}")
        realized = self.mark_pnl(price)
        delta = target - self.short_size
        fee = abs(delta) * price * self.fee_bps / 1e4
        self.realized_pnl += realized
        self.fees_paid += fee
        self.entry_mark = price
        self.short_size = target
        self.last_trade = HedgeTrade(
            traded_size=delta,
            short_size=target,
            mark=price,
            fee=fee,
            realized_pnl=realized,
        )
        return fee

    def accrue_funding(self, rate: float, mark: float) -> float:
        """Accrue one settlement; return the signed USDC received.

        ``short_size * mark * rate``: a positive rate pays a short (income), a
        negative rate charges it.  Returns ``0.0`` (without touching the book)
        when funding is disabled or the short is flat.
        """
        price = self._check_mark(mark)
        per_period = _as_finite(rate, "rate")
        if not self.funding_enabled or self.short_size == 0.0:
            return 0.0
        amount = self.short_size * price * per_period
        self.funding_pnl += amount
        return amount

    def equity(self, mark: float) -> float:
        """Total hedge equity in USDC: realised + unrealised + funding − fees."""
        price = self._check_mark(mark)
        return (
            self.realized_pnl
            + self.mark_pnl(price)
            + self.funding_pnl
            - self.fees_paid
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _check_mark(mark: float) -> float:
        price = _as_finite(mark, "mark")
        if price <= 0.0:
            raise HedgeError(f"mark must be > 0, got {mark!r}")
        return price
