"""S20 — ``HedgeAccount`` short-perp overlay accounting.

Every expected number is hand-computed in the comment above the assertion so a
regression in a sign convention fails here rather than propagating into the
environment, backtester or reward.  The account is pure arithmetic: no market
data, no look-ahead.
"""

from __future__ import annotations

import math

import pytest

from undertow.sim.hedge import (
    DEFAULT_HEDGE_FEE_BPS,
    HedgeAccount,
    HedgeError,
    HedgeState,
    HedgeTrade,
)


def _short_account(
    short_size: float = 2.0,
    entry_mark: float = 3000.0,
    *,
    fee_bps: float = 5.0,
    funding_enabled: bool = True,
) -> HedgeAccount:
    """An account already short ``short_size`` at ``entry_mark`` (no fee)."""
    account = HedgeAccount(fee_bps=fee_bps, funding_enabled=funding_enabled)
    # Enter the position with a zero-cost trade by starting flat: the trade's
    # fee is exactly ``size * mark * fee_bps / 1e4`` and is asserted separately.
    account.trade_to(short_size, entry_mark)
    return account


# ---------------------------------------------------------------------------
# Mark PnL signs
# ---------------------------------------------------------------------------
class TestMarkPnl:
    def test_short_gains_when_the_mark_falls(self) -> None:
        account = _short_account(short_size=2.0, entry_mark=3000.0)
        # 2 ETH short, 3000 -> 2900: 2 * (3000 - 2900) = +200.
        assert account.mark_pnl(2900.0) == pytest.approx(200.0)

    def test_short_loses_when_the_mark_rises(self) -> None:
        account = _short_account(short_size=2.0, entry_mark=3000.0)
        # 2 * (3000 - 3100) = -200.
        assert account.mark_pnl(3100.0) == pytest.approx(-200.0)

    def test_flat_mark_is_zero(self) -> None:
        account = _short_account(short_size=2.0, entry_mark=3000.0)
        assert account.mark_pnl(3000.0) == 0.0

    def test_flat_account_is_worth_zero(self) -> None:
        account = HedgeAccount()
        assert account.mark_pnl(2500.0) == 0.0
        assert account.equity(2500.0) == 0.0

    def test_fractional_short_marks_proportionally(self) -> None:
        account = _short_account(short_size=0.5, entry_mark=2500.0)
        # 0.5 * (2500 - 2400) = +50.
        assert account.mark_pnl(2400.0) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Trades: fee on |Δ|, realisation, entry reset
# ---------------------------------------------------------------------------
class TestTradeTo:
    def test_entry_fee_is_abs_delta_mark_bps(self) -> None:
        account = HedgeAccount(fee_bps=5.0)
        # 3 * 3000 * 5 / 1e4 = 4.5.
        fee = account.trade_to(3.0, 3000.0)
        assert fee == pytest.approx(4.5)
        assert account.fees_paid == pytest.approx(4.5)
        assert account.short_size == 3.0
        assert account.entry_mark == 3000.0
        assert account.realized_pnl == 0.0

    def test_trade_realises_mark_pnl_into_the_book(self) -> None:
        account = _short_account(short_size=1.0, entry_mark=3000.0)
        # Realise 1 * (3000 - 3100) = -100, then buy 2 more ETH at 3100:
        # fee = 2 * 3100 * 5 / 1e4 = 3.1.
        fee = account.trade_to(3.0, 3100.0)
        assert fee == pytest.approx(3.1)
        assert account.realized_pnl == pytest.approx(-100.0)
        assert account.entry_mark == 3100.0
        assert account.short_size == 3.0
        # The fixture's entry trade already paid 1 * 3000 * 5 / 1e4 = 1.5.
        assert account.fees_paid == pytest.approx(1.5 + 3.1)

    def test_trade_to_zero_flattens_the_short(self) -> None:
        account = _short_account(short_size=2.0, entry_mark=3000.0)
        # Close 2 ETH at 2800: realised = 2 * 200 = +400; fee = 2 * 2800 * 5e-4.
        fee = account.trade_to(0.0, 2800.0)
        assert fee == pytest.approx(2.0 * 2800.0 * 5.0 / 1e4)
        assert account.short_size == 0.0
        assert account.realized_pnl == pytest.approx(400.0)
        assert account.mark_pnl(2600.0) == 0.0

    def test_no_delta_charges_no_fee_but_rebases_the_mark(self) -> None:
        account = _short_account(short_size=2.0, entry_mark=3000.0)
        before = account.state
        fee = account.trade_to(2.0, 3050.0)
        assert fee == 0.0
        assert account.fees_paid == before.fees_paid
        # The 3000 -> 3050 move is realised into the book (2 * -50 = -100)
        # and the entry mark re-bases, so the unrealised PnL resets to zero.
        assert account.realized_pnl == pytest.approx(-100.0)
        assert account.entry_mark == 3050.0
        assert account.mark_pnl(3050.0) == 0.0

    def test_last_trade_records_the_executed_trade(self) -> None:
        account = _short_account(short_size=1.0, entry_mark=3000.0)
        account.trade_to(4.0, 3200.0)
        trade = account.last_trade
        assert isinstance(trade, HedgeTrade)
        assert trade.traded_size == pytest.approx(3.0)
        assert trade.short_size == pytest.approx(4.0)
        assert trade.mark == pytest.approx(3200.0)
        assert trade.fee == pytest.approx(3.0 * 3200.0 * 5.0 / 1e4)
        assert trade.realized_pnl == pytest.approx(1.0 * (3000.0 - 3200.0))

    def test_state_snapshot_matches_the_book(self) -> None:
        account = _short_account(short_size=2.0, entry_mark=3000.0)
        account.accrue_funding(0.0001, 3000.0)
        state = account.state
        assert isinstance(state, HedgeState)
        assert state.short_size == 2.0
        assert state.entry_mark == 3000.0
        assert state.realized_pnl == 0.0
        assert state.funding_pnl == pytest.approx(0.6)
        # The fixture's entry trade charged 2 * 3000 * 5 / 1e4 = 3.0 USDC.
        assert state.fees_paid == pytest.approx(3.0)

    def test_reset_clears_the_book_but_keeps_the_fee(self) -> None:
        account = _short_account(short_size=2.0, entry_mark=3000.0, fee_bps=7.0)
        account.accrue_funding(0.0001, 3000.0)
        account.reset()
        assert account.short_size == 0.0
        assert account.entry_mark == 0.0
        assert account.realized_pnl == 0.0
        assert account.funding_pnl == 0.0
        assert account.fees_paid == 0.0
        assert account.last_trade is None
        assert account.equity(3000.0) == 0.0
        assert account.fee_bps == 7.0


# ---------------------------------------------------------------------------
# Equity continuity across a trade
# ---------------------------------------------------------------------------
class TestEquityContinuity:
    def test_equity_drops_exactly_by_the_fee_at_the_same_mark(self) -> None:
        account = _short_account(short_size=2.0, entry_mark=3000.0)
        # 2 * (3000 - 2990) = +20 unrealised less the 3.0 entry fee.
        before = account.equity(2990.0)
        fee = account.trade_to(3.0, 2990.0)
        after = account.equity(2990.0)
        assert before == pytest.approx(17.0)
        assert after == pytest.approx(before - fee)
        assert fee == pytest.approx(1.0 * 2990.0 * 5.0 / 1e4)

    def test_equity_combines_realised_unrealised_funding_and_fees(self) -> None:
        account = _short_account(short_size=2.0, entry_mark=3000.0)
        account.trade_to(1.0, 3100.0)  # realise -200, fee 1*3100*5e-4 = 1.55
        account.accrue_funding(0.0002, 3100.0)  # 1 * 3100 * 2e-4 = +0.62
        # mark 3100 -> mark 3050: unrealised 1 * (3100 - 3050) = +50.
        # total = -200 + 50 + 0.62 - (3.0 + 1.55).
        assert account.equity(3050.0) == pytest.approx(-200.0 + 50.0 + 0.62 - 4.55)

    def test_equity_is_continuous_across_a_flat_trade(self) -> None:
        account = _short_account(short_size=2.0, entry_mark=3000.0)
        before = account.equity(3020.0)
        account.trade_to(2.0, 3020.0)
        assert account.equity(3020.0) == pytest.approx(before)


# ---------------------------------------------------------------------------
# Funding
# ---------------------------------------------------------------------------
class TestFunding:
    def test_positive_rate_pays_the_short(self) -> None:
        account = _short_account(short_size=2.0, entry_mark=3000.0)
        # 2 ETH * $3000 * 0.0001 = +0.6 USDC received.
        amount = account.accrue_funding(0.0001, 3000.0)
        assert amount == pytest.approx(0.6)
        assert account.funding_pnl == pytest.approx(0.6)
        assert account.equity(3000.0) == pytest.approx(0.6 - account.fees_paid)

    def test_negative_rate_charges_the_short(self) -> None:
        account = _short_account(short_size=1.5, entry_mark=2800.0)
        # 1.5 * 2800 * (-0.0002) = -0.84.
        assert account.accrue_funding(-0.0002, 2800.0) == pytest.approx(-0.84)
        assert account.funding_pnl == pytest.approx(-0.84)

    def test_zero_rate_is_neutral(self) -> None:
        account = _short_account()
        assert account.accrue_funding(0.0, 3000.0) == 0.0

    def test_flat_account_accrues_nothing(self) -> None:
        account = HedgeAccount()
        assert account.accrue_funding(0.001, 3000.0) == 0.0
        assert account.funding_pnl == 0.0

    def test_funding_disabled_is_a_no_op(self) -> None:
        account = _short_account(funding_enabled=False)
        assert account.accrue_funding(0.001, 3000.0) == 0.0
        assert account.funding_pnl == 0.0

    def test_funding_uses_the_supplied_mark_not_a_stored_one(self) -> None:
        account = _short_account(short_size=1.0, entry_mark=3000.0)
        # The settlement mark is 3200, not the entry mark: 1 * 3200 * 0.001.
        assert account.accrue_funding(0.001, 3200.0) == pytest.approx(3.2)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------
class TestErrors:
    def test_negative_target_raises(self) -> None:
        account = HedgeAccount()
        with pytest.raises(HedgeError, match="target_short"):
            account.trade_to(-0.1, 3000.0)

    @pytest.mark.parametrize("mark", [0.0, -1.0, math.nan, math.inf, -math.inf])
    def test_non_positive_or_non_finite_mark_raises(self, mark: float) -> None:
        account = _short_account()
        for method in (account.mark_pnl, account.equity):
            with pytest.raises(HedgeError, match="mark"):
                method(mark)
        with pytest.raises(HedgeError, match="mark"):
            account.trade_to(1.0, mark)
        with pytest.raises(HedgeError, match="mark"):
            account.accrue_funding(0.0001, mark)

    @pytest.mark.parametrize("rate", [math.nan, math.inf, -math.inf])
    def test_non_finite_rate_raises(self, rate: float) -> None:
        account = _short_account()
        with pytest.raises(HedgeError, match="rate"):
            account.accrue_funding(rate, 3000.0)

    @pytest.mark.parametrize("fee_bps", [-1.0, math.nan, math.inf])
    def test_bad_fee_bps_raises(self, fee_bps: float) -> None:
        with pytest.raises(HedgeError, match="fee_bps"):
            HedgeAccount(fee_bps=fee_bps)

    def test_default_fee_is_five_basis_points(self) -> None:
        assert HedgeAccount().fee_bps == DEFAULT_HEDGE_FEE_BPS == 5.0

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("short_size", -1.0),
            ("entry_mark", -1.0),
            ("fees_paid", -1.0),
            ("short_size", math.nan),
            ("funding_pnl", math.inf),
        ],
    )
    def test_invalid_state_snapshot_raises(self, field: str, value: float) -> None:
        with pytest.raises(HedgeError):
            HedgeState(**{field: value})

    def test_account_rejects_negative_initial_size(self) -> None:
        with pytest.raises(HedgeError, match="short_size"):
            HedgeAccount(short_size=-2.0)
