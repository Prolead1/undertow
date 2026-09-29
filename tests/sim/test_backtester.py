"""Behavioural tests for the S11 frozen-policy backtester (CONTRACTS.md §13).

These assert real behaviour on small golden tapes: the equity/PnL identity, the
exact Q128 fee accrual, the decision cadence, gas/slippage charged only on
rebalance steps, the decomposition identity, determinism, config-hash stability
and the frozen-policy guarantee.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import polars as pl
import pytest

from undertow.data import price_to_tick, tick_to_price, tick_to_sqrt_price_x96
from undertow.sim.backtest import (
    EQUITY_CURVE_COLUMNS,
    BacktestLedger,
    BacktestObservation,
    BacktestResult,
    run_backtest,
    summarize_backtest,
)
from undertow.sim.config import EpisodeConfig, GasConfig, SimConfig, SlippageConfig
from undertow.sim.core.position import initial_deposit
from undertow.sim.frictions import GasCostCalculator
from undertow.sim.marketview import MarketView
from undertow.sim.policies import HODLPolicy, PassiveNarrowPolicy, Policy, TauResetPolicy
from undertow.sim.types import Action, BacktestError, Tick, TickSpacing

UTC = UTC
DEC0 = 6
DEC1 = 18
ENTRY_TICK = 196242
TICK_SPACING = 60
CAPITAL = 100_000.0

_ENTRY_SQRT = tick_to_sqrt_price_x96(ENTRY_TICK) / (1 << 96)
_ENTRY_PRICE = float(tick_to_price(ENTRY_TICK, DEC0, DEC1))


def _market_view(
    *,
    minutes: list[int],
    prices: list[float],
    ticks: list[int | None] | None = None,
    fg0: list[int] | None = None,
    fg1: list[int] | None = None,
    base_fee: int = 20 * 10**9,
    priority_fee: int = 10**9,
    liquidity: int = 2**80,
) -> MarketView:
    """Build a small, self-consistent ``MarketView`` (train split) for tests.

    Reference closes and pool sqrts are consistent with ``ENTRY_TICK`` so the
    initial deposit and the HODL benchmark are reproducible by hand.
    """
    n = len(minutes)
    start = datetime(2022, 1, 1, tzinfo=UTC)
    times = [start + timedelta(minutes=m) for m in minutes]
    if ticks is None:
        ticks = [ENTRY_TICK] * n
    if fg0 is None:
        fg0 = [0] * n
    if fg1 is None:
        fg1 = [0] * n
    sqrt_x96 = [tick_to_sqrt_price_x96(t) if t is not None else None for t in ticks]

    tape = pl.DataFrame(
        {
            "seq": list(range(n)),
            "block_number": list(range(n)),
            "block_timestamp": times,
            "event_type": ["swap"] * n,
            "price_reference": [float(p) for p in prices],
            "price_pool": [float(p) for p in prices],
            "sqrt_price_x96": [str(s) if s is not None else None for s in sqrt_x96],
            "tick": ticks,
            "liquidity": [str(liquidity)] * n,
            "base_fee_per_gas": [str(base_fee)] * n,
            "priority_fee_p50_wei": [str(priority_fee)] * n,
            "fee_growth_global_0_x128": [str(v) if v is not None else None for v in fg0],
            "fee_growth_global_1_x128": [str(v) if v is not None else None for v in fg1],
            "regime": ["sideways"] * n,
        }
    )
    reference = pl.DataFrame(
        {
            "open_time": times,
            "close_time": [t + timedelta(minutes=1) for t in times],
            "close": [float(p) for p in prices],
        }
    )
    gas = pl.DataFrame({"block_number": list(range(n))})
    regimes = pl.DataFrame({"timestamp": times, "regime": ["sideways"] * n})
    end = times[-1] + timedelta(minutes=1)
    return MarketView(
        train=tape,
        eval=None,
        reference=reference,
        regimes=regimes,
        gas=gas,
        train_start_utc=times[0],
        train_end_utc=end,
        eval_start_utc=end,
        eval_end_utc=end + timedelta(days=1),
        is_train=True,
    )


def _flat_view(**kwargs: object) -> MarketView:
    """A 12-row tape at a constant price, one event per 10 minutes."""
    kwargs.setdefault("minutes", [i * 10 for i in range(12)])
    kwargs.setdefault("prices", [_ENTRY_PRICE] * 12)
    return _market_view(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1-4: run + ledger shape
# ---------------------------------------------------------------------------


def test_hodl_backtest_completes(tiny_market_view: MarketView) -> None:
    ledger = run_backtest(HODLPolicy(), tiny_market_view, SimConfig())
    assert isinstance(ledger, BacktestLedger)
    assert ledger.policy_name == "hodl"
    assert ledger.split == "train"
    assert ledger.n_steps == tiny_market_view.step_count()


def test_ledger_frames_non_empty_with_rebalancing_policy() -> None:
    ledger = run_backtest(PassiveNarrowPolicy(), _flat_view(), SimConfig())
    assert ledger.equity_curve.height > 0
    assert ledger.decision_log.height > 0
    assert ledger.cost_ledger.height > 0  # PassiveNarrow rebalances at t=0


def test_equity_curve_columns_exact(tiny_ledger: BacktestLedger) -> None:
    assert tuple(tiny_ledger.equity_curve.columns) == EQUITY_CURVE_COLUMNS


def test_hodl_decision_log_is_all_holds(tiny_ledger: BacktestLedger) -> None:
    actions = tiny_ledger.decision_log["action_type"].to_list()
    assert actions and set(actions) == {"hold"}
    assert tiny_ledger.decision_log["gas_paid"].to_list() == [0.0] * len(actions)
    assert tiny_ledger.decision_log["slippage_paid"].to_list() == [0.0] * len(actions)


# ---------------------------------------------------------------------------
# 5: passive narrow deploys on the first decision and holds thereafter
# ---------------------------------------------------------------------------


def test_passive_narrow_rebalances_first_decision_then_holds() -> None:
    ledger = run_backtest(PassiveNarrowPolicy(), _flat_view(), SimConfig())
    actions = ledger.decision_log["action_type"].to_list()
    assert actions[0] == "rebalance"
    assert set(actions[1:]) == {"hold"}
    first = ledger.decision_log.row(0, named=True)
    assert first["tick_lower"] < ENTRY_TICK < first["tick_upper"]


# ---------------------------------------------------------------------------
# 6-8: summary + HODL benchmark
# ---------------------------------------------------------------------------


def test_summarize_backtest_fields_present(tiny_ledger: BacktestLedger) -> None:
    result = summarize_backtest(tiny_ledger)
    assert isinstance(result, BacktestResult)
    for field in (
        "total_pnl",
        "annualized_return",
        "sharpe",
        "max_drawdown",
        "fee_income",
        "gas_paid",
        "slippage_paid",
        "il_realized",
        "hodl_return",
        "excess_vs_hodl",
    ):
        assert getattr(result, field) is not None


def test_hodl_return_matches_manual_benchmark() -> None:
    minutes = [i * 10 for i in range(12)]
    # A rising reference price exercises the HODL mark-to-market.
    prices = [_ENTRY_PRICE * (1.0 + 0.001 * i) for i in range(12)]
    view = _market_view(minutes=minutes, prices=prices)
    ledger = run_backtest(HODLPolicy(), view, SimConfig())

    lower = (ENTRY_TICK - 120) // TICK_SPACING * TICK_SPACING
    upper = -((-(ENTRY_TICK + 120)) // TICK_SPACING) * TICK_SPACING
    position = initial_deposit(
        _ENTRY_SQRT,
        _ENTRY_PRICE,
        Tick(lower),
        Tick(upper),
        CAPITAL,
        TickSpacing(TICK_SPACING),
    )
    expected = (
        position.hodl_value(_ENTRY_SQRT, _ENTRY_PRICE, prices[-1]) - CAPITAL
    )
    assert ledger.hodl_return == pytest.approx(expected, rel=1e-12)


def test_excess_vs_hodl_identity(tiny_ledger: BacktestLedger) -> None:
    result = summarize_backtest(tiny_ledger)
    assert result.excess_vs_hodl == pytest.approx(
        result.total_pnl - result.hodl_return
    )


# ---------------------------------------------------------------------------
# 9-10: determinism and config hash
# ---------------------------------------------------------------------------


def test_backtest_is_deterministic(tiny_market_view: MarketView) -> None:
    config = SimConfig(seed=7)
    first = run_backtest(HODLPolicy(), tiny_market_view, config)
    second = run_backtest(HODLPolicy(), tiny_market_view, config)
    assert first.equity_curve.equals(second.equity_curve)
    assert first.decision_log.equals(second.decision_log)
    assert first.cost_ledger.equals(second.cost_ledger)
    assert first.pnl_decomposition == second.pnl_decomposition


def test_config_hash_is_stable_and_config_sensitive(tiny_market_view: MarketView) -> None:
    a = run_backtest(HODLPolicy(), tiny_market_view, SimConfig(seed=0))
    b = run_backtest(HODLPolicy(), tiny_market_view, SimConfig(seed=0))
    c = run_backtest(HODLPolicy(), tiny_market_view, SimConfig(seed=1))
    assert a.config_hash == b.config_hash
    assert a.config_hash != c.config_hash
    assert len(a.config_hash) == 16


# ---------------------------------------------------------------------------
# Equity/PnL identity and decomposition
# ---------------------------------------------------------------------------


def test_equity_equals_capital_plus_cumulative_pnl(tiny_ledger: BacktestLedger) -> None:
    frame = tiny_ledger.equity_curve
    equity = frame["equity"].to_list()
    pnl = frame["pnl"].to_list()
    capital = tiny_ledger.initial_capital
    assert equity[0] == pytest.approx(capital)
    for e, p in zip(equity, pnl, strict=True):
        assert e == pytest.approx(capital + p, rel=1e-12, abs=1e-9)


def test_pnl_decomposition_keys_and_sum(tiny_ledger: BacktestLedger) -> None:
    decomposition = tiny_ledger.pnl_decomposition
    assert set(decomposition) == {
        "total_fees",
        "total_gas",
        "total_slippage",
        "total_il",
        "net",
    }
    assert decomposition["net"] == pytest.approx(
        decomposition["total_fees"]
        + decomposition["total_gas"]
        + decomposition["total_slippage"]
        + decomposition["total_il"]
    )


def test_decomposition_terms_match_equity_columns(tiny_ledger: BacktestLedger) -> None:
    frame = tiny_ledger.equity_curve
    decomposition = tiny_ledger.pnl_decomposition
    assert decomposition["total_fees"] == pytest.approx(float(frame["fees"].sum()))
    assert decomposition["total_gas"] == pytest.approx(float(frame["gas"].sum()))
    assert decomposition["total_slippage"] == pytest.approx(
        float(frame["slippage"].sum())
    )
    assert decomposition["total_il"] == pytest.approx(float(frame["il"].sum()))


def test_decomposition_net_equals_excess_vs_hodl(tiny_ledger: BacktestLedger) -> None:
    result = summarize_backtest(tiny_ledger)
    assert tiny_ledger.pnl_decomposition["net"] == pytest.approx(
        result.excess_vs_hodl, rel=1e-9, abs=1e-6
    )


def test_decomposition_net_equals_excess_for_rebalancing_policy() -> None:
    prices = [_ENTRY_PRICE * (1.0 + 0.005 * i) for i in range(8)]
    ticks = [int(price_to_tick(Decimal(str(p)), DEC0, DEC1)) for p in prices]
    view = _market_view(minutes=[i * 60 for i in range(8)], prices=prices, ticks=ticks)
    ledger = run_backtest(_RebalanceEveryStep(width=5), view, SimConfig())
    result = summarize_backtest(ledger)
    assert ledger.pnl_decomposition["net"] == pytest.approx(
        result.excess_vs_hodl, rel=1e-6, abs=1e-3
    )


# ---------------------------------------------------------------------------
# 11: exact Q128 fee accrual
# ---------------------------------------------------------------------------


def test_fee_accrual_is_exact_integer_q128() -> None:
    # Global fee growth jumps by a known Q128 amount at the third event; the
    # position is in range for the whole tape.
    delta = 1 << 96
    fg0 = [0, 0, delta, delta]
    view = _market_view(
        minutes=[0, 10, 20, 30],
        prices=[_ENTRY_PRICE] * 4,
        fg0=fg0,
    )
    ledger = run_backtest(HODLPolicy(), view, SimConfig())

    lower = (ENTRY_TICK - 120) // TICK_SPACING * TICK_SPACING
    upper = -((-(ENTRY_TICK + 120)) // TICK_SPACING) * TICK_SPACING
    position = initial_deposit(
        _ENTRY_SQRT,
        _ENTRY_PRICE,
        Tick(lower),
        Tick(upper),
        CAPITAL,
        TickSpacing(TICK_SPACING),
    )
    liquidity = int(round(position.liquidity))
    expected_raw = (liquidity * delta) >> 128
    expected_fee = expected_raw / 10**DEC0

    fees = ledger.equity_curve["fees"].to_list()
    assert fees[0] == 0.0 and fees[1] == 0.0
    assert fees[2] == pytest.approx(expected_fee, rel=1e-12)
    assert fees[3] == 0.0
    assert sum(fees) == pytest.approx(expected_fee, rel=1e-12)


def test_fees_only_accrue_while_in_range() -> None:
    # Move the pool tick far above the position's upper bound: no fees accrue.
    delta = 1 << 96
    far_tick = ENTRY_TICK + 10_000
    view = _market_view(
        minutes=[0, 10, 20],
        prices=[_ENTRY_PRICE] * 3,
        ticks=[ENTRY_TICK, far_tick, far_tick],
        fg0=[0, delta, delta],
    )
    ledger = run_backtest(HODLPolicy(), view, SimConfig())
    assert ledger.equity_curve["fees"].sum() == pytest.approx(0.0)


def test_leading_null_fee_growth_does_not_windfall() -> None:
    # The tape's global accumulators are nullable before the first snapshot; a
    # leading null must seed the baseline without crediting the unobservable gap.
    delta = 1 << 96
    view = _market_view(
        minutes=[0, 10, 20, 30],
        prices=[_ENTRY_PRICE] * 4,
        fg0=[None, None, delta, 2 * delta],
        fg1=[None, None, delta, 2 * delta],
    )
    ledger = run_backtest(HODLPolicy(), view, SimConfig())
    fees = ledger.equity_curve["fees"].to_list()
    assert fees[:3] == [0.0, 0.0, 0.0]

    lower = (ENTRY_TICK - 120) // TICK_SPACING * TICK_SPACING
    upper = -((-(ENTRY_TICK + 120)) // TICK_SPACING) * TICK_SPACING
    position = initial_deposit(
        _ENTRY_SQRT,
        _ENTRY_PRICE,
        Tick(lower),
        Tick(upper),
        CAPITAL,
        TickSpacing(TICK_SPACING),
    )
    expected_raw = (int(round(position.liquidity)) * delta) >> 128
    expected = expected_raw / 10**DEC0 + expected_raw / 10**DEC1 * _ENTRY_PRICE
    assert fees[3] == pytest.approx(expected, rel=1e-12)


def test_all_null_fee_growth_accrues_nothing() -> None:
    view = _market_view(
        minutes=[0, 10, 20],
        prices=[_ENTRY_PRICE] * 3,
        fg0=[None, None, None],
        fg1=[None, None, None],
    )
    ledger = run_backtest(HODLPolicy(), view, SimConfig())
    assert ledger.equity_curve["fees"].sum() == pytest.approx(0.0)


def test_equity_change_reconciles_with_decomposition_each_step() -> None:
    # The strongest form of the accounting identity: for every step,
    #   Δequity == fees + il + gas + slippage + ΔHODL
    # with HODL the fixed initial basket.  Exercised on a rebalancing policy with
    # nonzero WETH fee accrual and a moving price (the reviewer's counterexample).
    delta = 1 << 100
    prices = [
        _ENTRY_PRICE,
        _ENTRY_PRICE * 1.01,
        _ENTRY_PRICE * 1.03,
        _ENTRY_PRICE * 1.02,
    ]
    ticks = [int(price_to_tick(Decimal(str(p)), DEC0, DEC1)) for p in prices]
    view = _market_view(
        minutes=[i * 60 for i in range(4)],
        prices=prices,
        ticks=ticks,
        fg0=[0, 0, 0, 0],
        fg1=[0, delta, delta, delta],
    )
    ledger = run_backtest(_RebalanceEveryStep(width=5), view, SimConfig())

    # Reconstruct the fixed initial buy-and-hold basket exactly as run_backtest did.
    entry_tick = ticks[0]
    entry_sqrt = tick_to_sqrt_price_x96(entry_tick) / (1 << 96)
    lower = (entry_tick - 120) // TICK_SPACING * TICK_SPACING
    upper = -((-(entry_tick + 120)) // TICK_SPACING) * TICK_SPACING
    initial = initial_deposit(
        entry_sqrt,
        _ENTRY_PRICE,
        Tick(lower),
        Tick(upper),
        CAPITAL,
        TickSpacing(TICK_SPACING),
    )

    frame = ledger.equity_curve
    equity = frame["equity"].to_list()
    fees = frame["fees"].to_list()
    gas = frame["gas"].to_list()
    slippage = frame["slippage"].to_list()
    il = frame["il"].to_list()
    hodl = [initial.hodl_value(entry_sqrt, _ENTRY_PRICE, p) for p in prices]

    prev_equity = CAPITAL
    prev_hodl = CAPITAL
    for t in range(len(prices)):
        delta_equity = equity[t] - prev_equity
        delta_hodl = hodl[t] - prev_hodl
        expected = fees[t] + il[t] + gas[t] + slippage[t] + delta_hodl
        assert delta_equity == pytest.approx(expected, rel=1e-9, abs=1e-6)
        prev_equity = equity[t]
        prev_hodl = hodl[t]


def test_fee_revaluation_keeps_net_equals_excess() -> None:
    # Token1 (WETH) fees accrued at row 1 are marked-to-market by the +5% move at
    # row 2; the `fees` term must include that revaluation or `net` would diverge
    # from `excess_vs_hodl`.
    delta = 1 << 100
    prices = [_ENTRY_PRICE, _ENTRY_PRICE, _ENTRY_PRICE * 1.05]
    ticks = [int(price_to_tick(Decimal(str(p)), DEC0, DEC1)) for p in prices]
    view = _market_view(
        minutes=[0, 10, 20],
        prices=prices,
        ticks=ticks,
        fg0=[0, 0, 0],
        fg1=[0, delta, delta],
    )
    ledger = run_backtest(HODLPolicy(), view, SimConfig())
    result = summarize_backtest(ledger)
    fees = ledger.equity_curve["fees"].to_list()
    assert fees[1] > 0.0
    assert fees[2] > 0.0  # revaluation of the WETH fee leg
    assert ledger.pnl_decomposition["net"] == pytest.approx(
        result.excess_vs_hodl, rel=1e-9, abs=1e-6
    )


# ---------------------------------------------------------------------------
# 12-14: cadence, gas and slippage
# ---------------------------------------------------------------------------


def test_decision_cadence_respected() -> None:
    # Four events at 0/40/80/120 minutes with a 60-minute cadence -> 2 decisions.
    view = _market_view(
        minutes=[0, 40, 80, 120],
        prices=[_ENTRY_PRICE] * 4,
    )
    config = SimConfig(episode=EpisodeConfig(step_minutes=60))
    ledger = run_backtest(HODLPolicy(), view, config)
    assert ledger.decision_log.height == 2
    assert ledger.equity_curve.height == 4  # equity still recorded per event


def test_gas_only_charged_on_rebalance_steps() -> None:
    view = _flat_view()
    ledger = run_backtest(PassiveNarrowPolicy(), view, SimConfig())
    frame = ledger.equity_curve
    gas = frame["gas"].to_list()
    # PassiveNarrow rebalances exactly once (step 0), so exactly one non-zero gas.
    assert sum(1 for g in gas if g != 0.0) == 1
    assert gas[0] < 0.0

    calc = GasCostCalculator(SimConfig().gas)
    expected_gas = calc.gas_cost_usdc(
        "rebalance", 0, 20 * 10**9, 10**9, _ENTRY_PRICE
    )
    assert gas[0] == pytest.approx(-expected_gas, rel=1e-12)
    first_decision = ledger.decision_log.row(0, named=True)
    assert first_decision["gas_paid"] == pytest.approx(-expected_gas, rel=1e-12)


def test_slippage_only_charged_on_rebalance_steps() -> None:
    view = _flat_view()
    ledger = run_backtest(PassiveNarrowPolicy(), view, SimConfig())
    slippage = ledger.equity_curve["slippage"].to_list()
    assert sum(1 for s in slippage if s != 0.0) == 1
    assert slippage[0] < 0.0

    # ADR-012: notional * (3000/1e6 + 5/1e4) with notional ~ capital.
    position_value = CAPITAL  # at t=0 the initial position is worth the capital
    expected = position_value * (3000 / 1_000_000 + 5 / 10_000)
    assert slippage[0] == pytest.approx(-expected, rel=1e-6)


def test_cost_ledger_records_matching_cost_types() -> None:
    ledger = run_backtest(PassiveNarrowPolicy(), _flat_view(), SimConfig())
    types = ledger.cost_ledger["cost_type"].to_list()
    assert "gas" in types and "slippage" in types
    amounts = ledger.cost_ledger["amount_usdc"].to_list()
    assert all(a <= 0.0 for a in amounts)
    assert sum(amounts) == pytest.approx(
        float(ledger.equity_curve["gas"].sum() + ledger.equity_curve["slippage"].sum()),
        rel=1e-12,
    )


class _RebalanceEveryStep:
    """A deterministic policy that rebalances every decision (test-local)."""

    def __init__(self, width: int = 10) -> None:
        self._width = width

    @property
    def name(self) -> str:
        return "rebalance_every_step"

    def reset(self) -> None:
        """No episode-local state."""

    def act(self, observation: BacktestObservation, rng: object = None) -> Action:
        del observation, rng
        return Action(action_type="rebalance", center_offset=0, width=self._width)


def test_rebalance_carries_equity_forward() -> None:
    # A monotonic 1%/step rise with a full-range, cost-free rebalance must carry
    # the gains through each redeployment instead of resetting to the original
    # capital (the classic "redeploy initial capital" bug).
    prices = [_ENTRY_PRICE * (1.0 + 0.01 * i) for i in range(6)]
    ticks = [int(price_to_tick(Decimal(str(p)), DEC0, DEC1)) for p in prices]
    view = _market_view(minutes=[i * 60 for i in range(6)], prices=prices, ticks=ticks)
    free = SimConfig(
        episode=EpisodeConfig(fee_tier_bps=0, step_minutes=60),
        gas=GasConfig(
            mint_units=0, burn_units=0, collect_units=0, rebalance_swap_units=0
        ),
        slippage=SlippageConfig(fixed_impact_bps=0.0),
    )
    ledger = run_backtest(_RebalanceEveryStep(width=10_000), view, free)
    equity = ledger.equity_curve["equity"].to_list()
    assert equity[-1] > CAPITAL * 1.02
    assert all(e > CAPITAL for e in equity[1:]), "equity reset toward capital"


def test_rebalance_collects_accrued_fees() -> None:
    delta = 1 << 96
    view = _market_view(
        minutes=[0, 60, 120],
        prices=[_ENTRY_PRICE] * 3,
        fg0=[0, delta, delta],
    )
    config = SimConfig(episode=EpisodeConfig(step_minutes=60))
    ledger = run_backtest(_RebalanceEveryStep(), view, config)
    collected = ledger.decision_log["fees_collected"].to_list()
    assert collected[0] == 0.0  # no growth at the first decision
    assert collected[1] > 0.0  # growth from event 1 to 2 is collected at the rebalance


# ---------------------------------------------------------------------------
# 15: shared fixture
# ---------------------------------------------------------------------------


def test_tiny_ledger_fixture(tiny_ledger: BacktestLedger) -> None:
    assert tiny_ledger.n_steps == 100
    assert tiny_ledger.equity_curve.height == 100
    result = summarize_backtest(tiny_ledger)
    assert result.total_pnl == pytest.approx(
        tiny_ledger.equity_curve["pnl"][-1]
    )


# ---------------------------------------------------------------------------
# Frozen-policy guarantee
# ---------------------------------------------------------------------------


def test_policy_protocol_has_no_learning_api() -> None:
    assert not hasattr(Policy, "update")
    assert not hasattr(Policy, "learn")


def test_policy_update_is_never_called() -> None:
    class _SpyPolicy:
        def __init__(self) -> None:
            self.act_calls = 0
            self.update_calls = 0
            self._inner = PassiveNarrowPolicy()

        @property
        def name(self) -> str:
            return "spy"

        def reset(self) -> None:
            self.act_calls = 0
            self._inner.reset()

        def act(self, observation: BacktestObservation, rng: object = None) -> object:
            self.act_calls += 1
            return self._inner.act(observation, rng)  # type: ignore[arg-type]

        def update(self, *args: object, **kwargs: object) -> None:
            self.update_calls += 1
            raise AssertionError("the backtester must never call update()")

    spy = _SpyPolicy()
    ledger = run_backtest(spy, _flat_view(), SimConfig())  # type: ignore[arg-type]
    assert spy.act_calls == ledger.decision_log.height
    assert spy.update_calls == 0


# ---------------------------------------------------------------------------
# Position/validation edges
# ---------------------------------------------------------------------------


def test_tau_reset_policy_runs_on_volatile_tape() -> None:
    prices = [_ENTRY_PRICE * (1.0 + 0.002 * ((-1) ** i) * (i + 1)) for i in range(20)]
    view = _market_view(minutes=[i * 10 for i in range(20)], prices=prices)
    ledger = run_backtest(TauResetPolicy(120), view, SimConfig())
    assert ledger.equity_curve.height == 20
    assert set(ledger.decision_log["action_type"].to_list()) <= {"hold", "rebalance"}


def test_gas_price_uses_tape_columns() -> None:
    view = _market_view(
        minutes=[0, 10],
        prices=[_ENTRY_PRICE] * 2,
        base_fee=42 * 10**9,
        priority_fee=0,
    )
    ledger = run_backtest(PassiveNarrowPolicy(), view, SimConfig())
    calc = GasCostCalculator(SimConfig().gas)
    expected = calc.gas_cost_usdc("rebalance", 0, 42 * 10**9, 0, _ENTRY_PRICE)
    assert ledger.equity_curve["gas"][0] == pytest.approx(-expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_missing_tape_column_raises_backtest_error() -> None:
    view = _flat_view()
    assert view.train is not None
    broken = MarketView(
        train=view.train.drop("fee_growth_global_0_x128"),
        eval=None,
        reference=view.reference,
        regimes=view.regimes,
        gas=view.gas,
        train_start_utc=view.train_start_utc,
        train_end_utc=view.train_end_utc,
        eval_start_utc=view.eval_start_utc,
        eval_end_utc=view.eval_end_utc,
        is_train=True,
    )
    with pytest.raises(BacktestError, match="missing"):
        run_backtest(HODLPolicy(), broken, SimConfig())


def test_rebalance_wipeout_raises_backtest_error() -> None:
    config = SimConfig(episode=EpisodeConfig(agent_capital_usdc=1.0))
    with pytest.raises(BacktestError, match="wipe out"):
        run_backtest(PassiveNarrowPolicy(), _flat_view(), config)


def test_empty_tape_raises_backtest_error(monkeypatch: pytest.MonkeyPatch) -> None:
    view = _flat_view()
    assert view.train is not None
    empty = view.train.clear()
    monkeypatch.setattr(MarketView, "active_split_data", lambda self: empty)
    with pytest.raises(BacktestError, match="no tape rows"):
        run_backtest(HODLPolicy(), view, SimConfig())


def test_no_usable_reference_price_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    view = _flat_view()
    assert view.train is not None
    blanked = view.train.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("price_reference"),
        pl.lit(None, dtype=pl.Float64).alias("price_pool"),
    )
    broken = MarketView(
        train=blanked,
        eval=None,
        reference=view.reference,
        regimes=view.regimes,
        gas=view.gas,
        train_start_utc=view.train_start_utc,
        train_end_utc=view.train_end_utc,
        eval_start_utc=view.eval_start_utc,
        eval_end_utc=view.eval_end_utc,
        is_train=True,
    )
    monkeypatch.setattr(MarketView, "reference_close", lambda self, at_time: None)
    with pytest.raises(BacktestError, match="no usable reference price"):
        run_backtest(HODLPolicy(), broken, SimConfig())


def test_backtest_observation_to_vector_is_stable() -> None:
    obs = BacktestObservation(
        price=3000.0,
        sqrt_price=18244.0,
        tick=ENTRY_TICK,
        price_returns_1h=0.01,
        price_returns_24h=0.02,
        realized_vol_24h=0.5,
        position_in_range=True,
        position_value=100_000.0,
        uncollected_fees_token0=1.0,
        uncollected_fees_token1=0.0,
        il_vs_hodl=-0.01,
        pool_active_liquidity=float(2**80),
        pool_fee_tier_bps=3000,
        gas_price_recent_wei=21e9,
        eth_usd_price_recent=3000.0,
        regime="bear",
        step_in_episode=3,
        steps_remaining=7,
    )
    vector = obs.to_vector()
    assert isinstance(vector, np.ndarray)
    assert vector.dtype == np.float64
    assert len(vector) == 18
    assert vector[6] == 1.0  # in-range bool
    assert vector[12] == 3000.0  # fee tier
    assert vector[15] == 1.0  # bear ordinal
    assert vector[16] == 3.0 and vector[17] == 7.0


def test_no_lookahead_from_future_tape_rows() -> None:
    # PLAN §4: an S11 look-ahead probe.  Perturbing rows strictly after step 3
    # must leave the equity curve and decision log for steps 0..3 bit-identical.
    minutes = [i * 10 for i in range(8)]
    base_view = _market_view(minutes=minutes, prices=[_ENTRY_PRICE] * 8)
    future_prices = [_ENTRY_PRICE] * 4 + [_ENTRY_PRICE * 2.0] * 4
    future_view = _market_view(minutes=minutes, prices=future_prices)

    for policy in (HODLPolicy(), PassiveNarrowPolicy()):
        base = run_backtest(policy, base_view, SimConfig())
        changed = run_backtest(policy, future_view, SimConfig())
        assert base.equity_curve.head(4).equals(changed.equity_curve.head(4))
        assert base.decision_log.head(4).equals(changed.decision_log.head(4))
