"""S14 — parity & look-ahead validation (CONTRACTS.md §16).

Every test asserts real behavior: the frozen report shapes and their derived
``passed`` flag, a *detector* test that a deliberately-injected sim/backtester
mismatch fails parity (not just the happy path), and composed look-ahead probes
that report zero violations on the correct system and a non-zero count once a
leak is injected.

The parity fixture is a purpose-built, internally consistent replay scenario:
one 10-minute decision per reference bar, a swap whose two amount legs carry the
fee-bearing input and a negligible output, and a reference feed whose closes are
the exact ADR-009 human prices of the tape's raw ``sqrt_price_x96`` values.  That
isolation is what makes the int-vs-float comparison meaningful; the shared
``tiny_market_view`` fixture is intentionally *not* parity-shaped (its tape holds
a constant raw sqrt while its reference feed moves, and its equal-magnitude swap
legs charge two fee legs to the simulator's pool engine but one to the exact
tracker).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from undertow.data import (
    EVENT_TAPE_SCHEMA,
    GAS_SCHEMA,
    REFERENCE_SCHEMA,
    REGIME_SCHEMA,
    price_to_sqrt_price_x96,
    price_to_tick,
)
from undertow.sim.backtest import BacktestLedger, run_backtest
from undertow.sim.config import EpisodeConfig, HedgeConfig, RewardConfig, SimConfig
from undertow.sim.core.pool import PoolEngine, PoolState, TickState
from undertow.sim.core.position import DEFAULT_DEC0, DEFAULT_DEC1, calc_sqrt_price_a
from undertow.sim.env.lp_env import LpEnvironment
from undertow.sim.env.observations import ObservationBuilder
from undertow.sim.evaluate.protocol import action_to_index, observation_from_vector
from undertow.sim.frictions import ProportionalSlippageModel, ReplayGasModel
from undertow.sim.marketview import MarketView
from undertow.sim.policies import HODLPolicy, TauResetPolicy
from undertow.sim.prices.replay import ReplayPriceProcess
from undertow.sim.types import (
    LookAheadError,
    ParityError,
    Tick,
)
from undertow.sim.validation import (
    DEFAULT_PARITY_TOLERANCE,
    LookAheadReport,
    ParityReport,
    run_lookahead_probes,
    run_parity_check,
)
from undertow.sim.validation import parity as parity_module

# ---------------------------------------------------------------------------
# Purpose-built parity scenario
# ---------------------------------------------------------------------------

_PARITY_STEPS = 8
_PARITY_BASE_LIQUIDITY = 1.0e22
_PARITY_ENTRY_PRICE = 3000.0
_PARITY_SWAP_AMOUNT0 = "1000000000"  # 1000 USDC in raw units
_PARITY_SWAP_AMOUNT1 = "-1"  # negligible WETH out (keeps one fee-bearing leg)
_PARITY_START = datetime(2022, 1, 1, tzinfo=UTC)
_PARITY_STEP = timedelta(minutes=10)

#: Price path for the S22 rebalancing fixture.  The **first decision bar moves**
#: (3000 -> 3002.1), so the suite exercises ADR-031 decision-time execution rather
#: than the old one-bar-late path, and the 00:40 step change (3008 -> 3060)
#: pushes the tick out of the initial ±120-tick band, triggering a genuine
#: mid-episode ``TauResetPolicy`` recentre with a material swapped-imbalance
#: slippage charge (~$186).  The remaining bars are quiet, so exactly one
#: mid-episode rebalance executes and the fee/IL path is exercised after it.
#: The first S22 attempt hand-picked a flat first bar (3000 -> 3000) so a
#: one-bar-late simulator still agreed; this path cannot pass that way.
_REBALANCE_PRICES: tuple[float, ...] = (
    _PARITY_ENTRY_PRICE,
    _PARITY_ENTRY_PRICE * 1.0007,
    _PARITY_ENTRY_PRICE * 1.0017,
    _PARITY_ENTRY_PRICE * 1.0027,
    _PARITY_ENTRY_PRICE * 1.02,
    _PARITY_ENTRY_PRICE * 1.021,
    _PARITY_ENTRY_PRICE * 1.022,
    _PARITY_ENTRY_PRICE * 1.023,
)


def _sqrt_x96(price: float) -> str:
    return str(price_to_sqrt_price_x96(Decimal(str(price)), DEFAULT_DEC0, DEFAULT_DEC1))


def _parity_view(prices: Sequence[float] | None = None) -> MarketView:
    """A schema-valid, internally consistent replay window for parity.

    ``prices`` defaults to the S14 path (a 0.1 % rise per bar).  The S22
    rebalancing fixture passes :data:`_REBALANCE_PRICES`, whose first decision
    bar moves and whose mid-episode jump triggers a real recentre (see that
    constant for why).
    """
    if prices is None:
        # Human price rises -> raw tick falls -> a token0-in swap per event.
        prices = [
            _PARITY_ENTRY_PRICE * (1.0 + 0.001 * i) for i in range(_PARITY_STEPS)
        ]
    prices = [float(price) for price in prices]
    if len(prices) != _PARITY_STEPS:
        raise ValueError(
            f"parity fixture needs {_PARITY_STEPS} prices, got {len(prices)}"
        )
    times = [_PARITY_START + _PARITY_STEP * i for i in range(_PARITY_STEPS)]
    entry_tick = int(price_to_tick(Decimal(str(prices[0])), DEFAULT_DEC0, DEFAULT_DEC1))

    tape_rows: list[dict[str, object]] = [
        {
            "block_number": 0,
            "log_index": 0,
            "block_timestamp": times[0],
            "tx_hash": "0x" + "0" * 64,
            "pool_address": "0x" + "aa" * 20,
            "event_type": "mint",
            "amount0": None,
            "amount1": None,
            "sqrt_price_x96": None,
            "liquidity": None,
            "tick": None,
            "sender": "0x" + "bb" * 20,
            "recipient": None,
            "owner": "0x" + "cc" * 20,
            "tick_lower": entry_tick - 6000,
            "tick_upper": entry_tick + 6000,
            "liquidity_amount": str(int(_PARITY_BASE_LIQUIDITY)),
            "seq": 0,
            "price_pool": prices[0],
            "price_reference": prices[0],
            "regime": "sideways",
            "base_fee_per_gas": str(20 * 10**9),
            "priority_fee_p50_wei": str(10**9),
            "fee_growth_global_0_x128": "0",
            "fee_growth_global_1_x128": "0",
            "fee_growth_source": "exact",
        }
    ]
    for i, price in enumerate(prices):
        tape_rows.append(
            {
                "block_number": i + 1,
                "log_index": 0,
                "block_timestamp": times[i],
                "tx_hash": f"0x{i:064x}",
                "pool_address": "0x" + "aa" * 20,
                "event_type": "swap",
                "amount0": _PARITY_SWAP_AMOUNT0,
                "amount1": _PARITY_SWAP_AMOUNT1,
                "sqrt_price_x96": _sqrt_x96(price),
                "liquidity": str(int(_PARITY_BASE_LIQUIDITY)),
                "tick": int(price_to_tick(Decimal(str(price)), DEFAULT_DEC0, DEFAULT_DEC1)),
                "sender": "0x" + "bb" * 20,
                "recipient": "0x" + "dd" * 20,
                "owner": None,
                "tick_lower": None,
                "tick_upper": None,
                "liquidity_amount": None,
                "seq": i + 1,
                "price_pool": price,
                "price_reference": price,
                "regime": "sideways",
                "base_fee_per_gas": str(20 * 10**9),
                "priority_fee_p50_wei": str(10**9),
                "fee_growth_global_0_x128": "0",
                "fee_growth_global_1_x128": "0",
                "fee_growth_source": "exact",
            }
        )
    tape = pl.from_arrow(
        pa.table(
            {key: [row[key] for row in tape_rows] for key in tape_rows[0]},
            schema=EVENT_TAPE_SCHEMA,
        )
    )

    reference = pl.from_arrow(
        pa.table(
            {
                "open_time": [t - timedelta(minutes=1) for t in times],
                "close_time": times,
                "open": prices,
                "high": [p * 1.0001 for p in prices],
                "low": [p * 0.9999 for p in prices],
                "close": prices,
                "volume_base": [1.0] * _PARITY_STEPS,
                "quote_volume": [3000.0] * _PARITY_STEPS,
                "trades": [1] * _PARITY_STEPS,
                "symbol": ["ETHUSDC"] * _PARITY_STEPS,
                "is_gap_filled": [False] * _PARITY_STEPS,
            },
            schema=REFERENCE_SCHEMA,
        )
    )
    regimes = pl.from_arrow(
        pa.table(
            {
                "timestamp": times,
                "sigma_rv": [0.5] * _PARITY_STEPS,
                "mu": [0.0] * _PARITY_STEPS,
                "regime": ["sideways"] * _PARITY_STEPS,
                "window_complete": [True] * _PARITY_STEPS,
                "symbol": ["ETHUSDC"] * _PARITY_STEPS,
            },
            schema=REGIME_SCHEMA,
        )
    )
    gas = pl.from_arrow(
        pa.table(
            {
                "block_number": list(range(_PARITY_STEPS + 1)),
                "base_fee_per_gas": [str(20 * 10**9)] * (_PARITY_STEPS + 1),
                "gas_used": [15_000_000] * (_PARITY_STEPS + 1),
                "gas_limit": [30_000_000] * (_PARITY_STEPS + 1),
                "priority_fee_p50_wei": [str(10**9)] * (_PARITY_STEPS + 1),
                "priority_fee_p90_wei": [str(3 * 10**9)] * (_PARITY_STEPS + 1),
            },
            schema=GAS_SCHEMA,
        )
    )

    active_start = times[0]
    active_end = times[-1]
    return MarketView(
        train=tape,
        eval=None,
        reference=reference,
        regimes=regimes,
        gas=gas,
        train_start_utc=active_start,
        train_end_utc=active_end,
        eval_start_utc=active_end + timedelta(minutes=1),
        eval_end_utc=active_end + timedelta(days=1),
        is_train=True,
        episode_config=EpisodeConfig(duration_days=1, step_minutes=10),
        pool=None,
    )


def _parity_config() -> SimConfig:
    return SimConfig(episode=EpisodeConfig(duration_days=1, step_minutes=10), seed=0)


def _parity_env(
    view: MarketView,
    config: SimConfig,
    gas_model: object,
    slippage_model: object,
) -> LpEnvironment:
    entry_tick = int(
        price_to_tick(
            Decimal(str(_PARITY_ENTRY_PRICE)), DEFAULT_DEC0, DEFAULT_DEC1
        )
    )
    # The view's opening row is the ``mint`` that establishes the external
    # liquidity, so the pre-window base is zero and the engine builds it from
    # the tape — exactly as the backtester's empty-lattice fallback does
    # (ADR-019). The background ticks carry no net liquidity.
    engine = PoolEngine(
        state=PoolState(
            sqrt_price=calc_sqrt_price_a(Tick(entry_tick)),
            tick=Tick(entry_tick),
            liquidity=0.0,
            fee_growth_global_0=0.0,
            fee_growth_global_1=0.0,
            fee_tier_bps=3000,
            tick_spacing=60,
        ),
        ticks={
            Tick(196200 + step * 60): TickState(initialized=True)
            for step in range(-10, 11)
        },
    )
    process = ReplayPriceProcess(view, 0, view.step_count())
    # ``initial_width=2`` -> +/-120 ticks, matching the backtester's
    # ``DEFAULT_INITIAL_HALF_WIDTH_TICKS``; anything else deploys a different
    # initial position and parity is meaningless.
    return LpEnvironment(
        view,
        engine,
        process,
        gas_model,
        slippage_model,
        config,
        initial_width=2,
    )


class _ConstantPriceProcess:
    """A deterministic, non-replay price process for mode-guard tests."""

    def reset(self, rng: object) -> None:
        return None

    def step(self, rng: object) -> tuple[float, float, Tick]:
        from undertow.sim.prices.base import human_price_to_triplet

        return human_price_to_triplet(_PARITY_ENTRY_PRICE)

    @property
    def mode(self) -> str:
        return "calibrated"


def _calibrated_probe_env(
    view: MarketView, config: SimConfig
) -> LpEnvironment:
    """A calibrated-mode env used to exercise the non-replay guard paths."""
    engine = PoolEngine(
        state=PoolState(
            sqrt_price=calc_sqrt_price_a(Tick(196242)),
            tick=Tick(196242),
            liquidity=_PARITY_BASE_LIQUIDITY,
            fee_growth_global_0=0.0,
            fee_growth_global_1=0.0,
            fee_tier_bps=3000,
            tick_spacing=60,
        ),
        ticks={
            Tick(196200 + step * 60): TickState(initialized=True)
            for step in range(-10, 11)
        },
    )
    return LpEnvironment(
        view, engine, _ConstantPriceProcess(), object(), object(), config
    )


@pytest.fixture
def parity_market_view() -> MarketView:
    return _parity_view()


@pytest.fixture
def parity_config() -> SimConfig:
    return _parity_config()


@pytest.fixture
def parity_ledger(parity_market_view: MarketView, parity_config: SimConfig) -> BacktestLedger:
    return run_backtest(HODLPolicy(), parity_market_view, parity_config)


@pytest.fixture
def parity_env(
    parity_market_view: MarketView,
    parity_config: SimConfig,
    mock_gas_model: object,
    mock_slippage_model: object,
) -> LpEnvironment:
    return _parity_env(
        parity_market_view, parity_config, mock_gas_model, mock_slippage_model
    )


@pytest.fixture
def rebalance_market_view() -> MarketView:
    return _parity_view(_REBALANCE_PRICES)


@pytest.fixture
def rebalance_policy() -> TauResetPolicy:
    return TauResetPolicy(120)


@pytest.fixture
def rebalance_ledger(
    rebalance_market_view: MarketView,
    parity_config: SimConfig,
    rebalance_policy: TauResetPolicy,
) -> BacktestLedger:
    return run_backtest(rebalance_policy, rebalance_market_view, parity_config)


@pytest.fixture
def rebalance_env(
    rebalance_market_view: MarketView,
    parity_config: SimConfig,
) -> LpEnvironment:
    # Use the same friction models the backtester builds internally (the tape's
    # per-block base fee through ``ReplayGasModel`` and the pinned proportional
    # slippage model) so the gas/slippage series are directly comparable.
    return _parity_env(
        rebalance_market_view,
        parity_config,
        ReplayGasModel(rebalance_market_view, parity_config.gas),
        ProportionalSlippageModel(),
    )


@pytest.fixture
def tiny_backtest_fn(
    tiny_market_view: MarketView,
) -> Callable[[MarketView], BacktestLedger]:
    config = SimConfig()
    return lambda view: run_backtest(HODLPolicy(), view, config)


# ---------------------------------------------------------------------------
# ParityReport
# ---------------------------------------------------------------------------


class TestParityReport:
    def test_zero_drift_passes(self) -> None:
        report = ParityReport(
            fee_drift_mean=0.0,
            fee_drift_std=0.0,
            il_drift_mean=0.0,
            il_drift_std=0.0,
            pnl_drift_mean=0.0,
            pnl_drift_std=0.0,
            tolerance=1e-3,
        )
        assert report.passed is True

    def test_drift_exceeding_tolerance_fails(self) -> None:
        report = ParityReport(
            fee_drift_mean=0.0,
            fee_drift_std=0.0,
            il_drift_mean=2e-2,
            il_drift_std=0.0,
            pnl_drift_mean=0.0,
            pnl_drift_std=0.0,
            tolerance=1e-3,
        )
        assert report.passed is False

    def test_mean_at_tolerance_passes(self) -> None:
        on_boundary = ParityReport(
            fee_drift_mean=1e-3,
            fee_drift_std=0.0,
            il_drift_mean=-1e-3,
            il_drift_std=0.0,
            pnl_drift_mean=0.0,
            pnl_drift_std=0.0,
            tolerance=1e-3,
        )
        assert on_boundary.passed is True

    def test_passed_is_not_a_constructor_argument(self) -> None:
        # ``passed`` is a real frozen field (contract shape) but derived, so it
        # cannot be supplied by a caller trying to force a pass.
        with pytest.raises(TypeError):
            ParityReport(  # type: ignore[call-arg]
                fee_drift_mean=0.0,
                fee_drift_std=0.0,
                il_drift_mean=0.0,
                il_drift_std=0.0,
                pnl_drift_mean=0.0,
                pnl_drift_std=0.0,
                tolerance=1e-3,
                passed=True,
            )


# ---------------------------------------------------------------------------
# run_parity_check
# ---------------------------------------------------------------------------


class TestRunParityCheck:
    def test_initial_position_matches_backtester_bounds(
        self, parity_env: LpEnvironment
    ) -> None:
        """Guard the parity premise: both artifacts start from the same range."""
        parity_env.reset(seed=0)
        position = parity_env.pool_engine.positions[parity_env.position_id]
        # The backtester's deployment is +/-120 ticks snapped to the grid
        # (``DEFAULT_INITIAL_HALF_WIDTH_TICKS``); ``initial_width=2`` is the
        # matching 10-tick-spacing value at 60 spacing.
        assert position.tick_lower == 196080
        assert position.tick_upper == 196380

    def test_parity_passes_on_consistent_replay(
        self, parity_env: LpEnvironment, parity_ledger: BacktestLedger
    ) -> None:
        report = run_parity_check(
            parity_env, parity_ledger, num_episodes=2, seeds=(0, 1)
        )
        assert isinstance(report, ParityReport)
        assert report.tolerance == DEFAULT_PARITY_TOLERANCE
        assert report.justification
        assert abs(report.fee_drift_mean) < report.tolerance
        assert abs(report.il_drift_mean) < report.tolerance
        assert abs(report.pnl_drift_mean) < report.tolerance
        assert report.passed is True

    def test_measured_fee_drift_is_far_below_tolerance(
        self, parity_env: LpEnvironment, parity_ledger: BacktestLedger
    ) -> None:
        report = run_parity_check(
            parity_env, parity_ledger, num_episodes=2, seeds=(0, 1)
        )
        # The binding floor is the backtester's integer raw-unit truncation
        # (~1e-6 USDC/step for token0); the marginal-agent share term is a
        # secondary, scenario-dependent addition.  Both are well under 1e-4.
        assert abs(report.fee_drift_mean) < 1e-4
        assert abs(report.pnl_drift_mean) < 1e-4

    def test_parity_is_deterministic(
        self, parity_env: LpEnvironment, parity_ledger: BacktestLedger
    ) -> None:
        first = run_parity_check(
            parity_env, parity_ledger, num_episodes=2, seeds=(0, 1)
        )
        second = run_parity_check(
            parity_env, parity_ledger, num_episodes=2, seeds=(0, 1)
        )
        assert first == second

    def test_detector_flags_injected_fee_mismatch(
        self, parity_env: LpEnvironment, parity_ledger: BacktestLedger
    ) -> None:
        """A deliberate $1/step backtest fee mismatch must FAIL parity."""
        corrupted = replace(
            parity_ledger,
            equity_curve=parity_ledger.equity_curve.with_columns(
                (pl.col("fees") + 1.0).alias("fees")
            ),
        )
        report = run_parity_check(
            parity_env, corrupted, num_episodes=2, seeds=(0, 1)
        )
        assert report.fee_drift_mean < -0.9
        assert report.passed is False

    def test_detector_flags_injected_il_mismatch(
        self, parity_env: LpEnvironment, parity_ledger: BacktestLedger
    ) -> None:
        corrupted = replace(
            parity_ledger,
            equity_curve=parity_ledger.equity_curve.with_columns(
                (pl.col("il") + 100.0).alias("il")
            ),
        )
        report = run_parity_check(
            parity_env, corrupted, num_episodes=2, seeds=(0, 1)
        )
        assert abs(report.il_drift_mean) > 10.0
        assert report.passed is False

    def test_justification_names_the_truncation_floor(
        self, parity_env: LpEnvironment, parity_ledger: BacktestLedger
    ) -> None:
        report = run_parity_check(
            parity_env, parity_ledger, num_episodes=1, seeds=(0,)
        )
        assert "raw-unit truncation" in report.justification
        assert "2**-53" in report.justification
        assert "1e-06" in report.justification or "1e-6" in report.justification

    def test_justification_reflects_overridden_tolerance(
        self, parity_env: LpEnvironment, parity_ledger: BacktestLedger
    ) -> None:
        report = run_parity_check(
            parity_env, parity_ledger, num_episodes=1, seeds=(0,), tolerance=1e-4
        )
        assert report.tolerance == 1e-4
        assert "0.0001" in report.justification

    def test_rejects_ledger_with_different_policy(
        self, parity_env: LpEnvironment, parity_ledger: BacktestLedger
    ) -> None:
        wrong_policy = replace(parity_ledger, policy_name="passive_narrow")
        with pytest.raises(ParityError):
            run_parity_check(parity_env, wrong_policy, num_episodes=1, seeds=(0,))

    def test_rejects_ledger_with_different_config_hash(
        self, parity_env: LpEnvironment, parity_ledger: BacktestLedger
    ) -> None:
        wrong_config = replace(parity_ledger, config_hash="deadbeefdeadbeef")
        with pytest.raises(ParityError):
            run_parity_check(parity_env, wrong_config, num_episodes=1, seeds=(0,))

    def test_different_window_ledger_triggers_fallback(
        self,
        parity_env: LpEnvironment,
        parity_market_view: MarketView,
        parity_config: SimConfig,
    ) -> None:
        # A ledger over a sub-window cannot be reused; the harness must re-run
        # the backtester over the simulator's window and still pass.
        short = run_backtest(
            HODLPolicy(), parity_market_view.slice(0, 4), parity_config
        )
        assert short.n_steps == 4
        report = run_parity_check(parity_env, short, num_episodes=1, seeds=(0,))
        assert report.passed is True

    def test_different_window_non_hold_ledger_is_ignored(
        self,
        parity_env: LpEnvironment,
        parity_market_view: MarketView,
        parity_config: SimConfig,
    ) -> None:
        # The provenance guard applies only to a same-window reuse; a ledger
        # covering a different window is ignored and the backtester is re-run.
        short = run_backtest(
            HODLPolicy(), parity_market_view.slice(0, 4), parity_config
        )
        wrong = replace(short, policy_name="passive_narrow")
        report = run_parity_check(parity_env, wrong, num_episodes=1, seeds=(0,))
        assert report.passed is True

    def test_rejects_calibrated_env(
        self,
        parity_market_view: MarketView,
        parity_config: SimConfig,
        parity_ledger: BacktestLedger,
    ) -> None:
        env = _calibrated_probe_env(parity_market_view, parity_config)
        with pytest.raises(ParityError):
            run_parity_check(env, parity_ledger, num_episodes=1, seeds=(0,))

    def test_rejects_empty_seed_list(
        self, parity_env: LpEnvironment, parity_ledger: BacktestLedger
    ) -> None:
        with pytest.raises(ParityError):
            run_parity_check(parity_env, parity_ledger, num_episodes=0, seeds=())

    def test_rejects_non_positive_tolerance(
        self, parity_env: LpEnvironment, parity_ledger: BacktestLedger
    ) -> None:
        with pytest.raises(ParityError):
            run_parity_check(
                parity_env, parity_ledger, num_episodes=1, seeds=(0,), tolerance=0.0
            )


# ---------------------------------------------------------------------------
# S22 — rebalancing parity
# ---------------------------------------------------------------------------


class TestRebalancingParity:
    """The S14 harness driven by a live policy, not just the no-op HOLD.

    Revision 2 (ADR-031/S22): the fixture's first bar moves and a mid-episode
    jump forces a genuine recentre, the guard compares the action **executed in
    each numeric bucket**, and ``passed`` checks the per-step ``max|drift|`` so
    a one-step execution shift cannot fake a pass.
    """

    def test_policy_none_is_hold_with_zero_gas_and_slippage(
        self, parity_env: LpEnvironment, parity_ledger: BacktestLedger
    ) -> None:
        default = run_parity_check(
            parity_env, parity_ledger, num_episodes=2, seeds=(0, 1)
        )
        explicit_hold = run_parity_check(
            parity_env,
            parity_ledger,
            num_episodes=2,
            seeds=(0, 1),
            policy=HODLPolicy(),
        )
        assert default == explicit_hold
        assert default.passed is True
        assert default.gas_drift_mean == 0.0
        assert default.gas_drift_std == 0.0
        assert default.gas_drift_max == 0.0
        assert default.slippage_drift_mean == 0.0
        assert default.slippage_drift_std == 0.0
        assert default.slippage_drift_max == 0.0

    def test_rebalancing_parity_passes_over_three_seeds(
        self,
        rebalance_env: LpEnvironment,
        rebalance_ledger: BacktestLedger,
        rebalance_policy: TauResetPolicy,
    ) -> None:
        report = run_parity_check(
            rebalance_env,
            rebalance_ledger,
            num_episodes=3,
            seeds=(0, 1, 2),
            policy=rebalance_policy,
        )
        assert report.passed is True, report
        assert abs(report.fee_drift_mean) < report.tolerance
        assert abs(report.il_drift_mean) < report.tolerance
        assert abs(report.pnl_drift_mean) < report.tolerance
        # Both worlds call the same friction models on the same inputs, so the
        # gas/slippage drift is exactly zero -- including the per-step maximum,
        # which is what catches a one-step execution shift (ADR-031/S22).
        assert report.gas_drift_mean == 0.0
        assert report.gas_drift_std == 0.0
        assert report.gas_drift_max == 0.0
        assert report.slippage_drift_mean == 0.0
        assert report.slippage_drift_std == 0.0
        assert report.slippage_drift_max == 0.0

    def test_moving_path_triggers_a_genuine_mid_episode_rebalance(
        self,
        rebalance_env: LpEnvironment,
        rebalance_ledger: BacktestLedger,
        rebalance_policy: TauResetPolicy,
    ) -> None:
        """S22 T5: a real mid-episode recentre on the moving fixture passes.

        The first bar moves (so the old one-bar-late simulator cannot agree by
        construction) and the 00:40 step change triggers a rebalance with a
        non-zero swapped imbalance, exercised end-to-end with max|drift| < 1e-6
        gas/slippage agreement.
        """
        report = run_parity_check(
            rebalance_env,
            rebalance_ledger,
            num_episodes=3,
            seeds=(0, 1, 2),
            policy=rebalance_policy,
        )
        assert report.passed is True, report
        assert report.gas_drift_max < 1e-6
        assert report.slippage_drift_max < 1e-6

        # The backtester's decision log holds a rebalance after step 0 with a
        # material slippage charge (a real swapped imbalance, not a no-op).
        log = rebalance_ledger.decision_log
        first_seq = int(log["step"][0])
        mid_rebalances = log.filter(
            (pl.col("action_type") == "rebalance") & (pl.col("step") > first_seq)
        )
        assert mid_rebalances.height == 1, log
        mid_seq = int(mid_rebalances["step"][0])
        assert float(mid_rebalances["slippage_paid"][0]) < -1.0

        # The simulator, driven exactly as the harness drives it, executes the
        # same mid-episode rebalance out of the same bucket.
        decisions = self._sim_decisions(rebalance_env, rebalance_policy)
        bucket = list(log["step"]).index(mid_seq)
        step_number, action_type, slippage = decisions[bucket]
        assert step_number > 1  # not the step-0 deployment
        assert action_type == "rebalance"
        assert slippage < -1.0

    def test_rebalance_actually_executes(
        self,
        rebalance_env: LpEnvironment,
        rebalance_ledger: BacktestLedger,
        rebalance_policy: TauResetPolicy,
    ) -> None:
        # The backtester side records the executed rebalances...
        backtest_rebalances = int(
            (rebalance_ledger.decision_log["action_type"] == "rebalance").sum()
        )
        assert backtest_rebalances >= 2

        # ... and the simulator side, driven exactly as the harness drives it,
        # executes at least one too, so this cannot pass vacuously.
        decisions = self._sim_decisions(rebalance_env, rebalance_policy)
        assert sum(1 for _, kind, _ in decisions if kind == "rebalance") >= 2

    @staticmethod
    def _sim_decisions(
        env: LpEnvironment, policy: TauResetPolicy
    ) -> list[tuple[int, str, float]]:
        """Drive one episode as the harness does; return executed decisions."""
        policy.reset()
        rng = np.random.default_rng(0)
        vector, _info = env.reset(seed=0)
        out: list[tuple[int, str, float]] = []
        for _ in range(int(env.episode_length)):
            observation = observation_from_vector(vector.tolist())
            action = policy.act(observation, rng)
            index = action_to_index(env, action)
            vector, _reward, terminated, truncated, info = env.step(index)
            raw = info["reward_breakdown_raw"]
            out.append(
                (
                    int(info["step_in_episode"]),
                    str(info["action_type"]),
                    float(raw["slippage"]),
                )
            )
            if terminated or truncated:
                break
        return out

    def test_divergent_executed_actions_raise_naming_the_bucket(
        self,
        rebalance_env: LpEnvironment,
        rebalance_ledger: BacktestLedger,
        rebalance_policy: TauResetPolicy,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """S22 T7: the guard compares executed actions per bucket, by index.

        A live policy always re-runs the backtester, so the corruption is
        injected into the re-run (monkeypatched ``run_backtest``) rather than
        into the supplied ledger.
        """
        log = rebalance_ledger.decision_log
        first_seq = int(log["step"][0])
        mid_seq = int(
            log.filter(
                (pl.col("action_type") == "rebalance") & (pl.col("step") > first_seq)
            )["step"][0]
        )
        bucket = list(log["step"]).index(mid_seq)

        real = parity_module.run_backtest

        def corrupted(*args: object, **kwargs: object) -> BacktestLedger:
            ledger = real(*args, **kwargs)  # type: ignore[arg-type]
            return replace(
                ledger,
                decision_log=ledger.decision_log.with_columns(
                    pl.when(pl.col("step") == mid_seq)
                    .then(pl.lit("hold"))
                    .otherwise(pl.col("action_type"))
                    .alias("action_type")
                ),
            )

        monkeypatch.setattr(parity_module, "run_backtest", corrupted)
        with pytest.raises(ParityError, match=f"bucket {bucket}"):
            run_parity_check(
                rebalance_env,
                rebalance_ledger,
                num_episodes=1,
                seeds=(0,),
                policy=rebalance_policy,
            )

    def test_live_policy_ignores_the_supplied_ledger(
        self,
        rebalance_env: LpEnvironment,
        rebalance_ledger: BacktestLedger,
        rebalance_policy: TauResetPolicy,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """S22 provenance: a live policy always re-runs the backtester.

        A same-window ledger whose policy name/config hash would have been
        trusted by the first attempt must be ignored, not reused; the re-run is
        the only source of the backtester series.
        """
        mismatched = replace(
            rebalance_ledger, policy_name="hodl", config_hash="deadbeefdeadbeef"
        )
        calls: list[str] = []
        real = parity_module.run_backtest

        def recording(policy: object, view: MarketView, config: SimConfig) -> BacktestLedger:
            calls.append(policy.name)  # type: ignore[attr-defined]
            return real(policy, view, config)  # type: ignore[arg-type]

        monkeypatch.setattr(parity_module, "run_backtest", recording)
        report = run_parity_check(
            rebalance_env,
            mismatched,
            num_episodes=1,
            seeds=(0,),
            policy=rebalance_policy,
        )
        assert calls == [rebalance_policy.name]
        assert report.passed is True, report

    def test_one_step_shift_fails_even_though_the_mean_cancels(
        self,
        rebalance_env: LpEnvironment,
        rebalance_ledger: BacktestLedger,
        rebalance_policy: TauResetPolicy,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """S22 T6: a one-bucket execution shift must FAIL, not false-pass.

        Rolling the backtester's gas/slippage series one bucket moves the spike
        to the neighbouring step; the means cancel to ~0 (the first attempt's
        ``passed`` compared means only and returned True) while the per-step
        maxima expose the shift.
        """
        real = parity_module._per_step_backtest

        def shifted(*args: object, **kwargs: object) -> tuple[
            list[float], list[float], list[float], list[float], list[float]
        ]:
            fees, il, pnl, gas, slippage = real(*args, **kwargs)  # type: ignore[arg-type]
            return fees, il, pnl, gas[-1:] + gas[:-1], slippage[-1:] + slippage[:-1]

        monkeypatch.setattr(parity_module, "_per_step_backtest", shifted)
        report = run_parity_check(
            rebalance_env,
            rebalance_ledger,
            num_episodes=1,
            seeds=(0,),
            policy=rebalance_policy,
        )
        assert report.passed is False
        # The mean genuinely cancels: it is the per-step maximum that fails.
        assert abs(report.gas_drift_mean) <= report.tolerance
        assert abs(report.slippage_drift_mean) <= report.tolerance
        assert report.gas_drift_max > report.tolerance
        assert report.slippage_drift_max > report.tolerance


class TestHedgeParity:
    """S20: the hedge overlay is like-for-like across the two worlds."""

    def test_hedge_parity_passes_with_zero_drift(self) -> None:
        view = _parity_view(_REBALANCE_PRICES)
        settlement = _PARITY_START + timedelta(minutes=30)
        funding = pl.DataFrame(
            {
                "funding_time": [settlement],
                "symbol": ["ETHUSDC"],
                "funding_rate": [0.001],
                "mark_price": [None],
                "is_gap_filled": [False],
            }
        )
        view = replace(view, funding=funding)
        config = replace(
            _parity_config(), hedge=HedgeConfig(enabled=True, ratio=1.0)
        )
        policy = TauResetPolicy(120)
        env = _parity_env(
            view, config, ReplayGasModel(view, config.gas), ProportionalSlippageModel()
        )
        ledger = run_backtest(policy, view, config)

        # The fixture's overlay actually traded and accrued funding, so the
        # zero-drift assertion below is not vacuous.
        assert ledger.hedge_curve is not None
        assert float(ledger.hedge_curve["hedge_fee"].sum()) < 0.0
        assert float(ledger.hedge_curve["hedge_funding"].sum()) > 0.0

        report = run_parity_check(
            env,
            ledger,
            num_episodes=3,
            seeds=(0, 1, 2),
            policy=policy,
        )
        assert report.passed is True, report
        for attr in (
            "hedge_pnl_drift_mean",
            "hedge_funding_drift_mean",
            "hedge_fee_drift_mean",
            "hedge_pnl_drift_max",
            "hedge_funding_drift_max",
            "hedge_fee_drift_max",
        ):
            assert abs(getattr(report, attr)) < report.tolerance, (attr, report)

    def test_hedge_parity_passes_under_reward_hedge_ablation(self) -> None:
        """RQ1: ``[hedge] enabled=true`` + ``[reward] hedge_enabled=false``
        must still pass with zero hedge drift.

        This is the S20 parity regression: the simulator's gated
        ``reward_breakdown_raw`` zeroes the hedge terms under the ablation while
        the backtester's ledger keeps the full magnitudes, so a harness that
        read the gated breakdown reported every step as a full-magnitude hedge
        drift and ``passed=False``.  The harness now reads the ungated
        ``reward_breakdown_economics`` payload.
        """
        view = _parity_view(_REBALANCE_PRICES)
        settlement = _PARITY_START + timedelta(minutes=30)
        funding = pl.DataFrame(
            {
                "funding_time": [settlement],
                "symbol": ["ETHUSDC"],
                "funding_rate": [0.001],
                "mark_price": [None],
                "is_gap_filled": [False],
            }
        )
        view = replace(view, funding=funding)
        config = replace(
            _parity_config(),
            hedge=HedgeConfig(enabled=True, ratio=1.0),
            reward=RewardConfig(hedge_enabled=False),
        )
        policy = TauResetPolicy(120)
        env = _parity_env(
            view, config, ReplayGasModel(view, config.gas), ProportionalSlippageModel()
        )
        ledger = run_backtest(policy, view, config)

        # Non-vacuous: the ledger's overlay traded and accrued funding, and the
        # simulator's gated breakdown (the old harness source) reads zero.
        assert ledger.hedge_curve is not None
        assert float(ledger.hedge_curve["hedge_fee"].sum()) < 0.0
        assert float(ledger.hedge_curve["hedge_funding"].sum()) > 0.0
        env.reset(seed=0)
        _, _, _, _, gated_info = env.step(0)
        assert gated_info["reward_breakdown_raw"]["hedge_fee"] == 0.0
        assert gated_info["reward_breakdown_economics"]["hedge_fee"] != 0.0

        report = run_parity_check(
            env,
            ledger,
            num_episodes=3,
            seeds=(0, 1, 2),
            policy=policy,
        )
        assert report.passed is True, report
        for attr in (
            "hedge_pnl_drift_mean",
            "hedge_funding_drift_mean",
            "hedge_fee_drift_mean",
            "hedge_pnl_drift_max",
            "hedge_funding_drift_max",
            "hedge_fee_drift_max",
        ):
            assert abs(getattr(report, attr)) < report.tolerance, (attr, report)

    def test_parity_passes_under_gas_and_slippage_ablation(self) -> None:
        """The same economics fix covers the other gated friction terms.

        With ``gas_enabled=false`` and ``slippage_enabled=false`` the gated
        breakdown reads zero for a decision that genuinely paid both, so the
        pre-fix harness compared a zero series against the ledger's full costs
        and failed; reading the economics payload keeps parity exact.
        """
        view = _parity_view(_REBALANCE_PRICES)
        config = replace(
            _parity_config(),
            reward=RewardConfig(gas_enabled=False, slippage_enabled=False),
        )
        policy = TauResetPolicy(120)
        env = _parity_env(
            view, config, ReplayGasModel(view, config.gas), ProportionalSlippageModel()
        )
        ledger = run_backtest(policy, view, config)

        # Non-vacuous: the executed decisions paid material gas and slippage.
        assert float(ledger.decision_log["gas_paid"].sum()) < 0.0
        assert float(ledger.decision_log["slippage_paid"].sum()) < -1.0

        report = run_parity_check(
            env,
            ledger,
            num_episodes=3,
            seeds=(0, 1, 2),
            policy=policy,
        )
        assert report.passed is True, report
        assert report.gas_drift_max == 0.0
        assert report.slippage_drift_max == 0.0
        assert report.gas_drift_mean == 0.0
        assert report.slippage_drift_mean == 0.0


class TestExecutedDecisionGuard:
    """The per-bucket executed-action guard's error branches (S22 T9).

    These exercise the ``policy=None`` HOLD path, which reuses the supplied
    ledger, so the corruptions land in the decision log the guard reads.
    """

    def test_rejects_a_decision_log_step_outside_the_window(
        self, parity_env: LpEnvironment, parity_ledger: BacktestLedger
    ) -> None:
        first_seq = int(parity_ledger.decision_log["step"][0])
        corrupted = replace(
            parity_ledger,
            decision_log=parity_ledger.decision_log.with_columns(
                pl.when(pl.col("step") == first_seq)
                .then(pl.lit(999_999))
                .otherwise(pl.col("step"))
                .alias("step")
            ),
        )
        with pytest.raises(ParityError, match="outside the episode window"):
            run_parity_check(parity_env, corrupted, num_episodes=1, seeds=(0,))

    def test_rejects_duplicate_decisions_in_one_bucket(
        self, parity_env: LpEnvironment, parity_ledger: BacktestLedger
    ) -> None:
        # The same decision logged twice at one timestamp buckets to one step.
        duplicated = replace(
            parity_ledger,
            decision_log=pl.concat(
                [parity_ledger.decision_log, parity_ledger.decision_log.head(1)]
            ),
        )
        with pytest.raises(ParityError, match="more than one executed decision"):
            run_parity_check(parity_env, duplicated, num_episodes=1, seeds=(0,))

    def test_rejects_a_bucket_with_no_backtester_decision(
        self, parity_env: LpEnvironment, parity_ledger: BacktestLedger
    ) -> None:
        first_seq = int(parity_ledger.decision_log["step"][0])
        dropped = replace(
            parity_ledger,
            decision_log=parity_ledger.decision_log.filter(
                pl.col("step") != first_seq
            ),
        )
        with pytest.raises(ParityError, match="no backtester decision in sim bucket 0"):
            run_parity_check(parity_env, dropped, num_episodes=1, seeds=(0,))


# ---------------------------------------------------------------------------
# run_lookahead_probes
# ---------------------------------------------------------------------------


def _run_probes(
    view: MarketView,
    env: LpEnvironment,
    backtest_fn: Callable[[MarketView], BacktestLedger],
) -> LookAheadReport:
    return run_lookahead_probes(view, env, backtest_fn)


class TestLookAheadProbes:
    def test_report_shape(self) -> None:
        import dataclasses

        assert [f.name for f in dataclasses.fields(LookAheadReport)] == [
            "probes_run",
            "violations",
            "details",
        ]
        report = LookAheadReport(probes_run=7, violations=0, details=["ok"])
        assert report.violations == 0

    def test_zero_violations_on_correct_system(
        self, tiny_market_view: MarketView, tiny_env: LpEnvironment,
        tiny_backtest_fn: Callable[[MarketView], BacktestLedger],
    ) -> None:
        report = _run_probes(tiny_market_view, tiny_env, tiny_backtest_fn)
        assert report.violations == 0, report.details
        assert report.probes_run >= 7

    def test_split_wall_probe_uses_the_wall(
        self, tiny_market_view: MarketView, tiny_env: LpEnvironment,
        tiny_backtest_fn: Callable[[MarketView], BacktestLedger],
    ) -> None:
        # The MarketView mechanism itself rejects the out-of-bounds read...
        with pytest.raises(LookAheadError):
            tiny_market_view.slice(0, tiny_market_view.step_count() + 5)
        # ... and the probe reports a pass line for it.
        report = _run_probes(tiny_market_view, tiny_env, tiny_backtest_fn)
        assert any(
            detail.startswith("PASS split_wall") for detail in report.details
        )

    def test_probe_detects_disabled_split_wall(
        self, tiny_market_view: MarketView, tiny_env: LpEnvironment,
        tiny_backtest_fn: Callable[[MarketView], BacktestLedger],
    ) -> None:
        class _NoWallView(MarketView):
            """A real ``MarketView`` whose ``slice`` silently ignores the wall."""

            def slice(self, start_seq: int, end_seq: int) -> MarketView:
                try:
                    return super().slice(start_seq, end_seq)
                except LookAheadError:
                    return self

        inner = tiny_market_view
        no_wall = _NoWallView(
            train=inner.train,
            eval=inner.eval,
            reference=inner.reference,
            regimes=inner.regimes,
            gas=inner.gas,
            train_start_utc=inner.train_start_utc,
            train_end_utc=inner.train_end_utc,
            eval_start_utc=inner.eval_start_utc,
            eval_end_utc=inner.eval_end_utc,
            is_train=inner.is_train,
            episode_config=inner.episode_config,
            pool=inner.pool,
        )
        report = _run_probes(no_wall, tiny_env, tiny_backtest_fn)
        assert report.violations > 0
        assert any(
            "split_wall" in detail
            for detail in report.details
            if detail.startswith("VIOLATION")
        )

    def test_probe_detects_injected_observation_leak(
        self, tiny_market_view: MarketView, tiny_env: LpEnvironment,
        tiny_backtest_fn: Callable[[MarketView], BacktestLedger],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Break the wall inside the observation builder: use every reference bar
        # regardless of the decision time.  Future mutations then leak backwards.
        def leaky(self: ObservationBuilder, at_time: object) -> np.ndarray:
            closes = self.market_view.reference["close"].to_numpy().astype(float)
            return closes

        monkeypatch.setattr(ObservationBuilder, "_reference_closes_through", leaky)
        report = _run_probes(tiny_market_view, tiny_env, tiny_backtest_fn)
        assert report.violations > 0
        assert any(
            "observation" in detail or "sentinel" in detail
            for detail in report.details
            if detail.startswith("VIOLATION")
        )

    def test_probe_detects_injected_backtest_leak(
        self, tiny_market_view: MarketView, tiny_env: LpEnvironment,
    ) -> None:
        config = SimConfig()

        def leaky_backtest(view: MarketView) -> BacktestLedger:
            ledger = run_backtest(HODLPolicy(), view, config)
            # Leak the future: encode the view's full length into every decision.
            height = int(view.step_count())
            decision_log = ledger.decision_log.with_columns(
                pl.lit(height, dtype=pl.Int64).alias("center_offset")
            )
            return replace(ledger, decision_log=decision_log)

        report = _run_probes(tiny_market_view, tiny_env, leaky_backtest)
        assert report.violations > 0
        assert any(
            detail.startswith("VIOLATION backtester")
            for detail in report.details
        )

    def test_details_has_one_line_per_probe(
        self, tiny_market_view: MarketView, tiny_env: LpEnvironment,
        tiny_backtest_fn: Callable[[MarketView], BacktestLedger],
    ) -> None:
        report = _run_probes(tiny_market_view, tiny_env, tiny_backtest_fn)
        assert len(report.details) == report.probes_run

    def test_rejects_non_callable_backtest_fn(
        self, tiny_market_view: MarketView, tiny_env: LpEnvironment
    ) -> None:
        with pytest.raises(TypeError):
            run_lookahead_probes(tiny_market_view, tiny_env, None)  # type: ignore[arg-type]

    def test_calibrated_env_skips_composed_probe(
        self,
        parity_market_view: MarketView,
        parity_config: SimConfig,
        parity_ledger: BacktestLedger,
    ) -> None:
        env = _calibrated_probe_env(parity_market_view, parity_config)
        config = parity_config

        def backtest_fn(view: MarketView) -> BacktestLedger:
            return run_backtest(HODLPolicy(), view, config)

        report = run_lookahead_probes(parity_market_view, env, backtest_fn)
        assert report.violations == 0, report.details
        assert any("skipped: calibrated" in detail for detail in report.details)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestParityIntegration:
    def test_hold_only_parity_contract_to_contract(
        self, parity_env: LpEnvironment, parity_ledger: BacktestLedger
    ) -> None:
        report = run_parity_check(
            parity_env, parity_ledger, num_episodes=5, seeds=(0, 1, 2, 3, 4)
        )
        assert report.passed is True, report
        assert report.tolerance == DEFAULT_PARITY_TOLERANCE

    def test_lookahead_probes_on_tiny_fixtures(
        self, tiny_market_view: MarketView, tiny_env: LpEnvironment,
        tiny_backtest_fn: Callable[[MarketView], BacktestLedger],
    ) -> None:
        report = _run_probes(tiny_market_view, tiny_env, tiny_backtest_fn)
        assert report.violations == 0, report.details
        assert report.probes_run >= 7
