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

from collections.abc import Callable
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
from undertow.sim.config import EpisodeConfig, SimConfig
from undertow.sim.core.pool import PoolEngine, PoolState, TickState
from undertow.sim.core.position import DEFAULT_DEC0, DEFAULT_DEC1, calc_sqrt_price_a
from undertow.sim.env.lp_env import LpEnvironment
from undertow.sim.env.observations import ObservationBuilder
from undertow.sim.marketview import MarketView
from undertow.sim.policies import HODLPolicy
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


def _sqrt_x96(price: float) -> str:
    return str(price_to_sqrt_price_x96(Decimal(str(price)), DEFAULT_DEC0, DEFAULT_DEC1))


def _parity_view() -> MarketView:
    """A schema-valid, internally consistent replay window for parity."""
    times = [_PARITY_START + _PARITY_STEP * i for i in range(_PARITY_STEPS)]
    # Human price rises -> raw tick falls -> a token0-in swap for every event.
    prices = [_PARITY_ENTRY_PRICE * (1.0 + 0.001 * i) for i in range(_PARITY_STEPS)]
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
    engine = PoolEngine(
        state=PoolState(
            sqrt_price=calc_sqrt_price_a(Tick(entry_tick)),
            tick=Tick(entry_tick),
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
