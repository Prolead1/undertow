"""S06 — price processes: deterministic replay + calibrated MRSJD.

Every test asserts real behaviour pinned by ``CONTRACTS.md`` §7 and ADR-009:
the replay feed is reproduced exactly, ``(price, sqrt_price, tick)`` are
mutually consistent over many steps, both processes are deterministic under an
explicit RNG, and the MRSJD fit produces a valid row-stochastic parameter set
from the train split only.

Orientation (ADR-009): ``sqrt_price`` is the **raw** Uniswap sqrt price
(``10**((dec1-dec0)/2) / sqrt(price)`` ≈ 18244 at ETH≈$3004) and ``tick`` comes
from ``undertow.data.price_to_tick`` — never ``log(price)/log(1.0001)``.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import polars as pl
import pytest

from undertow.data import Q96, price_to_tick, tick_to_price, tick_to_sqrt_price_x96
from undertow.sim.config import EpisodeConfig
from undertow.sim.core.position import calc_sqrt_price_a
from undertow.sim.marketview import MarketView
from undertow.sim.prices import (
    CalibratedPriceProcess,
    MRSJDParams,
    PriceProcess,
    ReplayPriceProcess,
)
from undertow.sim.prices.base import human_price_to_triplet
from undertow.sim.prices.regime_jump import (
    DEFAULT_DT,
    _asof_regime_labels,
    _decision_dt_years,
    _empirical_transition,
    _fit_from_regime_labels,
)
from undertow.sim.prices.replay import _resample_decision_bars
from undertow.sim.types import LookAheadError, MarketViewError, Tick

_TRAIN_START = datetime(2022, 1, 1, tzinfo=UTC)
_TRAIN_END = datetime(2022, 2, 1, tzinfo=UTC)
_EVAL_START = datetime(2024, 1, 1, tzinfo=UTC)
_EVAL_END = datetime(2024, 12, 31, tzinfo=UTC)
_DEC0 = 6
_DEC1 = 18
_DT = 1.0 / (6.0 * 24.0 * 365.25)


def _uniform_params(
    *,
    drift: float = 0.0,
    diffusion: float = 0.0,
    jump_intensity: float = 0.0,
    jump_loc: float = 0.0,
    jump_scale: float = 0.01,
    jump_dof: float = 4.0,
) -> MRSJDParams:
    """Four identical regimes so switching cannot affect the assertions."""
    return MRSJDParams(
        state_names=("bull", "bear", "sideways", "high_vol"),
        transition_matrix=np.full((4, 4), 0.25),
        drift=(drift,) * 4,
        diffusion=(diffusion,) * 4,
        jump_intensity=(jump_intensity,) * 4,
        jump_loc=(jump_loc,) * 4,
        jump_scale=(jump_scale,) * 4,
        jump_dof=(jump_dof,) * 4,
    )


def _assert_triplet_consistent(price: float, sqrt_price: float, tick: int) -> None:
    """Assert the three ADR-009 relations and the floor-tick sandwich."""
    assert sqrt_price == pytest.approx(10.0 ** ((_DEC1 - _DEC0) / 2.0) / math.sqrt(price))
    assert price == pytest.approx(10.0 ** (_DEC1 - _DEC0) / sqrt_price**2)
    assert tick == price_to_tick(Decimal(str(price)), _DEC0, _DEC1)
    assert calc_sqrt_price_a(Tick(int(tick))) <= sqrt_price
    assert sqrt_price <= calc_sqrt_price_a(Tick(int(tick) + 1))


def _dense_train_view(n_minutes: int = 4000, seed: int = 3) -> MarketView:
    """A synthetic 1-minute reference feed for exercising the MRSJD fit.

    The shared ``tiny_market_view`` fixture is only ~100 bars, far too few for a
    meaningful 4-state EM; this local view keeps the same schema and train
    orientation but supplies a denser path with injected jumps.
    """
    start = _TRAIN_START
    end = start + timedelta(minutes=n_minutes)
    times = [start + timedelta(minutes=i) for i in range(n_minutes)]
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.001, size=n_minutes)
    jump_idx = rng.integers(0, n_minutes, size=25)
    returns[jump_idx] += rng.standard_t(3, size=25) * 0.01
    closes = 3000.0 * np.exp(np.cumsum(returns))

    reference = pl.DataFrame(
        {
            "open_time": times,
            "close_time": [t + timedelta(minutes=1) for t in times],
            "open": closes,
            "high": closes * 1.001,
            "low": closes * 0.999,
            "close": closes,
            "volume_base": [1.0] * n_minutes,
            "quote_volume": [3000.0] * n_minutes,
            "trades": [10] * n_minutes,
            "symbol": ["ETHUSDC"] * n_minutes,
            "is_gap_filled": [False] * n_minutes,
        }
    )
    tape = pl.DataFrame(
        {
            "seq": list(range(0, n_minutes, 10)),
            "block_timestamp": [t for t in times[::10]],
        }
    )
    return MarketView(
        train=tape,
        eval=None,
        reference=reference,
        regimes=pl.DataFrame(),
        gas=pl.DataFrame(),
        train_start_utc=start,
        train_end_utc=end,
        eval_start_utc=_EVAL_START,
        eval_end_utc=_EVAL_END,
        is_train=True,
    )


def _labelled_train_view(n_minutes: int = 4000, seed: int = 7) -> MarketView:
    """A dense 1-minute feed with per-state returns and data-module labels.

    Each regime label owns a contiguous block, so the regime-anchored fit has
    plenty of returns for every state and its per-state statistics are exactly
    reproducible in the test (which the unsupervised-HMM fallback would not be).
    """
    start = _TRAIN_START
    times = [start + timedelta(minutes=i) for i in range(n_minutes)]
    labels = [
        ("bull", "bear", "sideways", "high_vol")[min(i // max(1, n_minutes // 4), 3)]
        for i in range(n_minutes)
    ]
    # Strong per-state separation so the ordering assertions are not noise-bound.
    specs = {
        "bull": (5.0e-5, 2.0e-4),
        "bear": (-5.0e-5, 2.0e-4),
        "sideways": (0.0, 1.0e-4),
        "high_vol": (1.0e-5, 8.0e-4),
    }
    rng = np.random.default_rng(seed)
    returns = np.array([rng.normal(specs[label][0], specs[label][1]) for label in labels])
    closes = 3000.0 * np.exp(np.cumsum(returns))
    reference = pl.DataFrame(
        {
            "open_time": times,
            "close_time": [t + timedelta(minutes=1) for t in times],
            "open": closes,
            "high": closes * 1.001,
            "low": closes * 0.999,
            "close": closes,
            "volume_base": [1.0] * n_minutes,
            "quote_volume": [3000.0] * n_minutes,
            "trades": [10] * n_minutes,
            "symbol": ["ETHUSDT"] * n_minutes,
            "is_gap_filled": [False] * n_minutes,
        }
    )
    regimes = pl.DataFrame(
        {
            "timestamp": times,
            "sigma_rv": [0.5] * n_minutes,
            "mu": [0.0] * n_minutes,
            "regime": labels,
            "window_complete": [True] * n_minutes,
            "symbol": ["ETHUSDT"] * n_minutes,
        }
    )
    tape = pl.DataFrame(
        {"seq": list(range(0, n_minutes, 10)), "block_timestamp": times[::10]}
    )
    return MarketView(
        train=tape,
        eval=None,
        reference=reference,
        regimes=regimes,
        gas=pl.DataFrame(),
        train_start_utc=start,
        train_end_utc=start + timedelta(minutes=n_minutes),
        eval_start_utc=_EVAL_START,
        eval_end_utc=_EVAL_END,
        is_train=True,
    )


def _eval_view(train_view: MarketView) -> MarketView:
    """Rebound the train tape as an eval view (same reference spans it)."""
    return MarketView(
        train=None,
        eval=train_view.train,
        reference=train_view.reference,
        regimes=train_view.regimes,
        gas=train_view.gas,
        train_start_utc=_TRAIN_START,
        train_end_utc=_TRAIN_END,
        eval_start_utc=_TRAIN_START,
        eval_end_utc=_TRAIN_END,
        is_train=False,
    )


# ---------------------------------------------------------------------------
# Golden values (CONTRACTS §19)
# ---------------------------------------------------------------------------


def test_golden_tick_196242_price_and_sqrt_price() -> None:
    price = float(tick_to_price(196242, _DEC0, _DEC1))
    assert price == pytest.approx(3004.0, rel=1e-3)
    price_out, sqrt_price, tick = human_price_to_triplet(price, _DEC0, _DEC1)
    assert price_out == price
    assert sqrt_price == pytest.approx(18244.0, rel=1e-3)
    # The raw sqrt price equals sqrt(1.0001**tick) = sqrt_price_x96 / Q96.
    assert sqrt_price == pytest.approx(tick_to_sqrt_price_x96(196242) / Q96, rel=1e-9)
    assert tick == 196242


# ---------------------------------------------------------------------------
# ReplayPriceProcess
# ---------------------------------------------------------------------------


class TestReplayPriceProcess:
    def test_replays_reference_prices_in_order(
        self, tiny_market_view: MarketView
    ) -> None:
        proc = ReplayPriceProcess(tiny_market_view, 0, 500)
        assert proc.n_bars > 0
        rng = np.random.default_rng(0)
        prices = [proc.step(rng)[0] for _ in range(proc.n_bars)]

        expected = []
        for close_time in proc.close_times:
            row = tiny_market_view.reference.filter(pl.col("close_time") == close_time)
            assert row.height == 1
            expected.append(float(row["close"][0]))
        assert prices == pytest.approx(expected)

    def test_reset_restarts_from_first_bar(self, tiny_market_view: MarketView) -> None:
        proc = ReplayPriceProcess(tiny_market_view, 0, 500)
        rng = np.random.default_rng(0)
        first = proc.step(rng)[0]
        for _ in range(proc.n_bars - 1):
            proc.step(rng)
        proc.reset(rng)
        assert proc.step(rng)[0] == first

    def test_raises_stop_iteration_past_last_bar(
        self, tiny_market_view: MarketView
    ) -> None:
        proc = ReplayPriceProcess(tiny_market_view, 0, 500)
        rng = np.random.default_rng(0)
        for _ in range(proc.n_bars):
            proc.step(rng)
        with pytest.raises(StopIteration):
            proc.step(rng)

    def test_triplets_are_mutually_consistent(
        self, tiny_market_view: MarketView
    ) -> None:
        proc = ReplayPriceProcess(tiny_market_view, 0, 800)
        rng = np.random.default_rng(1)
        for _ in range(proc.n_bars):
            price, sqrt_price, tick = proc.step(rng)
            _assert_triplet_consistent(price, sqrt_price, tick)

    def test_mode_is_replay(self, tiny_market_view: MarketView) -> None:
        proc = ReplayPriceProcess(tiny_market_view, 0, 100)
        assert proc.mode == "replay"

    def test_satisfies_price_process_protocol(
        self, tiny_market_view: MarketView
    ) -> None:
        assert isinstance(ReplayPriceProcess(tiny_market_view, 0, 100), PriceProcess)

    def test_deterministic_given_feed(self, tiny_market_view: MarketView) -> None:
        a = ReplayPriceProcess(tiny_market_view, 0, 300)
        b = ReplayPriceProcess(tiny_market_view, 0, 300)
        rng_a = np.random.default_rng(5)
        rng_b = np.random.default_rng(5)
        assert [a.step(rng_a) for _ in range(a.n_bars)] == [
            b.step(rng_b) for _ in range(b.n_bars)
        ]

    def test_rejects_window_past_split_wall(
        self, tiny_market_view: MarketView
    ) -> None:
        with pytest.raises(LookAheadError):
            ReplayPriceProcess(tiny_market_view, 0, 10**9)

    def test_replayed_bars_stay_inside_train_split(
        self, tiny_market_view: MarketView
    ) -> None:
        proc = ReplayPriceProcess(tiny_market_view, 0, 500)
        for close_time in proc.close_times:
            assert tiny_market_view.active_start_utc < close_time
            assert close_time <= tiny_market_view.active_end_utc

    def test_rejects_inverted_window(self, tiny_market_view: MarketView) -> None:
        with pytest.raises(MarketViewError, match="start_seq"):
            ReplayPriceProcess(tiny_market_view, 500, 100)


# ---------------------------------------------------------------------------
# Decision-cadence resampling (ADR-010): 1-minute feed -> step_minutes bars
# ---------------------------------------------------------------------------


def _minute_bars(n: int) -> pl.DataFrame:
    start = datetime(2022, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "open_time": [start + timedelta(minutes=i) for i in range(n)],
            "close_time": [start + timedelta(minutes=i, seconds=59) for i in range(n)],
            "close": [float(i) for i in range(n)],
        }
    )


class TestDecisionResample:
    def test_groups_1min_bars_onto_decision_buckets(self) -> None:
        out = _resample_decision_bars(_minute_bars(25), 10)
        # Buckets [0,10), [10,20), [20,30): the last bar in each.
        assert out.height == 3
        assert out["close"].to_list() == [9.0, 19.0, 24.0]
        assert out["close_time"].to_list() == [
            datetime(2022, 1, 1, 0, 9, 59, tzinfo=UTC),
            datetime(2022, 1, 1, 0, 19, 59, tzinfo=UTC),
            datetime(2022, 1, 1, 0, 24, 59, tzinfo=UTC),
        ]

    def test_step_minutes_one_is_identity(self) -> None:
        bars = _minute_bars(5)
        assert _resample_decision_bars(bars, 1).equals(bars)

    def test_preserves_columns(self) -> None:
        bars = _minute_bars(25).with_columns(pl.lit("ETHUSDT").alias("symbol"))
        out = _resample_decision_bars(bars, 10)
        assert out.columns == bars.columns

    def test_replay_decision_cadence_leaves_coarse_bars_alone(
        self, tiny_market_view: MarketView
    ) -> None:
        # The fixture's bars are hours apart: each is already its own 10-min bucket,
        # so the resample must not merge them away.
        one = ReplayPriceProcess(tiny_market_view, 0, 500, step_minutes=1)
        ten = ReplayPriceProcess(tiny_market_view, 0, 500, step_minutes=10)
        assert ten.n_bars == one.n_bars

    def test_replay_rejects_non_positive_step_minutes(self, tiny_market_view: MarketView) -> None:
        with pytest.raises(MarketViewError, match="step_minutes"):
            ReplayPriceProcess(tiny_market_view, 0, 100, step_minutes=0)

    def test_replay_aggregates_1min_feed_to_decision_cadence(
        self, minute_market_view: MarketView
    ) -> None:
        view = minute_market_view
        ep = EpisodeConfig(duration_days=1, step_minutes=10)
        start, end = view.build_episode(0, episode_config=ep)
        proc = ReplayPriceProcess(view, start, end, step_minutes=10)
        # ~144 decisions for a 1-day window (the tape's last row can fall short of
        # the wall by up to one row spacing, so allow a few); crucially nowhere
        # near the ~1440 raw 1-minute bars.
        assert 139 <= proc.n_bars <= 145
        deltas = [b - a for a, b in zip(proc.close_times, proc.close_times[1:], strict=False)]
        assert deltas and all(
            timedelta(minutes=9) < delta < timedelta(minutes=11) for delta in deltas
        )


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def test_human_price_to_triplet_rejects_non_positive_price() -> None:
    with pytest.raises(ValueError, match="positive"):
        human_price_to_triplet(0.0)
    with pytest.raises(ValueError, match="positive"):
        human_price_to_triplet(-1.0)
    with pytest.raises(ValueError, match="positive"):
        human_price_to_triplet(float("nan"))


def test_decision_dt_years_tracks_bar_spacing() -> None:
    start = datetime(2022, 1, 1, tzinfo=UTC)
    one_minute_bars = [start + timedelta(minutes=i) for i in range(20)]
    assert _decision_dt_years(one_minute_bars, 10) == pytest.approx(_DT, rel=1e-9)
    two_minute_bars = [start + timedelta(minutes=2 * i) for i in range(20)]
    assert _decision_dt_years(two_minute_bars, 10) == pytest.approx(
        2.0 * _DT, rel=1e-9
    )
    # A single bar cannot supply a spacing: fall back to the pinned default.
    assert _decision_dt_years([start], 10) == _DT


def test_default_dt_is_ten_minutes() -> None:
    # CONTRACTS §7's literal ``1 / (6 * 365.25)`` is 4 hours; the pinned 10-min
    # cadence (144 steps/day) is ``1 / (6 * 24 * 365.25)``.
    assert math.isclose(DEFAULT_DT, 1.0 / (6.0 * 24.0 * 365.25), rel_tol=1e-12)
    assert math.isclose(DEFAULT_DT, 10.0 / (365.25 * 24.0 * 60.0), rel_tol=1e-12)
    assert math.isclose(
        CalibratedPriceProcess(_uniform_params()).dt, DEFAULT_DT, rel_tol=1e-12
    )


# ---------------------------------------------------------------------------
# MRSJDParams validation
# ---------------------------------------------------------------------------


class TestMRSJDParamsValidation:
    def test_accepts_valid_params(self) -> None:
        params = _uniform_params(drift=0.1, diffusion=0.5, jump_intensity=4.0)
        assert params.n_states == 4

    def test_rejects_rows_not_summing_to_one(self) -> None:
        matrix = np.full((4, 4), 0.25)
        matrix[0, 0] = 0.5  # row 0 now sums to 1.25
        with pytest.raises(ValueError, match="sum to 1"):
            MRSJDParams(
                state_names=("bull", "bear", "sideways", "high_vol"),
                transition_matrix=matrix,
                drift=(0.0,) * 4,
                diffusion=(0.1,) * 4,
                jump_intensity=(0.0,) * 4,
                jump_loc=(0.0,) * 4,
                jump_scale=(0.01,) * 4,
                jump_dof=(4.0,) * 4,
            )

    def test_rejects_negative_diffusion(self) -> None:
        with pytest.raises(ValueError, match="diffusion"):
            _uniform_params(diffusion=-0.1)

    def test_allows_zero_diffusion_for_deterministic_limit(self) -> None:
        # Documented deviation from the brief: sigma == 0 is the pure-drift /
        # pure-jump limit required by the deterministic tests below.
        params = _uniform_params(diffusion=0.0)
        assert params.diffusion == (0.0,) * 4

    def test_rejects_dof_at_or_below_two(self) -> None:
        with pytest.raises(ValueError, match="jump_dof"):
            _uniform_params(jump_dof=2.0)

    def test_rejects_non_square_matrix(self) -> None:
        with pytest.raises(ValueError, match="square"):
            MRSJDParams(
                state_names=("bull", "bear", "sideways", "high_vol"),
                transition_matrix=np.full((4, 3), 1.0 / 3.0),
                drift=(0.0,) * 4,
                diffusion=(0.1,) * 4,
                jump_intensity=(0.0,) * 4,
                jump_loc=(0.0,) * 4,
                jump_scale=(0.01,) * 4,
                jump_dof=(4.0,) * 4,
            )

    def test_rejects_mismatched_tuple_lengths(self) -> None:
        with pytest.raises(ValueError, match="drift"):
            MRSJDParams(
                state_names=("bull", "bear", "sideways", "high_vol"),
                transition_matrix=np.full((4, 4), 0.25),
                drift=(0.0,) * 3,
                diffusion=(0.1,) * 4,
                jump_intensity=(0.0,) * 4,
                jump_loc=(0.0,) * 4,
                jump_scale=(0.01,) * 4,
                jump_dof=(4.0,) * 4,
            )

    def test_rejects_state_name_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="state_names"):
            MRSJDParams(
                state_names=("bull", "bear"),
                transition_matrix=np.full((4, 4), 0.25),
                drift=(0.0,) * 4,
                diffusion=(0.1,) * 4,
                jump_intensity=(0.0,) * 4,
                jump_loc=(0.0,) * 4,
                jump_scale=(0.01,) * 4,
                jump_dof=(4.0,) * 4,
            )

    def test_rejects_non_finite_matrix(self) -> None:
        matrix = np.full((4, 4), 0.25)
        matrix[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            MRSJDParams(
                state_names=("bull", "bear", "sideways", "high_vol"),
                transition_matrix=matrix,
                drift=(0.0,) * 4,
                diffusion=(0.1,) * 4,
                jump_intensity=(0.0,) * 4,
                jump_loc=(0.0,) * 4,
                jump_scale=(0.01,) * 4,
                jump_dof=(4.0,) * 4,
            )

    def test_rejects_probability_entries_outside_unit_interval(self) -> None:
        matrix = np.full((4, 4), 0.25)
        matrix[0, 0] = -0.5
        matrix[0, 1] = 1.5  # row still sums to 1
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            MRSJDParams(
                state_names=("bull", "bear", "sideways", "high_vol"),
                transition_matrix=matrix,
                drift=(0.0,) * 4,
                diffusion=(0.1,) * 4,
                jump_intensity=(0.0,) * 4,
                jump_loc=(0.0,) * 4,
                jump_scale=(0.01,) * 4,
                jump_dof=(4.0,) * 4,
            )

    def test_rejects_negative_jump_intensity(self) -> None:
        with pytest.raises(ValueError, match="jump_intensity"):
            _uniform_params(jump_intensity=-1.0)

    def test_rejects_non_positive_jump_scale(self) -> None:
        with pytest.raises(ValueError, match="jump_scale"):
            _uniform_params(jump_scale=0.0)

    def test_rejects_non_finite_jump_loc(self) -> None:
        with pytest.raises(ValueError, match="jump_loc"):
            _uniform_params(jump_loc=float("nan"))


# ---------------------------------------------------------------------------
# CalibratedPriceProcess
# ---------------------------------------------------------------------------


class TestCalibratedPriceProcess:
    def test_zero_sigma_no_jumps_is_constant(self) -> None:
        proc = CalibratedPriceProcess(_uniform_params(), initial_price=3000.0)
        rng = np.random.default_rng(0)
        proc.reset(rng)
        triplets = [proc.step(rng) for _ in range(50)]
        assert all(t[0] == triplets[0][0] for t in triplets)
        assert all(t[1] == triplets[0][1] for t in triplets)
        assert all(t[2] == triplets[0][2] for t in triplets)

    def test_known_drift_matches_expected_growth(self) -> None:
        steps = 2000
        drift = 0.1
        proc = CalibratedPriceProcess(
            _uniform_params(drift=drift), initial_price=3000.0
        )
        rng = np.random.default_rng(7)
        proc.reset(rng)
        price = proc.step(rng)[0]
        for _ in range(steps - 1):
            price = proc.step(rng)[0]
        expected = 3000.0 * math.exp(drift * steps * _DT)
        assert price == pytest.approx(expected, rel=1e-9)

    def test_step_draws_return_from_entering_state_then_transitions(self) -> None:
        # One-hot cycle 0->1->2->3->0 with drift only in state 1 (bear).
        params = MRSJDParams(
            state_names=("bull", "bear", "sideways", "high_vol"),
            transition_matrix=np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            ),
            drift=(0.0, 1.0, 0.0, 0.0),
            diffusion=(0.0, 0.0, 0.0, 0.0),
            jump_intensity=(0.0, 0.0, 0.0, 0.0),
            jump_loc=(0.0, 0.0, 0.0, 0.0),
            jump_scale=(0.01, 0.01, 0.01, 0.01),
            jump_dof=(4.0, 4.0, 4.0, 4.0),
        )
        proc = CalibratedPriceProcess(params, initial_price=3000.0)
        # Force the entering state to bull (zero drift): a regression to the old
        # transition-first order would apply bear's drift to this very step.
        proc._state = 0
        rng = np.random.default_rng(0)
        proc.step(rng)
        assert proc.log_price == pytest.approx(math.log(3000.0))
        assert proc.state == 1
        # Next step is generated by bear (drift 1.0), then transitions to sideways.
        proc.step(rng)
        assert proc.log_price == pytest.approx(math.log(3000.0) + 1.0 * proc.dt)
        assert proc.state == 2

    def test_known_volatility_matches_expected_std(self) -> None:
        sigma = 0.5
        proc = CalibratedPriceProcess(
            _uniform_params(diffusion=sigma), initial_price=3000.0
        )
        rng = np.random.default_rng(123)
        proc.reset(rng)
        prices = np.array([proc.step(rng)[0] for _ in range(10_000)])
        returns = np.diff(np.log(prices))
        expected = sigma * math.sqrt(_DT)
        assert np.std(returns, ddof=1) == pytest.approx(expected, rel=0.05)
        standard_error = expected / math.sqrt(returns.size)
        assert abs(float(np.mean(returns))) < 4.0 * standard_error

    def test_triplets_are_mutually_consistent(self) -> None:
        proc = CalibratedPriceProcess(
            _uniform_params(drift=0.05, diffusion=0.3, jump_intensity=6.0),
            initial_price=3000.0,
        )
        rng = np.random.default_rng(99)
        proc.reset(rng)
        for _ in range(500):
            price, sqrt_price, tick = proc.step(rng)
            _assert_triplet_consistent(price, sqrt_price, tick)

    def test_mode_is_calibrated(self) -> None:
        assert CalibratedPriceProcess(_uniform_params()).mode == "calibrated"

    def test_satisfies_price_process_protocol(self) -> None:
        assert isinstance(CalibratedPriceProcess(_uniform_params()), PriceProcess)

    def test_deterministic_for_same_seed(self) -> None:
        params = _uniform_params(drift=0.05, diffusion=0.3, jump_intensity=5.0)

        def run(seed: int) -> list[tuple[float, float, int]]:
            proc = CalibratedPriceProcess(params, initial_price=3000.0)
            rng = np.random.default_rng(seed)
            proc.reset(rng)
            return [proc.step(rng) for _ in range(200)]

        assert run(1) == run(1)
        assert run(1) != run(2)

    def test_reset_is_reproducible(self) -> None:
        params = _uniform_params(drift=0.05, diffusion=0.3, jump_intensity=5.0)
        proc = CalibratedPriceProcess(params, initial_price=3000.0)
        rng_a = np.random.default_rng(11)
        proc.reset(rng_a)
        first = [proc.step(rng_a) for _ in range(100)]
        rng_b = np.random.default_rng(11)
        proc.reset(rng_b)
        second = [proc.step(rng_b) for _ in range(100)]
        assert first == second

    def test_different_seeds_reach_different_initial_regimes(self) -> None:
        params = _uniform_params()
        states = set()
        for seed in range(20):
            proc = CalibratedPriceProcess(params)
            proc.reset(np.random.default_rng(seed))
            states.add(proc.state)
        assert len(states) > 1

    def test_rejects_non_positive_dt(self) -> None:
        with pytest.raises(ValueError, match="dt"):
            CalibratedPriceProcess(_uniform_params(), dt=0.0)


# ---------------------------------------------------------------------------
# CalibratedPriceProcess.fit
# ---------------------------------------------------------------------------


class TestCalibratedFit:
    def test_rejects_eval_view(self, tiny_market_view: MarketView) -> None:
        with pytest.raises(ValueError, match="train"):
            CalibratedPriceProcess.fit(_eval_view(tiny_market_view))

    def test_asof_regime_labels_match_regime_at(self, tiny_market_view: MarketView) -> None:
        times = [
            tiny_market_view.active_start_utc + timedelta(minutes=10 * i)
            for i in range(1, 50)
        ]
        assert _asof_regime_labels(times, tiny_market_view.regimes) == [
            tiny_market_view.regime_at(t) for t in times
        ]

    def test_fit_warns_when_falling_back_to_hmm(self, caplog: pytest.LogCaptureFixture) -> None:
        # ``_dense_train_view`` has no regime labels, so the anchored path is
        # unavailable and the (known-degenerate) HMM fallback must announce itself.
        with caplog.at_level(logging.WARNING, logger="undertow.sim.prices.regime_jump"):
            CalibratedPriceProcess.fit(_dense_train_view())
        assert any("falling back" in record.message for record in caplog.records)

    def test_fit_produces_valid_params(self) -> None:
        params = CalibratedPriceProcess.fit(_dense_train_view())
        assert params.state_names == ("bull", "bear", "sideways", "high_vol")

        matrix = np.asarray(params.transition_matrix, dtype=float)
        assert matrix.shape == (4, 4)
        assert np.all(matrix >= 0.0)
        assert np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8)
        assert all(sigma > 0.0 for sigma in params.diffusion)
        assert all(scale > 0.0 for scale in params.jump_scale)
        assert all(dof > 2.0 for dof in params.jump_dof)
        assert all(lam >= 0.0 for lam in params.jump_intensity)
        assert all(math.isfinite(mu) for mu in params.drift)
        # Post-hoc labelling: bull is the highest-drift state, bear the lowest,
        # and the two remaining states split by volatility.
        assert params.drift[0] == max(params.drift)
        assert params.drift[1] == min(params.drift)
        assert params.diffusion[2] <= params.diffusion[3]

    def test_fit_anchors_to_regime_labels_when_populated(self) -> None:
        view = _labelled_train_view()
        params = CalibratedPriceProcess.fit(view)
        assert params.state_names == ("bull", "bear", "sideways", "high_vol")

        # Reproduce the label-conditioned statistics exactly: the HMM fallback
        # would not match.
        bars = view.reference_bars(view.active_start_utc, view.active_end_utc)
        closes = bars["close"].cast(pl.Float64).to_numpy()
        times = bars["close_time"].to_list()
        dt = _decision_dt_years(times, 10)
        returns = np.diff(np.log(closes[::10]))
        labels = [view.regime_at(t) for t in times[::10][:-1]]  # start of interval
        for i, name in enumerate(("bull", "bear", "sideways", "high_vol")):
            mask = np.array([label == name for label in labels])
            assert params.drift[i] == pytest.approx(float(returns[mask].mean() / dt), rel=1e-9)
            assert params.diffusion[i] == pytest.approx(
                float(returns[mask].std(ddof=1) / math.sqrt(dt)), rel=1e-9
            )
        assert params.drift[0] > params.drift[1]  # bull above bear
        assert params.diffusion[3] > params.diffusion[2]  # high_vol above sideways
        matrix = np.asarray(params.transition_matrix)
        assert np.allclose(matrix.sum(axis=1), 1.0)
        assert matrix[0, 0] > matrix[0, 1]  # persistent regimes

    def test_empirical_transition_breaks_on_unknown(self) -> None:
        transition = _empirical_transition(["bull"] * 5 + ["unknown"] + ["bear"] * 3)
        assert transition.shape == (4, 4)
        assert np.allclose(transition.sum(axis=1), 1.0)
        # The `unknown` gap must not create a bull -> bear transition.
        assert transition[0, 1] == pytest.approx(0.5 / (4.5 + 3 * 0.5))
        assert transition[0, 0] > transition[0, 1]

    def test_empirical_transition_smooths_unseen_rows(self) -> None:
        transition = _empirical_transition(["bull", "bear", "bear", "bull"])
        # sideways/high_vol never appear: their rows are the uniform smoothing only.
        assert transition[2].tolist() == pytest.approx([0.25] * 4)
        assert transition[3].tolist() == pytest.approx([0.25] * 4)

    def test_fit_from_regime_labels_annualises_start_state_means(self) -> None:
        dt = 1.0 / (6 * 24 * 365.25)
        labels = ["bull"] * 40 + ["bear"] * 40 + ["sideways"] * 40 + ["high_vol"] * 40
        values = np.array([1e-3] * 40 + [-2e-3] * 40 + [0.0] * 40 + [5e-3] * 40)
        params = _fit_from_regime_labels(values, labels, dt)
        assert params is not None
        assert params.state_names == ("bull", "bear", "sideways", "high_vol")
        assert [params.drift[i] for i in range(4)] == pytest.approx(
            [1e-3 / dt, -2e-3 / dt, 0.0, 5e-3 / dt]
        )

    def test_fit_from_regime_labels_returns_none_when_a_state_is_sparse(self) -> None:
        labels = ["bull"] * 100 + ["unknown"] * 10
        assert _fit_from_regime_labels(np.zeros(110), labels, 1e-5) is None

    def test_fit_from_regime_labels_returns_none_when_all_unknown(self) -> None:
        assert _fit_from_regime_labels(np.zeros(100), ["unknown"] * 100, 1e-5) is None

    def test_fit_from_regime_labels_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            _fit_from_regime_labels(np.zeros(5), ["bull"] * 4, 1e-5)

    def test_fit_on_shared_tiny_view_produces_valid_params(
        self, tiny_market_view: MarketView
    ) -> None:
        params = CalibratedPriceProcess.fit(tiny_market_view)
        matrix = np.asarray(params.transition_matrix, dtype=float)
        assert matrix.shape == (4, 4)
        assert np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8)
        assert all(sigma > 0.0 for sigma in params.diffusion)
        assert all(scale > 0.0 for scale in params.jump_scale)
        assert all(dof > 2.0 for dof in params.jump_dof)

    def test_fitted_params_drive_a_process(self) -> None:
        params = CalibratedPriceProcess.fit(_dense_train_view())
        proc = CalibratedPriceProcess(params, initial_price=3000.0)
        rng = np.random.default_rng(2)
        proc.reset(rng)
        for _ in range(100):
            price, sqrt_price, tick = proc.step(rng)
            _assert_triplet_consistent(price, sqrt_price, tick)
