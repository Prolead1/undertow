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

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import polars as pl
import pytest

from undertow.data import Q96, price_to_tick, tick_to_price, tick_to_sqrt_price_x96
from undertow.sim.core.position import calc_sqrt_price_a
from undertow.sim.marketview import MarketView
from undertow.sim.prices import (
    CalibratedPriceProcess,
    MRSJDParams,
    PriceProcess,
    ReplayPriceProcess,
)
from undertow.sim.prices.base import human_price_to_triplet
from undertow.sim.prices.regime_jump import DEFAULT_DT, _decision_dt_years
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
