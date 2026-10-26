"""S12 — observation builder + Gymnasium environment.

Every test asserts real behavior: the stable observation vector, the ordinal
regime encoding, the backward-looking (look-ahead) wall, exact action-index
mapping, deterministic resets/rollouts, the 5-tuple step contract with a full
``RewardBreakdown`` in ``info``, episode truncation, and the fee-share
identity for a hold step.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from tests.sim.synthetic_pool import tiny_episode
from undertow.data import (
    EVENT_TAPE_SCHEMA,
    FEE_GROWTH_SCHEMA,
    GAS_SCHEMA,
    REFERENCE_SCHEMA,
    REGIME_SCHEMA,
    tick_to_price,
    tick_to_sqrt_price_x96,
)
from undertow.sim.config import HedgeConfig, RewardConfig, SimConfig
from undertow.sim.core.pool import PoolEngine, PoolState, TickState
from undertow.sim.core.position import calc_sqrt_price_a, initial_deposit
from undertow.sim.env.lp_env import LpEnvironment
from undertow.sim.env.observations import (
    OBSERVATION_VECTOR_LENGTH,
    SCALED_VECTOR_INDICES,
    Observation,
    ObservationBuilder,
    scale_vector,
    unscale_vector,
)
from undertow.sim.env.reward import RewardBreakdown
from undertow.sim.marketview import MarketView
from undertow.sim.prices.base import human_price_to_triplet
from undertow.sim.prices.regime_jump import CalibratedPriceProcess, MRSJDParams
from undertow.sim.prices.replay import ReplayPriceProcess
from undertow.sim.types import (
    Action,
    EnvError,
    LookAheadError,
    PriceMode,
    Tick,
    TickSpacing,
)

_ENTRY_TICK = 196242
_FEE_TIER_BPS = 3000
_TICK_SPACING = 60
_BASE_LIQUIDITY = 1.0e18


def _fresh_pool_engine() -> PoolEngine:
    """A pristine external pool (no agent positions) at tick 196242."""
    return PoolEngine(
        state=PoolState(
            sqrt_price=calc_sqrt_price_a(Tick(_ENTRY_TICK)),
            tick=Tick(_ENTRY_TICK),
            liquidity=_BASE_LIQUIDITY,
            fee_growth_global_0=0.0,
            fee_growth_global_1=0.0,
            fee_tier_bps=_FEE_TIER_BPS,
            tick_spacing=_TICK_SPACING,
        ),
        ticks={
            Tick(196200 + step * 60): TickState(initialized=True)
            for step in range(-10, 11)
        },
    )


def _mrsjd_params() -> MRSJDParams:
    """A mild 4-state MRSJD for calibrated-mode tests (no network, no fit)."""
    return MRSJDParams(
        state_names=("bull", "bear", "sideways", "high_vol"),
        transition_matrix=np.full((4, 4), 0.25),
        drift=(0.0,) * 4,
        diffusion=(0.2,) * 4,
        jump_intensity=(0.0,) * 4,
        jump_loc=(0.0,) * 4,
        jump_scale=(0.01,) * 4,
        jump_dof=(4.0,) * 4,
    )


def _calibrated_env(
    tiny_market_view: MarketView,
    gas_model: object,
    slippage_model: object,
    *,
    duration_days: int = 1,
    step_minutes: int = 10,
    seed: int = 0,
) -> LpEnvironment:
    """A calibrated-mode env with a bounded, config-driven episode budget."""
    config = SimConfig(
        episode=tiny_episode(
            duration_days=duration_days, step_minutes=step_minutes
        ),
        price_mode=PriceMode.CALIBRATED,
        seed=seed,
    )
    price_process = CalibratedPriceProcess(_mrsjd_params(), initial_price=3004.0)
    return LpEnvironment(
        tiny_market_view,
        _fresh_pool_engine(),
        price_process,
        gas_model,
        slippage_model,
        config,
    )


def _window_start(
    view: MarketView,
    seed: int,
    *,
    duration_days: int = 1,
    step_minutes: int = 10,
) -> int:
    """Expected ``start_seq`` for a window seed, mirroring ``reset``'s call."""
    episode = tiny_episode(duration_days=duration_days, step_minutes=step_minutes)
    config_steps = (duration_days * 24 * 60) // step_minutes
    start_seq, _end_seq = view.build_episode(
        seed, max_steps=config_steps + 1, episode_config=episode
    )
    return int(start_seq)


# ---------------------------------------------------------------------------
# Observation dataclass
# ---------------------------------------------------------------------------


class TestObservationDataclass:
    def test_is_frozen_and_slotted(self) -> None:
        obs = Observation(
            price=3000.0,
            sqrt_price=1.0,
            tick=0,
            price_returns_1h=0.0,
            price_returns_24h=0.0,
            realized_vol_24h=0.0,
            position_in_range=True,
            position_value=1.0,
            uncollected_fees_token0=0.0,
            uncollected_fees_token1=0.0,
            il_vs_hodl=0.0,
            pool_active_liquidity=1.0,
            pool_fee_tier_bps=3000,
            gas_price_recent_wei=0.0,
            eth_usd_price_recent=3000.0,
            regime="bull",
            step_in_episode=0,
            steps_remaining=0,
            distance_to_lower_bound_spacings=0.0,
            distance_to_upper_bound_spacings=0.0,
            position_width_spacings=0.0,
        )
        assert not hasattr(obs, "__dict__")
        with pytest.raises(dataclasses.FrozenInstanceError):
            obs.price = 1.0  # type: ignore[misc]

    def test_vector_length_and_field_order(self) -> None:
        # Distinct values make any reordering detectable.
        values = list(range(1, OBSERVATION_VECTOR_LENGTH + 1))
        obs = Observation(
            price=float(values[0]),
            sqrt_price=float(values[1]),
            tick=int(values[2]),
            price_returns_1h=float(values[3]),
            price_returns_24h=float(values[4]),
            realized_vol_24h=float(values[5]),
            position_in_range=bool(values[6] % 2),
            position_value=float(values[7]),
            uncollected_fees_token0=float(values[8]),
            uncollected_fees_token1=float(values[9]),
            il_vs_hodl=float(values[10]),
            pool_active_liquidity=float(values[11]),
            pool_fee_tier_bps=int(values[12]),
            gas_price_recent_wei=float(values[13]),
            eth_usd_price_recent=float(values[14]),
            regime="sideways",
            step_in_episode=int(values[16]),
            steps_remaining=int(values[17]),
            distance_to_lower_bound_spacings=float(values[18]),
            distance_to_upper_bound_spacings=float(values[19]),
            position_width_spacings=float(values[20]),
            hedge_short_ratio=float(values[21]),
            hedge_equity_frac=float(values[22]),
            funding_rate_recent=float(values[23]),
        )
        vector = obs.to_vector()
        assert vector.dtype == np.float64
        assert vector.shape == (OBSERVATION_VECTOR_LENGTH,)
        expected = [float(v) for v in values]
        expected[6] = float(bool(values[6] % 2))
        expected[15] = 2.0  # sideways
        # ADR-034: the heavy-tailed positions are log-compressed in the vector.
        for index in SCALED_VECTOR_INDICES:
            expected[index] = math.log1p(expected[index])
        assert vector.tolist() == expected
        # regime ordinal sits at index 15: sideways -> 2.
        assert vector[15] == 2.0
        # ADR-029 geometry features are appended at indices 18..20; S20 hedge
        # features at 21..23.
        assert vector[18] == float(values[18])
        assert vector[19] == float(values[19])
        assert vector[20] == float(values[20])
        assert vector[21] == float(values[21])
        assert vector[22] == float(values[22])
        assert vector[23] == float(values[23])

    @pytest.mark.parametrize(
        ("regime", "ordinal"),
        [
            ("bull", 0.0),
            ("bear", 1.0),
            ("sideways", 2.0),
            ("high_vol", 3.0),
            ("unknown", 4.0),
            ("not_a_regime", 4.0),
        ],
    )
    def test_regime_ordinal(self, regime: str, ordinal: float) -> None:
        obs = Observation(
            price=3000.0,
            sqrt_price=1.0,
            tick=0,
            price_returns_1h=0.0,
            price_returns_24h=0.0,
            realized_vol_24h=0.0,
            position_in_range=True,
            position_value=1.0,
            uncollected_fees_token0=0.0,
            uncollected_fees_token1=0.0,
            il_vs_hodl=0.0,
            pool_active_liquidity=1.0,
            pool_fee_tier_bps=3000,
            gas_price_recent_wei=0.0,
            eth_usd_price_recent=3000.0,
            regime=regime,
            step_in_episode=0,
            steps_remaining=0,
            distance_to_lower_bound_spacings=0.0,
            distance_to_upper_bound_spacings=0.0,
            position_width_spacings=0.0,
        )
        assert obs.to_vector()[15] == ordinal


# ---------------------------------------------------------------------------
# ObservationBuilder
# ---------------------------------------------------------------------------


class TestObservationBuilder:
    def test_build_uses_live_pool_and_position(
        self,
        tiny_market_view: MarketView,
        tiny_pool_engine: PoolEngine,
        entry_position: object,
        mock_gas_model: object,
    ) -> None:
        builder = ObservationBuilder(tiny_market_view)
        position = tiny_pool_engine.positions[0]
        obs = builder.build(
            500,
            position,
            tiny_pool_engine,
            mock_gas_model,
            entry_price=3004.0,
            entry_sqrt_price=calc_sqrt_price_a(Tick(_ENTRY_TICK)),
            initial_capital=100_000.0,
        )
        assert obs.sqrt_price == tiny_pool_engine.state.sqrt_price
        assert obs.tick == tiny_pool_engine.state.tick
        assert obs.pool_active_liquidity == tiny_pool_engine.state.liquidity
        assert obs.pool_fee_tier_bps == _FEE_TIER_BPS
        assert obs.position_in_range is position.in_range(
            tiny_pool_engine.state.sqrt_price
        )
        assert obs.position_value == pytest.approx(
            position.value(tiny_pool_engine.state.sqrt_price, obs.price)
        )
        # ADR-029 band geometry is reported in tick-spacing units.
        spacing = float(position.tick_spacing)
        assert obs.distance_to_lower_bound_spacings == pytest.approx(
            (float(obs.tick) - float(position.tick_lower)) / spacing
        )
        assert obs.distance_to_upper_bound_spacings == pytest.approx(
            (float(position.tick_upper) - float(obs.tick)) / spacing
        )
        assert obs.position_width_spacings == pytest.approx(
            (float(position.tick_upper) - float(position.tick_lower)) / spacing
        )
        assert -1.0 <= obs.il_vs_hodl <= 1.0
        assert obs.step_in_episode == 500
        assert obs.gas_price_recent_wei > 0.0
        assert obs.eth_usd_price_recent == obs.price

    def test_rolling_features_are_backward_looking(
        self,
        tiny_market_view: MarketView,
        tiny_pool_engine: PoolEngine,
        entry_position: object,
        mock_gas_model: object,
    ) -> None:
        """A jump in a bar after `t` must not change the observation at `t`."""
        step = 300
        at_time = tiny_market_view.tape_at(step)["block_timestamp"]
        after = (
            tiny_market_view.reference.filter(pl.col("close_time") > at_time)
            .sort("close_time")
        )
        assert after.height > 0, "fixture needs a reference bar after the probe step"

        # Multiply every future close by 10: any leak would be blatant.
        leaked = tiny_market_view.reference.with_columns(
            pl.when(pl.col("close_time") > at_time)
            .then(pl.col("close") * 10.0)
            .otherwise(pl.col("close"))
            .alias("close")
        )
        modified_view = replace(tiny_market_view, reference=leaked)

        position = tiny_pool_engine.positions[0]
        kwargs = {
            "entry_price": 3004.0,
            "entry_sqrt_price": calc_sqrt_price_a(Tick(_ENTRY_TICK)),
            "initial_capital": 100_000.0,
        }
        before = ObservationBuilder(tiny_market_view).build(
            step, position, tiny_pool_engine, mock_gas_model, **kwargs
        )
        after_obs = ObservationBuilder(modified_view).build(
            step, position, tiny_pool_engine, mock_gas_model, **kwargs
        )
        assert before == after_obs
        assert np.array_equal(before.to_vector(), after_obs.to_vector())

        # Sanity: the leak *is* visible once the probe time passes the jump,
        # proving the mutation was real and the wall is what excluded it.
        first_future = after["close_time"][0]
        later_seq = int(
            tiny_market_view.active_split_data()
            .filter(pl.col("block_timestamp") >= first_future)["seq"]
            .min()
        )
        later_before = ObservationBuilder(tiny_market_view).build(
            later_seq, position, tiny_pool_engine, mock_gas_model, **kwargs
        )
        later_after = ObservationBuilder(modified_view).build(
            later_seq, position, tiny_pool_engine, mock_gas_model, **kwargs
        )
        assert later_before.price != pytest.approx(later_after.price)

    def test_band_geometry_signs_out_of_range(
        self,
        tiny_market_view: MarketView,
        tiny_pool_engine: PoolEngine,
        mock_gas_model: object,
    ) -> None:
        """Below the band both edges are negative/positive as signalled."""
        position = tiny_pool_engine.positions[0]
        below_tick = Tick(int(position.tick_lower) - 2 * int(position.tick_spacing))
        tiny_pool_engine.state.tick = below_tick
        tiny_pool_engine.state.sqrt_price = calc_sqrt_price_a(below_tick)
        obs = ObservationBuilder(tiny_market_view).build(
            500,
            position,
            tiny_pool_engine,
            mock_gas_model,
            entry_price=3004.0,
            entry_sqrt_price=calc_sqrt_price_a(Tick(_ENTRY_TICK)),
            initial_capital=100_000.0,
        )
        assert obs.position_in_range is False
        assert obs.distance_to_lower_bound_spacings == pytest.approx(-2.0)
        assert obs.distance_to_upper_bound_spacings > 0.0
        assert obs.position_width_spacings > 0.0

    def test_build_measures_il_against_the_supplied_hodl_basket(
        self,
        tiny_market_view: MarketView,
        tiny_pool_engine: PoolEngine,
        mock_gas_model: object,
    ) -> None:
        """The optional ``hodl_position`` is the IL benchmark (ADR-029)."""
        position = tiny_pool_engine.positions[0]
        other = dataclasses.replace(position, liquidity=position.liquidity * 2.0)
        builder = ObservationBuilder(tiny_market_view)
        entry_sqrt = calc_sqrt_price_a(Tick(_ENTRY_TICK))
        kwargs = {
            "entry_price": 3004.0,
            "entry_sqrt_price": entry_sqrt,
            "initial_capital": 100_000.0,
        }
        obs_own = builder.build(
            500, position, tiny_pool_engine, mock_gas_model, **kwargs
        )
        obs_fixed = builder.build(
            500,
            position,
            tiny_pool_engine,
            mock_gas_model,
            hodl_position=other,
            **kwargs,
        )
        assert obs_fixed.il_vs_hodl != pytest.approx(obs_own.il_vs_hodl)
        sqrt = tiny_pool_engine.state.sqrt_price
        expected = (
            position.value(sqrt, obs_fixed.price)
            - other.hodl_value(entry_sqrt, 3004.0, obs_fixed.price)
        ) / 100_000.0
        assert obs_fixed.il_vs_hodl == pytest.approx(expected)

    def test_rejects_non_positive_capital(
        self,
        tiny_market_view: MarketView,
        tiny_pool_engine: PoolEngine,
        mock_gas_model: object,
    ) -> None:
        builder = ObservationBuilder(tiny_market_view)
        position = tiny_pool_engine.positions[0]
        with pytest.raises(ValueError, match="initial_capital"):
            builder.build(
                100,
                position,
                tiny_pool_engine,
                mock_gas_model,
                entry_price=3004.0,
                entry_sqrt_price=calc_sqrt_price_a(Tick(_ENTRY_TICK)),
                initial_capital=0.0,
            )

    def test_episode_time_features(self, tiny_market_view: MarketView) -> None:
        builder = ObservationBuilder(tiny_market_view, episode_start=100, episode_length=10)
        assert builder.episode_start == 100
        assert builder.episode_length == 10
        builder.configure_episode(50, 7)
        assert builder.episode_start == 50
        assert builder.episode_length == 7

    def test_rolling_price_features_use_reference_bars(
        self, tiny_market_view: MarketView
    ) -> None:
        """The 1h/24h windows are 6/144 *reference bars*, not tape events."""
        import math

        base = datetime(2022, 1, 1, tzinfo=UTC)
        # 160 reference bars spread across the fixture's one-month split so the
        # view's span validation passes; the feature semantics are index-based.
        span = timedelta(days=31) / 159
        closes = [100.0 * (1.0 + 0.001 * i) for i in range(160)]
        open_times = [base + span * i for i in range(160)]
        reference = pl.DataFrame(
            {
                "open_time": open_times,
                "close_time": [t + timedelta(minutes=1) for t in open_times],
                "close": closes,
            }
        )
        view = dataclasses.replace(tiny_market_view, reference=reference)
        builder = ObservationBuilder(view)
        at_time = open_times[-1] + timedelta(minutes=1)
        price, r1h, r24h, vol = builder.rolling_price_features(at_time)
        assert price == pytest.approx(closes[-1])
        assert r1h == pytest.approx(math.log(closes[-1] / closes[-1 - 6]))
        assert r24h == pytest.approx(math.log(closes[-1] / closes[-1 - 144]))
        assert vol > 0.0

        # Closed on the right: a bar at ``at_time`` is included, one after is not.
        before, r1h_before, _r24h, _vol = builder.rolling_price_features(
            at_time - timedelta(minutes=2)
        )
        assert before == pytest.approx(closes[-2])
        assert r1h_before != pytest.approx(r1h)

    def test_reference_closes_index_matches_filter_semantics(
        self, tiny_market_view: MarketView
    ) -> None:
        """ADR-022: the sorted index returns exactly the filter+sort result,
        closed on the right (a bar exactly at ``at_time`` is included)."""
        builder = ObservationBuilder(tiny_market_view)
        close_times = tiny_market_view.reference["close_time"].sort().to_list()
        probes = (
            close_times[0],
            close_times[5],
            close_times[5] - timedelta(seconds=1),
        )
        for at_time in probes:
            expected = (
                tiny_market_view.reference.filter(pl.col("close_time") <= at_time)
                .sort("close_time")["close"]
                .to_numpy()
                .astype(float)
            )
            assert np.array_equal(builder._reference_closes_through(at_time), expected)

    def test_regime_index_is_as_of_inclusive_and_wall_checked(
        self, tiny_market_view: MarketView
    ) -> None:
        builder = ObservationBuilder(tiny_market_view)
        stamps = tiny_market_view.regimes["timestamp"].sort().to_list()
        assert builder._regime_at(stamps[3]) == tiny_market_view.regime_at(stamps[3])
        just_after = stamps[3] + timedelta(seconds=1)
        assert builder._regime_at(just_after) == tiny_market_view.regime_at(just_after)
        with pytest.raises(LookAheadError):
            builder._regime_at(tiny_market_view.active_end_utc + timedelta(days=1))

    def test_empty_regime_and_gas_feeds_yield_empty_indexes(
        self, tiny_market_view: MarketView
    ) -> None:
        """Empty regime/gas feeds give the unknown label / tape fallback, no crash."""
        view = replace(
            tiny_market_view,
            regimes=tiny_market_view.regimes.clear(),
            gas=tiny_market_view.gas.clear(),
        )
        builder = ObservationBuilder(view)
        at_time = tiny_market_view.active_start_utc

        assert builder._regime_at(at_time) == "unknown"
        total, base, priority = builder._gas_price_recent(
            {"block_number": 1, "base_fee_per_gas": "7", "priority_fee_p50_wei": "3"}
        )
        assert (total, base, priority) == (10.0, 7, 3)

    def test_naive_at_time_is_rejected(self, tiny_market_view: MarketView) -> None:
        builder = ObservationBuilder(tiny_market_view)
        with pytest.raises(ValueError, match="timezone-aware"):
            builder._reference_closes_through(datetime(2022, 1, 1))

    def test_gas_index_prefers_the_feed_and_falls_back_to_the_tape_row(
        self, tiny_market_view: MarketView
    ) -> None:
        builder = ObservationBuilder(tiny_market_view)
        row = tiny_market_view.gas.row(0, named=True)
        block = int(row["block_number"])

        _total, base, priority = builder._gas_price_recent({"block_number": block})
        assert (base, priority) == (
            int(row["base_fee_per_gas"]),
            int(row["priority_fee_p50_wei"]),
        )

        # A block with no gas row falls back to the tape row's own columns.
        tape_row = {
            "block_number": block + 10**9,
            "base_fee_per_gas": "7",
            "priority_fee_p50_wei": "3",
        }
        total, base, priority = builder._gas_price_recent(tape_row)
        assert (total, base, priority) == (10.0, 7, 3)


# ---------------------------------------------------------------------------
# LpEnvironment — spaces, reset, step contract
# ---------------------------------------------------------------------------


class TestSpacesAndReset:
    def test_observation_space_shape(
        self, tiny_env: LpEnvironment
    ) -> None:
        assert tiny_env.observation_space.shape == (OBSERVATION_VECTOR_LENGTH,)
        assert tiny_env.observation_space.dtype == np.float64

    def test_action_space_size(self, tiny_env: LpEnvironment) -> None:
        grid = tiny_env.config.action_grid
        assert tiny_env.action_space.n == 1 + len(grid.center_offsets) * len(grid.widths)
        assert tiny_env.action_space.n == 91  # 1 hold + 9 offsets * 10 primary widths

    def test_action_index_mapping(self, tiny_env: LpEnvironment) -> None:
        grid = tiny_env.config.action_grid
        assert tiny_env.action_for_index(0) == Action(action_type="hold")
        index = 1
        for offset in grid.center_offsets:
            for width in grid.widths:
                assert tiny_env.action_for_index(index) == Action(
                    action_type="rebalance", center_offset=offset, width=width
                )
                index += 1
        assert index == tiny_env.action_space.n
        with pytest.raises(EnvError):
            tiny_env.action_for_index(-1)
        with pytest.raises(EnvError):
            tiny_env.action_for_index(tiny_env.action_space.n)

    def test_reset_returns_vector_and_info(self, tiny_env: LpEnvironment) -> None:
        obs, info = tiny_env.reset(seed=0)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (OBSERVATION_VECTOR_LENGTH,)
        assert isinstance(info, dict)
        assert "reward_breakdown" in info
        assert info["equity"] == pytest.approx(
            tiny_env.config.episode.agent_capital_usdc
        )

    def test_reset_is_deterministic_with_seed(self, tiny_env: LpEnvironment) -> None:
        first, _ = tiny_env.reset(seed=42)
        second, _ = tiny_env.reset(seed=42)
        assert np.array_equal(first, second)

    def test_reset_seed_selects_episode_window(
        self,
        tiny_market_view: MarketView,
        mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        env = _calibrated_env(tiny_market_view, mock_gas_model, mock_slippage_model)
        first, _ = env.reset(seed=42)
        same, _ = env.reset(seed=42)
        assert np.array_equal(first, same)
        first_start = env._start_seq
        env.reset(seed=7)
        other_start = env._start_seq
        assert other_start != first_start

    def test_initial_equity_is_capital(self, tiny_env: LpEnvironment) -> None:
        _, info = tiny_env.reset(seed=7)
        assert info["equity"] == pytest.approx(
            tiny_env.config.episode.agent_capital_usdc
        )
        position = tiny_env.pool_engine.positions[tiny_env.position_id]
        assert position.value(
            tiny_env.pool_engine.state.sqrt_price, info["entry_price"]
        ) == pytest.approx(tiny_env.config.episode.agent_capital_usdc)

    def test_build_episode_uses_seed(self, tiny_env: LpEnvironment) -> None:
        tiny_env.reset(seed=1)
        start_a = tiny_env._start_seq
        tiny_env.reset(seed=1)
        assert tiny_env._start_seq == start_a

    def test_reset_without_seed_advances_window(
        self,
        tiny_market_view: MarketView,
        mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        env = _calibrated_env(tiny_market_view, mock_gas_model, mock_slippage_model)
        window_starts = [_window_start(tiny_market_view, seed) for seed in (0, 1, 2)]
        if len(set(window_starts)) < 3:
            pytest.skip(
                "tiny_market_view does not expose three distinct candidate "
                "windows for seeds 0-2; resampling cannot be distinguished"
            )

        observation, _ = env.reset(seed=0)
        assert env._start_seq == window_starts[0]
        # Two no-seed resets (the SB3 auto-reset path) must move to the next
        # window each time, and neither may replay the seeded window.
        first_unseeded, _ = env.reset()
        assert env._start_seq == window_starts[1]
        second_unseeded, _ = env.reset()
        assert env._start_seq == window_starts[2]
        assert window_starts[1] != window_starts[0]
        assert window_starts[2] not in (window_starts[0], window_starts[1])
        assert not np.array_equal(first_unseeded, observation)
        assert not np.array_equal(second_unseeded, first_unseeded)

    def test_reset_without_seed_is_reproducible_from_a_seeded_start(
        self,
        tiny_market_view: MarketView,
        mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        env = _calibrated_env(tiny_market_view, mock_gas_model, mock_slippage_model)

        def run_sequence() -> tuple[list[int], list[np.ndarray]]:
            starts: list[int] = []
            vectors: list[np.ndarray] = []
            observation, _ = env.reset(seed=3)
            starts.append(env._start_seq)
            vectors.append(observation)
            for _ in range(2):
                observation, _ = env.reset()
                starts.append(env._start_seq)
                vectors.append(observation)
            return starts, vectors

        first_starts, first_vectors = run_sequence()
        second_starts, second_vectors = run_sequence()
        assert first_starts == [
            _window_start(tiny_market_view, seed) for seed in (3, 4, 5)
        ]
        assert len(set(first_starts)) == 3
        assert first_starts == second_starts
        for first, second in zip(first_vectors, second_vectors, strict=True):
            assert np.array_equal(first, second)

    def test_reset_with_seed_returns_to_that_window(
        self,
        tiny_market_view: MarketView,
        mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        env = _calibrated_env(tiny_market_view, mock_gas_model, mock_slippage_model)
        seeded_observation, _ = env.reset(seed=3)
        seeded_start = env._start_seq
        assert seeded_start == _window_start(tiny_market_view, 3)

        for _ in range(4):
            env.reset()
        assert env._start_seq != seeded_start

        observation, _ = env.reset(seed=3)
        assert env._start_seq == seeded_start
        assert np.array_equal(observation, seeded_observation)

    def test_first_reset_without_seed_uses_config_seed(
        self,
        tiny_market_view: MarketView,
        mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        env = _calibrated_env(
            tiny_market_view, mock_gas_model, mock_slippage_model, seed=11
        )
        assert env._base_seed == 11
        assert env._episode_index == -1

        _observation, _ = env.reset()
        assert env._base_seed + env._episode_index == env.config.seed
        assert env._start_seq == _window_start(tiny_market_view, 11)


class TestStep:
    def test_step_returns_five_tuple(self, tiny_env: LpEnvironment) -> None:
        tiny_env.reset(seed=0)
        result = tiny_env.step(0)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (OBSERVATION_VECTOR_LENGTH,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_info_carries_full_reward_breakdown(
        self, tiny_env: LpEnvironment
    ) -> None:
        tiny_env.reset(seed=0)
        _, _, _, _, info = tiny_env.step(0)
        expected_keys = {f.name for f in dataclasses.fields(RewardBreakdown)}
        assert set(info["reward_breakdown"].keys()) == expected_keys
        assert set(info["reward_breakdown_raw"].keys()) == expected_keys
        # S20: the ungated economics payload (same shape, no ablation gating).
        assert set(info["reward_breakdown_economics"].keys()) == expected_keys
        assert (
            info["reward_breakdown_economics"]["equity"]
            == info["reward_breakdown_raw"]["equity"]
        )

    def test_hold_advances_price_keeps_position_and_accrues_fees(
        self, tiny_env: LpEnvironment
    ) -> None:
        tiny_env.reset(seed=42)
        position_id = tiny_env.position_id
        lower = tiny_env.pool_engine.positions[position_id].tick_lower
        upper = tiny_env.pool_engine.positions[position_id].tick_upper
        sqrt_before = tiny_env.pool_engine.state.sqrt_price

        _, _, _, _, info = tiny_env.step(0)

        assert tiny_env.position_id == position_id
        position = tiny_env.pool_engine.positions[position_id]
        assert position.tick_lower == lower
        assert position.tick_upper == upper
        # The fixture's tape holds a constant pool sqrt price, so the exact
        # piecewise engine (ADR-019) credits no fee; the nominal gross fee is
        # still reported.  Fee accrual is pinned in test_pool_exact_parity.py.
        assert tiny_env.pool_engine.state.sqrt_price != sqrt_before
        assert info["pool_fee_earned_0"] > 0.0
        assert tiny_env.pool_engine.position_fees(position_id) == (0.0, 0.0)
        assert info["reward_breakdown_raw"]["fees"] == 0.0

    def test_hold_step_has_zero_gas_and_slippage(
        self, tiny_env: LpEnvironment
    ) -> None:
        tiny_env.reset(seed=42)
        _, _, _, _, info = tiny_env.step(0)
        raw = info["reward_breakdown_raw"]
        assert raw["gas"] == 0.0
        assert raw["slippage"] == 0.0
        assert raw["reward"] == pytest.approx(raw["fees"] + raw["il_change"])
        assert raw["pnl_net"] == pytest.approx(
            raw["fees"] + raw["il_change"]
        )

    def test_hodl_pool_fee_is_reported_but_not_accrued_on_flat_tape(
        self, tiny_env: LpEnvironment
    ) -> None:
        # The shared ``tiny_market_view`` tape holds a constant pool sqrt price
        # (its reference feed moves, its pool does not), so the exact piecewise
        # engine (ADR-019) correctly accrues nothing even though the nominal
        # gross fee is reported.  The genuine fee-share identity is exercised in
        # ``test_pool_exact_parity.py`` and the ``_parity_env`` fixture.
        tiny_env.reset(seed=42)
        _, _, _, _, info = tiny_env.step(0)
        assert info["pool_fee_earned_0"] > 0.0
        raw, _ = tiny_env.pool_engine.position_fees(tiny_env.position_id)
        assert raw == 0.0
        assert info["reward_breakdown_raw"]["fees"] == 0.0

    def test_rebalance_changes_position_range(
        self, tiny_env: LpEnvironment
    ) -> None:
        tiny_env.reset(seed=42)
        first_id = tiny_env.position_id
        first_lower = tiny_env.pool_engine.positions[first_id].tick_lower
        first_upper = tiny_env.pool_engine.positions[first_id].tick_upper

        # The (center_offset=+4, width=1) action.
        action = Action(action_type="rebalance", center_offset=4, width=1)
        index = tiny_env._action_lookup.index(action)
        _, _, _, _, info = tiny_env.step(index)

        assert info["action"] == action
        assert tiny_env.position_id != first_id
        assert info["tick_lower"] != first_lower or info["tick_upper"] != first_upper
        assert info["tick_lower"] < info["tick_upper"]
        assert info["reward_breakdown_raw"]["gas"] < 0.0
        assert info["reward_breakdown_raw"]["slippage"] < 0.0

    def test_step_after_done_raises(self, tiny_env: LpEnvironment) -> None:
        tiny_env.reset(seed=0)
        for _ in range(tiny_env.episode_length):
            _, _, terminated, truncated, _ = tiny_env.step(0)
            if terminated or truncated:
                break
        with pytest.raises(EnvError):
            tiny_env.step(0)


# ---------------------------------------------------------------------------
# Episodes, determinism, calibrated mode
# ---------------------------------------------------------------------------


class TestEpisodesAndModes:
    def test_replay_episode_truncates_at_price_budget(
        self, tiny_env: LpEnvironment
    ) -> None:
        tiny_env.reset(seed=5)
        assert tiny_env.episode_length == 99
        steps = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = tiny_env.step(0)
            steps += 1
        assert steps == tiny_env.episode_length
        assert truncated is True
        assert terminated is False

    def test_replay_env_applies_decision_cadence_on_1min_feed(
        self,
        minute_market_view: MarketView,
        mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        """The replay env must decide every ``step_minutes``, not every bar.

        Exercises the wiring in ``LpEnvironment.reset`` (the price process gets
        ``step_minutes`` and the window is sampled at the env's cadence) — the
        default fixture's hour-spaced bars cannot catch a regression here.
        """
        config = SimConfig(episode=tiny_episode(duration_days=1, step_minutes=10))
        env = LpEnvironment(
            minute_market_view,
            _fresh_pool_engine(),
            ReplayPriceProcess(minute_market_view, 0, minute_market_view.step_count()),
            mock_gas_model,
            mock_slippage_model,
            config,
        )
        env.reset(seed=0)
        # ~144 decisions for a 1-day window (a few either way: the tape's last row
        # can fall short of the window end); never the ~1440 raw 1-minute bars.
        assert 139 <= len(env._bar_close_times) <= 146
        assert env.episode_length == min(144, len(env._bar_close_times))
        assert env.episode_length >= 139
        assert env.price_process.n_bars == len(env._bar_close_times)
        # Consecutive decision bars are step_minutes apart, not one minute.
        deltas = [
            b - a
            for a, b in zip(env._bar_close_times, env._bar_close_times[1:], strict=False)
        ]
        assert deltas and all(
            timedelta(minutes=9) < delta < timedelta(minutes=11) for delta in deltas
        )

    def test_episode_length_matches_config(
        self,
        tiny_market_view: MarketView,
        mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        env = _calibrated_env(
            tiny_market_view,
            mock_gas_model,
            mock_slippage_model,
            duration_days=1,
            step_minutes=10,
        )
        env.reset(seed=1)
        assert env.episode_length == 144  # 1 day / 10 min
        steps = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(0)
            steps += 1
        assert steps == 144
        assert truncated is True

    def test_replay_rollout_is_deterministic(self, tiny_env: LpEnvironment) -> None:
        def rollout() -> list[np.ndarray]:
            frames = [tiny_env.reset(seed=7)[0]]
            for _ in range(25):
                frames.append(tiny_env.step(0)[0])
            return frames

        first = rollout()
        second = rollout()
        assert len(first) == len(second) == 26
        for a, b in zip(first, second, strict=True):
            assert np.array_equal(a, b)

    def test_calibrated_mode_produces_seeded_sequence(
        self,
        tiny_market_view: MarketView,
        mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        env = _calibrated_env(tiny_market_view, mock_gas_model, mock_slippage_model)

        def rollout() -> list[np.ndarray]:
            frames = [env.reset(seed=3)[0]]
            for _ in range(15):
                frames.append(env.step(0)[0])
            return frames

        first = rollout()
        second = rollout()
        assert all(np.all(np.isfinite(frame)) for frame in first)
        for a, b in zip(first, second, strict=True):
            assert np.array_equal(a, b)

    def test_tiny_env_fixture_resets_and_steps(self, tiny_env: LpEnvironment) -> None:
        obs, info = tiny_env.reset(seed=0)
        assert obs.shape == (OBSERVATION_VECTOR_LENGTH,)
        assert info["action_type"] == "hold"
        result = tiny_env.step(0)
        assert len(result) == 5

    def test_replay_decision_row_matches_the_window_frame(
        self, tiny_env: LpEnvironment
    ) -> None:
        """The env keeps the window frame and extracts single rows on demand; the
        decision row after an interval must equal the row ``to_dicts()`` held."""
        tiny_env.reset(seed=0)
        for _ in range(5):
            tiny_env.step(0)
            if tiny_env._event_idx:
                break

        assert tiny_env._ev_window is not None
        assert tiny_env._event_idx > 0
        # The row ``to_dicts()`` would have held at the interval's last index.
        old_style_row = tiny_env._ev_window.to_dicts()[tiny_env._event_idx - 1]
        assert tiny_env._last_tape_row == old_style_row


# ---------------------------------------------------------------------------
# Position factory sanity (the env funds its initial position correctly)
# ---------------------------------------------------------------------------


class TestInitialDeployment:
    def test_initial_position_consumes_capital(
        self, tiny_env: LpEnvironment
    ) -> None:
        _, info = tiny_env.reset(seed=11)
        entry_price = info["entry_price"]
        position = tiny_env.pool_engine.positions[tiny_env.position_id]
        assert position.value(
            tiny_env.pool_engine.state.sqrt_price, entry_price
        ) == pytest.approx(tiny_env.config.episode.agent_capital_usdc)

    def test_deploy_action_bounds_are_snapped(
        self, tiny_env: LpEnvironment
    ) -> None:
        _, info = tiny_env.reset(seed=11)
        assert info["tick_lower"] % _TICK_SPACING == 0
        assert info["tick_upper"] % _TICK_SPACING == 0
        assert info["tick_lower"] < info["tick_upper"]
        # The deployment is centred on the entry tick within one spacing.
        midpoint = (info["tick_lower"] + info["tick_upper"]) // 2
        assert abs(midpoint - _ENTRY_TICK) <= _TICK_SPACING


# ---------------------------------------------------------------------------
# Review follow-ups: event-granularity, equity continuity, validation, exhaustion
# ---------------------------------------------------------------------------


class _RecordingGasModel:
    """A gas model that records the decision context it is priced from.

    Returns a small fixed USDC cost (the tests pin the decision *row* the env
    selected, not the cost arithmetic, which ``test_frictions.py`` owns).
    """

    def __init__(self, cost: float = 7.0) -> None:
        self.cost = float(cost)
        self.calls: list[tuple[str, int, int, int, float]] = []

    def gas_cost_usdc(
        self,
        action_type: str,
        block_number: int,
        base_fee_per_gas_wei: int,
        priority_fee_p50_wei: int,
        eth_usd_price: float,
    ) -> float:
        self.calls.append(
            (
                str(action_type),
                int(block_number),
                int(base_fee_per_gas_wei),
                int(priority_fee_p50_wei),
                float(eth_usd_price),
            )
        )
        return self.cost


class _ConstantPriceProcess:
    """A deterministic, unbounded price process for equity-continuity tests."""

    def __init__(self, price: float) -> None:
        self._price = float(price)

    def reset(self, rng: object) -> None:
        return None

    def step(self, rng: object) -> tuple[float, float, Tick]:
        return human_price_to_triplet(self._price)

    @property
    def mode(self) -> str:
        return "calibrated"


class _SequencePriceProcess:
    """A deterministic price process that walks a fixed price sequence.

    ``step`` repeats the final price once the sequence is exhausted, so a test
    can move the market through a known path and then keep it flat.
    """

    def __init__(self, prices: list[float]) -> None:
        self._prices = [float(p) for p in prices]
        self._i = 0

    def reset(self, rng: object) -> None:
        self._i = 0

    def step(self, rng: object) -> tuple[float, float, Tick]:
        price = self._prices[min(self._i, len(self._prices) - 1)]
        self._i += 1
        return human_price_to_triplet(price)

    @property
    def mode(self) -> str:
        return "calibrated"


class _ExhaustingPriceProcess:
    """A price process that raises ``StopIteration`` after ``n`` steps."""

    def __init__(self, price: float, n: int) -> None:
        self._price = float(price)
        self._n = int(n)
        self._i = 0

    def reset(self, rng: object) -> None:
        self._i = 0

    def step(self, rng: object) -> tuple[float, float, Tick]:
        if self._i >= self._n:
            raise StopIteration
        self._i += 1
        return human_price_to_triplet(self._price)

    @property
    def mode(self) -> str:
        return "calibrated"


class _SequenceExhaustingPriceProcess:
    """Yields a fixed price sequence, then raises ``StopIteration``."""

    def __init__(self, prices: list[float]) -> None:
        self._prices = [float(p) for p in prices]
        self._i = 0

    def reset(self, rng: object) -> None:
        self._i = 0

    def step(self, rng: object) -> tuple[float, float, Tick]:
        if self._i >= len(self._prices):
            raise StopIteration
        price = self._prices[self._i]
        self._i += 1
        return human_price_to_triplet(price)

    @property
    def mode(self) -> str:
        return "calibrated"


class TestReplayEventGranularity:
    def test_pool_is_stepped_multiple_times_per_decision(
        self, tiny_env: LpEnvironment
    ) -> None:
        """Replay consumes every tape event in the interval, not one per step."""
        tiny_env.reset(seed=42)
        for _ in range(5):
            tiny_env.step(0)
        assert tiny_env.step_count == 5
        # The tiny tape ~10 events per reference bar, so the event cursor must
        # run well ahead of the decision counter.
        assert tiny_env._event_idx > tiny_env.step_count

    def test_interval_fee_is_sum_of_swap_events(
        self, tiny_env: LpEnvironment
    ) -> None:
        tiny_env.reset(seed=42)
        _, _, _, _, info = tiny_env.step(0)
        # The reported gross fee is the sum over the interval's swaps; the
        # constant-price tape means the agent accrues none of it (ADR-019).
        assert info["pool_fee_earned_0"] > 0.0
        actual_raw, _ = tiny_env.pool_engine.position_fees(tiny_env.position_id)
        assert actual_raw == 0.0

    def test_non_swap_events_are_not_charged(
        self, tiny_env: LpEnvironment
    ) -> None:
        """Mint/burn rows in the interval contribute no swap fee.

        The tiny tape alternates swap/mint. If non-swap rows were charged the
        aggregated fee would be roughly double the swap-only fee; compare
        against the sum over swap rows only.
        """
        tiny_env.reset(seed=42)
        _, _, _, _, info = tiny_env.step(0)
        # Recompute the swap-only fee for the first interval at the entry price.
        bar_close = tiny_env._bar_close_times[0]
        swap_fee = 0.0
        for i, when in enumerate(tiny_env._ev_times):
            if when > bar_close:
                break
            if tiny_env._ev_types[i] != "swap":
                continue
            v0, _ = tiny_env._event_volumes(i)
            swap_fee += v0 * tiny_env.config.episode.fee_tier_bps / 1_000_000
        assert info["pool_fee_earned_0"] == pytest.approx(swap_fee, rel=1e-9)


class TestEquityContinuity:
    def test_rebalance_carries_collected_fees_into_equity(
        self, tiny_market_view: MarketView, mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        """Equity after a rebalance equals equity before + pnl_net (fees kept)."""
        entry = 3004.0
        process = _ConstantPriceProcess(entry)
        config = SimConfig(
            episode=tiny_episode(duration_days=1, step_minutes=10),
            price_mode=PriceMode.CALIBRATED,
        )
        env = LpEnvironment(
            tiny_market_view,
            _fresh_pool_engine(),
            process,
            mock_gas_model,
            mock_slippage_model,
            config,
        )
        env.reset(seed=0)
        _, _, _, _, hold_info = env.step(0)
        equity_before = hold_info["equity"]

        rebalance_action = Action(action_type="rebalance", center_offset=0, width=1)
        index = env._action_lookup.index(rebalance_action)
        _, _, _, _, rebalance_info = env.step(index)
        raw = rebalance_info["reward_breakdown_raw"]
        assert raw["gas"] < 0.0
        assert raw["slippage"] < 0.0
        # Constant price => IL change is ~0 and equity moves by pnl_net exactly.
        assert rebalance_info["equity"] == pytest.approx(
            equity_before + raw["pnl_net"], abs=1e-6
        )
        # The old position's collected fees are not silently dropped: under
        # ADR-031 the rebalance executes at the decision state and collects the
        # fees accrued through the hold step.  The rebalance step's own interval
        # then accrues on the new position (zero at the constant price), so it
        # is reported separately rather than added to ``collected``.
        hold_fees = hold_info["reward_breakdown_raw"]["fees"]
        assert rebalance_info["collected_fees_usdc"] == pytest.approx(
            hold_fees, rel=1e-6, abs=1e-9
        )
        assert raw["fees"] == pytest.approx(0.0, abs=1e-12)

    def test_reward_measures_excess_vs_fixed_initial_hodl(
        self, tiny_market_view: MarketView, mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        """A rebalance does not reset the IL benchmark (ADR-029).

        The level carried into the next step is the current position value minus
        the *initial* deposit's HODL value, so rebalancing costs and accumulated
        under/outperformance stay in the level.  The previous rolling-entry
        benchmark reset it to ~0 at every rebalance.
        """
        prices = [3004.0, 3010.0, 3016.0, 3020.0]
        env = LpEnvironment(
            tiny_market_view,
            _fresh_pool_engine(),
            _SequencePriceProcess(prices),
            mock_gas_model,
            mock_slippage_model,
            SimConfig(
                episode=tiny_episode(duration_days=1, step_minutes=10),
                price_mode=PriceMode.CALIBRATED,
            ),
        )
        env.reset(seed=0)
        initial = env._initial_position
        assert initial is not None
        env.step(0)  # 3004
        env.step(0)  # 3010
        rebalance = Action(action_type="rebalance", center_offset=0, width=2)
        index = env._action_lookup.index(rebalance)
        vector, _, _, _, _ = env.step(index)  # 3016

        # The returned observation measures IL against the same fixed basket.
        from undertow.sim.evaluate import observation_from_vector

        obs = observation_from_vector(vector)
        fixed_basket = (
            env._position.value(env.pool_engine.state.sqrt_price, obs.price)
            - initial.hodl_value(env._entry_sqrt, env._entry_price, obs.price)
        ) / env._capital
        own_basket = (
            env._position.value(env.pool_engine.state.sqrt_price, obs.price)
            - env._position.hodl_value(env._entry_sqrt, env._entry_price, obs.price)
        ) / env._capital
        assert obs.il_vs_hodl == pytest.approx(fixed_basket)
        assert obs.il_vs_hodl != pytest.approx(own_basket)

        # ADR-031 marks the rebalance step at its post-advance bar: the action
        # executes at 3010, then the step advances and values the position at
        # 3016.
        price = prices[2]
        _, sqrt, _ = human_price_to_triplet(price)
        expected = env._position.value(sqrt, price) - initial.hodl_value(
            env._entry_sqrt, env._entry_price, price
        )
        assert env._last_il_total == pytest.approx(expected)
        # The fixed benchmark keeps the rebalancing PnL in the level; a
        # rolling-entry benchmark would be ~0 right after the rebalance.
        own_entry_level = env._position.value(sqrt, price) - env._position.hodl_value(
            env._entry_sqrt, env._entry_price, price
        )
        assert env._last_il_total != pytest.approx(own_entry_level)

    def test_episode_reward_sum_is_excess_over_initial_hodl(
        self, tiny_market_view: MarketView, mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        """Summing ``pnl_net`` telescopes to equity minus the initial HODL."""
        prices = [3004.0, 3010.0, 3016.0]
        env = LpEnvironment(
            tiny_market_view,
            _fresh_pool_engine(),
            _SequencePriceProcess(prices),
            mock_gas_model,
            mock_slippage_model,
            SimConfig(
                episode=tiny_episode(duration_days=1, step_minutes=10),
                price_mode=PriceMode.CALIBRATED,
            ),
        )
        env.reset(seed=0)
        initial = env._initial_position
        assert initial is not None
        rebalance = Action(action_type="rebalance", center_offset=0, width=2)
        index = env._action_lookup.index(rebalance)
        total = 0.0
        info: dict[str, object] = {}
        for step in range(10):
            action = index if step == 2 else 0
            _, _, terminated, truncated, info = env.step(action)
            total += float(info["reward_breakdown_raw"]["pnl_net"])
            if terminated or truncated:
                break
        final_price = env._last_mark_price
        hodl_final = initial.hodl_value(
            env._entry_sqrt, env._entry_price, final_price
        )
        assert total == pytest.approx(float(info["equity"]) - hodl_final)

    def test_unaffordable_rebalance_reverts_without_changing_the_book(
        self, tiny_market_view: MarketView, mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        """A rebalance the position cannot pay for is a no-op (on-chain revert).

        The env used to silently recapitalise the position with
        ``max(capital, 1.0)``, which let a bankrupt policy keep rebalancing for
        free and diverged from the backtester's ground truth.  Two identical
        envs step the same state: one holds, the other asks for the unaffordable
        rebalance; every book term must agree.
        """
        config = SimConfig(
            episode=tiny_episode(
                duration_days=1, step_minutes=10, agent_capital_usdc=1.0
            ),
            price_mode=PriceMode.CALIBRATED,
        )

        def _make_env() -> LpEnvironment:
            return LpEnvironment(
                tiny_market_view,
                _fresh_pool_engine(),
                _ConstantPriceProcess(3004.0),
                mock_gas_model,
                mock_slippage_model,
                config,
            )

        hold_env = _make_env()
        reject_env = _make_env()
        hold_env.reset(seed=0)
        reject_env.reset(seed=0)
        hold_env.step(0)  # one hold step after the entry decision
        reject_env.step(0)
        assert reject_env._position is not None
        lower = reject_env._position.tick_lower
        upper = reject_env._position.tick_upper
        position_id = reject_env.position_id

        _, hold_reward, _, _, hold_info = hold_env.step(0)
        rebalance = Action(action_type="rebalance", center_offset=0, width=1)
        index = reject_env._action_lookup.index(rebalance)
        _, reject_reward, _, _, reject_info = reject_env.step(index)

        raw = reject_info["reward_breakdown_raw"]
        assert reject_info["action_type"] == "rebalance_rejected"
        assert reject_info["rebalance_rejected"] is True
        assert raw["gas"] == 0.0
        assert raw["slippage"] == 0.0
        # The rejected action is book-identical to the hold on the same state.
        assert reject_reward == hold_reward
        assert reject_info["equity"] == hold_info["equity"]
        assert reject_env.position_id == position_id == hold_env.position_id
        assert reject_env._position.tick_lower == lower
        assert reject_env._position.tick_upper == upper
        assert reject_env._last_il_total == hold_env._last_il_total
        assert reject_env._collected_fees_usdc == hold_env._collected_fees_usdc
        assert reject_env._last_fees_usdc == hold_env._last_fees_usdc


class TestDecisionTimeExecution:
    """ADR-031/S22 T1-T3: the action executes at the decision state it observed.

    The old ordering advanced the market first and executed one decision bar
    late, so a first-step rebalance centred on the *next* bar's tick and a
    mid-episode rebalance centred on the following bar's price.
    """

    @staticmethod
    def _calibrated_test_env(
        tiny_market_view: MarketView,
        price_process: object,
        gas_model: object,
        slippage_model: object,
    ) -> LpEnvironment:
        return LpEnvironment(
            tiny_market_view,
            _fresh_pool_engine(),
            price_process,
            gas_model,
            slippage_model,
            SimConfig(
                episode=tiny_episode(duration_days=1, step_minutes=10),
                price_mode=PriceMode.CALIBRATED,
            ),
        )

    def test_first_step_rebalance_executes_at_the_entry_price(
        self,
        tiny_market_view: MarketView,
        mock_slippage_model: object,
    ) -> None:
        """S22 T1: a step-0 rebalance uses the entry tick, not the next bar's.

        The first advance moves to 4000 (hundreds of ticks away), so the old
        one-bar-late execution would have centred the band there.  Gas must
        come from the entry decision row, not the advanced row.
        """
        gas_model = _RecordingGasModel()
        env = self._calibrated_test_env(
            tiny_market_view,
            _SequencePriceProcess([4000.0]),
            gas_model,
            mock_slippage_model,
        )
        _obs, reset_info = env.reset(seed=0)
        entry_tick = int(reset_info["tick"])
        entry_row = env._last_tape_row
        assert entry_row  # T3: reset populates the decision state
        assert entry_row["block_number"] == env.market_view.tape_at(
            env._start_seq
        )["block_number"]

        action = Action(action_type="rebalance", center_offset=0, width=2)
        index = env._action_lookup.index(action)
        _, _, _, _, info = env.step(index)

        spacing = TickSpacing(int(env.config.episode.tick_spacing))
        position = env.pool_engine.positions[env.position_id]
        assert position.tick_lower == action.lower_tick(Tick(entry_tick), spacing)
        assert position.tick_upper == action.upper_tick(Tick(entry_tick), spacing)
        # The market did advance, so the entry tick and the executed range are
        # distinguishable from the old (next-bar) behaviour.
        assert env.pool_engine.state.tick != Tick(entry_tick)

        # The rebalance priced gas from the entry decision row's block.
        assert len(gas_model.calls) == 1
        gas_action, gas_block, *_ = gas_model.calls[0]
        assert gas_action == "rebalance"
        assert gas_block == int(entry_row["block_number"])
        assert info["reward_breakdown_raw"]["gas"] == pytest.approx(-gas_model.cost)

    def test_mid_episode_rebalance_executes_at_the_decision_price(
        self,
        tiny_market_view: MarketView,
        mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        """S22 T2: a mid-episode rebalance centres on the decision mark.

        The step change 3010 -> 3070 happens *after* the decision, so the old
        ordering would have opened the range around 3070's tick.
        """
        env = self._calibrated_test_env(
            tiny_market_view,
            _SequencePriceProcess([3004.0, 3010.0, 3070.0]),
            mock_gas_model,
            mock_slippage_model,
        )
        env.reset(seed=0)
        env.step(0)  # decision at entry -> mark 3004
        env.step(0)  # decision at 3004 -> mark 3010
        decision_tick = Tick(env.pool_engine.state.tick)
        assert float(tick_to_price(int(decision_tick), 6, 18)) == pytest.approx(
            3010.0, rel=1e-3
        )

        action = Action(action_type="rebalance", center_offset=0, width=2)
        index = env._action_lookup.index(action)
        env.step(index)  # decision at 3010 -> mark 3070

        spacing = TickSpacing(int(env.config.episode.tick_spacing))
        position = env.pool_engine.positions[env.position_id]
        assert position.tick_lower == action.lower_tick(decision_tick, spacing)
        assert position.tick_upper == action.upper_tick(decision_tick, spacing)
        # The range's centre is the decision mark, not the post-advance mark.
        assert env.pool_engine.state.tick != decision_tick

    def test_reset_populates_the_decision_state(
        self, tiny_env: LpEnvironment
    ) -> None:
        """S22 T3: ``_last_tape_row``/time/mark are set before the first step.

        The old ``reset`` left ``_last_tape_row`` empty until the first advance,
        so a first-step rebalance would have priced gas from an empty row.
        """
        _obs, info = tiny_env.reset(seed=3)
        assert tiny_env._last_tape_row
        assert tiny_env._last_tape_row == tiny_env.market_view.tape_at(
            tiny_env._start_seq
        )
        assert tiny_env._last_at_time == info["timestamp"]
        assert tiny_env._last_mark_price == pytest.approx(float(info["price"]))


class TestRebalanceSwapNotional:
    """S18: rebalance slippage charges the swapped inventory imbalance only."""

    def test_balanced_recentre_charges_no_slippage(
        self, tiny_market_view: MarketView, mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        env = _calibrated_env(tiny_market_view, mock_gas_model, mock_slippage_model)
        env.reset(seed=0)
        assert env._position is not None
        # The reset deposit was built by ``initial_deposit`` on the env's own
        # band, so rebalancing to that exact band at the same mark swaps zero
        # inventory.  The old full-position-value charge (notional * 0.0035)
        # made this common balanced recentre the worst case.
        action = Action(
            action_type="rebalance",
            center_offset=env._initial_center_offset,
            width=env._initial_width,
        )
        sqrt = env.pool_engine.state.sqrt_price
        price = env._entry_price
        notional = env._position.value(sqrt, price)

        _gas, slippage_cost, executed = env._rebalance(
            action, sqrt, price, env._last_tape_row
        )

        assert executed is True
        assert slippage_cost < 1e-6 * notional
        assert slippage_cost == pytest.approx(0.0, abs=1e-9)

    def test_out_of_range_recentre_charges_about_half_notional(
        self, tiny_market_view: MarketView, mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        env = _calibrated_env(tiny_market_view, mock_gas_model, mock_slippage_model)
        env.reset(seed=0)
        spacing = TickSpacing(int(env.config.episode.tick_spacing))
        tick = env.pool_engine.state.tick
        sqrt = env.pool_engine.state.sqrt_price
        # Pair the raw sqrt with its own human price: the fixture's reference
        # close is not exactly the tick's price, and the target band's ~50/50
        # split is what pins the half-notional result.
        price = float(tick_to_price(int(tick), 6, 18))
        capital = float(env.config.episode.agent_capital_usdc)

        # Park the live position entirely below the current tick: it holds only
        # token1 (WETH), so its token0 leg is zero and the target band's ~50 %
        # token0 leg is the whole swapped imbalance.
        base = (int(tick) // int(spacing)) * int(spacing)
        old_lower = Tick(base - 20 * int(spacing))
        old_upper = Tick(base - 10 * int(spacing))
        old = initial_deposit(sqrt, price, old_lower, old_upper, capital, spacing)
        assert old.amount0(sqrt) == 0.0
        assert env._position_id is not None
        env.pool_engine.close_position(env._position_id)
        env._position_id = env.pool_engine.open_position(
            old_lower, old_upper, old.liquidity
        )
        env._position = env.pool_engine.positions[env._position_id]

        action = Action(action_type="rebalance", center_offset=0, width=10)
        target = initial_deposit(
            sqrt,
            price,
            action.lower_tick(tick, spacing),
            action.upper_tick(tick, spacing),
            capital,
            spacing,
        )
        expected_swap = abs(old.amount0(sqrt) - target.amount0(sqrt))
        assert expected_swap == pytest.approx(capital / 2.0, rel=0.02)

        _gas, slippage_cost, executed = env._rebalance(
            action, sqrt, price, env._last_tape_row
        )

        assert executed is True
        rate = _FEE_TIER_BPS / 1_000_000 + 5 / 10_000
        assert slippage_cost == pytest.approx(expected_swap * rate, rel=1e-9)
        # The pre-S18 formula charged ``capital``; the swap is ~half of it.
        assert slippage_cost == pytest.approx(capital * rate / 2.0, rel=0.02)


class TestValidationAndExhaustion:
    def test_rejects_non_positive_initial_width(
        self, tiny_market_view: MarketView, mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        with pytest.raises(ValueError, match="initial_width"):
            LpEnvironment(
                tiny_market_view,
                _fresh_pool_engine(),
                ReplayPriceProcess(tiny_market_view, 0, 100),
                mock_gas_model,
                mock_slippage_model,
                SimConfig(),
                initial_width=0,
            )

    def test_rejects_engine_with_preexisting_positions(
        self, tiny_market_view: MarketView, mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        engine = _fresh_pool_engine()
        engine.open_position(Tick(196200), Tick(196320), 1.0)
        with pytest.raises(EnvError, match="no agent positions"):
            LpEnvironment(
                tiny_market_view,
                engine,
                ReplayPriceProcess(tiny_market_view, 0, 100),
                mock_gas_model,
                mock_slippage_model,
                SimConfig(),
            )

    def test_rejects_non_positive_episode_budget(
        self, tiny_market_view: MarketView, mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        config = SimConfig(episode=tiny_episode(duration_days=0, step_minutes=10))
        env = LpEnvironment(
            tiny_market_view,
            _fresh_pool_engine(),
            _ConstantPriceProcess(3004.0),
            mock_gas_model,
            mock_slippage_model,
            config,
        )
        with pytest.raises(EnvError, match="non-positive step budget"):
            env.reset(seed=0)

    def test_current_il_total_is_zero_before_reset(
        self, tiny_market_view: MarketView, mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        """The defensive pre-reset guard returns 0.0 rather than raising."""
        env = LpEnvironment(
            tiny_market_view,
            _fresh_pool_engine(),
            _ConstantPriceProcess(3004.0),
            mock_gas_model,
            mock_slippage_model,
            SimConfig(),
        )
        assert env._current_il_total(1.0, 3004.0) == 0.0

    def test_price_exhaustion_terminates_episode(
        self, tiny_market_view: MarketView, mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        config = SimConfig(episode=tiny_episode(duration_days=1, step_minutes=10))
        env = LpEnvironment(
            tiny_market_view,
            _fresh_pool_engine(),
            _ExhaustingPriceProcess(3004.0, 1),
            mock_gas_model,
            mock_slippage_model,
            config,
        )
        env.reset(seed=0)
        env.step(0)
        _, _, terminated, truncated, _ = env.step(0)
        assert terminated is True
        assert truncated is False
        # A terminated episode rejects further steps.
        with pytest.raises(EnvError):
            env.step(0)

    def test_rebalance_at_price_exhaustion_executes_at_the_decision_price(
        self, tiny_market_view: MarketView, mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        """ADR-031: the action no longer needs a next bar, so it executes.

        The old ordering decided on the (missing) next bar: an exhausted market
        skipped the rebalance and reported it rejected.  The action now executes
        at the decision state first and the episode then terminates, so the
        turnover is real and gas is charged -- matching the backtester.
        """
        config = SimConfig(episode=tiny_episode(duration_days=1, step_minutes=10))
        env = LpEnvironment(
            tiny_market_view,
            _fresh_pool_engine(),
            _ExhaustingPriceProcess(3004.0, 1),
            mock_gas_model,
            mock_slippage_model,
            config,
        )
        env.reset(seed=0)
        env.step(0)
        first_id = env.position_id
        assert first_id is not None
        first_lower = env.pool_engine.positions[first_id].tick_lower
        rebalance = Action(action_type="rebalance", center_offset=0, width=1)
        index = env._action_lookup.index(rebalance)
        _, _, terminated, _, info = env.step(index)
        assert terminated is True
        assert info["action_type"] == "rebalance"
        assert "rebalance_rejected" not in info
        assert info["reward_breakdown_raw"]["gas"] < 0.0
        assert env.position_id != first_id
        assert info["tick_lower"] != first_lower

    def test_exhaustion_fallback_marks_at_the_last_price(
        self, tiny_market_view: MarketView, mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        """The terminal fallback reuses the last mark, not the episode entry.

        The mark price is no longer rebased at rebalances, so using
        ``_entry_price`` in the exhaustion branch would pair the episode-entry
        human price with the engine's current raw sqrt (ADR-029 follow-up).
        """
        env = LpEnvironment(
            tiny_market_view,
            _fresh_pool_engine(),
            _SequenceExhaustingPriceProcess([3004.0, 3010.0]),
            mock_gas_model,
            mock_slippage_model,
            SimConfig(
                episode=tiny_episode(duration_days=1, step_minutes=10),
                price_mode=PriceMode.CALIBRATED,
            ),
        )
        env.reset(seed=0)
        env.step(0)  # 3004
        env.step(0)  # 3010
        price, _, _, _, exhausted = env._advance_calibrated()
        assert exhausted is True
        assert price == pytest.approx(3010.0)


# ---------------------------------------------------------------------------
# External pool seeding + warmup (ADR-018/ADR-019, reviewer finding 3)
# ---------------------------------------------------------------------------
_SEEDED_T = 196242
_SEEDED_STEP = timedelta(minutes=10)
_SEEDED_START = datetime(2024, 6, 1, tzinfo=UTC)
_SEEDED_BASE_L = 5.0e6
_SEEDED_NEW_L = 2.0e6
_SEEDED_POOL = "0x" + "aa" * 20
_SEEDED_GLOBAL_SENTINEL = -(2**31)


def _seeded_tape_row(
    *,
    seq: int,
    block: int,
    when: datetime,
    event_type: str,
    tick: int | None = None,
    amount0: str | None = None,
    amount1: str | None = None,
    tick_lower: int | None = None,
    tick_upper: int | None = None,
    liquidity_amount: str | None = None,
) -> dict[str, object]:
    price = (
        float(tick_to_price(tick, 6, 18)) if tick is not None else float(
            tick_to_price(_SEEDED_T, 6, 18)
        )
    )
    return {
        "block_number": block,
        "log_index": 0,
        "block_timestamp": when,
        "tx_hash": f"0x{seq:064x}",
        "pool_address": _SEEDED_POOL,
        "event_type": event_type,
        "amount0": amount0,
        "amount1": amount1,
        "sqrt_price_x96": str(tick_to_sqrt_price_x96(tick)) if tick is not None else None,
        "liquidity": str(int(_SEEDED_BASE_L)) if event_type == "swap" else None,
        "tick": tick,
        "sender": "0x" + "bb" * 20,
        "recipient": "0x" + "dd" * 20 if event_type == "swap" else None,
        "owner": "0x" + "cc" * 20 if event_type in ("mint", "burn") else None,
        "tick_lower": tick_lower,
        "tick_upper": tick_upper,
        "liquidity_amount": liquidity_amount,
        "seq": seq,
        # The as-of reference close is the exact human price of the swap's x96.
        "price_pool": price,
        "price_reference": price,
        "regime": "sideways",
        "base_fee_per_gas": str(20 * 10**9),
        "priority_fee_p50_wei": str(10**9),
        "fee_growth_global_0_x128": "0",
        "fee_growth_global_1_x128": "0",
        "fee_growth_source": "exact",
    }


def _seeded_market_view() -> MarketView:
    """A synthetic lane with a real ``pool_state`` snapshot + warmup rows.

    The lane starts at tick ``T`` with ``5e6`` external liquidity in
    ``[T-600, T+600]``.  Three warmup rows then move the pool to tick ``T+300``,
    mint a second ``2e6`` position ``[T+240, T+6000]`` and cross its lower bound,
    so the warmed window-start state has ``7e6`` active liquidity and a non-zero
    token1 global accumulator.  The window rows are direction-consistent swaps
    (token1 in => tick up, token0 in => tick down) that move within the agent's
    initial range.
    """
    t = _SEEDED_T
    warm0 = _SEEDED_START - 3 * _SEEDED_STEP
    warm1 = _SEEDED_START - 2 * _SEEDED_STEP
    warm2 = _SEEDED_START - _SEEDED_STEP
    warm_rows = [
        _seeded_tape_row(
            seq=0, block=0, when=warm0, event_type="swap", tick=t + 120,
            amount0="-1", amount1="1000000000",
        ),
        _seeded_tape_row(
            seq=1, block=1, when=warm1, event_type="mint",
            tick_lower=t + 240, tick_upper=t + 6000,
            liquidity_amount=str(int(_SEEDED_NEW_L)),
        ),
        _seeded_tape_row(
            seq=2, block=2, when=warm2, event_type="swap", tick=t + 300,
            amount0="-1", amount1="500000000",
        ),
    ]
    # (from_tick, to_tick, direction); ``token1 in => tick up``, ``token0 in =>
    # tick down`` (CONTRACTS.md §4.1).
    window_moves = [
        (t + 300, t + 300, "up"),
        (t + 300, t + 240, "down"),
        (t + 240, t + 360, "up"),
        (t + 360, t + 290, "down"),
        (t + 290, t + 390, "up"),
    ]
    window_rows = [
        _seeded_tape_row(
            seq=3 + i,
            block=3 + i,
            when=_SEEDED_START + i * _SEEDED_STEP,
            event_type="swap",
            tick=to_tick,
            amount0="-1" if direction == "up" else "1000000000",
            amount1="1000000000" if direction == "up" else "-1",
        )
        for i, (_from_tick, to_tick, direction) in enumerate(window_moves)
    ]
    window_ticks = [to_tick for _from, to_tick, _dir in window_moves]
    tape = pl.from_arrow(
        pa.table(
            {k: [r[k] for r in window_rows] for k in window_rows[0]},
            schema=EVENT_TAPE_SCHEMA,
        )
    )
    warmup = pl.from_arrow(
        pa.table(
            {k: [r[k] for r in warm_rows] for k in warm_rows[0]},
            schema=EVENT_TAPE_SCHEMA,
        )
    )

    pool_rows: list[dict[str, object]] = [
        {
            "block_number": 0,
            "tick": _SEEDED_GLOBAL_SENTINEL,
            "fee_growth_outside_0_x128": "0",
            "fee_growth_outside_1_x128": "0",
            "liquidity_gross": "0",
            "liquidity_net": "0",
            "initialized": False,
            "fee_growth_global_0_x128": "0",
            "fee_growth_global_1_x128": "0",
            "current_tick": t,
            "current_liquidity": str(int(_SEEDED_BASE_L)),
            "source": "onchain",
            "pool_address": _SEEDED_POOL,
            "fee_protocol": 0,
        },
        {
            "block_number": 0,
            "tick": t - 600,
            "fee_growth_outside_0_x128": "0",
            "fee_growth_outside_1_x128": "0",
            "liquidity_gross": str(int(_SEEDED_BASE_L)),
            "liquidity_net": str(int(_SEEDED_BASE_L)),
            "initialized": True,
            "fee_growth_global_0_x128": "0",
            "fee_growth_global_1_x128": "0",
            "current_tick": t,
            "current_liquidity": str(int(_SEEDED_BASE_L)),
            "source": "onchain",
            "pool_address": _SEEDED_POOL,
            "fee_protocol": 0,
        },
        {
            "block_number": 0,
            "tick": t + 600,
            "fee_growth_outside_0_x128": "0",
            "fee_growth_outside_1_x128": "0",
            "liquidity_gross": str(int(_SEEDED_BASE_L)),
            "liquidity_net": str(-int(_SEEDED_BASE_L)),
            "initialized": True,
            "fee_growth_global_0_x128": "0",
            "fee_growth_global_1_x128": "0",
            "current_tick": t,
            "current_liquidity": str(int(_SEEDED_BASE_L)),
            "source": "onchain",
            "pool_address": _SEEDED_POOL,
            "fee_protocol": 0,
        },
    ]
    pool_state = pl.from_arrow(
        pa.table(
            {k: [r[k] for r in pool_rows] for k in pool_rows[0]},
            schema=FEE_GROWTH_SCHEMA,
        )
    )

    window_times = [_SEEDED_START + i * _SEEDED_STEP for i in range(len(window_rows))]
    reference = pl.from_arrow(
        pa.table(
            {
                "open_time": [wt - timedelta(seconds=1) for wt in window_times],
                "close_time": window_times,
                "open": [float(tick_to_price(tk, 6, 18)) for tk in window_ticks],
                "high": [float(tick_to_price(tk, 6, 18)) * 1.0001 for tk in window_ticks],
                "low": [float(tick_to_price(tk, 6, 18)) * 0.9999 for tk in window_ticks],
                "close": [float(tick_to_price(tk, 6, 18)) for tk in window_ticks],
                "volume_base": [1.0] * len(window_rows),
                "quote_volume": [3000.0] * len(window_rows),
                "trades": [1] * len(window_rows),
                "symbol": ["ETHUSDC"] * len(window_rows),
                "is_gap_filled": [False] * len(window_rows),
            },
            schema=REFERENCE_SCHEMA,
        )
    )
    regimes = pl.from_arrow(
        pa.table(
            {
                "timestamp": window_times,
                "sigma_rv": [0.5] * len(window_rows),
                "mu": [0.0] * len(window_rows),
                "regime": ["sideways"] * len(window_rows),
                "window_complete": [True] * len(window_rows),
                "symbol": ["ETHUSDC"] * len(window_rows),
            },
            schema=REGIME_SCHEMA,
        )
    )
    gas = pl.from_arrow(
        pa.table(
            {
                "block_number": list(range(len(window_rows))),
                "base_fee_per_gas": [str(20 * 10**9)] * len(window_rows),
                "gas_used": [15_000_000] * len(window_rows),
                "gas_limit": [30_000_000] * len(window_rows),
                "priority_fee_p50_wei": [str(10**9)] * len(window_rows),
                "priority_fee_p90_wei": [str(3 * 10**9)] * len(window_rows),
            },
            schema=GAS_SCHEMA,
        )
    )
    return MarketView(
        train=tape,
        eval=None,
        reference=reference,
        regimes=regimes,
        gas=gas,
        train_start_utc=window_times[0],
        train_end_utc=window_times[-1],
        eval_start_utc=window_times[-1] + timedelta(minutes=1),
        eval_end_utc=window_times[-1] + timedelta(days=1),
        is_train=True,
        episode_config=tiny_episode(duration_days=1, step_minutes=10),
        pool=None,
        pool_state=pool_state,
        warmup=warmup,
    )


def _seeded_env(view: MarketView) -> LpEnvironment:
    from undertow.sim.evaluate.protocol import build_replay_env

    config = SimConfig(
        episode=tiny_episode(duration_days=1, step_minutes=10), seed=0
    )
    return build_replay_env(view, config)


def _pool_fingerprint(env: LpEnvironment) -> tuple[object, ...]:
    state = env.pool_engine.state
    ticks = tuple(
        sorted(
            (
                int(tick),
                ts.fee_growth_outside_0,
                ts.fee_growth_outside_1,
                ts.liquidity_gross,
                ts.liquidity_net,
                ts.initialized,
            )
            for tick, ts in env.pool_engine.ticks.items()
        )
    )
    return (
        env._start_seq,
        env._end_seq,
        state.sqrt_price,
        state.tick,
        state.sqrt_price_x96,
        state.liquidity,
        state.fee_growth_global_0,
        state.fee_growth_global_1,
        state.fee_protocol,
        ticks,
    )


class TestExternalPoolSeeding:
    """ADR-018/019: reset seeds from the snapshot and replays warmup to the window."""

    def test_reset_seeds_and_warms_to_window_start(self) -> None:
        env = _seeded_env(_seeded_market_view())
        env.reset(seed=11)

        assert env._external_seeded is True
        state = env.pool_engine.state
        # The warmup moved T -> T+300 and crossed the new 2e6 mint's lower bound.
        assert state.tick == Tick(_SEEDED_T + 300)
        assert state.liquidity == pytest.approx(_SEEDED_BASE_L + _SEEDED_NEW_L)
        assert state.fee_growth_global_1 > 0.0
        minted_tick = Tick(_SEEDED_T + 240)
        assert minted_tick in env.pool_engine.ticks
        assert env.pool_engine.ticks[minted_tick].liquidity_net == pytest.approx(
            _SEEDED_NEW_L
        )
        # The per-episode base captured at reset is the warmed state, not the raw
        # snapshot (5e6 at tick T).
        assert env._base_state.liquidity == pytest.approx(
            _SEEDED_BASE_L + _SEEDED_NEW_L
        )
        assert env._base_state.tick == Tick(_SEEDED_T + 300)

    def test_reset_is_reproducible_and_rebuilds_from_snapshot(self) -> None:
        env = _seeded_env(_seeded_market_view())
        env.reset(seed=7)
        first = _pool_fingerprint(env)

        # Advance the episode, then reset the SAME env: it must rebuild the
        # external pool from the snapshot + warmup, not accumulate on the live
        # engine.
        env.step(0)
        env.step(1)
        env.reset(seed=7)

        assert _pool_fingerprint(env) == first

    @pytest.mark.parametrize("start_seq", [3, 4, 5, 6, 7, 8])
    def test_checkpoint_warmup_matches_full_replay(
        self, monkeypatch: pytest.MonkeyPatch, start_seq: int
    ) -> None:
        """ADR-020: warming from a checkpoint is identical to a full warmup."""

        def fingerprint(stride: int) -> tuple[object, ...]:
            monkeypatch.setattr(
                "undertow.sim.env.lp_env.WARMUP_CHECKPOINT_STRIDE", stride
            )
            env = _seeded_env(_seeded_market_view())
            env._start_seq = start_seq
            env._reset_external_pool()
            return _pool_fingerprint(env)

        assert fingerprint(2) == fingerprint(10**9)

    def test_checkpoint_warmup_replays_at_most_stride(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ADR-020: a reset replays only the gap from the nearest checkpoint."""
        monkeypatch.setattr("undertow.sim.env.lp_env.WARMUP_CHECKPOINT_STRIDE", 2)
        env = _seeded_env(_seeded_market_view())
        env._ensure_warmup_checkpoints()  # build the timeline once, uncounted

        original = LpEnvironment._apply_warmup_row
        replayed: list[dict[str, object]] = []

        def counting(engine: PoolEngine, row: dict[str, object]) -> None:
            replayed.append(row)
            original(engine, row)

        monkeypatch.setattr(LpEnvironment, "_apply_warmup_row", staticmethod(counting))
        env._start_seq = 5  # nearest checkpoint is seq 3 in the 8-row fixture
        env._reset_external_pool()

        assert 0 < len(replayed) <= 2

    def test_alignment_gate_skips_zero_volume_step_for_seeded_views(
        self, tiny_env: LpEnvironment
    ) -> None:
        # A seeded view is tape-driven: the engine's exact x96 survives the
        # step (the alignment step would reset it to None).
        seeded = _seeded_env(_seeded_market_view())
        seeded.reset(seed=3)
        seeded.step(0)
        assert seeded.pool_engine.state.sqrt_price_x96 is not None
        assert seeded.pool_engine.state.sqrt_price_x96 == int(
            seeded._last_tape_row["sqrt_price_x96"]
        )

        # A fixture view (no snapshot) still aligns onto the reference bar, which
        # clears the exact x96.
        tiny_env.reset(seed=42)
        tiny_env.step(0)
        assert tiny_env.pool_engine.state.sqrt_price_x96 is None

    def test_seeded_mark_price_uses_last_tape_event(self) -> None:
        # ADR-019: for a snapshot-seeded view the position is marked at the last
        # processed tape event's own reference close (paired with that event's
        # raw sqrt), not the decision bar's close — a bar-close/event-sqrt mix
        # would inject a one-bar mark-to-market error into IL/PnL parity.
        env = _seeded_env(_seeded_market_view())
        env.reset(seed=5)
        mark_price, _sqrt, _at_time, row, done = env._advance_replay()
        assert done is False
        assert mark_price == pytest.approx(float(row["price_reference"]))

    def test_moving_tape_accrues_fees_and_carries_them_across_rebalance(self) -> None:
        env = _seeded_env(_seeded_market_view())
        env.reset(seed=5)

        fees_total = 0.0
        for _ in range(2):
            _, _, _, _, info = env.step(0)
            fees_total += float(info["reward_breakdown_raw"]["fees"])
        assert fees_total > 0.0
        raw_before, _ = env.pool_engine.position_fees(env.position_id)
        assert raw_before > 0.0

        # ADR-031: the rebalance executes at the decision state, so it collects
        # the fees accrued through the two prior steps.  The step's own interval
        # then accrues to the *new* position -- zero here because the recentred
        # band sits out of range, so the old "collected = prior + this step"
        # identity no longer applies.
        _, _, _, _, rebalance_info = env.step(1)
        collected = float(rebalance_info["collected_fees_usdc"])
        this_step = float(rebalance_info["reward_breakdown_raw"]["fees"])
        assert collected == pytest.approx(fees_total, rel=1e-9, abs=1e-9)
        assert this_step == 0.0
        raw_after, _ = env.pool_engine.position_fees(env.position_id)
        assert raw_after == 0.0


# ---------------------------------------------------------------------------
# S20 — hedge overlay
# ---------------------------------------------------------------------------


def _flat_reference_view(view: MarketView, price: float = 3004.0) -> MarketView:
    """``view`` with every reference close pinned to ``price``.

    Calibrated-mode envs take their entry price from the reference close, so a
    flat reference feed makes the entry mark and the ordered price process
    agree (the tiny fixture's reference feed is a random walk an arbitrary
    distance from any ``_SequencePriceProcess``).
    """
    return replace(
        view,
        reference=view.reference.with_columns(pl.lit(float(price)).alias("close")),
    )


def _hedge_env(
    tiny_market_view: MarketView,
    price_process: object,
    gas_model: object,
    slippage_model: object,
    *,
    enabled: bool,
    ratio: float = 1.0,
    funding_enabled: bool = True,
    reward_hedge: bool = True,
    reward_gas: bool = True,
    reward_slippage: bool = True,
) -> LpEnvironment:
    """A calibrated env with the hedge overlay configured explicitly."""
    return LpEnvironment(
        tiny_market_view,
        _fresh_pool_engine(),
        price_process,
        gas_model,
        slippage_model,
        SimConfig(
            episode=tiny_episode(duration_days=31, step_minutes=10),
            price_mode=PriceMode.CALIBRATED,
            hedge=HedgeConfig(
                enabled=enabled,
                ratio=ratio,
                funding_enabled=funding_enabled,
            ),
            reward=RewardConfig(
                hedge_enabled=reward_hedge,
                gas_enabled=reward_gas,
                slippage_enabled=reward_slippage,
            ),
        ),
    )


def _funding_frame(
    times: list[datetime],
    rates: list[float],
    marks: list[float | None] | None = None,
) -> pl.DataFrame:
    """A minimal funding feed (the three columns the sim reads)."""
    return pl.DataFrame(
        {
            "funding_time": times,
            "symbol": ["ETHUSDC"] * len(times),
            "funding_rate": rates,
            "mark_price": marks if marks is not None else [None] * len(times),
            "is_gap_filled": [False] * len(times),
        }
    )


class TestHedgeOverlay:
    def test_disabled_and_zero_ratio_are_byte_identical(
        self,
        tiny_market_view: MarketView,
        mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        """A disabled hedge changes nothing: 24 dims, last three zero.

        An overlay configured with ``ratio=0`` trades no ETH and must match the
        disabled env exactly on the full vector and the raw breakdown, which is
        the regression guard for the first 21 LP-only dimensions.
        """
        view = _flat_reference_view(tiny_market_view)
        prices = [3004.0, 3008.0, 3002.0, 3010.0]
        disabled = _hedge_env(
            view,
            _SequencePriceProcess(prices),
            mock_gas_model,
            mock_slippage_model,
            enabled=False,
        )
        zero = _hedge_env(
            view,
            _SequencePriceProcess(prices),
            mock_gas_model,
            mock_slippage_model,
            enabled=True,
            ratio=0.0,
        )
        obs_a, info_a = disabled.reset(seed=0)
        obs_b, info_b = zero.reset(seed=0)
        assert np.array_equal(obs_a, obs_b)
        assert obs_a.shape == (OBSERVATION_VECTOR_LENGTH,)
        assert obs_a[21:].tolist() == [0.0, 0.0, 0.0]
        assert info_a["reward_breakdown_raw"] == info_b["reward_breakdown_raw"]

        for _ in range(3):
            obs_a, reward_a, _, _, info_a = disabled.step(0)
            obs_b, reward_b, _, _, info_b = zero.step(0)
            assert np.array_equal(obs_a, obs_b)
            assert reward_a == reward_b
            assert info_a["reward_breakdown_raw"] == info_b["reward_breakdown_raw"]
            assert info_a["hedge_equity"] == 0.0
            for key in ("hedge_pnl", "hedge_funding", "hedge_fee"):
                assert info_a["reward_breakdown_raw"][key] == 0.0

    def test_hedge_offsets_a_large_price_move(
        self,
        tiny_market_view: MarketView,
        mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        """With ``ratio=1`` the short tracks the LP ETH delta and variance falls.

        A pure (fee-free) price rise is the cleanest delta-neutrality probe: the
        LP loses on its WETH leg and the short gains the same magnitude, so the
        total equity move is far smaller than the unhedged one.
        """
        view = _flat_reference_view(tiny_market_view)
        prices = [3004.0, 3050.0, 3100.0, 3150.0, 3200.0, 3250.0, 3300.0]
        unhedged = _hedge_env(
            view,
            _SequencePriceProcess(prices),
            mock_gas_model,
            mock_slippage_model,
            enabled=False,
        )
        hedged = _hedge_env(
            view,
            _SequencePriceProcess(prices),
            mock_gas_model,
            mock_slippage_model,
            enabled=True,
            ratio=1.0,
        )
        capital = hedged.config.episode.agent_capital_usdc
        unhedged.reset(seed=0)
        reset_vector, _ = hedged.reset(seed=0)
        hedge_pnl_rising = 0.0
        for _ in range(len(prices)):
            _, _, _, _, unhedged_info = unhedged.step(0)
            _, _, _, _, hedged_info = hedged.step(0)
            hedge_pnl_rising += float(
                hedged_info["reward_breakdown_raw"]["hedge_pnl"]
            )
        unhedged_pnl = float(unhedged_info["equity"]) - capital
        hedged_pnl = (
            float(hedged_info["equity"]) + float(hedged_info["hedge_equity"])
        ) - capital
        assert abs(unhedged_pnl) > 100.0  # the rise genuinely moved the LP value
        # The overlay removes the directional move; the residual is the LP's
        # convexity/IL cost, well under 1 % of capital here.
        assert abs(hedged_pnl) < abs(unhedged_pnl) * 0.5
        assert abs(hedged_pnl) < 0.01 * capital
        # The short loses in total as the price rises (before funding).
        assert hedge_pnl_rising < 0.0
        from undertow.sim.evaluate import observation_from_vector

        obs = observation_from_vector(reset_vector.tolist())
        assert obs.hedge_short_ratio == pytest.approx(1.0, rel=1e-9)

    def test_hedge_trade_executes_at_the_decision_price(
        self,
        tiny_market_view: MarketView,
        mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        """ADR-031: the re-hedge is priced at the decision mark, not the next bar."""
        env = _hedge_env(
            _flat_reference_view(tiny_market_view),
            _SequencePriceProcess([3004.0, 3010.0, 4000.0]),
            mock_gas_model,
            mock_slippage_model,
            enabled=True,
        )
        env.reset(seed=0)
        env.step(0)  # decision at entry -> mark 3004
        env.step(0)  # decision at 3004 -> mark 3010
        assert env._last_mark_price == pytest.approx(3010.0)
        rebalance = Action(action_type="rebalance", center_offset=0, width=2)
        index = env._action_lookup.index(rebalance)
        env.step(index)  # decision at 3010 -> advance to 4000
        assert env._last_mark_price == pytest.approx(4000.0)
        trade = env._hedge.last_trade
        assert trade is not None
        assert trade.mark == pytest.approx(3010.0)
        assert trade.mark != pytest.approx(4000.0)

    def test_funding_accrues_only_at_settlements(
        self,
        tiny_market_view: MarketView,
        mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        """Funding is charged once per settlement, at the sim's own mark."""
        closes = tiny_market_view.reference.sort("close_time")
        close_times = closes["close_time"].to_list()
        settlement_times = [close_times[10], close_times[40]]
        rates = [0.001, -0.002]
        # Flat reference closes pin the funding marks at 3004 exactly.
        view = replace(
            _flat_reference_view(tiny_market_view, 3004.0),
            funding=_funding_frame(settlement_times, rates),
        )
        env = _hedge_env(
            view,
            _ConstantPriceProcess(3004.0),
            mock_gas_model,
            mock_slippage_model,
            enabled=True,
        )
        env.reset(seed=0)
        # Let the pool settle onto the constant process price so the short is
        # constant over the settlements (the first step re-bases the entry).
        for _ in range(5):
            env.step(0)
        short = env._hedge.short_size
        assert short > 0.0

        funding_steps = 0
        funding_total = 0.0
        for _ in range(500):
            _, _, terminated, truncated, info = env.step(0)
            accrual = float(info["reward_breakdown_raw"]["hedge_funding"])
            if accrual != 0.0:
                funding_steps += 1
                funding_total += accrual
            if terminated or truncated:
                break
        assert funding_steps == 2
        expected = short * 3004.0 * (rates[0] + rates[1])
        assert funding_total == pytest.approx(expected, rel=1e-12)
        # The stream's nullable mark price is not used: the settlements were
        # accrued at the reference closes, not at a stream mark.
        assert env._hedge.funding_pnl == pytest.approx(funding_total)

    def test_funding_disabled_gates_the_accrual(
        self,
        tiny_market_view: MarketView,
        mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        closes = tiny_market_view.reference.sort("close_time")
        view = replace(
            _flat_reference_view(tiny_market_view),
            funding=_funding_frame(
                [closes["close_time"][10]], [0.001]
            ),
        )
        env = _hedge_env(
            view,
            _ConstantPriceProcess(3004.0),
            mock_gas_model,
            mock_slippage_model,
            enabled=True,
            funding_enabled=False,
        )
        env.reset(seed=0)
        for _ in range(500):
            _, _, terminated, truncated, info = env.step(0)
            assert info["reward_breakdown_raw"]["hedge_funding"] == 0.0
            if terminated or truncated:
                break

    def test_hedge_terms_gated_by_reward_flag_but_pnl_net_is_not(
        self,
        tiny_market_view: MarketView,
        mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        view = _flat_reference_view(tiny_market_view)
        prices = [3004.0, 3060.0]
        gated_env = _hedge_env(
            view,
            _SequencePriceProcess(prices),
            mock_gas_model,
            mock_slippage_model,
            enabled=True,
            reward_hedge=False,
        )
        ungated_env = _hedge_env(
            view,
            _SequencePriceProcess(prices),
            mock_gas_model,
            mock_slippage_model,
            enabled=True,
            reward_hedge=True,
        )
        gated_env.reset(seed=0)
        ungated_env.reset(seed=0)
        _, _, _, _, gated_info = gated_env.step(0)
        _, _, _, _, ungated_info = ungated_env.step(0)
        gated = gated_info["reward_breakdown_raw"]
        ungated = ungated_info["reward_breakdown_raw"]

        # The gate zeroes the terms in the learning reward...
        assert gated["hedge_pnl"] == 0.0
        assert gated["hedge_funding"] == 0.0
        assert gated["hedge_fee"] == 0.0
        assert ungated["hedge_pnl"] != 0.0
        assert ungated["reward"] - gated["reward"] == pytest.approx(
            ungated["hedge_pnl"] + ungated["hedge_funding"] + ungated["hedge_fee"]
        )
        # ... but the ungated economic total is identical either way.
        assert gated["pnl_net"] == pytest.approx(ungated["pnl_net"])

    def test_hedge_equity_frac_is_reported(
        self,
        tiny_market_view: MarketView,
        mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        env = _hedge_env(
            _flat_reference_view(tiny_market_view),
            _ConstantPriceProcess(3004.0),
            mock_gas_model,
            mock_slippage_model,
            enabled=True,
        )
        vector, info = env.reset(seed=0)
        from undertow.sim.evaluate import observation_from_vector

        obs = observation_from_vector(vector.tolist())
        # The entry trade paid the perp fee, so hedge equity is -fee / capital.
        assert obs.hedge_equity_frac == pytest.approx(
            float(info["hedge_equity"]) / env.config.episode.agent_capital_usdc
        )
        assert obs.hedge_equity_frac < 0.0
        assert obs.hedge_short_ratio == pytest.approx(1.0, rel=1e-3)

    def test_disabled_hedge_zeroes_appended_dims_with_funding_present(
        self,
        tiny_market_view: MarketView,
        mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        """S20 test item 5: all three appended dims are zero when disabled.

        The view carries a funding settlement at the window start, so an
        ungated ``funding_rate_recent`` would be non-zero; the disabled run
        must still report the LP-only vector and its first 21 dims must equal
        the enabled run's (the overlay never touches the LP state).
        """
        settlement = tiny_market_view.active_start_utc
        view = replace(
            _flat_reference_view(tiny_market_view),
            funding=_funding_frame([settlement], [0.001]),
        )
        disabled = _hedge_env(
            view,
            _ConstantPriceProcess(3004.0),
            mock_gas_model,
            mock_slippage_model,
            enabled=False,
        )
        enabled = _hedge_env(
            view,
            _ConstantPriceProcess(3004.0),
            mock_gas_model,
            mock_slippage_model,
            enabled=True,
        )
        disabled_obs, disabled_info = disabled.reset(seed=0)
        enabled_obs, _enabled_info = enabled.reset(seed=0)
        # Non-vacuous: the enabled run reads the stream's rate at the same time.
        assert enabled_obs[23] == pytest.approx(0.001)
        assert disabled_obs[21:].tolist() == [0.0, 0.0, 0.0]
        assert np.array_equal(disabled_obs[:21], enabled_obs[:21])
        assert disabled_info["reward_breakdown_economics"]["hedge_funding"] == 0.0

        for _ in range(3):
            disabled_obs, _, terminated, truncated, _ = disabled.step(0)
            assert disabled_obs[21:].tolist() == [0.0, 0.0, 0.0]
            if terminated or truncated:
                break

    def test_economics_payload_is_ungated_by_reward_flags(
        self,
        tiny_market_view: MarketView,
        mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        """The ungated payload keeps every term an RQ1 ablation zeroes.

        This is the environment-side half of the S20 parity fix: with gas,
        slippage and hedge ablated in the *learning* reward, the raw breakdown
        reads zero for all three, while the economics payload the parity
        harness consumes still holds the true magnitudes.
        """
        env = _hedge_env(
            _flat_reference_view(tiny_market_view),
            _SequencePriceProcess([3004.0, 3060.0, 3090.0]),
            mock_gas_model,
            mock_slippage_model,
            enabled=True,
            reward_hedge=False,
            reward_gas=False,
            reward_slippage=False,
        )
        env.reset(seed=0)
        rebalance = Action(action_type="rebalance", center_offset=0, width=2)
        index = env._action_lookup.index(rebalance)
        _, _, _, _, info = env.step(index)
        raw = info["reward_breakdown_raw"]
        economics = info["reward_breakdown_economics"]

        # The gated view zeroes the ablated terms...
        assert raw["gas"] == 0.0
        assert raw["slippage"] == 0.0
        assert raw["hedge_pnl"] == 0.0
        assert raw["hedge_funding"] == 0.0
        assert raw["hedge_fee"] == 0.0
        # ... while the economics payload keeps them (the rebalance paid gas
        # and slippage; the mark move and reset trade moved the hedge).
        assert economics["gas"] < 0.0
        assert economics["slippage"] < 0.0
        assert economics["hedge_pnl"] != 0.0
        assert economics["hedge_fee"] < 0.0
        # The ungated net is shared: `pnl_net` never depended on the flags.
        assert economics["pnl_net"] == pytest.approx(raw["pnl_net"])
        assert economics["fees"] == raw["fees"]
        assert economics["il_change"] == raw["il_change"]


# ---------------------------------------------------------------------------
# S20 golden guard: the disabled build reproduces the LP-only output exactly
# ---------------------------------------------------------------------------
# Captured with the scenario in ``TestDisabledGoldenGuard`` below: shared tiny
# fixture, flat reference feed, ``_SequencePriceProcess([3004, 3010, 2995,
# 3005, 3015])``, calibrated mode, reset(seed=0) then holds at steps 1--2, a
# (center_offset=0, width=2) rebalance at step 3, a hold at step 4.  Both the
# observation vector and the (gated/ungated identical) USDC reward breakdown
# were re-captured after ADR-035 changed the fee share, so any accidental
# change to the disabled path fails on the raw float values rather than on a
# loose tolerance.
_GOLDEN_DISABLED_OBS: dict[str, list[float]] = {
    "reset": [
        3004.307167600141,
        18244.32638576171,
        196242.0000000001,
        0.0,
        0.0,
        0.0,
        1.0,
        99999.9999998899,
        0.0,
        0.0,
        0.0,
        9.999999999999986e+17,
        3000.0,
        21000000000.000008,
        3004.307167600141,
        0.0,
        0.0,
        4463.0,
        10.7,
        10.3,
        21.0,
    ],
    "step1": [
        3003.9999999999977,
        18245.259129220653,
        196242.99999999994,
        0.0,
        0.0,
        0.0,
        1.0,
        99994.78781637354,
        8.573579989668916e-05,
        0.0,
        -4.2143421451328325e-08,
        9.999999999999986e+17,
        3000.0,
        21000000000.000008,
        3003.9999999999977,
        0.0,
        1.0,
        4462.0,
        10.716666666666667,
        10.283333333333333,
        21.0,
    ],
    "step2": [
        3003.9999999999977,
        18227.065414412238,
        196223.0,
        0.0,
        0.0,
        0.0,
        1.0,
        99996.39275793757,
        0.00017147159979337832,
        0.0,
        1.600727221928537e-05,
        9.999999999999986e+17,
        3000.0,
        21000000000.000008,
        3003.9999999999977,
        0.0,
        2.0,
        4461.0,
        10.383333333333333,
        10.616666666666667,
        21.0,
    ],
    "step3": [
        3003.9999999999977,
        18272.652143588803,
        196273.00000000012,
        0.0,
        0.0,
        0.0,
        1.0,
        99925.73301608098,
        0.0003261846826820426,
        0.0,
        -0.0006905901463470945,
        9.999999999999986e+17,
        3000.0,
        21000000000.000008,
        3003.9999999999977,
        0.0,
        3.0,
        4460.0,
        3.216666666666667,
        1.7833333333333334,
        5.0,
    ],
    "step4": [
        3003.9999999999977,
        18242.223059788008,
        196238.99999999994,
        0.0,
        0.0,
        0.0,
        1.0,
        99910.86910225668,
        0.0006523693653640852,
        0.0,
        -0.0008392292845899647,
        9.999999999999986e+17,
        3000.0,
        21000000000.000008,
        3003.9999999999977,
        0.0,
        4.0,
        4459.0,
        2.65,
        2.35,
        5.0,
    ],
}

_GOLDEN_DISABLED_RAW: dict[str, dict[str, float]] = {
    "reset": {
        "fees": 0.0,
        "gas": 0.0,
        "slippage": 0.0,
        "il_change": 0.0,
        "risk_penalty": 0.0,
        "reward": 0.0,
        "pnl_net": 0.0,
        "equity": 100000.0,
    },
    "step1": {
        "fees": 8.573579989668916e-05,
        "gas": -0.0,
        "slippage": -0.0,
        "il_change": -0.004214342145132832,
        "risk_penalty": 0.0,
        "reward": -0.0041286063452361434,
        "pnl_net": -0.0041286063452361434,
        "equity": 99994.78790210928,
    },
    "step2": {
        "fees": 8.573579989668916e-05,
        "gas": -0.0,
        "slippage": -0.0,
        "il_change": -1.4418992520513711,
        "risk_penalty": 0.0,
        "reward": -1.4418135162514745,
        "pnl_net": -1.4418135162514745,
        "equity": 100095.07496569946,
    },
    "step3": {
        "fees": 0.0003261846826820426,
        "gas": -76.4775,
        "slippage": -6.16750327591709,
        "il_change": -25.06211177939258,
        "risk_penalty": 0.0,
        "reward": -107.706788870627,
        "pnl_net": -107.706788870627,
        "equity": 99733.04598406279,
    },
    "step4": {
        "fees": 0.0003261846826820426,
        "gas": -0.0,
        "slippage": -0.0,
        "il_change": 25.97291066442267,
        "risk_penalty": 0.0,
        "reward": 25.97323684910535,
        "pnl_net": 25.97323684910535,
        "equity": 99928.56734942259,
    },
}


class TestDisabledGoldenGuard:
    """The disabled overlay reproduces the LP-only numbers exactly.

    Golden constants captured with the scenario below and re-captured after
    ADR-035 changed the fee share; see the comments above the constants.  The
    first 21 observation dims must match the recorded values pushed through the
    deterministic ADR-034 log compression bit for bit, the eight original
    reward fields must match raw bit for bit, and the three appended hedge dims
    must be exactly zero.
    """

    def test_disabled_path_reproduces_head_observations_and_reward(
        self,
        tiny_market_view: MarketView,
        mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        env = _hedge_env(
            _flat_reference_view(tiny_market_view),
            _SequencePriceProcess([3004.0, 3010.0, 2995.0, 3005.0, 3015.0]),
            mock_gas_model,
            mock_slippage_model,
            enabled=False,
        )
        observation, info = env.reset(seed=0)

        def _ledger() -> tuple[float, float]:
            assert env.position_id is not None
            return env.pool_engine.position_fees(env.position_id)

        rows: list[
            tuple[
                str,
                np.ndarray,
                dict[str, float],
                dict[str, float],
                tuple[float, float],
            ]
        ] = [
            (
                "reset",
                observation,
                dict(info["reward_breakdown_raw"]),
                dict(info["reward_breakdown_economics"]),
                _ledger(),
            )
        ]
        rebalance = Action(action_type="rebalance", center_offset=0, width=2)
        rebalance_index = env._action_lookup.index(rebalance)
        for step, action_idx in enumerate([0, 0, rebalance_index, 0]):
            observation, _reward, terminated, truncated, info = env.step(action_idx)
            rows.append(
                (
                    f"step{step + 1}",
                    observation,
                    dict(info["reward_breakdown_raw"]),
                    dict(info["reward_breakdown_economics"]),
                    _ledger(),
                )
            )
            if terminated or truncated:  # pragma: no cover - long episode budget
                break

        assert [label for label, *_rest in rows] == list(_GOLDEN_DISABLED_OBS)
        for label, obs, raw, economics, ledger in rows:
            assert obs.shape == (OBSERVATION_VECTOR_LENGTH,)
            # ADR-034 log-compresses the heavy-tailed dims, so compare in the
            # compressed space: the current vector must be bit-for-bit the
            # recorded HEAD values pushed through the same deterministic
            # transform.  The reward breakdown below stays bit-for-bit exact
            # in raw units.
            assert np.array_equal(
                obs[:21],
                scale_vector(np.asarray(_GOLDEN_DISABLED_OBS[label])),
            )
            assert obs[21:].tolist() == [0.0, 0.0, 0.0]
            # ADR-035 invariant: the observation's fee dims must be the same
            # ledger the reward reads.  This stops a future re-capture from
            # silently freezing a growth-path fallback mismatch.  Indices 8/9
            # are ADR-034 log-compressed, so compare the raw values.
            raw_obs = unscale_vector(obs)
            assert raw_obs[8] == pytest.approx(
                ledger[0] / 10.0**env._position.dec0, rel=1e-9, abs=1e-12
            )
            assert raw_obs[9] == pytest.approx(
                ledger[1] / 10.0**env._position.dec1, rel=1e-9, abs=1e-12
            )
            golden_raw = _GOLDEN_DISABLED_RAW[label]
            assert set(golden_raw) <= set(raw)
            for key, expected in golden_raw.items():
                assert raw[key] == expected, (label, key, raw[key], expected)
            for key in ("hedge_pnl", "hedge_funding", "hedge_fee"):
                assert raw[key] == 0.0
            # No ablation is configured, so the ungated economics view is
            # field-for-field the same as the gated raw breakdown.
            assert economics == raw
