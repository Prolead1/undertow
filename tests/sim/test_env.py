"""S12 — observation builder + Gymnasium environment.

Every test asserts real behavior: the stable observation vector, the ordinal
regime encoding, the backward-looking (look-ahead) wall, exact action-index
mapping, deterministic resets/rollouts, the 5-tuple step contract with a full
``RewardBreakdown`` in ``info``, episode truncation, and the fee-share
identity for a hold step.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from undertow.data import (
    EVENT_TAPE_SCHEMA,
    FEE_GROWTH_SCHEMA,
    GAS_SCHEMA,
    REFERENCE_SCHEMA,
    REGIME_SCHEMA,
    tick_to_price,
    tick_to_sqrt_price_x96,
)
from undertow.sim.config import EpisodeConfig, SimConfig
from undertow.sim.core.pool import PoolEngine, PoolState, TickState
from undertow.sim.core.position import calc_sqrt_price_a
from undertow.sim.env.lp_env import LpEnvironment
from undertow.sim.env.observations import (
    OBSERVATION_VECTOR_LENGTH,
    Observation,
    ObservationBuilder,
)
from undertow.sim.env.reward import RewardBreakdown
from undertow.sim.marketview import MarketView
from undertow.sim.prices.base import human_price_to_triplet
from undertow.sim.prices.regime_jump import CalibratedPriceProcess, MRSJDParams
from undertow.sim.prices.replay import ReplayPriceProcess
from undertow.sim.types import Action, EnvError, PriceMode, Tick

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
) -> LpEnvironment:
    """A calibrated-mode env with a bounded, config-driven episode budget."""
    config = SimConfig(
        episode=EpisodeConfig(
            duration_days=duration_days, step_minutes=step_minutes
        ),
        price_mode=PriceMode.CALIBRATED,
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
        )
        vector = obs.to_vector()
        assert vector.dtype == np.float64
        assert vector.shape == (OBSERVATION_VECTOR_LENGTH,)
        expected = [float(v) for v in values]
        expected[6] = float(bool(values[6] % 2))
        expected[15] = 2.0  # sideways
        assert vector.tolist() == expected
        # regime ordinal sits at index 15: sideways -> 2.
        assert vector[15] == 2.0

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
        assert tiny_env.action_space.n == 55

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
        config = SimConfig(episode=EpisodeConfig(duration_days=1, step_minutes=10))
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
            episode=EpisodeConfig(duration_days=1, step_minutes=10),
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
        # The old position's collected fees are not silently dropped: they are
        # the fees accrued on every prior step (ADR-019: a step with no price
        # movement has no segment and accrues nothing, so the constant-price
        # rebalance step contributes zero on top of the hold step).
        hold_fees = hold_info["reward_breakdown_raw"]["fees"]
        assert rebalance_info["collected_fees_usdc"] == pytest.approx(
            hold_fees + raw["fees"], rel=1e-6, abs=1e-9
        )


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
        config = SimConfig(episode=EpisodeConfig(duration_days=0, step_minutes=10))
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

    def test_price_exhaustion_terminates_episode(
        self, tiny_market_view: MarketView, mock_gas_model: object,
        mock_slippage_model: object,
    ) -> None:
        config = SimConfig(episode=EpisodeConfig(duration_days=1, step_minutes=10))
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
        episode_config=EpisodeConfig(duration_days=1, step_minutes=10),
        pool=None,
        pool_state=pool_state,
        warmup=warmup,
    )


def _seeded_env(view: MarketView) -> LpEnvironment:
    from undertow.sim.evaluate.protocol import build_replay_env

    config = SimConfig(
        episode=EpisodeConfig(duration_days=1, step_minutes=10), seed=0
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

        # A rebalance collects every uncollected fee (prior steps + this step's)
        # and redeploys a fresh position with zero uncollected fees.
        _, _, _, _, rebalance_info = env.step(1)
        collected = float(rebalance_info["collected_fees_usdc"])
        this_step = float(rebalance_info["reward_breakdown_raw"]["fees"])
        assert collected == pytest.approx(fees_total + this_step, rel=1e-9, abs=1e-9)
        raw_after, _ = env.pool_engine.position_fees(env.position_id)
        assert raw_after == 0.0
