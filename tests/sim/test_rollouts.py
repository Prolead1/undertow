"""U06 — deterministic rollout recording (RolloutRecorder + ppo_runner wiring).

Covers the frozen parquet schema (``CONTRACTS.md`` §2.1) round-trip, the
single-place ``info`` → row mapping, the schema validation error paths, the
post-training wiring (contiguous steps, ``in_range`` invariant, opt-out and
log-and-continue on failure) and the enriched env ``info`` the recorder reads.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import gymnasium as gym
import numpy as np
import polars as pl
import pytest
from gymnasium import spaces

from tests.sim.synthetic_pool import tiny_episode
from undertow.sim.config import SimConfig, TrainingConfig
from undertow.sim.train import RolloutRecorder, ppo_runner, train_single_seed
from undertow.sim.train.rollouts import ROLLOUT_SCHEMA, rows_from_env
from undertow.sim.types import Action

_START = datetime(2026, 1, 1, tzinfo=UTC)


#: Synthetic initial capital used to derive the normalised breakdown from the
#: raw USDC one, mirroring ``RewardConfig.normalize_by_capital``.
_CAPITAL = 1_000_000.0

#: Raw USDC cost decomposition (per CONTRACTS.md §0), before normalisation.
_RAW = {
    "fees": 1.5,
    "gas": -0.2,
    "slippage": -0.1,
    "il_change": -0.05,
    "risk_penalty": 0.0,
    "reward": 1.15,
}


def _info(
    step: int,
    *,
    tick: int = 200_000,
    lower: int = 199_000,
    upper: int = 201_000,
    action_index: int = 0,
    action: Action | None = None,
    regime: str = "bull",
    price: float | None = None,
) -> dict[str, object]:
    """A synthetic ``info`` dict with every field ``rows_from_env`` reads.

    Models the env's two breakdowns exactly: ``reward_breakdown_raw`` is USDC and
    ``reward_breakdown`` is the same decomposition divided by ``_CAPITAL``.  A
    fixture that put USDC values in the normalised dict would mask the U06 units
    bug, so the two are deliberately different scales.
    """
    return {
        "reward_breakdown": {
            key: value / _CAPITAL for key, value in _RAW.items()
        }
        | {
            "pnl_net": _RAW["reward"] / _CAPITAL,
            "equity": (1000.0 + step) / _CAPITAL,
        },
        "reward_breakdown_raw": dict(_RAW)
        | {
            "pnl_net": _RAW["reward"],
            "equity": 1000.0 + step,
        },
        "action": action if action is not None else Action(action_type="hold"),
        "step_in_episode": step,
        "equity": 1000.0 + step,
        "tick_lower": lower,
        "tick_upper": upper,
        "timestamp": _START + timedelta(minutes=step),
        "price": 3000.0 + step if price is None else price,
        "tick": tick,
        "position_value": 999.0 + step,
        "action_index": action_index,
        "regime": regime,
        "gas_price_wei": 12.5,
        "active_liquidity": 1.0e18,
        "position_liquidity": 2.5e17,
        "pool_fee_earned_0": 0.75,
        "pool_fee_earned_1": 0.25,
    }


def test_rows_from_env_maps_info_and_recomputes_in_range() -> None:
    infos = [
        _info(0, tick=200_000, lower=199_000, upper=201_000),
        _info(
            1,
            tick=202_000,
            lower=199_000,
            upper=201_000,
            action_index=3,
            action=Action(action_type="rebalance", center_offset=-1, width=4),
        ),
    ]
    rows = rows_from_env(infos, run_id="ppo_s0_deadbeef", seed=0, episode=2)

    assert [row["step"] for row in rows] == [0, 1]
    assert all(row["run_id"] == "ppo_s0_deadbeef" for row in rows)
    assert all(row["seed"] == 0 and row["episode"] == 2 for row in rows)
    assert rows[0]["in_range"] is True
    # 202000 is outside [199000, 201000) even though it is above the lower tick.
    assert rows[1]["in_range"] is False
    assert rows[1]["action_index"] == 3
    assert rows[1]["action_type"] == "rebalance"
    assert rows[1]["center_offset"] == -1
    assert rows[1]["width"] == 4
    # Costs come from the raw USDC breakdown and ``il_change`` is renamed to
    # the contract's ``il`` column.
    assert rows[0]["il"] == pytest.approx(-0.05)
    assert rows[0]["fees"] == pytest.approx(1.5)
    assert rows[0]["gas"] == pytest.approx(-0.2)
    assert rows[0]["slippage"] == pytest.approx(-0.1)
    # Only ``reward`` is the normalised learning signal.
    assert rows[0]["reward"] == pytest.approx(1.15 / _CAPITAL)
    # Pool state is carried through so the UI can chart the pool over time.
    assert rows[0]["pool_active_liquidity"] == pytest.approx(1.0e18)
    assert rows[0]["position_liquidity"] == pytest.approx(2.5e17)
    assert rows[0]["pool_fee_earned_0"] == pytest.approx(0.75)
    assert rows[0]["pool_fee_earned_1"] == pytest.approx(0.25)


def test_rows_from_env_records_rejected_rebalance() -> None:
    """A reverted rebalance is recorded via the env's ``action_type`` override."""
    info = _info(
        0, action=Action(action_type="rebalance", center_offset=0, width=1)
    )
    info["action_type"] = "rebalance_rejected"
    rows = rows_from_env([info], run_id="ppo_s0_deadbeef", seed=0, episode=0)
    assert rows[0]["action_type"] == "rebalance_rejected"


def test_record_episode_round_trips_exact_schema(tmp_path: Path) -> None:
    rows = rows_from_env(
        [_info(0), _info(1, tick=202_500)], run_id="run", seed=7, episode=1
    )
    recorder = RolloutRecorder("run", 7, tmp_path, max_episodes=3)
    path = recorder.record_episode(1, rows)

    assert path == tmp_path / "rollouts" / "episode_1.parquet"
    frame = pl.read_parquet(path)
    assert list(frame.schema) == list(ROLLOUT_SCHEMA)
    assert dict(frame.schema) == ROLLOUT_SCHEMA
    assert frame["step"].to_list() == [0, 1]
    assert frame["run_id"].to_list() == ["run", "run"]
    assert frame["seed"].to_list() == [7, 7]
    assert frame["episode"].to_list() == [1, 1]
    assert frame["price"].to_list() == pytest.approx([3000.0, 3001.0])
    assert frame["in_range"].to_list() == [True, False]
    assert frame["timestamp"].to_list() == [
        _START,
        _START + timedelta(minutes=1),
    ]


def test_record_episode_rejects_unknown_missing_and_empty(tmp_path: Path) -> None:
    recorder = RolloutRecorder("run", 0, tmp_path, max_episodes=3)
    valid = dict(rows_from_env([_info(0)], run_id="run", seed=0, episode=0)[0])

    unknown = dict(valid)
    unknown["surprise"] = 1
    with pytest.raises(ValueError, match="unknown rollout column.*surprise"):
        recorder.record_episode(0, [unknown])

    missing = dict(valid)
    del missing["regime"]
    with pytest.raises(ValueError, match="regime"):
        recorder.record_episode(0, [missing])

    with pytest.raises(ValueError, match="empty"):
        recorder.record_episode(0, [])


def test_record_episode_validates_episode_bounds_and_close_is_idempotent(
    tmp_path: Path,
) -> None:
    recorder = RolloutRecorder("run", 0, tmp_path, max_episodes=1)
    rows = rows_from_env([_info(0)], run_id="run", seed=0, episode=0)
    with pytest.raises(ValueError, match="outside"):
        recorder.record_episode(1, rows)

    recorder.close()
    recorder.close()  # idempotent
    with pytest.raises(ValueError, match="after close"):
        recorder.record_episode(0, rows)


# ---------------------------------------------------------------------------
# ppo_runner wiring
# ---------------------------------------------------------------------------


class _RolloutBanditEnv(gym.Env):
    """A 3-step bandit whose ``info`` is a valid rollout record each step."""

    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)
    action_space = spaces.Discrete(2)

    def __init__(self) -> None:
        self._step = 0
        self._seed = 0

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        del options
        self._seed = 0 if seed is None else int(seed)
        self._step = 0
        return np.zeros(2, dtype=np.float64), self._info(0)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step += 1
        terminated = self._step >= 3
        return (
            np.zeros(2, dtype=np.float64),
            -0.05,
            terminated,
            False,
            self._info(int(action)),
        )

    def _info(self, action_index: int) -> dict[str, object]:
        step = self._step
        # Alternate inside/outside the deployed range so ``in_range`` is tested.
        tick = 200_000 + step
        lower = 199_000 if step % 2 == 0 else tick + 1
        return _info(
            step,
            tick=tick,
            lower=lower,
            upper=201_000,
            action_index=action_index,
            action=Action(action_type="hold"),
            # Make the trajectory depend on the reset seed so a multi-episode
            # recording can be shown to produce non-identical episodes.
            price=3000.0 + 100.0 * self._seed + step,
        )


def _training_config(
    *,
    record_rollouts: bool = False,
    rollout_max_episodes: int = 1,
    total_timesteps: int = 64,
) -> SimConfig:
    return SimConfig(
        episode=tiny_episode(duration_days=1, step_minutes=10),
        training=TrainingConfig(
            seeds=(0,),
            total_timesteps=total_timesteps,
            log_freq_steps=32,
            checkpoint_freq_steps=10_000,
            parallel_envs=1,
            record_rollouts=record_rollouts,
            rollout_max_episodes=rollout_max_episodes,
        ),
    )


def test_opt_out_writes_no_rollout_directory(tmp_path: Path) -> None:
    train_single_seed(_RolloutBanditEnv, _training_config(), seed=0, log_dir=tmp_path)
    assert not (tmp_path / "rollouts").exists()


def test_recording_produces_contiguous_in_range_episode(tmp_path: Path) -> None:
    train_single_seed(
        _RolloutBanditEnv,
        _training_config(record_rollouts=True, rollout_max_episodes=1),
        seed=0,
        log_dir=tmp_path,
    )
    path = tmp_path / "rollouts" / "episode_0.parquet"
    assert path.exists()
    frame = pl.read_parquet(path)
    assert frame.height == 4  # reset decision + 3 stepped decisions
    assert frame["step"].to_list() == list(range(frame.height))
    assert frame["in_range"].to_list() == [
        lower <= tick < upper
        for lower, tick, upper in zip(
            frame["tick_lower"].to_list(),
            frame["tick"].to_list(),
            frame["tick_upper"].to_list(),
            strict=True,
        )
    ]


def test_recording_multiple_episodes_are_distinct(tmp_path: Path) -> None:
    """Each episode is reseeded, so ``rollout_max_episodes > 1`` is not N copies.

    ``LpEnvironment.reset(seed=None)`` keeps the previous seed and rebuilds the
    same RNG, so a pass that only seeds episode 0 would write identical windows
    under every filename.  The env fixture's trajectory varies with the reset
    seed, making that regression observable here.
    """
    train_single_seed(
        _RolloutBanditEnv,
        _training_config(record_rollouts=True, rollout_max_episodes=2),
        seed=0,
        log_dir=tmp_path,
    )
    episode_0 = pl.read_parquet(tmp_path / "rollouts" / "episode_0.parquet")
    episode_1 = pl.read_parquet(tmp_path / "rollouts" / "episode_1.parquet")
    assert episode_0["price"].to_list() != episode_1["price"].to_list()
    assert episode_0["episode"].to_list() == [0] * episode_0.height
    assert episode_1["episode"].to_list() == [1] * episode_1.height


def test_recording_failure_does_not_fail_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _BrokenRecorder:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("recorder boom")

    monkeypatch.setattr(ppo_runner, "RolloutRecorder", _BrokenRecorder)
    manifest = train_single_seed(
        _RolloutBanditEnv,
        _training_config(record_rollouts=True),
        seed=0,
        log_dir=tmp_path,
    )
    assert manifest.checkpoint_path.exists()
    assert not (tmp_path / "rollouts").exists()


# ---------------------------------------------------------------------------
# Real-environment info enrichment (the fields the recorder reads)
# ---------------------------------------------------------------------------


def test_lp_env_info_carries_rollout_fields(tiny_env: object) -> None:
    env = tiny_env
    _obs, info = env.reset(seed=0)  # type: ignore[attr-defined]
    expected = {
        "action_index",
        "timestamp",
        "price",
        "tick",
        "position_value",
        "in_range",
        "regime",
        "gas_price_wei",
        "tick_lower",
        "tick_upper",
        "equity",
        "active_liquidity",
        "position_liquidity",
        "pool_fee_earned_0",
        "pool_fee_earned_1",
    }
    assert expected <= set(info)
    _obs, _reward, _terminated, _truncated, next_info = env.step(0)  # type: ignore[attr-defined]
    assert expected <= set(next_info)


# ---------------------------------------------------------------------------
# Real-environment integration (slow)
# ---------------------------------------------------------------------------


def _real_env_factory(
    market_view: object, gas_model: object, slippage_model: object
) -> Callable[[], gym.Env]:
    """Fresh tiny ``LpEnvironment`` per call (mirrors ``test_training``)."""
    from undertow.sim.core.pool import PoolEngine, PoolState, TickState
    from undertow.sim.core.position import calc_sqrt_price_a
    from undertow.sim.env.lp_env import LpEnvironment
    from undertow.sim.prices.replay import ReplayPriceProcess
    from undertow.sim.types import Tick

    config = SimConfig(
        episode=tiny_episode(duration_days=31, step_minutes=10), seed=0
    )

    def factory() -> gym.Env:
        engine = PoolEngine(
            state=PoolState(
                sqrt_price=calc_sqrt_price_a(Tick(196242)),
                tick=Tick(196242),
                liquidity=1.0e18,
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
            market_view,  # type: ignore[arg-type]
            engine,
            ReplayPriceProcess(market_view, 0, market_view.step_count()),  # type: ignore[attr-defined]
            gas_model,  # type: ignore[arg-type]
            slippage_model,  # type: ignore[arg-type]
            config,
        )

    return factory


@pytest.mark.slow
def test_train_single_seed_records_real_rollout(
    tiny_market_view: object,
    mock_gas_model: object,
    mock_slippage_model: object,
    tmp_path: Path,
) -> None:
    env_fn = _real_env_factory(tiny_market_view, mock_gas_model, mock_slippage_model)
    train_single_seed(
        env_fn,
        _training_config(record_rollouts=True, rollout_max_episodes=1, total_timesteps=64),
        seed=0,
        log_dir=tmp_path,
    )
    frame = pl.read_parquet(tmp_path / "rollouts" / "episode_0.parquet")
    assert frame.height > 1
    assert frame["step"].to_list() == list(range(frame.height))
    assert frame["in_range"].to_list() == [
        lower <= tick < upper
        for lower, tick, upper in zip(
            frame["tick_lower"].to_list(),
            frame["tick"].to_list(),
            frame["tick_upper"].to_list(),
            strict=True,
        )
    ]


@pytest.mark.slow
def test_real_env_rows_carry_raw_usdc_costs(
    tiny_market_view: object,
    mock_gas_model: object,
    mock_slippage_model: object,
) -> None:
    """Costs are the raw USDC breakdown, not the capital-normalised one (U06).

    ``reward_breakdown`` divides every monetary field by the initial capital, so
    reading costs from it would shrink a 76.48 USDC rebalance gas cost to
    ~1e-4 and mix units with the USDC ``equity`` column.  Action index 1 is the
    first rebalance in the default grid, whose mocked gas cost is 76.4775 USDC.
    """
    env = _real_env_factory(tiny_market_view, mock_gas_model, mock_slippage_model)()
    try:
        env.reset(seed=0)  # type: ignore[attr-defined]
        _obs, _reward, _terminated, _truncated, info = env.step(1)  # type: ignore[attr-defined]
    finally:
        env.close()  # type: ignore[attr-defined]

    rows = rows_from_env([info], run_id="run", seed=0, episode=0)
    raw = cast(Mapping[str, object], info["reward_breakdown_raw"])
    normalised = cast(Mapping[str, object], info["reward_breakdown"])
    for column, key in (
        ("fees", "fees"),
        ("gas", "gas"),
        ("slippage", "slippage"),
        ("il", "il_change"),
        ("risk_penalty", "risk_penalty"),
    ):
        assert abs(float(rows[0][column]) - float(raw[key])) < 1e-9  # type: ignore[arg-type]
    # A rebalance costs real USDC; a normalised value would be orders smaller.
    assert abs(float(rows[0]["gas"])) > 1.0
    # ``reward`` remains the normalised learning signal.
    assert float(rows[0]["reward"]) == pytest.approx(float(normalised["reward"]))
