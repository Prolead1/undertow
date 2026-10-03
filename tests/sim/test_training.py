"""S13 — PPO training harness: single-seed, multi-seed, checkpoints, provenance.

Every test asserts real behaviour on tiny CPU configurations (hundreds of
timesteps, no GPU/network): manifest provenance, checkpoints that load, a policy
that actually learns a known-optimal action, identical learning curves for
identical seeds, multi-seed aggregation, the one-env-per-seed factory contract
and a smoke test on the real :class:`~undertow.sim.env.LpEnvironment`.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3 import PPO

from undertow.sim.config import EpisodeConfig, SimConfig, TrainingConfig
from undertow.sim.core.pool import PoolEngine, PoolState, TickState
from undertow.sim.core.position import calc_sqrt_price_a
from undertow.sim.env.lp_env import LpEnvironment
from undertow.sim.prices.replay import ReplayPriceProcess
from undertow.sim.train import (
    RunManifest,
    TrainingResult,
    train_ppo,
    train_single_seed,
)
from undertow.sim.train.ppo_runner import _derive_rollout, _train_single_seed
from undertow.sim.train.runs import compute_config_hash, library_versions
from undertow.sim.types import Tick

#: Optimal arm index and reward in the deterministic bandit below.
_OPTIMAL_ARM = 1
_OPTIMAL_REWARD = 1.0


class _BanditEnv(gym.Env):
    """A deterministic 2-armed bandit with a known optimal arm.

    Arm ``0`` pays 0, arm ``1`` pays 1; every episode is a single step, so the
    optimal policy selects arm 1 with probability 1. The environment carries no
    randomness of its own, which makes it a clean probe for PPO's policy
    updates (only the model's seeded RNG drives exploration).
    """

    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)
    action_space = spaces.Discrete(2)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        del seed, options
        return np.zeros(2, dtype=np.float64), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        reward = _OPTIMAL_REWARD if int(action) == _OPTIMAL_ARM else 0.0
        return np.zeros(2, dtype=np.float64), reward, True, False, {}


class _DelayedBanditEnv(gym.Env):
    """A bandit whose episodes last 10 steps, so short runs complete none."""

    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)
    action_space = spaces.Discrete(2)

    def __init__(self) -> None:
        self._step = 0

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        del seed, options
        self._step = 0
        return np.zeros(2, dtype=np.float64), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step += 1
        reward = _OPTIMAL_REWARD if int(action) == _OPTIMAL_ARM else 0.0
        terminated = self._step >= 10
        return np.zeros(2, dtype=np.float64), reward, terminated, False, {}


def _training_config(
    *,
    seeds: tuple[int, ...] = (0,),
    total_timesteps: int = 256,
    log_freq_steps: int = 64,
    checkpoint_freq_steps: int = 128,
    parallel_envs: int = 1,
) -> SimConfig:
    """A tiny, fast CPU training config (hundreds of steps, one env by default)."""
    return SimConfig(
        episode=EpisodeConfig(duration_days=1, step_minutes=10),
        training=TrainingConfig(
            seeds=seeds,
            total_timesteps=total_timesteps,
            log_freq_steps=log_freq_steps,
            checkpoint_freq_steps=checkpoint_freq_steps,
            parallel_envs=parallel_envs,
        ),
    )


def _fresh_pool_engine() -> PoolEngine:
    """A pristine external pool (no agent positions) mirroring the conftest env."""
    return PoolEngine(
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


def _real_env_factory(
    market_view: object, gas_model: object, slippage_model: object
) -> Callable[[], LpEnvironment]:
    """Build a fresh ``LpEnvironment`` on every call (one env per request)."""
    config = SimConfig(
        episode=EpisodeConfig(duration_days=1, step_minutes=10),
        seed=0,
    )

    def factory() -> LpEnvironment:
        return LpEnvironment(
            market_view,  # type: ignore[arg-type]
            _fresh_pool_engine(),
            ReplayPriceProcess(market_view, 0, market_view.step_count()),  # type: ignore[arg-type]
            gas_model,
            slippage_model,
            config,
        )

    return factory


class _CountingFactory:
    """Wrap a factory and count how many times it is invoked."""

    def __init__(self, base: Callable[[], gym.Env]) -> None:
        self.base = base
        self.calls = 0

    def __call__(self) -> gym.Env:
        self.calls += 1
        return self.base()


# ---------------------------------------------------------------------------
# train_single_seed
# ---------------------------------------------------------------------------


class TestTrainSingleSeed:
    def test_writes_checkpoint_and_manifest_with_provenance(self, tmp_path: Path) -> None:
        config = _training_config(total_timesteps=256, checkpoint_freq_steps=128)
        manifest = train_single_seed(_BanditEnv, config, seed=3, log_dir=tmp_path)

        assert isinstance(manifest, RunManifest)
        assert manifest.seed == 3
        assert manifest.algorithm == "ppo"
        assert manifest.total_timesteps == 256
        assert manifest.config_hash == compute_config_hash(config)
        assert manifest.git_commit  # never empty/None
        assert manifest.run_id == f"ppo_s3_{manifest.config_hash}"
        assert isinstance(manifest.best_reward, float)
        assert manifest.checkpoint_path.exists()
        assert manifest.checkpoint_path.name == "ppo_seed_3.zip"

        versions = dict(manifest.library_versions)
        assert {"stable_baselines3", "torch", "gymnasium", "numpy"} <= set(versions)
        assert versions == library_versions()

        sidecar = tmp_path / "manifest.json"
        assert sidecar.exists()
        payload = json.loads(sidecar.read_text())
        assert payload["seed"] == 3
        assert payload["config_hash"] == manifest.config_hash
        assert payload["checkpoint_path"] == str(manifest.checkpoint_path)
        assert payload["algorithm"] == "ppo"
        assert len(payload["learning_curve"]) == 4  # 256 / 64

    def test_checkpoints_written_at_configured_frequency(self, tmp_path: Path) -> None:
        config = _training_config(total_timesteps=256, checkpoint_freq_steps=128)
        train_single_seed(_BanditEnv, config, seed=0, log_dir=tmp_path)

        checkpoints = sorted((tmp_path / "checkpoints").glob("*.zip"))
        names = {p.name for p in checkpoints}
        assert "ppo_seed_0_128_steps.zip" in names
        assert "ppo_seed_0_256_steps.zip" in names

    def test_checkpoint_is_loadable(self, tmp_path: Path) -> None:
        config = _training_config()
        manifest = train_single_seed(_BanditEnv, config, seed=1, log_dir=tmp_path)
        model = PPO.load(str(manifest.checkpoint_path), device="cpu")
        action, _ = model.predict(np.zeros(2, dtype=np.float64), deterministic=True)
        assert int(action) in (0, 1)

    def test_learns_the_optimal_arm(self, tmp_path: Path) -> None:
        config = _training_config(total_timesteps=512, log_freq_steps=64)
        manifest = train_single_seed(_BanditEnv, config, seed=7, log_dir=tmp_path)
        assert manifest.best_reward > 0.9

        model = PPO.load(str(manifest.checkpoint_path), device="cpu")
        obs = np.zeros(2, dtype=np.float64)
        choices = [int(model.predict(obs, deterministic=True)[0]) for _ in range(100)]
        assert choices.count(_OPTIMAL_ARM) / len(choices) > 0.9

    def test_same_seed_reproduces_learning_curve(self, tmp_path: Path) -> None:
        config = _training_config(total_timesteps=256)
        _, curve_a = _train_single_seed(_BanditEnv, config, 5, tmp_path / "a")
        _, curve_b = _train_single_seed(_BanditEnv, config, 5, tmp_path / "b")
        assert curve_a == curve_b

        manifest_a = json.loads((tmp_path / "a" / "manifest.json").read_text())
        manifest_b = json.loads((tmp_path / "b" / "manifest.json").read_text())
        assert manifest_a["learning_curve"] == manifest_b["learning_curve"]
        assert manifest_a["best_reward"] == manifest_b["best_reward"]

    def test_distinct_seeds_get_distinct_run_ids(self, tmp_path: Path) -> None:
        config = _training_config()
        first = train_single_seed(_BanditEnv, config, 0, tmp_path / "s0")
        second = train_single_seed(_BanditEnv, config, 1, tmp_path / "s1")
        assert first.run_id != second.run_id
        assert first.seed == 0 and second.seed == 1

    def test_log_dir_none_uses_a_temp_artifact_dir(self) -> None:
        config = _training_config()
        manifest = train_single_seed(_BanditEnv, config, seed=0)
        try:
            assert manifest.checkpoint_path.exists()
            assert "undertow-ppo-seed0" in str(manifest.checkpoint_path.parent)
        finally:
            shutil.rmtree(manifest.checkpoint_path.parent, ignore_errors=True)

    def test_manifest_is_frozen_and_slotted(self) -> None:
        manifest = RunManifest(
            run_id="r",
            seed=0,
            config_hash="abc",
            git_commit="deadbeef",
            algorithm="ppo",
            total_timesteps=1,
            best_reward=0.0,
            checkpoint_path=Path("x.zip"),
        )
        assert not hasattr(manifest, "__dict__")
        with pytest.raises(dataclasses.FrozenInstanceError):
            manifest.seed = 1  # type: ignore[misc]

    def test_stays_converged_after_zero_extra_steps(self, tmp_path: Path) -> None:
        config = _training_config(total_timesteps=512, log_freq_steps=64)
        manifest = train_single_seed(_BanditEnv, config, seed=11, log_dir=tmp_path)
        model = PPO.load(
            str(manifest.checkpoint_path), env=_BanditEnv(), device="cpu"
        )
        model.learn(total_timesteps=0)  # no extra optimisation
        obs = np.zeros(2, dtype=np.float64)
        choices = [int(model.predict(obs, deterministic=True)[0]) for _ in range(100)]
        assert choices.count(_OPTIMAL_ARM) / len(choices) > 0.9

    def test_empty_curve_falls_back_when_no_episode_completes(
        self, tmp_path: Path
    ) -> None:
        config = _training_config(
            total_timesteps=4, log_freq_steps=8, checkpoint_freq_steps=2
        )
        manifest, curve = _train_single_seed(_DelayedBanditEnv, config, 0, tmp_path)
        assert curve == []
        assert manifest.best_reward == 0.0
        payload = json.loads((tmp_path / "manifest.json").read_text())
        assert payload["learning_curve"] == []


class TestRolloutDerivation:
    def test_reproduces_anchor_defaults_at_production_budget(self) -> None:
        config = TrainingConfig(total_timesteps=1_000_000, parallel_envs=4)
        assert _derive_rollout(config) == (2048, 64, 10)

    @pytest.mark.parametrize("parallel_envs", [1, 2, 3, 4, 8])
    @pytest.mark.parametrize(
        "total_timesteps",
        [2, 4, 12, 14, 20, 23, 58, 100, 256, 1000],
    )
    def test_batch_size_satisfies_sb3_invariants(
        self, total_timesteps: int, parallel_envs: int
    ) -> None:
        config = TrainingConfig(
            total_timesteps=total_timesteps, parallel_envs=parallel_envs
        )
        n_steps, batch_size, _ = _derive_rollout(config)
        rollout = n_steps * parallel_envs
        assert rollout >= 2
        assert 1 < batch_size <= rollout
        assert rollout % batch_size == 0

    @pytest.mark.parametrize("total_timesteps", [12, 14, 23])
    def test_odd_budget_trains_end_to_end(
        self, total_timesteps: int, tmp_path: Path
    ) -> None:
        # Regression guard: these budgets used to emit batch_size == 1 and
        # crash SB3 with ``AssertionError: batch_size must be greater than 1``.
        config = _training_config(
            total_timesteps=total_timesteps,
            log_freq_steps=4,
            checkpoint_freq_steps=1000,
        )
        manifest = train_single_seed(_BanditEnv, config, seed=0, log_dir=tmp_path)
        assert manifest.checkpoint_path.exists()

    def test_vec_env_is_closed_when_ppo_construction_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        closed = {"count": 0}

        class _ClosableBandit(_BanditEnv):
            def close(self) -> None:
                closed["count"] += 1

        def _boom(*args: object, **kwargs: object) -> object:
            raise RuntimeError("constructor failed")

        monkeypatch.setattr("undertow.sim.train.ppo_runner.PPO", _boom)
        config = _training_config()
        with pytest.raises(RuntimeError, match="constructor failed"):
            train_single_seed(_ClosableBandit, config, 0, tmp_path)
        assert closed["count"] == 1

    def test_tensorboard_hook_detects_optional_package(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys
        import types

        from undertow.sim.train.ppo_runner import _tensorboard_log

        monkeypatch.setitem(sys.modules, "tensorboard", types.ModuleType("tensorboard"))
        assert _tensorboard_log(tmp_path) == str(tmp_path / "tensorboard")
        monkeypatch.delitem(sys.modules, "tensorboard")
        assert _tensorboard_log(tmp_path) is None


# ---------------------------------------------------------------------------
# train_ppo
# ---------------------------------------------------------------------------


class TestTrainPpo:
    def test_runs_every_seed_and_aggregates(self, tmp_path: Path) -> None:
        config = _training_config(seeds=(0, 1, 2), total_timesteps=256)
        result = train_ppo(_BanditEnv, config, log_dir=tmp_path)

        assert isinstance(result, TrainingResult)
        assert tuple(result.seeds) == (0, 1, 2)
        assert len(result.manifests) == 3
        assert [m.seed for m in result.manifests] == [0, 1, 2]
        assert all(m.config_hash == compute_config_hash(config) for m in result.manifests)

        curves = result.learning_curves
        assert set(curves) >= {"mean_reward", "std_reward"}
        assert len(curves["mean_reward"]) == 4  # 256 / 64
        assert len(curves["std_reward"]) == 4
        for seed in (0, 1, 2):
            assert len(curves[f"seed_{seed}_reward"]) == 4

    def test_best_checkpoint_is_highest_reward_seed(self, tmp_path: Path) -> None:
        config = _training_config(seeds=(0, 1, 2), total_timesteps=256)
        result = train_ppo(_BanditEnv, config, log_dir=tmp_path)
        expected = max(result.manifests, key=lambda m: m.best_reward)
        assert result.best_checkpoint == expected.checkpoint_path
        assert result.best_checkpoint.exists()

    def test_env_factory_called_once_per_seed(self, tmp_path: Path) -> None:
        config = _training_config(
            seeds=(0, 1, 2), total_timesteps=128, parallel_envs=1
        )
        factory = _CountingFactory(_BanditEnv)
        result = train_ppo(factory, config, log_dir=tmp_path)
        assert factory.calls == 3
        assert len(result.manifests) == 3

    def test_multi_env_seeds_each_slot(self, tmp_path: Path) -> None:
        config = _training_config(
            seeds=(0,), total_timesteps=128, parallel_envs=2
        )
        factory = _CountingFactory(_BanditEnv)
        result = train_ppo(factory, config, log_dir=tmp_path)
        assert factory.calls == 2
        assert len(result.manifests) == 1
        assert result.manifests[0].checkpoint_path.exists()
        # checkpoint_freq=128 with 2 envs -> save_freq=64 callback calls = 128 steps
        checkpoints = tmp_path / "seed_0" / "checkpoints"
        assert (checkpoints / "ppo_seed_0_128_steps.zip").exists()

    def test_same_seed_set_reproduces_aggregate_curve(self, tmp_path: Path) -> None:
        config = _training_config(seeds=(0, 1), total_timesteps=128)
        first = train_ppo(_BanditEnv, config, log_dir=tmp_path / "one")
        second = train_ppo(_BanditEnv, config, log_dir=tmp_path / "two")
        assert first.learning_curves == second.learning_curves
        assert [m.best_reward for m in first.manifests] == [
            m.best_reward for m in second.manifests
        ]

    def test_empty_seed_sequence_is_rejected(self) -> None:
        config = _training_config(seeds=())
        with pytest.raises(ValueError, match="at least one seed"):
            train_ppo(_BanditEnv, config)


# ---------------------------------------------------------------------------
# Real environment smoke test
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRealEnvironmentSmoke:
    def test_train_single_seed_on_tiny_lp_env(
        self,
        tiny_market_view: object,
        mock_gas_model: object,
        mock_slippage_model: object,
        tmp_path: Path,
    ) -> None:
        config = _training_config(
            total_timesteps=128, log_freq_steps=64, checkpoint_freq_steps=64
        )
        env_fn = _CountingFactory(
            _real_env_factory(tiny_market_view, mock_gas_model, mock_slippage_model)
        )
        manifest = train_single_seed(env_fn, config, seed=0, log_dir=tmp_path)

        assert env_fn.calls == 1
        assert manifest.checkpoint_path.exists()
        PPO.load(str(manifest.checkpoint_path), device="cpu")
        payload = json.loads((tmp_path / "manifest.json").read_text())
        assert payload["learning_curve"]  # at least one evaluation recorded