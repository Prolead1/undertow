"""Tests for ``undertow.sim.env`` — the Gymnasium environment."""

from __future__ import annotations

import numpy as np
import pytest

from undertow.sim.config import SimConfig
from undertow.sim.env import UndertowEnv
from undertow.sim.price import GBMPrice


@pytest.fixture
def env() -> UndertowEnv:
    rng = np.random.default_rng(42)
    gbm = GBMPrice(mu=0.0, sigma=0.2, dt=1.0 / (365 * 24 * 60))
    gbm.reset(rng, 3000.0)
    return UndertowEnv(
        config=SimConfig(),
        price_process=gbm,
        decision_freq=10,  # faster for testing
    )


class TestUndertowEnv:
    def test_reset(self, env: UndertowEnv) -> None:
        obs, info = env.reset(seed=42)
        assert obs.shape == (6,)
        assert obs.dtype == np.float32
        assert "portfolio_value" in info

    def test_step(self, env: UndertowEnv) -> None:
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (6,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_multiple_steps(self, env: UndertowEnv) -> None:
        env.reset(seed=42)
        for _ in range(20):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            assert env.observation_space.contains(obs)
            assert isinstance(reward, float)
            if term or trunc:
                break

    def test_hold_action(self, env: UndertowEnv) -> None:
        env.reset(seed=42)
        hold_action = env.cfg.action_grid.n_actions - 1
        obs, reward, term, trunc, info = env.step(hold_action)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)

    def test_termination(self, env: UndertowEnv) -> None:
        env.reset(seed=42)
        # Force termination by hacking internal state
        env._step_idx = env.steps_per_episode
        _, _, terminated, _, _ = env.step(0)
        assert terminated

    def test_action_space(self, env: UndertowEnv) -> None:
        assert env.action_space.n == env.cfg.action_grid.n_actions

    def test_observation_space(self, env: UndertowEnv) -> None:
        env.reset(seed=42)
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_reproducibility(self) -> None:
        """Same seed produces same trajectory."""
        gbm1 = GBMPrice(mu=0.0, sigma=0.2, dt=1.0 / (365 * 24 * 60))
        gbm1.reset(np.random.default_rng(42), 3000.0)
        env1 = UndertowEnv(config=SimConfig(), price_process=gbm1, decision_freq=10)

        gbm2 = GBMPrice(mu=0.0, sigma=0.2, dt=1.0 / (365 * 24 * 60))
        gbm2.reset(np.random.default_rng(42), 3000.0)
        env2 = UndertowEnv(config=SimConfig(), price_process=gbm2, decision_freq=10)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        assert np.allclose(obs1, obs2)

        obs1, _, _, _, _ = env1.step(0)
        obs2, _, _, _, _ = env2.step(0)
        assert np.allclose(obs1, obs2)

    def test_check_env(self) -> None:
        """Verify the env passes Gymnasium's built-in checker."""
        from gymnasium.utils.env_checker import check_env
        gbm = GBMPrice(mu=0.0, sigma=0.2, dt=1.0 / (365 * 24 * 60))
        gbm.reset(np.random.default_rng(42), 3000.0)
        env = UndertowEnv(config=SimConfig(), price_process=gbm, decision_freq=10)
        check_env(env, skip_render_check=True)