"""Tests for ``undertow.sim.config``."""

from __future__ import annotations

import pytest

from undertow.sim.config import (
    ActionGrid,
    EpisodeConfig,
    SimConfig,
    default_sim_config,
    load_sim_config,
)


class TestEpisodeConfig:
    def test_defaults(self) -> None:
        ep = EpisodeConfig()
        assert ep.duration_days == 30
        assert ep.step_minutes == 60
        assert ep.steps_per_episode == 30 * 24  # 720 hourly steps

    def test_steps_per_episode(self) -> None:
        ep = EpisodeConfig(duration_days=7, step_minutes=30)
        assert ep.steps_per_episode == 7 * 24 * 2  # 336 half-hourly steps

    def test_invalid_fee_tier(self) -> None:
        with pytest.raises(ValueError):
            EpisodeConfig(fee_tier_bps=42)

    def test_tick_spacing_mismatch(self) -> None:
        with pytest.raises(ValueError):
            EpisodeConfig(fee_tier_bps=30, tick_spacing=10)


class TestActionGrid:
    def test_defaults(self) -> None:
        grid = ActionGrid()
        assert grid.n_actions > 0
        assert grid.include_hold

    def test_decode_hold(self) -> None:
        grid = ActionGrid()
        hold_action = grid.n_actions - 1
        assert grid.decode(hold_action) is None

    def test_decode_rebalance(self) -> None:
        grid = ActionGrid()
        result = grid.decode(0)
        assert result is not None
        offset, width = result
        assert offset == -0.20
        assert width == 0.02

    def test_no_hold(self) -> None:
        grid = ActionGrid(include_hold=False)
        result = grid.decode(grid.n_actions - 1)
        assert result is not None  # last action is not hold


class TestSimConfig:
    def test_default(self) -> None:
        cfg = default_sim_config()
        assert cfg.episode.fee_tier_bps == 30

    def test_construct(self) -> None:
        cfg = SimConfig()
        assert cfg.episode.steps_per_episode == 720


class TestLoadSimConfig:
    def test_load_defaults(self, tmp_path) -> None:
        """Loading a minimal TOML with no sections should give defaults."""
        import tomli_w
        path = tmp_path / "cfg.toml"
        path.write_text(tomli_w.dumps({}))
        cfg = load_sim_config(str(path))
        assert cfg.episode.fee_tier_bps == 30

    def test_load_override(self, tmp_path) -> None:
        """Section values override defaults."""
        import tomli_w
        path = tmp_path / "cfg.toml"
        path.write_text(tomli_w.dumps({
            "episode": {"duration_days": 7, "step_minutes": 30},
        }))
        cfg = load_sim_config(str(path))
        assert cfg.episode.duration_days == 7
        assert cfg.episode.step_minutes == 30
        # Unspecified field falls back to default
        assert cfg.episode.fee_tier_bps == 30

    def test_load_full(self) -> None:
        """Loading the shipped default TOML matches the programmatic default."""
        import inspect, os
        repo_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        cfg_file = os.path.join(repo_root, "configs", "sim_default.toml")
        from_file = load_sim_config(cfg_file)
        from_code = default_sim_config()
        assert from_file == from_code