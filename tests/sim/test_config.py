"""Tests for ``undertow.sim.config``."""
# ruff: noqa: ANN001  # pytest fixtures (tmp_path, ...)

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from undertow.sim.config import (
    GAS_BURN,
    GAS_COLLECT,
    GAS_MINT,
    GAS_REBALANCE_SWAP,
    ActionGridConfig,
    BacktestConfig,
    EpisodeConfig,
    GasConfig,
    RewardConfig,
    SimConfig,
    SlippageConfig,
    SplitConfig,
    TrainingConfig,
    load_sim_config,
)
from undertow.sim.types import PriceMode, SimConfigError


def _write_toml(path: Path, content: str) -> None:
    path.write_text(content)


# ---------------------------------------------------------------------------
# Gas constants
# ---------------------------------------------------------------------------
class TestGasConstants:
    def test_all_positive(self) -> None:
        for name, val in [
            ("GAS_MINT", GAS_MINT),
            ("GAS_BURN", GAS_BURN),
            ("GAS_COLLECT", GAS_COLLECT),
            ("GAS_REBALANCE_SWAP", GAS_REBALANCE_SWAP),
        ]:
            assert isinstance(val, int), f"{name} should be int"
            assert val > 0, f"{name} should be positive"


# ---------------------------------------------------------------------------
# EpisodeConfig
# ---------------------------------------------------------------------------
class TestEpisodeConfig:
    def test_defaults(self) -> None:
        ep = EpisodeConfig()
        assert ep.duration_days == 30
        assert ep.step_minutes == 10
        assert ep.agent_capital_usdc == 100_000.0
        assert ep.marginal_agent is True
        assert ep.fee_tier_bps == 3000
        assert ep.tick_spacing == 60


# ---------------------------------------------------------------------------
# ActionGridConfig
# ---------------------------------------------------------------------------
class TestActionGridConfig:
    def test_defaults(self) -> None:
        grid = ActionGridConfig()
        assert grid.include_hold is True
        assert len(grid.center_offsets) == 9
        assert len(grid.widths) == 6
        assert grid.center_offsets[0] == -4
        assert grid.widths[-1] == 50

    def test_no_duplicate_center_offsets(self) -> None:
        with pytest.raises(SimConfigError, match="center_offsets"):
            ActionGridConfig(center_offsets=(0, 0, 1))

    def test_no_duplicate_widths(self) -> None:
        with pytest.raises(SimConfigError, match="widths"):
            ActionGridConfig(widths=(1, 2, 2))


# ---------------------------------------------------------------------------
# GasConfig
# ---------------------------------------------------------------------------
class TestGasConfig:
    def test_defaults(self) -> None:
        gc = GasConfig()
        assert gc.mint_units == GAS_MINT
        assert gc.burn_units == GAS_BURN
        assert gc.collect_units == GAS_COLLECT
        assert gc.rebalance_swap_units == GAS_REBALANCE_SWAP


# ---------------------------------------------------------------------------
# SlippageConfig
# ---------------------------------------------------------------------------
class TestSlippageConfig:
    def test_default(self) -> None:
        sc = SlippageConfig()
        assert sc.fixed_impact_bps == 5.0


# ---------------------------------------------------------------------------
# RewardConfig
# ---------------------------------------------------------------------------
class TestRewardConfig:
    def test_defaults(self) -> None:
        rc = RewardConfig()
        assert rc.risk_penalty_lambda == 0.0
        assert rc.risk_rolling_window_steps == 144
        assert rc.normalize_by_capital is True
        assert rc.fees_enabled is True
        assert rc.gas_enabled is True
        assert rc.slippage_enabled is True
        assert rc.il_enabled is True
        assert rc.risk_penalty_enabled is False


# ---------------------------------------------------------------------------
# SplitConfig
# ---------------------------------------------------------------------------
class TestSplitConfig:
    def test_defaults(self) -> None:
        sc = SplitConfig()
        assert sc.train_start_utc == datetime(2022, 1, 1, tzinfo=UTC)
        assert sc.train_end_utc == datetime(2023, 12, 31, tzinfo=UTC)
        assert sc.eval_start_utc == datetime(2024, 1, 1, tzinfo=UTC)
        assert sc.eval_end_utc == datetime(2024, 12, 31, tzinfo=UTC)

    def test_train_end_before_eval_start(self) -> None:
        with pytest.raises(SimConfigError, match="train_end_utc"):
            SplitConfig(
                train_end_utc=datetime(2024, 6, 1, tzinfo=UTC),
                eval_start_utc=datetime(2024, 1, 1, tzinfo=UTC),
            )

    def test_train_end_equals_eval_start_is_valid(self) -> None:
        """Boundary: adjacent splits are valid."""
        sc = SplitConfig(
            train_end_utc=datetime(2024, 1, 1, tzinfo=UTC),
            eval_start_utc=datetime(2024, 1, 1, tzinfo=UTC),
        )
        assert sc.train_end_utc == sc.eval_start_utc


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------
class TestTrainingConfig:
    def test_defaults(self) -> None:
        tc = TrainingConfig()
        assert tc.algorithm == "ppo"
        assert tc.seeds == (0, 1, 2, 3, 4)
        assert tc.discount_gamma == 0.99
        assert tc.gae_lambda == 0.95
        assert tc.clip_epsilon == 0.15
        assert tc.hidden_size == 256
        assert tc.n_hidden == 2
        assert tc.parallel_envs == 4
        assert tc.total_timesteps == 1_000_000
        assert tc.checkpoint_freq_steps == 100_000
        assert tc.log_freq_steps == 10_000


# ---------------------------------------------------------------------------
# BacktestConfig
# ---------------------------------------------------------------------------
class TestBacktestConfig:
    def test_defaults(self) -> None:
        bc = BacktestConfig()
        assert bc.compute_decomposition is True
        assert bc.log_decision_every_step is False


# ---------------------------------------------------------------------------
# SimConfig
# ---------------------------------------------------------------------------
class TestSimConfig:
    def test_default_constructs(self) -> None:
        cfg = SimConfig()
        assert cfg.episode.duration_days == 30
        assert cfg.price_mode == PriceMode.REPLAY
        assert cfg.seed == 0
        assert cfg.output_dir is None

    def test_rng_reproducible(self) -> None:
        cfg = SimConfig(seed=42)
        rng1 = cfg.rng()
        rng2 = cfg.rng()
        assert rng1.standard_normal(10).tolist() == rng2.standard_normal(10).tolist()

    def test_rng_different_seeds_different(self) -> None:
        cfg1 = SimConfig(seed=0)
        cfg2 = SimConfig(seed=1)
        a = cfg1.rng().standard_normal(100)
        b = cfg2.rng().standard_normal(100)
        assert not np.allclose(a, b)


# ---------------------------------------------------------------------------
# load_sim_config
# ---------------------------------------------------------------------------
class TestLoadSimConfig:
    def test_unknown_section_raises(self, tmp_path) -> None:
        path = tmp_path / "bad.toml"
        _write_toml(path, "[not_a_section]\nkey = 1\n")
        with pytest.raises(SimConfigError, match="Unknown config section"):
            load_sim_config(str(path))

    def test_empty_file_gives_defaults(self, tmp_path) -> None:
        path = tmp_path / "empty.toml"
        _write_toml(path, "")
        cfg = load_sim_config(str(path))
        assert cfg.episode.duration_days == 30

    def test_partial_override(self, tmp_path) -> None:
        path = tmp_path / "partial.toml"
        _write_toml(
            path,
            '[episode]\nduration_days = 7\nstep_minutes = 30\n'
            '[reward]\nfees_enabled = false\n'
            'risk_penalty_enabled = true\n',
        )
        cfg = load_sim_config(str(path))
        assert cfg.episode.duration_days == 7
        assert cfg.episode.step_minutes == 30
        # Unspecified field falls back to default
        assert cfg.episode.fee_tier_bps == 3000
        # Reward overrides
        assert cfg.reward.fees_enabled is False
        assert cfg.reward.risk_penalty_enabled is True
        # Unspecified reward flags still default
        assert cfg.reward.gas_enabled is True

    def test_price_mode_override(self, tmp_path) -> None:
        path = tmp_path / "calibrated.toml"
        _write_toml(path, '[price]\nmode = "calibrated"\n')
        cfg = load_sim_config(str(path))
        assert cfg.price_mode == PriceMode.CALIBRATED

    def test_action_grid_sequence_converted_to_tuple(self, tmp_path) -> None:
        path = tmp_path / "grid.toml"
        _write_toml(path, '[action_grid]\ncenter_offsets = [-2, 0, 2]\nwidths = [1, 5]\n')
        cfg = load_sim_config(str(path))
        assert isinstance(cfg.action_grid.center_offsets, tuple)
        assert isinstance(cfg.action_grid.widths, tuple)
        assert cfg.action_grid.center_offsets == (-2, 0, 2)

    def test_split_config_validation_applies(self, tmp_path) -> None:
        path = tmp_path / "bad_split.toml"
        _write_toml(
            path,
            "[split]\ntrain_end_utc = 2024-06-01T00:00:00Z\n"
            "eval_start_utc = 2024-01-01T00:00:00Z\n",
        )
        with pytest.raises(SimConfigError, match="train_end_utc"):
            load_sim_config(str(path))
