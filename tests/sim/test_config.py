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
    HedgeConfig,
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
        assert ep.fee_tier_bps == 500  # primary 0.05% study pool
        assert ep.tick_spacing == 10

    def test_defaults_match_the_primary_pool_fields(self) -> None:
        """The dataclass defaults are the primary 0.05 % pool, not the 0.30 %
        contrast, and the action grid matches ``configs/sim_default.toml``."""
        from undertow.data.config import default_pools

        grid = ActionGridConfig()
        primary = default_pools()["USDC_WETH_500"]
        assert EpisodeConfig().fee_tier_bps == primary.fee_tier.value
        assert EpisodeConfig().tick_spacing == primary.tick_spacing
        assert tuple(grid.widths) == (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000)


# ---------------------------------------------------------------------------
# ActionGridConfig
# ---------------------------------------------------------------------------
class TestActionGridConfig:
    def test_defaults(self) -> None:
        grid = ActionGridConfig()
        assert grid.include_hold is True
        assert len(grid.center_offsets) == 9
        # Matches the primary 0.05 % pool's spacing-10 ladder.
        assert len(grid.widths) == 10
        assert grid.center_offsets[0] == -4
        assert grid.widths[-1] == 1000

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
        # S20: the hedge terms are part of the learning reward by default; the
        # RQ1 ablation sets this false (only meaningful with [hedge] enabled).
        assert rc.hedge_enabled is True


# ---------------------------------------------------------------------------
# HedgeConfig
# ---------------------------------------------------------------------------
class TestHedgeConfig:
    def test_defaults(self) -> None:
        hc = HedgeConfig()
        assert hc.enabled is False
        assert hc.fee_bps == 5.0
        assert hc.funding_enabled is True
        assert hc.ratio == 1.0
        assert tuple(hc.ratios) == (0.0, 0.25, 0.5, 0.75, 1.0)

    @pytest.mark.parametrize(
        "fee_bps", [-1.0, -1e-9, float("nan"), float("inf"), float("-inf")]
    )
    def test_negative_or_non_finite_fee_bps_raises(self, fee_bps: float) -> None:
        with pytest.raises(SimConfigError, match="fee_bps"):
            HedgeConfig(fee_bps=fee_bps)

    @pytest.mark.parametrize(
        "ratio",
        [-0.01, 1.01, -1.0, 2.0, float("nan"), float("inf"), float("-inf")],
    )
    def test_out_of_range_or_non_finite_ratio_raises(self, ratio: float) -> None:
        with pytest.raises(SimConfigError, match="ratio"):
            HedgeConfig(ratio=ratio)

    @pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0])
    def test_ratio_boundaries_are_valid(self, ratio: float) -> None:
        assert HedgeConfig(ratio=ratio).ratio == ratio

    @pytest.mark.parametrize(
        "bad",
        [-0.01, 1.01, -1.0, 2.0, float("nan"), float("inf"), float("-inf")],
    )
    def test_out_of_range_or_non_finite_ratios_entry_raises(self, bad: float) -> None:
        with pytest.raises(SimConfigError, match="ratios"):
            HedgeConfig(ratios=(0.0, bad, 1.0))

    def test_empty_ratios_is_valid(self) -> None:
        # ``ratios`` is the S20b action head; this task only consumes ``ratio``,
        # so an empty discrete head is a legal (if degenerate) configuration.
        assert tuple(HedgeConfig(ratios=()).ratios) == ()

    def test_error_message_names_the_offending_value(self) -> None:
        with pytest.raises(SimConfigError, match="3.5"):
            HedgeConfig(ratio=3.5)
        with pytest.raises(SimConfigError, match="-2.0"):
            HedgeConfig(fee_bps=-2.0)
        with pytest.raises(SimConfigError, match="1.25"):
            HedgeConfig(ratios=(0.5, 1.25))


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
        assert tc.vec_env == "dummy"
        assert tc.telemetry is True
        # Rollouts are recorded by default so the replay views have data.
        assert tc.record_rollouts is True
        assert tc.rollout_max_episodes == 5

    def test_rollouts_can_still_be_disabled(self) -> None:
        assert TrainingConfig(record_rollouts=False).record_rollouts is False

    def test_vec_env_accepts_subproc_and_rejects_unknown(self) -> None:
        assert TrainingConfig(vec_env="subproc").vec_env == "subproc"
        with pytest.raises(SimConfigError, match="vec_env"):
            TrainingConfig(vec_env="threads")  # type: ignore[arg-type]


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
        assert cfg.episode.fee_tier_bps == 500
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

    def test_hedge_section_loads(self, tmp_path) -> None:
        path = tmp_path / "hedge.toml"
        _write_toml(
            path,
            "[hedge]\nenabled = true\nfee_bps = 3.5\n"
            "funding_enabled = false\nratio = 0.5\n"
            "ratios = [0.0, 0.5, 1.0]\n",
        )
        cfg = load_sim_config(str(path))
        assert cfg.hedge.enabled is True
        assert cfg.hedge.fee_bps == 3.5
        assert cfg.hedge.funding_enabled is False
        assert cfg.hedge.ratio == 0.5
        # TOML arrays become tuples for the ``Sequence``-typed field, exactly
        # like the action grid.
        assert isinstance(cfg.hedge.ratios, tuple)
        assert cfg.hedge.ratios == (0.0, 0.5, 1.0)

    def test_hedge_validation_applies_through_the_loader(self, tmp_path) -> None:
        path = tmp_path / "bad_hedge.toml"
        _write_toml(path, "[hedge]\nenabled = true\nratio = 1.5\n")
        with pytest.raises(SimConfigError, match="ratio"):
            load_sim_config(str(path))

    def test_empty_hedge_section_uses_defaults(self, tmp_path) -> None:
        path = tmp_path / "empty_hedge.toml"
        _write_toml(path, "[hedge]\n")
        cfg = load_sim_config(str(path))
        assert cfg.hedge.enabled is False
        assert cfg.hedge.fee_bps == 5.0
        assert cfg.hedge.ratio == 1.0


# ---------------------------------------------------------------------------
# Shipped config/data pairing
# ---------------------------------------------------------------------------


def test_shipped_sim_configs_pair_with_their_data_pools() -> None:
    """Every shipped sim config matches its data config's pool tier/spacing.

    Guards the easy mistake of running a 0.30 % dataset with the primary
    0.05 % sim config (or vice versa), which silently prices the replay at the
    wrong fee tier and snaps its ranges on the wrong grid.
    """
    from undertow.data import load_data_config

    repo_root = Path(__file__).resolve().parents[2]
    for sim_name, data_name in (
        ("sim_default.toml", "eth_usdc_500.toml"),
        ("sim_eth_usdc_500.toml", "eth_usdc_500.toml"),
        ("sim_eth_usdc_3000.toml", "eth_usdc_3000.toml"),
    ):
        sim = load_sim_config(repo_root / "configs" / sim_name)
        data = load_data_config(
            repo_root / "configs" / data_name, expand_endpoints=False
        )
        assert sim.episode.fee_tier_bps == data.pool.fee_tier.value, sim_name
        assert sim.episode.tick_spacing == data.pool.tick_spacing, sim_name
