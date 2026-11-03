"""Tests for ``undertow.sim.config``."""
# ruff: noqa: ANN001  # pytest fixtures (tmp_path, ...)

from __future__ import annotations

import math
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
        # 0 = unconstrained: historical behaviour (ADR-037).
        assert ep.min_rebalance_interval_steps == 0

    def test_min_rebalance_interval_accepts_large_values(self) -> None:
        ep = EpisodeConfig(min_rebalance_interval_steps=1_000_000)
        assert ep.min_rebalance_interval_steps == 1_000_000

    @pytest.mark.parametrize("value", [-1, -100])
    def test_min_rebalance_interval_rejects_negative(self, value: int) -> None:
        with pytest.raises(
            SimConfigError, match=r"episode\.min_rebalance_interval_steps"
        ):
            EpisodeConfig(min_rebalance_interval_steps=value)

    @pytest.mark.parametrize(
        "value", [1.5, float("nan"), float("inf"), float("-inf")]
    )
    def test_min_rebalance_interval_rejects_non_integral_or_non_finite(
        self, value: float
    ) -> None:
        with pytest.raises(
            SimConfigError, match=r"episode\.min_rebalance_interval_steps"
        ):
            EpisodeConfig(min_rebalance_interval_steps=value)  # type: ignore[arg-type]

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
        # No hold prior by default: behaviour is unchanged unless opted in.
        assert tc.hold_bias == 0.0

    def test_optimizer_anchor_defaults(self) -> None:
        """PLAN.md §7 anchors; ``ent_coef`` follows SB3's 0.0 (ADR-036)."""
        tc = TrainingConfig()
        assert tc.learning_rate == 3e-4
        assert tc.batch_size == 64
        assert tc.n_epochs == 10
        assert tc.ent_coef == 0.0
        assert tc.target_kl is None

    def test_optimizer_knob_boundaries_are_accepted(self) -> None:
        """The smallest mini-batch and a zero entropy bonus both validate."""
        tc = TrainingConfig(batch_size=2, ent_coef=0.0)
        assert tc.batch_size == 2
        assert tc.ent_coef == 0.0

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("learning_rate", 0),
            ("learning_rate", -1e-4),
            ("batch_size", 1),
            ("n_epochs", 0),
            ("ent_coef", -0.1),
            ("target_kl", 0),
            ("target_kl", -1),
        ],
    )
    def test_optimizer_knobs_reject_invalid_values(self, field, value) -> None:
        with pytest.raises(SimConfigError, match=field):
            TrainingConfig(**{field: value})

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("learning_rate", math.nan),
            ("learning_rate", math.inf),
            ("learning_rate", -math.inf),
            ("batch_size", math.nan),
            ("batch_size", math.inf),
            ("batch_size", -math.inf),
            ("n_epochs", math.nan),
            ("n_epochs", math.inf),
            ("n_epochs", -math.inf),
            ("ent_coef", math.nan),
            ("ent_coef", math.inf),
            ("ent_coef", -math.inf),
            ("target_kl", math.nan),
            ("target_kl", math.inf),
            ("target_kl", -math.inf),
        ],
    )
    def test_optimizer_knobs_reject_non_finite_values(self, field, value) -> None:
        with pytest.raises(SimConfigError, match=field):
            TrainingConfig(**{field: value})

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("batch_size", 2.9),
            ("batch_size", 64.5),
            ("n_epochs", 1.5),
            ("n_epochs", 10.2),
        ],
    )
    def test_optimizer_knobs_reject_non_integral_values(self, field, value) -> None:
        with pytest.raises(SimConfigError, match=field):
            TrainingConfig(**{field: value})

    def test_rollouts_can_still_be_disabled(self) -> None:
        assert TrainingConfig(record_rollouts=False).record_rollouts is False

    def test_vec_env_accepts_subproc_and_rejects_unknown(self) -> None:
        assert TrainingConfig(vec_env="subproc").vec_env == "subproc"
        with pytest.raises(SimConfigError, match="vec_env"):
            TrainingConfig(vec_env="threads")  # type: ignore[arg-type]

    def test_hold_bias_accepts_a_positive_prior(self) -> None:
        assert TrainingConfig(hold_bias=2.5).hold_bias == 2.5

    @pytest.mark.parametrize("hold_bias", [-1.0, -1e-9, math.nan, math.inf])
    def test_hold_bias_must_be_finite_and_non_negative(self, hold_bias) -> None:
        with pytest.raises(SimConfigError, match="hold_bias"):
            TrainingConfig(hold_bias=hold_bias)


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

    def test_min_rebalance_interval_round_trips_through_the_episode_section(
        self, tmp_path
    ) -> None:
        path = tmp_path / "interval.toml"
        _write_toml(path, "[episode]\nmin_rebalance_interval_steps = 3\n")
        cfg = load_sim_config(str(path))
        # ``_populate`` picks the new dataclass field up without a loader change.
        assert cfg.episode.min_rebalance_interval_steps == 3
        assert cfg.episode.duration_days == 30  # other keys default

    def test_omitted_min_rebalance_interval_defaults_to_zero(self, tmp_path) -> None:
        path = tmp_path / "no_interval.toml"
        _write_toml(path, "[episode]\nduration_days = 7\n")
        cfg = load_sim_config(str(path))
        assert cfg.episode.min_rebalance_interval_steps == 0

    @pytest.mark.parametrize("literal", ["-1", "1.5", "nan", "inf"])
    def test_min_rebalance_interval_validation_applies_through_the_loader(
        self, tmp_path, literal
    ) -> None:
        path = tmp_path / "bad_interval.toml"
        _write_toml(path, f"[episode]\nmin_rebalance_interval_steps = {literal}\n")
        with pytest.raises(SimConfigError, match="min_rebalance_interval_steps"):
            load_sim_config(str(path))

    def test_hold_bias_round_trips_through_the_training_section(self, tmp_path) -> None:
        path = tmp_path / "hold_bias.toml"
        _write_toml(path, "[training]\nhold_bias = 2.5\n")
        cfg = load_sim_config(str(path))
        # ``_populate`` picks the new dataclass field up without a loader change.
        assert cfg.training.hold_bias == 2.5
        assert cfg.training.total_timesteps == 1_000_000  # other keys default

    @pytest.mark.parametrize("literal", ["-0.5", "nan", "-inf"])
    def test_hold_bias_validation_applies_through_the_loader(
        self, tmp_path, literal
    ) -> None:
        path = tmp_path / "bad_hold_bias.toml"
        _write_toml(path, f"[training]\nhold_bias = {literal}\n")
        with pytest.raises(SimConfigError, match="hold_bias"):
            load_sim_config(str(path))

    def test_optimizer_knobs_round_trip_through_the_training_section(self, tmp_path) -> None:
        path = tmp_path / "optimizer.toml"
        _write_toml(
            path,
            "[training]\n"
            "learning_rate = 0.0001\n"
            "batch_size = 512\n"
            "n_epochs = 4\n"
            "ent_coef = 0.001\n"
            "target_kl = 0.01\n",
        )
        cfg = load_sim_config(str(path))
        # ``_populate`` picks the new dataclass fields up without a loader
        # change: values load and unspecified keys keep their defaults.
        assert cfg.training.learning_rate == pytest.approx(1e-4)
        assert cfg.training.batch_size == 512
        assert cfg.training.n_epochs == 4
        assert cfg.training.ent_coef == pytest.approx(0.001)
        assert cfg.training.target_kl == pytest.approx(0.01)
        assert cfg.training.total_timesteps == 1_000_000

    @pytest.mark.parametrize(
        ("field", "literal"),
        [
            ("learning_rate", "0.0"),
            ("batch_size", "1"),
            ("batch_size", "2.9"),
            ("batch_size", "nan"),
            ("n_epochs", "0"),
            ("n_epochs", "1.5"),
            ("n_epochs", "inf"),
            ("ent_coef", "-0.1"),
            ("target_kl", "0.0"),
        ],
    )
    def test_optimizer_validation_applies_through_the_loader(
        self, tmp_path, field, literal
    ) -> None:
        path = tmp_path / "bad_optimizer.toml"
        _write_toml(path, f"[training]\n{field} = {literal}\n")
        with pytest.raises(SimConfigError, match=field):
            load_sim_config(str(path))


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
        ("sim_eth_usdc_500_rl.toml", "eth_usdc_500.toml"),
        ("sim_eth_usdc_3000.toml", "eth_usdc_3000.toml"),
    ):
        sim = load_sim_config(repo_root / "configs" / sim_name)
        data = load_data_config(
            repo_root / "configs" / data_name, expand_endpoints=False
        )
        assert sim.episode.fee_tier_bps == data.pool.fee_tier.value, sim_name
        assert sim.episode.tick_spacing == data.pool.tick_spacing, sim_name


def test_primary_pool_config_carries_the_stabilized_recipe() -> None:
    """`configs/sim_eth_usdc_500.toml` lands ADR-036's overrides, not defaults."""
    repo_root = Path(__file__).resolve().parents[2]
    cfg = load_sim_config(repo_root / "configs" / "sim_eth_usdc_500.toml")

    grid = cfg.action_grid
    assert grid.center_offsets == (-1, 0, 1)
    assert grid.widths == (2, 5, 10, 25, 50, 100, 250, 500, 1000)
    # Factored grid size: hold + offsets x widths; the primary pool uses the
    # reduced 28-action grid so `hold` is explorable (ADR-036).
    assert 1 + len(grid.center_offsets) * len(grid.widths) == 28

    training = cfg.training
    assert training.learning_rate == pytest.approx(1e-4)
    assert training.batch_size == 512
    assert training.n_epochs == 4
    assert training.ent_coef == pytest.approx(0.001)
    assert training.target_kl == pytest.approx(0.01)
    assert training.hold_bias == pytest.approx(4.0)


def test_shipped_rl_config_carries_the_band_selection_recipe() -> None:
    """`configs/sim_eth_usdc_500_rl.toml` lands ADR-038's recipe, not defaults."""
    repo_root = Path(__file__).resolve().parents[2]
    cfg = load_sim_config(repo_root / "configs" / "sim_eth_usdc_500_rl.toml")

    grid = cfg.action_grid
    assert grid.center_offsets == (0,)
    assert grid.widths == (50, 250, 1000)
    # One-shot band choice: hold + 1 offset x 3 widths (ADR-038).
    assert 1 + len(grid.center_offsets) * len(grid.widths) == 4

    episode = cfg.episode
    # One rebalance per episode: the policy deploys once and holds (ADR-037).
    assert episode.min_rebalance_interval_steps == 1_000_000_000

    training = cfg.training
    # Near-undiscounted gamma so the step-0 band choice stays in the value
    # targets over the 4320-step episode.
    assert training.discount_gamma == pytest.approx(0.9999)
    assert training.hold_bias == pytest.approx(0.0)
    assert training.learning_rate == pytest.approx(1e-4)
    assert training.batch_size == 512
    assert training.n_epochs == 4
    assert training.ent_coef == pytest.approx(0.001)
    assert training.target_kl == pytest.approx(0.01)
    # Seeds, budget and vectorisation stay at the primary values (ADR-038).
    assert training.seeds == (0, 1, 2, 3, 4)
    assert training.total_timesteps == 1_000_000
    assert training.checkpoint_freq_steps == 100_000
    assert training.parallel_envs == 4
    assert training.vec_env == "subproc"
