"""Tests for ``undertow.sim.env.reward`` (S09, CONTRACTS.md §9).

Each expected number is hand-computed in the comment above the assertion so a
regression in the reward gating or sign convention fails here rather than
propagating into every training result.
"""

from __future__ import annotations

import dataclasses

import pytest

from undertow.sim.config import RewardConfig
from undertow.sim.env import RewardBreakdown, compute_reward, normalize_reward


def _config(**overrides: object) -> RewardConfig:
    """All terms enabled by default so each test can flip exactly one flag."""
    defaults: dict[str, object] = {
        "fees_enabled": True,
        "gas_enabled": True,
        "slippage_enabled": True,
        "il_enabled": True,
        "risk_penalty_enabled": True,
        "risk_penalty_lambda": 0.1,
        "normalize_by_capital": True,
    }
    defaults.update(overrides)
    return RewardConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Full breakdown and reward arithmetic
# ---------------------------------------------------------------------------
class TestFullBreakdown:
    def test_all_terms_enabled(self) -> None:
        cfg = _config()
        # fees=100, gas=10 -> -10, slip=5 -> -5, il=-40, risk=-0.1*6=-0.6
        b = compute_reward(100.0, 10.0, 5.0, -40.0, 6.0, cfg)
        assert b.fees == 100.0
        assert b.gas == -10.0
        assert b.slippage == -5.0
        assert b.il_change == -40.0
        assert b.risk_penalty == pytest.approx(-0.6)
        # reward = 100 - 10 - 5 - 40 - 0.6
        assert b.reward == pytest.approx(44.4)
        # pnl_net = 100 - 10 - 5 - 40 (ungated)
        assert b.pnl_net == pytest.approx(45.0)
        assert b.equity == 0.0

    def test_reward_equals_sum_of_terms(self) -> None:
        cfg = _config()
        b = compute_reward(123.4, 17.0, 8.5, -61.25, 9.9, cfg)
        assert b.reward == pytest.approx(
            b.fees + b.gas + b.slippage + b.il_change + b.risk_penalty
        )

    def test_reward_sum_holds_when_some_flags_off(self) -> None:
        cfg = _config(gas_enabled=False, risk_penalty_enabled=False)
        b = compute_reward(50.0, 7.0, 3.0, -20.0, 5.0, cfg)
        assert b.reward == pytest.approx(
            b.fees + b.gas + b.slippage + b.il_change + b.risk_penalty
        )
        assert b.gas == 0.0
        assert b.risk_penalty == 0.0


# ---------------------------------------------------------------------------
# Sign conventions
# ---------------------------------------------------------------------------
class TestSignConventions:
    def test_costs_stored_negative(self) -> None:
        cfg = _config()
        b = compute_reward(0.0, 10.0, 5.0, -1.0, 3.0, cfg)
        assert b.gas == -10.0
        assert b.slippage == -5.0
        assert b.gas <= 0.0
        assert b.slippage <= 0.0
        assert b.risk_penalty <= 0.0

    def test_fees_stored_positive(self) -> None:
        b = compute_reward(42.0, 0.0, 0.0, 0.0, 0.0, _config())
        assert b.fees == 42.0
        assert b.fees >= 0.0

    def test_il_passthrough_is_signed(self) -> None:
        cfg = _config(risk_penalty_enabled=False)
        loss = compute_reward(0.0, 0.0, 0.0, -25.0, 0.0, cfg)
        gain = compute_reward(0.0, 0.0, 0.0, 11.0, 0.0, cfg)
        assert loss.il_change == -25.0
        assert gain.il_change == 11.0

    def test_risk_penalty_formula(self) -> None:
        cfg = _config(risk_penalty_lambda=0.1)
        b = compute_reward(0.0, 0.0, 0.0, 0.0, 100.0, cfg)
        assert b.risk_penalty == pytest.approx(-10.0)

    def test_risk_penalty_default_config_is_off(self) -> None:
        # Contract defaults: lambda=0 and risk_penalty_enabled=False.
        b = compute_reward(0.0, 0.0, 0.0, 0.0, 999.0, RewardConfig())
        assert b.risk_penalty == 0.0


# ---------------------------------------------------------------------------
# Ablation gating: each flag zeroes only its own term
# ---------------------------------------------------------------------------
class TestAblationFlags:
    def test_gas_disabled_zeroes_gas_only(self) -> None:
        cfg = _config(gas_enabled=False)
        b = compute_reward(100.0, 10.0, 5.0, -40.0, 6.0, cfg)
        assert b.gas == 0.0
        # every other term untouched, raw economics still in pnl_net
        assert b.fees == 100.0
        assert b.slippage == -5.0
        assert b.il_change == -40.0
        assert b.risk_penalty == pytest.approx(-0.6)
        assert b.pnl_net == pytest.approx(45.0)

    def test_slippage_disabled_zeroes_slippage_only(self) -> None:
        cfg = _config(slippage_enabled=False)
        b = compute_reward(100.0, 10.0, 5.0, -40.0, 6.0, cfg)
        assert b.slippage == 0.0
        assert b.fees == 100.0
        assert b.gas == -10.0
        assert b.il_change == -40.0
        assert b.pnl_net == pytest.approx(45.0)

    def test_fees_disabled_zeroes_fees_only(self) -> None:
        cfg = _config(fees_enabled=False)
        b = compute_reward(100.0, 10.0, 5.0, -40.0, 6.0, cfg)
        assert b.fees == 0.0
        assert b.gas == -10.0
        assert b.slippage == -5.0
        assert b.il_change == -40.0
        assert b.pnl_net == pytest.approx(45.0)

    def test_il_disabled_zeroes_il_only(self) -> None:
        cfg = _config(il_enabled=False)
        b = compute_reward(100.0, 10.0, 5.0, -40.0, 6.0, cfg)
        assert b.il_change == 0.0
        assert b.fees == 100.0
        assert b.gas == -10.0
        assert b.slippage == -5.0
        assert b.pnl_net == pytest.approx(45.0)

    def test_risk_disabled_zeroes_risk_only(self) -> None:
        cfg = _config(risk_penalty_enabled=False)
        b = compute_reward(100.0, 10.0, 5.0, -40.0, 6.0, cfg)
        assert b.risk_penalty == 0.0
        assert b.fees == 100.0
        assert b.gas == -10.0
        assert b.slippage == -5.0
        assert b.il_change == -40.0
        assert b.pnl_net == pytest.approx(45.0)

    def test_pnl_net_is_flag_invariant(self) -> None:
        # Same raw inputs, every combination of ablation flags -> identical pnl_net.
        raw = (100.0, 10.0, 5.0, -40.0, 6.0)
        expected = 100.0 - 10.0 - 5.0 - 40.0
        for kwargs in (
            {},
            {"fees_enabled": False},
            {"gas_enabled": False},
            {"slippage_enabled": False},
            {"il_enabled": False},
            {"risk_penalty_enabled": False},
            {"fees_enabled": False, "gas_enabled": False, "il_enabled": False},
        ):
            b = compute_reward(*raw, _config(**kwargs))
            assert b.pnl_net == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Golden case (CONTRACTS.md §19): IL/LVR flips the sign of the reward
# ---------------------------------------------------------------------------
class TestGoldenILSignFlip:
    def test_full_vs_no_il(self) -> None:
        # fees=10, gas=5 -> -5, slippage=3 -> -3, il=-15, no risk penalty.
        # full:   10 - 5 - 3 - 15 = -13 (net negative: costs + IL exceed fees)
        # no_il:  10 - 5 - 3      =  +2 (net positive: IL not penalized)
        raw = (10.0, 5.0, 3.0, -15.0, 0.0)
        full = compute_reward(*raw, _config(risk_penalty_enabled=False))
        no_il = compute_reward(
            *raw, _config(risk_penalty_enabled=False, il_enabled=False)
        )
        assert full.reward == pytest.approx(-13.0)
        assert full.reward < 0.0
        assert no_il.reward == pytest.approx(2.0)
        assert no_il.reward > 0.0
        # The economic record is identical either way.
        assert full.pnl_net == pytest.approx(-13.0)
        assert no_il.pnl_net == pytest.approx(-13.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_all_zero_inputs(self) -> None:
        b = compute_reward(0.0, 0.0, 0.0, 0.0, 0.0, _config())
        assert b.reward == 0.0
        assert b.pnl_net == 0.0
        assert b.fees == 0.0
        assert b.gas == 0.0
        assert b.slippage == 0.0
        assert b.il_change == 0.0
        assert b.risk_penalty == 0.0

    def test_zero_volatility_gives_zero_risk_penalty(self) -> None:
        b = compute_reward(10.0, 0.0, 0.0, 0.0, 0.0, _config())
        assert b.risk_penalty == 0.0
        assert b.reward == pytest.approx(10.0)

    def test_zero_il_change_is_neutral(self) -> None:
        b = compute_reward(10.0, 1.0, 1.0, 0.0, 0.0, _config())
        assert b.il_change == 0.0
        assert b.reward == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Normalization by initial capital (applied at the env boundary)
# ---------------------------------------------------------------------------
class TestNormalizeReward:
    def test_scales_every_field_by_inverse_capital(self) -> None:
        cfg = _config()
        b = compute_reward(100.0, 10.0, 5.0, -40.0, 6.0, cfg)
        n = normalize_reward(b, 100_000.0, cfg)
        factor = 1.0 / 100_000.0
        assert n.fees == pytest.approx(b.fees * factor)
        assert n.gas == pytest.approx(b.gas * factor)
        assert n.slippage == pytest.approx(b.slippage * factor)
        assert n.il_change == pytest.approx(b.il_change * factor)
        assert n.risk_penalty == pytest.approx(b.risk_penalty * factor)
        assert n.reward == pytest.approx(b.reward * factor)
        assert n.pnl_net == pytest.approx(b.pnl_net * factor)
        assert n.equity == pytest.approx(b.equity * factor)

    def test_normalization_halves_when_capital_doubles(self) -> None:
        cfg = _config()
        b = compute_reward(100.0, 10.0, 5.0, -40.0, 6.0, cfg)
        n1 = normalize_reward(b, 50_000.0, cfg)
        n2 = normalize_reward(b, 100_000.0, cfg)
        assert n2.reward == pytest.approx(n1.reward / 2.0)
        assert n1.reward > 0.0

    def test_flag_off_returns_unchanged(self) -> None:
        cfg = _config(normalize_by_capital=False)
        b = compute_reward(100.0, 10.0, 5.0, -40.0, 6.0, cfg)
        n = normalize_reward(b, 100_000.0, cfg)
        assert n is b

    def test_zero_capital_raises(self) -> None:
        cfg = _config()
        b = compute_reward(10.0, 0.0, 0.0, 0.0, 0.0, cfg)
        with pytest.raises(ValueError, match="initial_capital"):
            normalize_reward(b, 0.0, cfg)

    def test_negative_capital_raises(self) -> None:
        cfg = _config()
        b = compute_reward(10.0, 0.0, 0.0, 0.0, 0.0, cfg)
        with pytest.raises(ValueError, match="initial_capital"):
            normalize_reward(b, -1.0, cfg)


# ---------------------------------------------------------------------------
# Dataclass contract / field accessibility
# ---------------------------------------------------------------------------
class TestRewardBreakdownShape:
    def test_fields_accessible_with_expected_types(self) -> None:
        b = compute_reward(1.0, 2.0, 3.0, -4.0, 5.0, _config())
        for name in (
            "fees",
            "gas",
            "slippage",
            "il_change",
            "risk_penalty",
            "reward",
            "pnl_net",
            "equity",
        ):
            value = getattr(b, name)
            assert isinstance(value, float), name

    @pytest.mark.parametrize(
        "name",
        [
            "fees",
            "gas",
            "slippage",
            "il_change",
            "risk_penalty",
            "reward",
            "pnl_net",
            "equity",
        ],
    )
    def test_defaults_are_zero(self, name: str) -> None:
        b = RewardBreakdown()
        assert getattr(b, name) == 0.0

    def test_is_frozen(self) -> None:
        b = RewardBreakdown()
        with pytest.raises(dataclasses.FrozenInstanceError):
            b.reward = 1.0  # type: ignore[misc]

    def test_is_slotted(self) -> None:
        assert not hasattr(RewardBreakdown(), "__dict__")