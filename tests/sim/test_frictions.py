"""S08 — friction models (gas + slippage).

Every test asserts real, hand-computed behavior: gas arithmetic per action type,
the ADR-005 flat tip surcharge (and the ignored ``priority_fee_p50_wei``
argument), tape replay, spikes, slippage proportionality, protocol conformance,
fixture wiring and non-negativity.
"""

from __future__ import annotations

import numpy as np
import pytest

from undertow.sim.config import (
    GAS_BURN,
    GAS_COLLECT,
    GAS_MINT,
    GAS_REBALANCE_SWAP,
    GasConfig,
)
from undertow.sim.frictions import (
    DEFAULT_TIP_SURCHARGE_PCT,
    FlatGasModel,
    GasCostCalculator,
    GasModel,
    ProportionalSlippageModel,
    ReplayGasModel,
    SlippageModel,
    SpikeGasModel,
)
from undertow.sim.marketview import MarketView

# Hand-computed reference inputs (ADR-005: base fee only, +3% tip surcharge).
_BASE_FEE_WEI = 30 * 10**9
_ETH_USD = 3000.0
_WEI_PER_ETH = 1e18


def _expected_usdc(gas_units: int, base_fee_wei: int = _BASE_FEE_WEI,
                   tip_pct: float = DEFAULT_TIP_SURCHARGE_PCT,
                   eth_usd: float = _ETH_USD) -> float:
    """Reference implementation of the ADR-005 gas formula."""
    return base_fee_wei * (1.0 + tip_pct / 100.0) * gas_units * eth_usd / _WEI_PER_ETH


# ---------------------------------------------------------------------------
# GasCostCalculator
# ---------------------------------------------------------------------------


def test_gas_calculator_mint_exact() -> None:
    """`mint` = GAS_MINT units at the uplifted base fee."""
    calc = GasCostCalculator(GasConfig())
    cost = calc.gas_cost_usdc("mint", 0, _BASE_FEE_WEI, 0, _ETH_USD)
    assert cost == pytest.approx(_expected_usdc(GAS_MINT))
    # hand-computed literal: 460_000 * 30e9 * 1.03 * 3000 / 1e18 == 42.642
    assert cost == pytest.approx(42.642)


def test_gas_calculator_rebalance_is_burn_plus_mint_plus_swap() -> None:
    """`rebalance` = burn + mint + swap units (S08 mapping)."""
    calc = GasCostCalculator(GasConfig())
    units = GAS_BURN + GAS_MINT + GAS_REBALANCE_SWAP
    assert calc.gas_units_for("rebalance") == units
    cost = calc.gas_cost_usdc("rebalance", 0, _BASE_FEE_WEI, 0, _ETH_USD)
    assert cost == pytest.approx(_expected_usdc(units))
    # hand-computed literal: 825_000 * 30e9 * 1.03 * 3000 / 1e18 == 76.4775
    assert cost == pytest.approx(76.4775)


@pytest.mark.parametrize(
    ("action_type", "units"),
    [
        ("mint", GAS_MINT),
        ("burn", GAS_BURN),
        ("collect", GAS_COLLECT),
        ("rebalance_swap", GAS_REBALANCE_SWAP),
        ("rebalance", GAS_BURN + GAS_MINT + GAS_REBALANCE_SWAP),
    ],
)
def test_gas_calculator_each_action_type(action_type: str, units: int) -> None:
    calc = GasCostCalculator(GasConfig())
    assert calc.gas_units_for(action_type) == units
    assert calc.gas_cost_usdc(
        action_type, 0, _BASE_FEE_WEI, 0, _ETH_USD
    ) == pytest.approx(_expected_usdc(units))


def test_gas_calculator_hold_is_free() -> None:
    calc = GasCostCalculator(GasConfig())
    assert calc.gas_units_for("hold") == 0
    assert calc.gas_cost_usdc("hold", 7, _BASE_FEE_WEI, 0, _ETH_USD) == 0.0


def test_gas_calculator_unknown_action_raises() -> None:
    calc = GasCostCalculator(GasConfig())
    with pytest.raises(ValueError, match="unknown action_type"):
        calc.gas_cost_usdc("teleport", 0, _BASE_FEE_WEI, 0, _ETH_USD)
    with pytest.raises(ValueError, match="unknown action_type"):
        calc.gas_units_for("")


def test_gas_calculator_flat_surcharge_arithmetic() -> None:
    """tip_surcharge_pct scales the base fee linearly (ADR-005)."""
    no_tip = GasCostCalculator(GasConfig(), tip_surcharge_pct=0)
    tip_10 = GasCostCalculator(GasConfig(), tip_surcharge_pct=10)
    base = no_tip.gas_cost_usdc("mint", 0, _BASE_FEE_WEI, 0, _ETH_USD)
    uplifted = tip_10.gas_cost_usdc("mint", 0, _BASE_FEE_WEI, 0, _ETH_USD)
    assert base == pytest.approx(41.4)  # 460_000 * 30e9 * 3000 / 1e18
    assert uplifted == pytest.approx(45.54)  # 460_000 * 33e9 * 3000 / 1e18
    assert uplifted == pytest.approx(base * 1.10)


def test_gas_calculator_ignores_priority_fee() -> None:
    """ADR-005/006 mismatch: the priority fee argument must not affect cost."""
    calc = GasCostCalculator(GasConfig())
    without = calc.gas_cost_usdc("mint", 0, _BASE_FEE_WEI, 0, _ETH_USD)
    with_huge = calc.gas_cost_usdc(
        "mint", 0, _BASE_FEE_WEI, 10**18, _ETH_USD
    )
    assert without == with_huge


def test_gas_calculator_rejects_negative_inputs() -> None:
    calc = GasCostCalculator(GasConfig())
    with pytest.raises(ValueError):
        calc.gas_cost_usdc("mint", 0, -1, 0, _ETH_USD)
    with pytest.raises(ValueError):
        calc.gas_cost_usdc("mint", 0, _BASE_FEE_WEI, 0, -1.0)
    with pytest.raises(ValueError):
        GasCostCalculator(GasConfig(), tip_surcharge_pct=-1)


# ---------------------------------------------------------------------------
# ReplayGasModel
# ---------------------------------------------------------------------------


def test_replay_uses_tape_base_fee(tiny_market_view: MarketView) -> None:
    """Block 0 carries base_fee 20e9 in the fixture; the passed arg is ignored."""
    model = ReplayGasModel(tiny_market_view)
    cost = model.gas_cost_usdc("mint", 0, base_fee_per_gas_wei=999, priority_fee_p50_wei=999,
                               eth_usd_price=_ETH_USD)
    # 460_000 * 20e9 * 1.03 * 3000 / 1e18 == 28.428
    assert cost == pytest.approx(28.428)
    assert cost == pytest.approx(
        _expected_usdc(GAS_MINT, base_fee_wei=20 * 10**9)
    )


def test_replay_rebalance_uses_tape(tiny_market_view: MarketView) -> None:
    model = ReplayGasModel(tiny_market_view)
    cost = model.gas_cost_usdc("rebalance", 3, 0, 0, _ETH_USD)
    units = GAS_BURN + GAS_MINT + GAS_REBALANCE_SWAP
    assert cost == pytest.approx(
        _expected_usdc(units, base_fee_wei=20 * 10**9)
    )


def test_replay_missing_block_is_zero(tiny_market_view: MarketView) -> None:
    model = ReplayGasModel(tiny_market_view)
    assert model.gas_cost_usdc("mint", 999_999, 0, 0, _ETH_USD) == 0.0


def test_replay_tip_surcharge_forwarded(tiny_market_view: MarketView) -> None:
    base_only = ReplayGasModel(tiny_market_view, tip_surcharge_pct=0)
    tipped = ReplayGasModel(tiny_market_view, tip_surcharge_pct=10)
    b = base_only.gas_cost_usdc("mint", 0, 0, 0, _ETH_USD)
    t = tipped.gas_cost_usdc("mint", 0, 0, 0, _ETH_USD)
    assert t == pytest.approx(b * 1.10)


# ---------------------------------------------------------------------------
# FlatGasModel
# ---------------------------------------------------------------------------


def test_flat_is_block_independent() -> None:
    model = FlatGasModel(GasConfig(), eth_usd_price=3000.0, base_fee_gwei=30.0)
    for block in (0, 1, 999_999):
        assert model.gas_cost_usdc("mint", block, 0, 0, 0.0) == pytest.approx(42.642)


def test_flat_expected_action_costs() -> None:
    model = FlatGasModel(GasConfig(), eth_usd_price=3000.0, base_fee_gwei=30.0)
    assert model.gas_cost_usdc("mint", 0, 0, 0, 0.0) == pytest.approx(42.642)
    assert model.gas_cost_usdc("rebalance", 0, 0, 0, 0.0) == pytest.approx(76.4775)
    assert model.gas_cost_usdc("hold", 0, 0, 0, 0.0) == 0.0


def test_flat_custom_price_and_fee() -> None:
    model = FlatGasModel(
        GasConfig(), eth_usd_price=2000.0, base_fee_gwei=10.0, tip_surcharge_pct=0
    )
    # 460_000 * 10e9 * 2000 / 1e18 == 9.2
    assert model.gas_cost_usdc("mint", 0, 0, 0, 0.0) == pytest.approx(9.2)


# ---------------------------------------------------------------------------
# SpikeGasModel
# ---------------------------------------------------------------------------


def test_spike_probability_one_always_spikes() -> None:
    base = FlatGasModel(GasConfig())
    model = SpikeGasModel(
        base, spike_multiplier=10.0, spike_probability=1.0,
        rng=np.random.default_rng(123),
    )
    expected = base.gas_cost_usdc("mint", 0, 0, 0, 0.0) * 10.0
    for _ in range(5):
        assert model.gas_cost_usdc("mint", 0, 0, 0, 0.0) == pytest.approx(expected)


def test_spike_probability_zero_never_spikes() -> None:
    base = FlatGasModel(GasConfig())
    model = SpikeGasModel(
        base, spike_multiplier=10.0, spike_probability=0.0,
        rng=np.random.default_rng(123),
    )
    expected = base.gas_cost_usdc("mint", 0, 0, 0, 0.0)
    for _ in range(5):
        assert model.gas_cost_usdc("mint", 0, 0, 0, 0.0) == pytest.approx(expected)


def test_spike_is_deterministic_for_a_seed() -> None:
    base = FlatGasModel(GasConfig())

    def run(seed: int) -> list[float]:
        model = SpikeGasModel(
            base, spike_multiplier=10.0, spike_probability=0.5,
            rng=np.random.default_rng(seed),
        )
        return [model.gas_cost_usdc("mint", 0, 0, 0, 0.0) for _ in range(20)]

    assert run(7) == run(7)
    assert run(7) != run(8)


def test_spike_rejects_bad_probability() -> None:
    base = FlatGasModel(GasConfig())
    with pytest.raises(ValueError):
        SpikeGasModel(base, spike_probability=1.5)
    with pytest.raises(ValueError):
        SpikeGasModel(base, spike_multiplier=-1.0)


# ---------------------------------------------------------------------------
# ProportionalSlippageModel
# ---------------------------------------------------------------------------


def test_slippage_exact_formula() -> None:
    model = ProportionalSlippageModel()
    # ADR-012: pool fee in pips (3000/1e6) + impact in bps (5/1e4).
    # 100_000 * (3000/1_000_000 + 5/10_000) == 100_000 * 0.0035 == 350.
    assert model.slippage_cost_usdc(100_000.0, 3000, 5.0) == pytest.approx(350.0)
    assert model.slippage_cost_usdc(100_000.0, 3000, 5.0) == pytest.approx(
        100_000.0 * (3000 / 1_000_000.0 + 5.0 / 10_000.0)
    )


def test_slippage_zero_notional_is_zero() -> None:
    model = ProportionalSlippageModel()
    assert model.slippage_cost_usdc(0.0, 3000, 5.0) == 0.0


def test_slippage_rejects_negative_inputs() -> None:
    model = ProportionalSlippageModel()
    with pytest.raises(ValueError):
        model.slippage_cost_usdc(-1.0, 3000, 5.0)
    with pytest.raises(ValueError):
        model.slippage_cost_usdc(1.0, -1, 5.0)
    with pytest.raises(ValueError):
        model.slippage_cost_usdc(1.0, 3000, -1.0)


# ---------------------------------------------------------------------------
# Protocol conformance, fixtures, non-negativity
# ---------------------------------------------------------------------------


def test_protocols_are_runtime_checkable(tiny_market_view: MarketView) -> None:
    assert isinstance(GasCostCalculator(GasConfig()), GasModel)
    assert isinstance(ReplayGasModel(tiny_market_view), GasModel)
    assert isinstance(FlatGasModel(GasConfig()), GasModel)
    assert isinstance(
        SpikeGasModel(FlatGasModel(GasConfig())), GasModel
    )
    assert isinstance(ProportionalSlippageModel(), SlippageModel)
    assert not isinstance(object(), GasModel)


def test_fixtures_are_importable(
    mock_gas_model: FlatGasModel,
    mock_slippage_model: ProportionalSlippageModel,
) -> None:
    assert mock_gas_model.gas_cost_usdc("mint", 0, 0, 0, 0.0) == pytest.approx(42.642)
    assert mock_gas_model.gas_cost_usdc("rebalance", 0, 0, 0, 0.0) == pytest.approx(
        76.4775
    )
    assert mock_gas_model.gas_cost_usdc("hold", 0, 0, 0, 0.0) == 0.0
    assert mock_slippage_model.slippage_cost_usdc(
        100_000.0, 3000, 5.0
    ) == pytest.approx(350.0)


def test_gas_cost_never_negative(tiny_market_view: MarketView) -> None:
    models = [
        GasCostCalculator(GasConfig()),
        FlatGasModel(GasConfig()),
        ReplayGasModel(tiny_market_view),
        SpikeGasModel(FlatGasModel(GasConfig()), spike_probability=1.0),
    ]
    for model in models:
        for action in ("mint", "burn", "collect", "rebalance_swap", "rebalance", "hold"):
            cost = model.gas_cost_usdc(action, 0, _BASE_FEE_WEI, 0, _ETH_USD)
            assert cost >= 0.0


def test_slippage_cost_never_negative() -> None:
    model = ProportionalSlippageModel()
    for notional in (0.0, 1.0, 100_000.0, 10**9):
        assert model.slippage_cost_usdc(notional, 3000, 5.0) >= 0.0
