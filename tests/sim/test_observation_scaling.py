"""Observation-vector scaling tests (ADR-034).

The §11 vector feeds the policy network, so its heavy-tailed features are
log-compressed by :func:`scale_vector` and restored by :func:`unscale_vector`.
These tests pin the compression, the exact inverse, and the simulator/backtest
agreement that trained checkpoints depend on.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from undertow.sim.backtest.engine import BacktestObservation
from undertow.sim.env.observations import (
    SCALED_VECTOR_INDICES,
    Observation,
    scale_vector,
    unscale_vector,
)
from undertow.sim.evaluate import observation_from_vector

#: One realistic raw §11 observation, shared by the simulator/backtest checks.
RAW_VALUES: dict[str, object] = {
    "price": 1900.0,
    "sqrt_price": 22_941.0,
    "tick": 200_844,
    "price_returns_1h": 0.01,
    "price_returns_24h": -0.02,
    "realized_vol_24h": 0.55,
    "position_in_range": True,
    "position_value": 100_000.0,
    "uncollected_fees_token0": 1_500_000.0,
    "uncollected_fees_token1": 3.0e17,
    "il_vs_hodl": -0.01,
    "pool_active_liquidity": 3.08e19,
    "pool_fee_tier_bps": 500,
    "gas_price_recent_wei": 1.37e10,
    "eth_usd_price_recent": 1900.0,
    "regime": "sideways",
    "step_in_episode": 12,
    "steps_remaining": 4_308,
    "distance_to_lower_bound_spacings": 2.0,
    "distance_to_upper_bound_spacings": -1.0,
    "position_width_spacings": 4.0,
    "hedge_short_ratio": 0.0,
    "hedge_equity_frac": 0.0,
    "funding_rate_recent": 0.0,
}


def test_scaled_features_are_order_one_and_the_rest_are_untouched() -> None:
    vector = Observation(**RAW_VALUES).to_vector()  # type: ignore[arg-type]
    for index in SCALED_VECTOR_INDICES:
        assert abs(vector[index]) < 50.0
    assert vector[0] == pytest.approx(math.log1p(1900.0))
    assert vector[2] == pytest.approx(math.log1p(200_844))
    assert vector[11] == pytest.approx(math.log1p(3.08e19))
    # Positions outside the scaling set keep their raw value.
    assert vector[3] == pytest.approx(0.01)
    assert vector[12] == 500.0
    assert vector[15] == 2.0
    assert vector[18] == 2.0
    assert vector[21] == 0.0


def test_unscale_vector_inverts_scale_vector_to_machine_precision() -> None:
    raw = np.arange(-5.0, 19.0)
    assert np.allclose(unscale_vector(scale_vector(raw)), raw, rtol=1e-12, atol=1e-12)


def test_observation_from_vector_recovers_raw_fields() -> None:
    obs = observation_from_vector(Observation(**RAW_VALUES).to_vector().tolist())  # type: ignore[arg-type]
    assert obs.price == pytest.approx(1900.0, rel=1e-12)
    assert obs.sqrt_price == pytest.approx(22_941.0, rel=1e-12)
    assert obs.tick == 200_844
    assert obs.position_value == pytest.approx(100_000.0, rel=1e-12)
    assert obs.uncollected_fees_token1 == pytest.approx(3.0e17, rel=1e-12)
    assert obs.pool_active_liquidity == pytest.approx(3.08e19, rel=1e-12)
    assert obs.gas_price_recent_wei == pytest.approx(1.37e10, rel=1e-12)
    assert obs.eth_usd_price_recent == pytest.approx(1900.0, rel=1e-12)


def test_backtest_and_simulator_vectors_agree() -> None:
    """A checkpoint must see the same scale in training and in the backtester."""
    sim_vector = Observation(**RAW_VALUES).to_vector()  # type: ignore[arg-type]
    backtest_vector = BacktestObservation(**RAW_VALUES).to_vector()  # type: ignore[arg-type]
    assert np.array_equal(sim_vector, backtest_vector)
