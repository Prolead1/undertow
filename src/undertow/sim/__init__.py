"""``undertow.sim`` — AMM / concentrated-liquidity simulator and RL environment.

Self-contained package. Does NOT import from ``undertow.data``.
"""

from __future__ import annotations

from undertow.sim.config import (
    ActionGrid,
    EpisodeConfig,
    GasConfig,
    PriceConfig,
    RewardConfig,
    SimConfig,
    SlippageConfig,
    TrainingConfig,
    default_sim_config,
    load_sim_config,
)
from undertow.sim.math import (
    MAX_TICK,
    MIN_TICK,
    Q96,
    Q128,
    align_tick_down,
    align_tick_up,
    amounts_for_liquidity,
    get_amount0_delta,
    get_amount1_delta,
    liquidity_for_amounts,
    price_to_sqrt_price_x96,
    price_to_tick,
    q128_to_decimal,
    sqrt_price_x96_to_price,
    sqrt_price_x96_to_tick,
    tick_to_price,
    tick_to_sqrt_price_x96,
    wrapping_add_256,
    wrapping_sub_256,
)
from undertow.sim.metrics import (
    annualized_return,
    annualized_volatility,
    decomposition,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    var_cvar,
)
from undertow.sim.pool import ConcentratedLiquidityPool, Position, TickState
from undertow.sim.price import GBMPrice, ReplayPrice, RegimeSwitchingPrice

__all__ = [
    # Config
    "SimConfig",
    "EpisodeConfig",
    "ActionGrid",
    "GasConfig",
    "PriceConfig",
    "RewardConfig",
    "SlippageConfig",
    "TrainingConfig",
    "load_sim_config",
    # Math
    "Q96",
    "Q128",
    "MIN_TICK",
    "MAX_TICK",
    "sqrt_price_x96_to_price",
    "price_to_sqrt_price_x96",
    "tick_to_sqrt_price_x96",
    "sqrt_price_x96_to_tick",
    "tick_to_price",
    "price_to_tick",
    "align_tick_down",
    "align_tick_up",
    "wrapping_sub_256",
    "wrapping_add_256",
    "q128_to_decimal",
    "get_amount0_delta",
    "get_amount1_delta",
    "liquidity_for_amounts",
    "amounts_for_liquidity",
    # Pool
    "Position",
    "TickState",
    # Price
    "GBMPrice",
    "RegimeSwitchingPrice",
    "ReplayPrice",
    # Metrics
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "var_cvar",
    "decomposition",
    "annualized_return",
    "annualized_volatility",
]
