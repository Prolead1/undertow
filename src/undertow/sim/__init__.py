"""``undertow.sim`` — AMM / concentrated-liquidity simulator and RL environment.

This module is the package's public API surface (S16, ``PLAN.md`` §2.1).  Every
name below resolves at import time and is re-exported from the sub-package that
owns it; the simulator's internal modules remain importable by full path for
consumers that need a type that is deliberately not part of this surface.

The package boundary is law: ``undertow.sim`` reads historical data only through
the top-level ``undertow.data`` public API and never imports a ``undertow.data``
internal module (``PLAN.md`` §4).

Public groups
-------------
* **config** — :class:`SimConfig`, :func:`load_sim_config`.
* **types** — domain aliases, enums and the exception root.
* **core** — position valuation and the tick-lattice pool engine.
* **marketview** — the look-ahead-safe window onto the dataset.
* **prices** — replay and calibrated price processes.
* **frictions** — gas and slippage protocols.
* **policies** — the frozen :class:`Policy` protocol and the baseline ladder.
* **env** — the Gymnasium environment, observation and reward.
* **backtest** — the frozen-policy event-replay backtester.
* **metrics** — performance metrics and PnL decomposition.
* **train** — the PPO training harness.
* **validation** — parity and look-ahead probes.
* **evaluate** — the RQ1–RQ3 evaluation runner.
"""

from __future__ import annotations

# -- Backtester -------------------------------------------------------------
from undertow.sim.backtest import (
    BacktestLedger,
    BacktestResult,
    run_backtest,
    summarize_backtest,
)

# -- Config -----------------------------------------------------------------
from undertow.sim.config import SimConfig, load_sim_config

# -- Core -------------------------------------------------------------------
from undertow.sim.core.pool import PoolEngine, PoolState, TickState
from undertow.sim.core.position import Position, initial_deposit

# -- Environment ------------------------------------------------------------
from undertow.sim.env import (
    LpEnvironment,
    Observation,
    RewardBreakdown,
    compute_reward,
)

# -- Evaluation -------------------------------------------------------------
from undertow.sim.evaluate import (
    run_ablations,
    run_gap_analysis,
    run_regime_evaluation,
    write_results,
)

# -- Frictions --------------------------------------------------------------
from undertow.sim.frictions import GasModel, SlippageModel

# -- Market data ------------------------------------------------------------
from undertow.sim.marketview import MarketView, build_market_view

# -- Metrics ----------------------------------------------------------------
from undertow.sim.metrics import (
    all_metrics,
    annualized_return,
    max_drawdown,
    per_regime_metrics,
    pnl_decomposition,
    sharpe_ratio,
    sortino_ratio,
)

# -- Policies ---------------------------------------------------------------
from undertow.sim.policies import (
    CostAwareRebalancePolicy,
    FullRangeV2Policy,
    HODLPolicy,
    PassiveNarrowPolicy,
    PassiveWidePolicy,
    Policy,
    TauResetPolicy,
)

# -- Prices -----------------------------------------------------------------
from undertow.sim.prices import (
    CalibratedPriceProcess,
    MRSJDParams,
    PriceProcess,
    ReplayPriceProcess,
)

# -- Training ---------------------------------------------------------------
from undertow.sim.train import (
    RunManifest,
    TrainingResult,
    train_ppo,
    train_single_seed,
)

# -- Types ------------------------------------------------------------------
from undertow.sim.types import (
    Action,
    Price,
    PriceMode,
    Regime,
    SqrtPrice,
    Tick,
    TickSpacing,
    UndertowSimError,
    Wealth,
)

# -- Validation -------------------------------------------------------------
from undertow.sim.validation import run_lookahead_probes, run_parity_check

__all__ = [
    # Config
    "SimConfig",
    "load_sim_config",
    # Types
    "Action",
    "Tick",
    "TickSpacing",
    "SqrtPrice",
    "Price",
    "Wealth",
    "Regime",
    "PriceMode",
    "UndertowSimError",
    # Core
    "Position",
    "initial_deposit",
    "PoolEngine",
    "PoolState",
    "TickState",
    # Market data
    "MarketView",
    "build_market_view",
    # Prices
    "PriceProcess",
    "ReplayPriceProcess",
    "CalibratedPriceProcess",
    "MRSJDParams",
    # Frictions
    "GasModel",
    "SlippageModel",
    # Policies
    "Policy",
    "HODLPolicy",
    "PassiveNarrowPolicy",
    "PassiveWidePolicy",
    "FullRangeV2Policy",
    "TauResetPolicy",
    "CostAwareRebalancePolicy",
    # Environment
    "LpEnvironment",
    "Observation",
    "compute_reward",
    "RewardBreakdown",
    # Backtester
    "run_backtest",
    "summarize_backtest",
    "BacktestLedger",
    "BacktestResult",
    # Metrics
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "annualized_return",
    "all_metrics",
    "pnl_decomposition",
    "per_regime_metrics",
    # Training
    "train_ppo",
    "train_single_seed",
    "TrainingResult",
    "RunManifest",
    # Validation
    "run_parity_check",
    "run_lookahead_probes",
    # Evaluation
    "run_ablations",
    "run_regime_evaluation",
    "run_gap_analysis",
    "write_results",
]
