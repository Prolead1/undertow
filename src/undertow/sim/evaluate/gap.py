"""RQ3 simulation-to-reality gap (S15, CONTRACTS.md §17).

The gap report quantifies ``Δ = sim_PnL − onchain_PnL`` for one frozen policy
over **identical** walk-forward episodes, decomposed by cost term.  The
backtester is the on-chain (ground-truth) measurement instrument; the replay
simulator is the float64 world model.  S14 established that the two agree within
the stated tolerance on identical inputs, so this module is an aggregation of
that drift over full episodes, not a re-validation.

Honesty
-------
The gap is reported even when large.  It is a thesis finding, not a failure: it
is exactly the residual realism left after fees, gas, slippage and IL have been
modelled.  ``gap == sim_pnl - onchain_pnl`` holds by construction and
``gap_by_cost_type`` reports the per-term difference with the ledger's sign
convention (fees ``>= 0``; gas/slippage/IL ``<= 0``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from undertow.sim.config import SimConfig
from undertow.sim.evaluate.protocol import (
    EpisodeRecord,
    EvaluationError,
    evaluate_episodes,
    walk_forward_windows,
)

if TYPE_CHECKING:
    from undertow.sim.env.lp_env import LpEnvironment
    from undertow.sim.marketview import MarketView
    from undertow.sim.policies import Policy

__all__ = ["GapReport", "run_gap_analysis"]

#: Decomposition keys reported by the gap report, mapped to their ledger field
#: in :meth:`EpisodeRecord.decomposition`.
_COST_KEYS: tuple[tuple[str, str], ...] = (
    ("fees", "total_fees"),
    ("gas", "total_gas"),
    ("slippage", "total_slippage"),
    ("il", "total_il"),
)


@dataclass(frozen=True, slots=True)
class GapReport:
    """RQ3: simulation-to-reality gap (CONTRACTS §17)."""

    policy_name: str
    sim_pnl: float
    onchain_pnl: float
    gap: float  # sim_pnl - onchain_pnl
    gap_by_cost_type: dict[str, float]


def _mean(values: list[float]) -> float:
    """Arithmetic mean of a possibly-empty list (0.0 when empty)."""
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _mean_decomposition(
    records: list[EpisodeRecord], key: str
) -> float:
    """Mean of one decomposition field across episodes."""
    return _mean([float(record.decomposition.get(key, 0.0)) for record in records])


def run_gap_analysis(
    config: SimConfig,
    policy: Policy,
    market_view: MarketView,
    *,
    env_factory: Callable[[MarketView, SimConfig], LpEnvironment] | None = None,
    num_seeds: int = 1,
) -> GapReport:
    """Quantify the sim-to-reality gap over the same walk-forward episodes.

    The simulator is run first, and the backtester is then run on the simulator's
    **exact** sampled window for each episode.  That makes the comparison literal:
    both artifacts see the same tape rows even when the tape carries more events
    than one episode's decision budget (the environment samples a sub-window in
    that case) and even when ``num_seeds`` produces several samples.
    ``num_seeds`` defaults to one.
    """
    windows = walk_forward_windows(market_view, config)
    sim_records = evaluate_episodes(
        policy,
        market_view,
        config,
        num_seeds=num_seeds,
        backend="replay_sim",
        env_factory=env_factory,
        windows=windows,
    )
    unique_windows = list(
        dict.fromkeys((record.start_seq, record.end_seq) for record in sim_records)
    )
    onchain_by_window = {
        (record.start_seq, record.end_seq): record
        for record in evaluate_episodes(
            policy,
            market_view,
            config,
            num_seeds=1,
            backend="backtest",
            windows=unique_windows,
        )
    }
    onchain_records: list[EpisodeRecord] = []
    for record in sim_records:
        key = (record.start_seq, record.end_seq)
        matched = onchain_by_window.get(key)
        if matched is None:  # pragma: no cover - defensive
            raise EvaluationError(
                f"run_gap_analysis: no backtest episode for window {key}"
            )
        onchain_records.append(matched)

    sim_pnl = _mean([float(record.metrics["total_pnl"]) for record in sim_records])
    onchain_pnl = _mean(
        [float(record.metrics["total_pnl"]) for record in onchain_records]
    )
    gap_by_cost_type = {
        name: _mean_decomposition(sim_records, key)
        - _mean_decomposition(onchain_records, key)
        for name, key in _COST_KEYS
    }

    return GapReport(
        policy_name=policy.name,
        sim_pnl=sim_pnl,
        onchain_pnl=onchain_pnl,
        gap=sim_pnl - onchain_pnl,
        gap_by_cost_type=gap_by_cost_type,
    )
