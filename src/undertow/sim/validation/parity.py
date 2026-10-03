"""Sim-vs-backtester parity harness (S14, ``CONTRACTS.md`` §16).

The simulator and the backtester live in two different numeric worlds: the
training simulator (S12) accumulates fee growth in **float64**, while the
backtester (S11) accrues fees through ``undertow.data``'s **exact-integer Q128**
``FeeGrowthReplay`` facade (ADR-013).  The thesis's credibility depends on being
able to show that, fed the *identical* replay inputs (same window, same initial
position, same reference price path, same gas, and the no-op HOLD policy so no
rebalance costs are paid), the two agree up to a stated, justified tolerance.

:func:`run_parity_check` runs one replay-mode simulator episode per seed and the
backtester over the **same** episode window, aligns the two per-step series by
decision time, and reports the mean/std of the per-step fee, IL and PnL drift.

Tolerance (ADR-013)
-------------------
The correct comparison is exact-int vs float64, not float-vs-float.  Two
independent error sources bound the drift:

* **Integer raw-unit truncation (binding floor for a marginal agent).**  The
  backtester's final fee step is
  ``fees = (L * (g_inside_now - g_inside_last)) >> 128``: the right shift floors
  the result to a whole raw token unit.  For a token0 fee that is at most ``1``
  raw unit = ``1e-6`` USDC **per accrual** (token1's raw-unit quantum is
  ``1e-18`` WETH, roughly ``3e-15`` USDC at $3000); a decision step that
  aggregates ``N`` tape events is bounded by ``N`` raw units, still far below
  the stated tolerance.  This is the dominant residual in the marginal regime.
* **Float64 round-trip.**  The simulator accumulates the same fee-growth
  quantity as a float64; the conversion loses at most one half-ulp (relative
  ``2**-53``), orders of magnitude below the raw-unit truncation.
* **Marginal-agent fee share (secondary).**  The simulator distributes a step's
  fees over ``L_agent + L_pool_active`` while the backtester's tracker
  distributes over ``L_pool_active`` alone (the synthetic position is marginal
  and is never minted).  The absolute difference is approximately
  ``fee * (L_agent / L_pool_active)**2``.  For a genuinely marginal agent
  (``L_agent << L_pool_active``) this is far below the truncation floor; the
  stated tolerance is justified in that regime, and a non-marginal agent can
  exceed it.

:data:`DEFAULT_PARITY_TOLERANCE` (``1e-3`` USDC/step) clears the ~``1e-6``
USDC/step truncation floor with a wide safety margin while remaining many orders
of magnitude above the float64 round-trip floor.  The chosen value is reported
back in :attr:`ParityReport.tolerance` together with the justification prose,
and the test suite measures the actual drift on a controlled replay scenario.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from undertow.sim.backtest import BacktestLedger, run_backtest
from undertow.sim.backtest.ledger import config_hash
from undertow.sim.config import SimConfig
from undertow.sim.policies import HODLPolicy
from undertow.sim.types import ParityError

if TYPE_CHECKING:
    from datetime import datetime

    from undertow.sim.env.lp_env import LpEnvironment
    from undertow.sim.marketview import MarketView

__all__ = ["DEFAULT_PARITY_TOLERANCE", "ParityReport", "run_parity_check"]

#: Stated tolerance for the mean per-step fee/IL/PnL drift, in USDC.  See the
#: module docstring for the bound: in the marginal-agent regime the binding
#: floor is the backtester's integer ``(L * dg) >> 128`` raw-unit truncation
#: (~1e-6 USDC per accrual), which this value clears with a wide margin; the
#: float64 round-trip is far smaller and the marginal-agent share term is a
#: secondary order-``(L_agent/L_pool)**2`` addition.
DEFAULT_PARITY_TOLERANCE: float = 1e-3

_JUSTIFICATION_TEMPLATE = (
    "Backtester fee accrual is exact-integer Q128 through FeeGrowthReplay "
    "(ADR-013); the simulator accumulates the same fee-growth quantity in "
    "float64. In the marginal regime (L_agent << L_pool_active) the binding "
    "residual is the backtester's final raw-unit truncation: "
    "fees = (L * (g_inside_now - g_inside_last)) >> 128 floors every accrual to "
    "a whole raw token unit, bounding the fee difference by one raw token0 unit "
    "(1e-6 USDC) per accrual plus a negligible token1 term. The float64 "
    "round-trip adds only relative error ~2**-53, far below that. The "
    "marginal-agent fee-share convention is a secondary term of order "
    "fee*(L_agent/L_pool_active)**2: negligible for a genuinely marginal agent, "
    "but it can exceed this tolerance if the agent is not small relative to the "
    "pool. A tolerance of {tolerance:g} USDC/step covers the raw-unit "
    "truncation floor with a wide safety margin in the marginal regime."
)


def _justification(tolerance: float) -> str:
    """Render the tolerance rationale for the value actually used."""
    return _JUSTIFICATION_TEMPLATE.format(tolerance=tolerance)


@dataclass(frozen=True, slots=True)
class ParityReport:
    """Sim-vs-backtester agreement on identical inputs (CONTRACTS §16).

    The six drift fields are the mean and population std of the per-step
    ``sim - backtest`` difference for fees, IL and PnL, in USDC.  ``passed`` is
    derived from the three *mean* drifts and ``tolerance`` and is exposed as a
    real (``init=False``) dataclass field so it is part of the frozen shape.
    """

    fee_drift_mean: float
    fee_drift_std: float
    il_drift_mean: float
    il_drift_std: float
    pnl_drift_mean: float
    pnl_drift_std: float
    tolerance: float
    passed: bool = field(init=False)
    justification: str = ""

    def __post_init__(self) -> None:
        means = (self.fee_drift_mean, self.il_drift_mean, self.pnl_drift_mean)
        within = all(math.isfinite(m) and abs(m) <= self.tolerance for m in means)
        object.__setattr__(self, "passed", bool(within))


def _ledger_window(ledger: BacktestLedger) -> tuple[int, int] | None:
    """Inclusive-exclusive seq window covered by a ledger, or ``None`` if empty."""
    steps = ledger.equity_curve["step"].to_list()
    if not steps:
        return None
    return int(min(steps)), int(max(steps)) + 1


def _backtest_for_window(
    market_view: MarketView,
    window: tuple[int, int],
    config: SimConfig,
    reference_ledger: BacktestLedger,
) -> BacktestLedger:
    """Return the ledger for ``window``, reusing the reference one when it matches.

    The supplied ``backtest_ledger`` is the caller's authoritative backtest for
    the window it covers; when the simulator episode samples exactly that window
    it is used directly (so the report is literally sim-vs-that ledger).  A
    matching window is only reused when the ledger also matches the HOLD policy
    and the simulator's config hash -- a same-window ledger from a different
    policy/config would otherwise yield false parity.  When the episode window
    differs, the backtester is re-run over the identical window with the HOLD
    policy and the same config.
    """
    if _ledger_window(reference_ledger) == window:
        if reference_ledger.policy_name != HODLPolicy().name:
            raise ParityError(
                "run_parity_check: the supplied backtest_ledger is not a HOLD-policy "
                f"run (policy_name={reference_ledger.policy_name!r}); the parity "
                "check requires the no-op dummy policy"
            )
        if reference_ledger.config_hash != config_hash(config):
            raise ParityError(
                "run_parity_check: the supplied backtest_ledger was produced with a "
                "different SimConfig (config_hash mismatch); parity requires the "
                "simulator and backtester to share identical inputs"
            )
        return reference_ledger
    return run_backtest(HODLPolicy(), market_view.slice(window[0], window[1]), config)


def _per_step_backtest(
    ledger: BacktestLedger,
    market_view: MarketView,
    window: tuple[int, int],
    decision_times: Sequence[datetime],
) -> tuple[list[float], list[float], list[float]]:
    """Aggregate a per-event ledger into the simulator's per-decision-step series.

    A simulator decision at time ``T`` observes every tape event with
    ``block_timestamp <= T``.  The backtester's ``fees``/``il`` columns are
    already per-event deltas; its ``equity`` column is cumulative.  Summing the
    deltas and marking the last equity in each ``(T_{k-1}, T_k]`` interval gives
    series directly comparable to the simulator's per-step fee/IL/PnL.
    """
    view = market_view.slice(window[0], window[1])
    tape = view.active_split_data()
    time_by_seq = dict(
        zip(tape["seq"].to_list(), tape["block_timestamp"].to_list(), strict=True)
    )
    curve = ledger.equity_curve
    bt_times = [time_by_seq[int(s)] for s in curve["step"].to_list()]
    bt_fees = [float(v) for v in curve["fees"].to_list()]
    bt_il = [float(v) for v in curve["il"].to_list()]
    bt_equity = [float(v) for v in curve["equity"].to_list()]

    capital = float(ledger.initial_capital) if ledger.initial_capital > 0.0 else (
        bt_equity[0] if bt_equity else 0.0
    )
    fee_cum = 0.0
    il_cum = 0.0
    equity = capital
    prev_fee = 0.0
    prev_il = 0.0
    prev_equity = capital
    idx = 0
    total = len(bt_times)
    out_fees: list[float] = []
    out_il: list[float] = []
    out_pnl: list[float] = []
    for decision_time in decision_times:
        while idx < total and bt_times[idx] <= decision_time:
            fee_cum += bt_fees[idx]
            il_cum += bt_il[idx]
            equity = bt_equity[idx]
            idx += 1
        out_fees.append(fee_cum - prev_fee)
        out_il.append(il_cum - prev_il)
        out_pnl.append(equity - prev_equity)
        prev_fee, prev_il, prev_equity = fee_cum, il_cum, equity
    return out_fees, out_il, out_pnl


def _sim_step_series(
    sim_env: LpEnvironment,
) -> tuple[list[float], list[float], list[float], list[datetime]]:
    """Step a freshly reset HOLD episode, returning per-step fees/IL/PnL/times."""
    episode_length = int(sim_env.episode_length)
    decision_times = list(sim_env.price_process.close_times)[:episode_length]
    sim_fees: list[float] = []
    sim_il: list[float] = []
    sim_pnl: list[float] = []
    equity_prev = float(sim_env.config.episode.agent_capital_usdc)
    for _ in range(episode_length):
        _obs, _reward, terminated, truncated, info = sim_env.step(0)
        raw = info["reward_breakdown_raw"]
        sim_fees.append(float(raw["fees"]))
        sim_il.append(float(raw["il_change"]))
        equity = float(raw["equity"])
        sim_pnl.append(equity - equity_prev)
        equity_prev = equity
        if terminated or truncated:
            break
    return sim_fees, sim_il, sim_pnl, decision_times[: len(sim_fees)]


def run_parity_check(
    sim_env: LpEnvironment,
    backtest_ledger: BacktestLedger,
    num_episodes: int = 10,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    *,
    tolerance: float = DEFAULT_PARITY_TOLERANCE,
) -> ParityReport:
    """Run sim and backtester on identical replay inputs and quantify the drift.

    For each seed a replay-mode simulator episode is run under the no-op HOLD
    policy; the backtester is run over the identical episode window with the same
    config and HOLD policy.  The two per-step series are aligned by decision time
    and the per-step ``sim - backtest`` differences are pooled across every
    episode.  ``passed`` is true when all three mean drifts lie within
    ``tolerance``.

    One episode is run per supplied seed, so the effective episode count is
    ``min(num_episodes, len(seeds))``.  When the supplied ``backtest_ledger``
    covers the exact episode window *and* matches the HOLD policy and the
    simulator's config hash it is used directly; otherwise the backtester is
    re-run over the identical window.  ``tolerance`` is a keyword-only additive
    extension (PLAN.md §0.2); the default is :data:`DEFAULT_PARITY_TOLERANCE`.

    Raises
    ------
    ParityError
        If ``sim_env`` is not in replay mode, the seed list is empty, the
        tolerance is non-positive, a reused ledger has mismatched provenance, or
        no comparable steps could be aligned.
    """
    if not getattr(sim_env, "is_replay", False):
        raise ParityError(
            "run_parity_check requires a replay-mode LpEnvironment so the "
            "simulator and backtester replay the same reference-feed price path"
        )
    if tolerance <= 0.0 or not math.isfinite(tolerance):
        raise ParityError(f"tolerance must be finite and > 0, got {tolerance!r}")

    market_view = sim_env.market_view
    config = sim_env.config
    chosen = tuple(int(s) for s in seeds)[: max(0, int(num_episodes))]
    if not chosen:
        raise ParityError("run_parity_check requires at least one seed")

    fee_drifts: list[float] = []
    il_drifts: list[float] = []
    pnl_drifts: list[float] = []

    for seed in chosen:
        sim_env.reset(seed=seed)
        start_seq = int(sim_env.price_process.start_seq)
        end_seq = int(sim_env.price_process.end_seq)
        window = (start_seq, end_seq)

        sim_fees, sim_il, sim_pnl, decision_times = _sim_step_series(sim_env)
        ledger = _backtest_for_window(
            market_view, window, config, backtest_ledger
        )
        bt_fees, bt_il, bt_pnl = _per_step_backtest(
            ledger, market_view, window, decision_times
        )

        for i in range(min(len(sim_fees), len(bt_fees))):
            fee_drifts.append(sim_fees[i] - bt_fees[i])
            il_drifts.append(sim_il[i] - bt_il[i])
            pnl_drifts.append(sim_pnl[i] - bt_pnl[i])

    if not fee_drifts:
        raise ParityError(
            "run_parity_check produced no comparable steps; check that the "
            "simulator episode window overlaps the backtester's tape"
        )

    return ParityReport(
        fee_drift_mean=float(np.mean(fee_drifts)),
        fee_drift_std=float(np.std(fee_drifts)),
        il_drift_mean=float(np.mean(il_drifts)),
        il_drift_std=float(np.std(il_drifts)),
        pnl_drift_mean=float(np.mean(pnl_drifts)),
        pnl_drift_std=float(np.std(pnl_drifts)),
        tolerance=float(tolerance),
        justification=_justification(float(tolerance)),
    )
