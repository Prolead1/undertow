"""Sim-vs-backtester parity harness (S14, ``CONTRACTS.md`` §16).

The simulator and the backtester live in two different numeric worlds: the
training simulator (S12) accumulates fee growth in **float64**, while the
backtester (S11) accrues fees through ``undertow.data``'s **exact-integer Q128**
``FeeGrowthReplay`` facade (ADR-013).  The thesis's credibility depends on being
able to show that, fed the *identical* replay inputs (same window, same initial
position, same reference price path, same gas and the same policy), the two
agree up to a stated, justified tolerance.

:func:`run_parity_check` runs one replay-mode simulator episode per seed and the
backtester over the **same** episode window, aligns the two per-step series by
decision time, and reports the mean/std (and per-step maximum) of the fee, IL,
PnL, gas and slippage drift.

Decision-time execution (ADR-031, S22)
--------------------------------------
``LpEnvironment.step`` executes an action at the **decision state** it was
chosen from and measures the following interval on the post-action position,
exactly as :class:`~undertow.sim.backtest.engine.BacktestEngine` executes at the
decision event.  The harness enforces that alignment:

* **Executed-action guard.**  The action **executed in each numeric bucket**
  (type, params and executed/rejected) must equal the backtester's decision-log
  entry for that bucket.  The simulator's decision times are reference-bar
  closes and the backtester's are first-event times, so the decision log is
  bucketed with the same ``<=`` rule used for the numeric series and a
  one-to-one mapping is asserted; a divergence raises
  :class:`~undertow.sim.types.ParityError` naming the first divergent step.
  Comparing the observation-time action instead is nearly tautological and was
  the first S22 attempt's core flaw.
* **Per-step pass criterion.**  ``passed`` requires the per-step ``max|drift|``
  of the gas and slippage series to lie within ``tolerance``, not just the
  means: a one-step execution shift moves a gas/slippage spike to another step,
  cancels in the mean, and must still fail.  The mean/std fields (and the
  fee/IL/PnL maxima) stay for reporting.
* **Provenance.**  A supplied ledger is reused only on the ``policy=None`` HOLD
  path (exactly as before S22).  When ``policy is not None`` the backtester is
  **always re-run** over the identical window with that policy: a policy
  ``name`` is not a fingerprint, and a same-named policy with different
  parameters would otherwise produce false parity.

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
* **Model alignment (ADR-019).**  As of ADR-019 the simulator's ``PoolEngine``
  is a float64 replica of the backtester's exact engine: agent positions are
  *marginal* sentinels that never contribute to ``state.liquidity``, fees are
  apportioned piecewise across crossed ticks, and fee growth is distributed
  over the external ``L_pool_active`` alone — the same convention as the
  tracker.  The former order-``(L_agent / L_pool_active)**2`` share-convention
  term therefore no longer exists.

:data:`DEFAULT_PARITY_TOLERANCE` (``1e-3`` USDC/step) clears the ~``1e-6``
USDC/step truncation floor with a wide safety margin while remaining many orders
of magnitude above the float64 round-trip floor.  The chosen value is reported
back in :attr:`ParityReport.tolerance` together with the justification prose,
and the test suite measures the actual drift on a controlled replay scenario.

RNG note
--------
The simulator drives the policy with ``np.random.default_rng(seed)`` while the
backtester uses ``config.rng()``.  For a deterministic policy (HODL,
``TauResetPolicy``) this is irrelevant; a stochastic policy would need both
seeds threaded explicitly so the two worlds sample the same actions.  That is
left to S20 (hedge leg) — the harness already accepts any deterministic
:class:`~undertow.sim.policies.Policy`.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from undertow.sim.backtest import BacktestLedger, run_backtest
from undertow.sim.backtest.ledger import config_hash
from undertow.sim.config import SimConfig
from undertow.sim.evaluate.protocol import action_to_index, observation_from_vector
from undertow.sim.policies import HODLPolicy, Policy
from undertow.sim.prices import ReplayPriceProcess
from undertow.sim.types import ParityError

if TYPE_CHECKING:
    from undertow.sim.env.lp_env import LpEnvironment
    from undertow.sim.marketview import MarketView

__all__ = ["DEFAULT_PARITY_TOLERANCE", "ParityReport", "run_parity_check"]

#: Stated tolerance for the per-step fee/IL/PnL/gas/slippage drift, in USDC.
#: See the module docstring for the bound: in the marginal-agent regime the
#: binding floor is the backtester's integer ``(L * dg) >> 128`` raw-unit
#: truncation (~1e-6 USDC per accrual), which this value clears with a wide
#: margin; the float64 round-trip is far smaller and the marginal-agent share
#: term is a secondary order-``(L_agent/L_pool)**2`` addition.  Gas and
#: slippage use the same tolerance: both worlds call the same friction models
#: with the same inputs, so their expected drift is exactly zero.
DEFAULT_PARITY_TOLERANCE: float = 1e-3

_JUSTIFICATION_TEMPLATE = (
    "Backtester fee accrual is exact-integer Q128 through FeeGrowthReplay "
    "(ADR-013); the simulator replicates the same external-liquidity, "
    "piecewise mechanism in float64 (ADR-019), with a marginal agent position "
    "that never contributes to the external pool liquidity. The binding "
    "residual is the backtester's final raw-unit truncation: "
    "fees = (L * (g_inside_now - g_inside_last)) >> 128 floors every accrual to "
    "a whole raw token unit, bounding the fee difference by one raw token0 unit "
    "(1e-6 USDC) per accrual plus a negligible token1 term. The float64 "
    "round-trip adds only relative error ~2**-53, far below that. A tolerance "
    "of {tolerance:g} USDC/step covers the raw-unit truncation floor with a wide "
    "safety margin. Gas and slippage use the same tolerance: both worlds call "
    "the same friction models with the same inputs, so their expected drift is "
    "zero. The gate checks the per-step max|drift| of the gas and slippage "
    "series as well as every mean, so a one-step execution shift cannot cancel "
    "into a false pass."
)


def _justification(tolerance: float) -> str:
    """Render the tolerance rationale for the value actually used."""
    return _JUSTIFICATION_TEMPLATE.format(tolerance=tolerance)


@dataclass(frozen=True, slots=True)
class ParityReport:
    """Sim-vs-backtester agreement on identical inputs (CONTRACTS §16).

    The drift fields are the mean and population std of the per-step
    ``sim - backtest`` difference for fees, IL, PnL, gas, slippage and the S20
    hedge series (mark PnL, funding, fee), in USDC; the ``*_max`` fields are
    the worst single-step absolute drift of each series.  ``passed`` requires
    every mean **and** the per-step ``max|drift|`` of the spike-prone
    gas/slippage/hedge series to lie within ``tolerance`` (ADR-031/S22: a
    one-step shift cancels in the mean) and is exposed as a real
    (``init=False``) dataclass field so it is part of the frozen shape.  The
    gas/slippage, hedge and max fields are additive, defaulted extensions
    (PLAN.md §0.2) appended after the frozen fields.
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
    # --- S22 additive extensions (defaulted; appended after the frozen fields) ---
    gas_drift_mean: float = 0.0
    gas_drift_std: float = 0.0
    slippage_drift_mean: float = 0.0
    slippage_drift_std: float = 0.0
    #: Worst single-step ``|sim - backtest|`` for each series (ADR-031/S22).
    fee_drift_max: float = 0.0
    il_drift_max: float = 0.0
    pnl_drift_max: float = 0.0
    gas_drift_max: float = 0.0
    slippage_drift_max: float = 0.0
    # --- S20 hedge additive extensions (defaulted; appended last) ---
    #: Mean/std of the per-step hedge mark PnL drift (USDC).
    hedge_pnl_drift_mean: float = 0.0
    hedge_pnl_drift_std: float = 0.0
    #: Mean/std of the per-step hedge funding drift (USDC).
    hedge_funding_drift_mean: float = 0.0
    hedge_funding_drift_std: float = 0.0
    #: Mean/std of the per-step hedge fee drift (USDC, signed ``<= 0``).
    hedge_fee_drift_mean: float = 0.0
    hedge_fee_drift_std: float = 0.0
    #: Worst single-step ``|sim - backtest|`` for each hedge series.
    hedge_pnl_drift_max: float = 0.0
    hedge_funding_drift_max: float = 0.0
    hedge_fee_drift_max: float = 0.0

    def __post_init__(self) -> None:
        means = (
            self.fee_drift_mean,
            self.il_drift_mean,
            self.pnl_drift_mean,
            self.gas_drift_mean,
            self.slippage_drift_mean,
            self.hedge_pnl_drift_mean,
            self.hedge_funding_drift_mean,
            self.hedge_fee_drift_mean,
        )
        # The per-step criterion is on the spike-prone series: a one-step
        # execution shift moves a gas/slippage/funding spike and cancels in the
        # means.
        maxima = (
            self.gas_drift_max,
            self.slippage_drift_max,
            self.hedge_pnl_drift_max,
            self.hedge_funding_drift_max,
            self.hedge_fee_drift_max,
        )
        within = all(math.isfinite(m) and abs(m) <= self.tolerance for m in means)
        within = within and all(
            math.isfinite(m) and abs(m) <= self.tolerance for m in maxima
        )
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
    active_policy: Policy,
    *,
    reuse_ledger: bool,
) -> BacktestLedger:
    """Return the ledger for ``window``, reusing the supplied one only for HOLD.

    The supplied ``backtest_ledger`` is the caller's authoritative backtest for
    the window it covers; on the ``policy=None`` HOLD path (``reuse_ledger``) a
    matching window is used directly — so the report is literally sim-vs-that
    ledger — after checking it is a HOLD run with the simulator's config hash.
    A same-window ledger from a different policy/config would otherwise yield
    false parity.  When ``policy is not None`` the backtester is **always
    re-run** over the identical window with that policy: the supplied ledger is
    ignored because a policy ``name`` alone is not a fingerprint and same-named
    policies with different parameters would fake a pass (S22 review).
    """
    if reuse_ledger and _ledger_window(reference_ledger) == window:
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
    return run_backtest(
        active_policy, market_view.slice(window[0], window[1]), config
    )


@dataclass(frozen=True, slots=True)
class _Decision:
    """One simulator decision, for the cross-world executed-action guard.

    ``time`` is the timestamp of the **decision state** the policy acted on and
    the action executed at (ADR-031) — the observation boundary, not the
    end-of-step timestamp the numeric series are bucketed on.
    """

    time: datetime
    action_type: str
    center_offset: int
    width: int


def _bucket_backtest_decisions(
    ledger: BacktestLedger,
    market_view: MarketView,
    window: tuple[int, int],
    decisions: Sequence[_Decision],
) -> list[dict[str, Any] | None]:
    """Map each backtester decision to the simulator's step bucket (one-to-one).

    The simulator executes the action chosen at decision time ``D_k`` in step
    ``k``; the backtester logs the decision it executed at the first event at or
    after its own previous decision.  Both are bucketed with the ``<=`` rule:
    bucket ``k`` covers ``(D_{k-1}, D_k]`` (the simulator's decision time is
    the bucket's right edge), so a backtester decision at a first-event time
    slightly before a reference-bar close still lands in the same bucket as the
    simulator step that executed it.  Backtester decisions after the simulator
    episode's last decision are expected (the backtester replays the full
    window) and ignored.  A decision-log step outside the window, a duplicate
    decision inside one bucket, or a bucket with no backtester counterpart
    raises :class:`ParityError`.
    """
    view = market_view.slice(window[0], window[1])
    tape = view.active_split_data()
    time_by_seq = dict(
        zip(tape["seq"].to_list(), tape["block_timestamp"].to_list(), strict=True)
    )
    decision_times = [decision.time for decision in decisions]
    buckets: list[dict[str, Any] | None] = [None] * len(decisions)
    for row in ledger.decision_log.iter_rows(named=True):
        seq = int(row["step"])
        if seq not in time_by_seq:
            raise ParityError(
                "run_parity_check: the backtester's decision log references tape "
                f"step {seq} outside the episode window {window}"
            )
        at_time = time_by_seq[seq]
        bucket = bisect_left(decision_times, at_time)
        if bucket >= len(decisions):
            # Extra backtester decision after the simulator's last decision.
            continue
        if buckets[bucket] is not None:
            raise ParityError(
                "run_parity_check: the backtester logged more than one executed "
                f"decision in bucket {bucket} ({decision_times[bucket]}); the "
                "decision sequence cannot be aligned one-to-one"
            )
        buckets[bucket] = row
    return buckets


def _assert_decisions_match(
    decisions: Sequence[_Decision],
    decision_rows: Sequence[dict[str, Any] | None],
) -> None:
    """Assert the executed action in each numeric bucket is identical.

    A numeric comparison across different decisions is meaningless, so this is a
    hard guard, not a warning: the first bucket whose executed action (type,
    center offset, width, executed/rejected) differs — or has no backtester
    counterpart — raises :class:`~undertow.sim.types.ParityError` naming the
    step.  This compares the action each world **executed**, not the
    observation-time action, which is what let the first S22 attempt pass while
    the simulator executed one bar late.
    """
    for step, decision in enumerate(decisions):
        backtest_row = decision_rows[step]
        if backtest_row is None:
            raise ParityError(
                "run_parity_check: no backtester decision in sim bucket "
                f"{step} ({decision.time}); the two worlds have different "
                "decision cadences and cannot be compared"
            )
        backtest = (
            str(backtest_row["action_type"]),
            int(backtest_row["center_offset"]),
            int(backtest_row["width"]),
        )
        simulated = (decision.action_type, decision.center_offset, decision.width)
        if backtest != simulated:
            raise ParityError(
                "run_parity_check: executed-action sequence diverges at sim bucket "
                f"{step} ({decision.time}): simulator executed {simulated!r} but "
                f"the backtester executed {backtest!r}"
            )


def _sim_step_series(
    sim_env: LpEnvironment,
    policy: Policy,
    seed: int,
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[datetime],
    list[_Decision],
]:
    """Reset and drive one episode with ``policy``, returning the per-step series.

    The policy is reset, then each decision converts the environment's returned
    vector back to an :class:`~undertow.sim.env.observations.Observation` with
    :func:`~undertow.sim.evaluate.protocol.observation_from_vector`, calls
    ``policy.act`` and maps the :class:`~undertow.sim.types.Action` back to the
    environment's discrete index.  Returns ``(fees, il, pnl, gas, slippage,
    hedge_pnl, hedge_funding, hedge_fee, step_end_times, decisions)``: the
    numeric series are the per-step **ungated economics** terms in USDC
    (``info["reward_breakdown_economics"]``; the gated
    ``reward_breakdown_raw`` would read ``0`` for an ablated friction while the
    backtester still records its full magnitude -- S20) with the hedge terms
    zero when the overlay is disabled, ``step_end_times`` the bar-close
    timestamps the numeric series are bucketed on, and ``decisions`` the
    executed-action records (at the decision state each action was executed at)
    the cross-world guard compares against the backtester.
    """
    policy.reset()
    rng = np.random.default_rng(int(seed))
    vector, info = sim_env.reset(seed=int(seed))
    decision_time = cast(datetime, info["timestamp"])

    episode_length = int(sim_env.episode_length)
    replay_process = cast(ReplayPriceProcess, sim_env.price_process)
    step_end_times = list(replay_process.close_times)[:episode_length]

    sim_fees: list[float] = []
    sim_il: list[float] = []
    sim_pnl: list[float] = []
    sim_gas: list[float] = []
    sim_slippage: list[float] = []
    sim_hedge_pnl: list[float] = []
    sim_hedge_funding: list[float] = []
    sim_hedge_fee: list[float] = []
    decisions: list[_Decision] = []
    equity_prev = float(sim_env.config.episode.agent_capital_usdc)
    for _ in range(episode_length):
        observation = observation_from_vector(vector.tolist())
        action = policy.act(observation, rng)
        action_index = action_to_index(sim_env, action)
        vector, _reward, terminated, truncated, info = sim_env.step(action_index)
        # Ungated economics (S20): an RQ1 ablation zeroes the term in the gated
        # ``reward_breakdown_raw`` while the backtester's series is unchanged.
        economics = info["reward_breakdown_economics"]
        sim_fees.append(float(economics["fees"]))
        sim_il.append(float(economics["il_change"]))
        equity = float(economics["equity"])
        sim_pnl.append(equity - equity_prev)
        equity_prev = equity
        sim_gas.append(float(economics["gas"]))
        sim_slippage.append(float(economics["slippage"]))
        sim_hedge_pnl.append(float(economics["hedge_pnl"]))
        sim_hedge_funding.append(float(economics["hedge_funding"]))
        sim_hedge_fee.append(float(economics["hedge_fee"]))
        applied = info["action"]
        decisions.append(
            _Decision(
                time=decision_time,
                action_type=str(info["action_type"]),
                center_offset=int(applied.center_offset),
                width=int(applied.width),
            )
        )
        decision_time = cast(datetime, info["timestamp"])
        if terminated or truncated:
            break
    return (
        sim_fees,
        sim_il,
        sim_pnl,
        sim_gas,
        sim_slippage,
        sim_hedge_pnl,
        sim_hedge_funding,
        sim_hedge_fee,
        step_end_times[: len(sim_fees)],
        decisions,
    )


def _per_step_backtest(
    ledger: BacktestLedger,
    market_view: MarketView,
    window: tuple[int, int],
    decision_times: Sequence[datetime],
    decision_rows: Sequence[dict[str, Any] | None],
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    """Aggregate a per-event ledger into the simulator's per-step series.

    Step ``k`` is the simulator's step ``k``: the action executed at decision
    time ``D_k`` followed by the tape events in ``(D_k, D_{k+1}]``.  The event
    deltas (fees and IL) are bucketed with the existing ``<=`` rule, so bucket
    ``k`` sums the events up to ``decision_times[k]``.  Gas and slippage are the
    **executed decision's** own costs from ``decision_rows[k]``: the simulator
    pays them at the interval's start while the backtester records them on the
    decision row at the interval's end, so reading them off the event delta
    columns would attribute the cost to the previous step (the S22 review's
    one-step shift).  PnL is the pre-decision equity change: the backtester's
    equity on a decision row is post-decision, so the decision's costs are
    added back to recover the mark the simulator observes at the interval end.
    """
    view = market_view.slice(window[0], window[1])
    tape = view.active_split_data()
    time_by_seq = dict(
        zip(tape["seq"].to_list(), tape["block_timestamp"].to_list(), strict=True)
    )
    curve = ledger.equity_curve
    curve_seqs = [int(s) for s in curve["step"].to_list()]
    bt_times = [time_by_seq[s] for s in curve_seqs]
    bt_fees = [float(v) for v in curve["fees"].to_list()]
    bt_il = [float(v) for v in curve["il"].to_list()]
    bt_equity = [float(v) for v in curve["equity"].to_list()]
    # The cost a decision row carries (always 0 for a hold/rejected decision).
    cost_by_seq: dict[int, float] = {}
    for row in ledger.decision_log.iter_rows(named=True):
        seq = int(row["step"])
        cost_by_seq[seq] = float(row["gas_paid"]) + float(row["slippage_paid"])
    bt_costs = [cost_by_seq.get(seq, 0.0) for seq in curve_seqs]

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
    out_gas: list[float] = []
    out_slippage: list[float] = []
    for step, decision_time in enumerate(decision_times):
        while idx < total and bt_times[idx] <= decision_time:
            fee_cum += bt_fees[idx]
            il_cum += bt_il[idx]
            # Undo the decision's own costs: the simulator's end-of-step mark is
            # pre-decision, so the backtester's post-decision equity overstates
            # it.
            equity = bt_equity[idx] - bt_costs[idx]
            idx += 1
        out_fees.append(fee_cum - prev_fee)
        out_il.append(il_cum - prev_il)
        out_pnl.append(equity - prev_equity)
        # The costs of the decision this step executed (the bucket's action).
        decision_row = decision_rows[step] if step < len(decision_rows) else None
        if decision_row is None:
            out_gas.append(0.0)
            out_slippage.append(0.0)
        else:
            out_gas.append(float(decision_row["gas_paid"]))
            out_slippage.append(float(decision_row["slippage_paid"]))
        prev_fee, prev_il = fee_cum, il_cum
        prev_equity = equity
    return out_fees, out_il, out_pnl, out_gas, out_slippage


def _per_step_hedge(
    ledger: BacktestLedger,
    market_view: MarketView,
    window: tuple[int, int],
    decision_times: Sequence[datetime],
    decision_rows: Sequence[dict[str, Any] | None],
) -> tuple[list[float], list[float], list[float]]:
    """Aggregate the ledger's hedge series into the sim's per-step buckets.

    Mark PnL and funding are interval accruals, bucketed with the same ``<=``
    rule as the fee/IL series, so bucket ``k`` holds the contribution over the
    interval ending at the simulator's step-end time ``D_k`` — exactly the
    interval the simulator measured (its short was established at the
    interval's start).  The hedge **fee** is a decision cost, like gas and
    slippage: the simulator pays it at the decision state that opens bucket
    ``k``, while the backtester logs it on the decision row that
    ``_bucket_backtest_decisions`` maps to bucket ``k``, so it is read from
    there.  A ledger without a hedge curve (the overlay disabled) yields zeros,
    which is what the simulator's gated breakdown reports too.
    """
    steps = len(decision_times)
    curve = ledger.hedge_curve
    if curve is None or curve.height == 0:
        return [0.0] * steps, [0.0] * steps, [0.0] * steps

    view = market_view.slice(window[0], window[1])
    tape = view.active_split_data()
    time_by_seq = dict(
        zip(tape["seq"].to_list(), tape["block_timestamp"].to_list(), strict=True)
    )
    rows = sorted(
        (time_by_seq[int(seq)], float(pnl), float(funding))
        for seq, pnl, funding in zip(
            curve["step"].to_list(),
            curve["hedge_pnl"].to_list(),
            curve["hedge_funding"].to_list(),
            strict=True,
        )
    )

    out_pnl: list[float] = []
    out_funding: list[float] = []
    out_fee: list[float] = []
    idx = 0
    total = len(rows)
    pnl_cum = funding_cum = 0.0
    prev_pnl = prev_funding = 0.0
    for step, decision_time in enumerate(decision_times):
        while idx < total and rows[idx][0] <= decision_time:
            pnl_cum += rows[idx][1]
            funding_cum += rows[idx][2]
            idx += 1
        out_pnl.append(pnl_cum - prev_pnl)
        out_funding.append(funding_cum - prev_funding)
        row = decision_rows[step] if step < len(decision_rows) else None
        out_fee.append(float(row.get("hedge_fee", 0.0)) if row is not None else 0.0)
        prev_pnl, prev_funding = pnl_cum, funding_cum
    return out_pnl, out_funding, out_fee


def run_parity_check(
    sim_env: LpEnvironment,
    backtest_ledger: BacktestLedger,
    num_episodes: int = 10,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    *,
    policy: Policy | None = None,
    tolerance: float = DEFAULT_PARITY_TOLERANCE,
) -> ParityReport:
    """Run sim and backtester on identical replay inputs and quantify the drift.

    For each seed a replay-mode simulator episode is run under ``policy`` (the
    no-op HOLD policy when ``policy is None``); the backtester is run over the
    identical episode window with the same config and policy.  The two per-step
    series are aligned by decision time and the per-step ``sim - backtest``
    differences are pooled across every episode (including the S20 hedge mark
    PnL, funding and fee series when the overlay is enabled).  ``passed`` is
    true when every mean drift lies within ``tolerance`` and the per-step
    ``max|drift|`` of the spike-prone gas/slippage/hedge series does too; the
    fee/IL/PnL maxima are reported alongside (a one-step execution shift must
    fail, not cancel into a false pass).

    Before the numeric comparison the action **executed in each numeric bucket**
    is checked to be identical across the two worlds -- same action type and
    parameters at every simulator step, including executed/rejected rebalances
    -- and a divergence raises :class:`ParityError` naming the first divergent
    step.  One episode is run per supplied seed, so the effective episode count
    is ``min(num_episodes, len(seeds))``.  When ``policy is None`` and the
    supplied ``backtest_ledger`` covers the exact episode window, is a HOLD run
    and matches the simulator's config hash it is used directly; otherwise — and
    always when ``policy is not None`` — the backtester is re-run over the
    identical window.  ``policy`` and ``tolerance`` are keyword-only additive
    extensions (PLAN.md §0.2); the defaults are the no-op HOLD policy and
    :data:`DEFAULT_PARITY_TOLERANCE`.

    A stochastic policy would need the simulator's ``default_rng(seed)`` and the
    backtester's ``config.rng()`` seeds threaded together (see the module RNG
    note); this harness assumes a deterministic policy.

    Raises
    ------
    ParityError
        If ``sim_env`` is not in replay mode, the seed list is empty, the
        tolerance is non-positive, a reused ledger has mismatched provenance,
        the two worlds' executed decisions diverge, or no comparable steps
        could be aligned.
    """
    active_policy: Policy = HODLPolicy() if policy is None else policy
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
    gas_drifts: list[float] = []
    slippage_drifts: list[float] = []
    hedge_pnl_drifts: list[float] = []
    hedge_funding_drifts: list[float] = []
    hedge_fee_drifts: list[float] = []

    for seed in chosen:
        (
            sim_fees,
            sim_il,
            sim_pnl,
            sim_gas,
            sim_slippage,
            sim_hedge_pnl,
            sim_hedge_funding,
            sim_hedge_fee,
            decision_times,
            decisions,
        ) = _sim_step_series(sim_env, active_policy, seed)
        replay_process = cast(ReplayPriceProcess, sim_env.price_process)
        window = (int(replay_process.start_seq), int(replay_process.end_seq))

        ledger = _backtest_for_window(
            market_view,
            window,
            config,
            backtest_ledger,
            active_policy,
            reuse_ledger=policy is None,
        )
        decision_rows = _bucket_backtest_decisions(
            ledger, market_view, window, decisions
        )
        _assert_decisions_match(decisions, decision_rows)
        bt_fees, bt_il, bt_pnl, bt_gas, bt_slippage = _per_step_backtest(
            ledger, market_view, window, decision_times, decision_rows
        )
        bt_hedge_pnl, bt_hedge_funding, bt_hedge_fee = _per_step_hedge(
            ledger, market_view, window, decision_times, decision_rows
        )

        for i in range(min(len(sim_fees), len(bt_fees))):
            fee_drifts.append(sim_fees[i] - bt_fees[i])
            il_drifts.append(sim_il[i] - bt_il[i])
            pnl_drifts.append(sim_pnl[i] - bt_pnl[i])
            gas_drifts.append(sim_gas[i] - bt_gas[i])
            slippage_drifts.append(sim_slippage[i] - bt_slippage[i])
            hedge_pnl_drifts.append(sim_hedge_pnl[i] - bt_hedge_pnl[i])
            hedge_funding_drifts.append(
                sim_hedge_funding[i] - bt_hedge_funding[i]
            )
            hedge_fee_drifts.append(sim_hedge_fee[i] - bt_hedge_fee[i])

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
        gas_drift_mean=float(np.mean(gas_drifts)),
        gas_drift_std=float(np.std(gas_drifts)),
        slippage_drift_mean=float(np.mean(slippage_drifts)),
        slippage_drift_std=float(np.std(slippage_drifts)),
        fee_drift_max=float(np.max(np.abs(fee_drifts))),
        il_drift_max=float(np.max(np.abs(il_drifts))),
        pnl_drift_max=float(np.max(np.abs(pnl_drifts))),
        gas_drift_max=float(np.max(np.abs(gas_drifts))),
        slippage_drift_max=float(np.max(np.abs(slippage_drifts))),
        hedge_pnl_drift_mean=float(np.mean(hedge_pnl_drifts)),
        hedge_pnl_drift_std=float(np.std(hedge_pnl_drifts)),
        hedge_funding_drift_mean=float(np.mean(hedge_funding_drifts)),
        hedge_funding_drift_std=float(np.std(hedge_funding_drifts)),
        hedge_fee_drift_mean=float(np.mean(hedge_fee_drifts)),
        hedge_fee_drift_std=float(np.std(hedge_fee_drifts)),
        hedge_pnl_drift_max=float(np.max(np.abs(hedge_pnl_drifts))),
        hedge_funding_drift_max=float(np.max(np.abs(hedge_funding_drifts))),
        hedge_fee_drift_max=float(np.max(np.abs(hedge_fee_drifts))),
    )
