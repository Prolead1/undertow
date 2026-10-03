"""RQ1 friction-ablation matrix (S15, CONTRACTS.md §17).

The ablation table answers "what is the contribution of each friction term to
performance?" for the five variants pinned by the brief:

======================  =====  ====  ========  ====
Ablation                Fees   Gas   Slippage  IL
======================  =====  ====  ========  ====
``full``                on     on    on        on
``no_gas``              on     off   on        on
``no_slippage``         on     on    off       on
``no_il``               on     on    on        off
``static_fee``          on     flat  on        on
======================  =====  ====  ========  ====

How the ablation is realised
----------------------------
For deterministic baseline policies the ablation is a **measurement** change,
not a training change: the evaluation layer rebuilds the per-step equity curve
with the ablated cost terms zeroed (and, for ``static_fee``, re-prices every
rebalance at a flat gas cost) using the backtester's exact cost ledger.  Because
the backtester's equity change reconciles step-by-step with
``fees + il + gas + slippage + ΔHODL``, the reconstruction is exact: the
untouched components and the HODL mark are preserved, and only the gated terms
change.  The variant's four flags are stored on the result and gate the ledger at
this layer; they are the same flags the simulator's :func:`compute_reward` honors
(:class:`~undertow.sim.config.RewardConfig`, S09), and for the RL branch they are
also written into the variant config handed to ``train_fn``.  The baseline branch
can share one backtest pass because the backtester does not read ``RewardConfig``
— it records the gross cost terms, and the flags decide which are subtracted.

For an RL policy the reward function *is* the training objective, so each
variant needs its own training run.  That is out of scope for the test suite and
is never done silently: an optional ``train_fn`` hook is invoked per variant to
produce the frozen policy to evaluate, and when it is absent and the policy is
not a known deterministic baseline, :func:`run_ablations` raises
:class:`~undertow.sim.evaluate.protocol.EvaluationError`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from undertow.sim.config import SimConfig
from undertow.sim.evaluate.protocol import (
    REGIME_LABELS,
    EpisodeRecord,
    EvaluationError,
    episode_metrics,
    evaluate_episodes,
    per_regime_frame,
)
from undertow.sim.frictions import FlatGasModel
from undertow.sim.policies import (
    CostAwareRebalancePolicy,
    FullRangeV2Policy,
    HODLPolicy,
    PassiveNarrowPolicy,
    PassiveWidePolicy,
    TauResetPolicy,
)

if TYPE_CHECKING:
    from undertow.sim.env.lp_env import LpEnvironment
    from undertow.sim.marketview import MarketView
    from undertow.sim.policies import Policy

__all__ = [
    "ABLATION_VARIANTS",
    "AblationResult",
    "RegimeMatrix",
    "regime_matrix_wide",
    "run_ablations",
    "run_regime_evaluation",
]

#: The five RQ1 variants: ``(name, flags, static_gas)``.  The four flags are
#: exactly the brief's table; ``static_fee`` differs from ``full`` only by the
#: flat gas model (``static_gas=True``), which the variant name also records.
ABLATION_VARIANTS: tuple[tuple[str, dict[str, bool], bool], ...] = (
    ("full", {"fees": True, "gas": True, "slippage": True, "il": True}, False),
    ("no_gas", {"fees": True, "gas": False, "slippage": True, "il": True}, False),
    (
        "no_slippage",
        {"fees": True, "gas": True, "slippage": False, "il": True},
        False,
    ),
    ("no_il", {"fees": True, "gas": True, "slippage": True, "il": False}, False),
    ("static_fee", {"fees": True, "gas": True, "slippage": True, "il": True}, True),
)

#: The known deterministic baselines (CONTRACTS §10).  A policy of one of these
#: types is evaluated directly under each variant; anything else is an RL policy
#: and requires the caller's ``train_fn``.
_BASELINE_TYPES: tuple[type, ...] = (
    HODLPolicy,
    PassiveNarrowPolicy,
    PassiveWidePolicy,
    FullRangeV2Policy,
    TauResetPolicy,
    CostAwareRebalancePolicy,
)

#: Flat gas re-pricing used by ``static_fee``: the S08 defaults (30 gwei base
#: fee, 3 % tip surcharge, $3000 ETH).  Tests sensitivity to gas *volatility*
#: without changing the gas units.
_STATIC_GAS_BASE_FEE_GWEI: float = 30.0
_STATIC_GAS_ETH_PRICE: float = 3000.0

#: Per-variant training hook: ``(name, config, market_view, checkpoint_dir) ->
#: Policy``.  The hook must return a *frozen* policy for the variant's
#: ``RewardConfig``; the evaluation runner never trains.
TrainFn = Callable[["str", SimConfig, "MarketView", "Path | None"], "Policy"]


@dataclass(frozen=True, slots=True)
class AblationResult:
    """RQ1: one row of the ablation table (CONTRACTS §17)."""

    ablation_name: str  # "full", "no_gas", "no_slippage", "no_il", "static_fee"
    flags: dict[str, bool]
    metrics: dict[str, float]  # mean over episodes/seeds
    per_regime: pl.DataFrame | None = None


def _is_baseline(policy: Policy) -> bool:
    """True when ``policy`` is a deterministic baseline (no training needed).

    Only the six concrete baseline types qualify.  A duck-typed ``is_baseline``
    escape hatch was deliberately rejected: it would let a non-baseline policy
    skip the per-variant training the RL path requires.
    """
    return isinstance(policy, _BASELINE_TYPES)


def _variant_config(config: SimConfig, flags: dict[str, bool]) -> SimConfig:
    """The config whose ``RewardConfig`` carries the variant's gating flags."""
    reward = replace(
        config.reward,
        fees_enabled=bool(flags["fees"]),
        gas_enabled=bool(flags["gas"]),
        slippage_enabled=bool(flags["slippage"]),
        il_enabled=bool(flags["il"]),
    )
    return replace(config, reward=reward)


def _flat_rebalance_cost(config: SimConfig) -> float:
    """Flat USDC gas cost of one rebalance under the ``static_fee`` model."""
    model = FlatGasModel(
        config.gas,
        eth_usd_price=_STATIC_GAS_ETH_PRICE,
        base_fee_gwei=_STATIC_GAS_BASE_FEE_GWEI,
    )
    return float(model.gas_cost_usdc("rebalance", 0, 0, 0, _STATIC_GAS_ETH_PRICE))


def _gated_records(
    records: Sequence[EpisodeRecord],
    flags: dict[str, bool],
    *,
    static_gas: bool,
    static_gas_cost: float,
) -> list[EpisodeRecord]:
    """Rebuild each backtest episode with the variant's cost terms gated.

    The backtester equity change reconciles with
    ``fees + il + gas + slippage + ΔHODL``; the HODL mark and the untouched
    terms are recovered as the residual, so only the gated terms change.  A
    ``static_fee`` record replaces each rebalance's actual gas with the flat
    per-rebalance cost.
    """
    gated: list[EpisodeRecord] = []
    for record in records:
        curve = record.equity_curve
        missing = [
            column
            for column in ("equity", "fees", "gas", "slippage", "il")
            if column not in curve.columns
        ]
        if missing:
            raise EvaluationError(
                "run_ablations requires a backtest equity curve with columns "
                f"{('equity', 'fees', 'gas', 'slippage', 'il')}; missing {missing}"
            )
        fees = curve["fees"].to_numpy().astype(float)
        gas = curve["gas"].to_numpy().astype(float)
        slippage = curve["slippage"].to_numpy().astype(float)
        il = curve["il"].to_numpy().astype(float)
        equity = curve["equity"].to_numpy().astype(float)

        full = fees + gas + slippage + il
        residual = equity - record.initial_capital - np.cumsum(full)

        gated_fees = fees if flags["fees"] else np.zeros_like(fees)
        gated_gas = gas if flags["gas"] else np.zeros_like(gas)
        if static_gas:
            gated_gas = np.where(gas != 0.0, -float(static_gas_cost), 0.0)
        gated_slippage = slippage if flags["slippage"] else np.zeros_like(slippage)
        gated_il = il if flags["il"] else np.zeros_like(il)
        gated_sum = gated_fees + gated_gas + gated_slippage + gated_il
        ablated_equity = record.initial_capital + residual + np.cumsum(gated_sum)

        decomposition = {
            "total_fees": float(gated_fees.sum()),
            "total_gas": float(gated_gas.sum()),
            "total_slippage": float(gated_slippage.sum()),
            "total_il": float(gated_il.sum()),
            "net": float(gated_sum.sum()),
        }
        equity_series = pl.Series("equity", ablated_equity)
        metrics = episode_metrics(
            equity_series,
            record.initial_capital,
            decomposition,
            record.periods_per_year,
            int(record.metrics.get("n_rebalances", 0)),
        )
        gated.append(
            replace(
                record,
                metrics=metrics,
                decomposition=decomposition,
                equity_curve=pl.DataFrame({"equity": ablated_equity}),
            )
        )
    return gated


def _aggregate(records: Sequence[EpisodeRecord], metric_names: Sequence[str]) -> dict[str, float]:
    """Mean of each metric across the gated episodes."""
    means: dict[str, float] = {}
    for name in metric_names:
        values = [float(record.metrics[name]) for record in records if name in record.metrics]
        means[name] = float(np.mean(values)) if values else 0.0
    return means


def run_ablations(
    config: SimConfig,
    policy: Policy,
    market_view: MarketView,
    *,
    train_fn: TrainFn | None = None,
    checkpoint_dir: Path | None = None,
    num_seeds: int | None = None,
) -> list[AblationResult]:
    """Run the RQ1 ablation table (CONTRACTS §17).

    Deterministic baselines are evaluated under each variant; an RL policy must
    supply ``train_fn`` so each variant gets its own training run (never done
    silently).  The returned list is in :data:`ABLATION_VARIANTS` order.
    """
    baseline = _is_baseline(policy)
    if not baseline and train_fn is None:
        raise EvaluationError(
            f"run_ablations: policy {policy.name!r} is not a deterministic baseline; "
            "each RQ1 variant changes the reward function, so an RL policy must be "
            "retrained per variant. Pass train_fn=(name, config, market_view, "
            "checkpoint_dir) -> Policy; run_ablations will not train silently."
        )

    # For a deterministic baseline the flags only change *what is measured*: the
    # backtester's ledger is identical across variants, so one pass is shared and
    # each variant re-gates it.  An RL policy changes across variants (different
    # reward), so it is trained and evaluated once per variant.
    shared_records: list[EpisodeRecord] | None = None
    if train_fn is None:
        shared_records = evaluate_episodes(
            policy, market_view, config, num_seeds=num_seeds, backend="backtest"
        )

    results: list[AblationResult] = []
    for name, flags, static_gas in ABLATION_VARIANTS:
        variant_config = _variant_config(config, flags)
        if shared_records is not None:
            records = shared_records
        else:
            assert train_fn is not None  # guaranteed by the guard above
            variant_policy = train_fn(name, variant_config, market_view, checkpoint_dir)
            records = evaluate_episodes(
                variant_policy,
                market_view,
                variant_config,
                num_seeds=num_seeds,
                backend="backtest",
            )
        gated = _gated_records(
            records,
            flags,
            static_gas=static_gas,
            static_gas_cost=_flat_rebalance_cost(variant_config),
        )
        results.append(
            AblationResult(
                ablation_name=name,
                flags=dict(flags),
                metrics=_aggregate(gated, list(gated[0].metrics) if gated else []),
                per_regime=per_regime_frame(gated),
            )
        )
    return results


@dataclass(frozen=True, slots=True)
class RegimeMatrix:
    """RQ2: per-regime results for one policy (CONTRACTS §17).

    ``metrics_per_regime`` is the long-format frame produced by
    :func:`~undertow.sim.evaluate.protocol.per_regime_frame`:
    ``regime, metric, mean, std, n``.  ``regimes`` always contains the four
    canonical labels first, followed by any non-canonical label observed.
    """

    policy_name: str
    regimes: list[str]
    metrics_per_regime: pl.DataFrame


def regime_matrix_wide(matrix: RegimeMatrix) -> pl.DataFrame:
    """Pivot a long-format :class:`RegimeMatrix` to the brief's wide shape.

    One row per regime and two columns per metric (``<metric>_mean``,
    ``<metric>_std``), matching the S15 brief's "rows = regimes, columns =
    metrics + mean + std".  ``RegimeMatrix.metrics_per_regime`` keeps the
    canonical long format (``regime, metric, mean, std, n``) so downstream
    consumers can pick either view.
    """
    frame = matrix.metrics_per_regime
    lookup = {
        (str(row["regime"]), str(row["metric"])): (
            float(row["mean"]),
            float(row["std"]),
        )
        for row in frame.iter_rows(named=True)
    }
    metrics = list(dict.fromkeys(str(value) for value in frame["metric"].to_list()))
    rows: list[dict[str, object]] = []
    for regime in matrix.regimes:
        row: dict[str, object] = {"regime": regime}
        for metric in metrics:
            mean, std = lookup.get((regime, metric), (0.0, 0.0))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        rows.append(row)
    return pl.DataFrame(rows)


def run_regime_evaluation(
    config: SimConfig,
    policies: dict[str, Policy],
    market_view: MarketView,
    *,
    num_seeds: int | None = None,
    backend: str = "backtest",
    env_factory: Callable[[MarketView, SimConfig], LpEnvironment] | None = None,
) -> list[RegimeMatrix]:
    """Run the RQ2 regime matrix: every policy × regime × metrics.

    Each policy is evaluated over the walk-forward episodes; every episode is
    tagged by its dominant regime via ``market_view.regime_at`` and grouped, and
    per-regime mean/std are computed over episodes and seeds.  Returns one
    :class:`RegimeMatrix` per policy in the iteration order of ``policies``.
    """
    results: list[RegimeMatrix] = []
    for name, policy in policies.items():
        records = evaluate_episodes(
            policy,
            market_view,
            config,
            num_seeds=num_seeds,
            backend=backend,
            env_factory=env_factory,
        )
        frame = per_regime_frame(records)
        observed = [
            str(label)
            for label in frame["regime"].unique(maintain_order=True).to_list()
            if label not in REGIME_LABELS
        ]
        results.append(
            RegimeMatrix(
                policy_name=name,
                regimes=list(REGIME_LABELS) + observed,
                metrics_per_regime=frame,
            )
        )
    return results
