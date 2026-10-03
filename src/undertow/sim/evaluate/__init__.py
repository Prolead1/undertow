"""``undertow.sim.evaluate`` — the RQ1–RQ3 evaluation runner (S15, CONTRACTS §17).

Public surface:

* :class:`AblationResult` / :func:`run_ablations` — RQ1 friction ablation.
* :class:`RegimeMatrix` / :func:`run_regime_evaluation` — RQ2 per-regime matrix.
* :class:`GapReport` / :func:`run_gap_analysis` — RQ3 sim-to-reality gap.
* :func:`write_results` — markdown/CSV artifacts for all three.
* :func:`evaluate_policy` / :func:`evaluate_episodes` — the walk-forward,
  multi-seed protocol shared by the three deliverables (ground truth is the
  backtester; the replay simulator is compared against it).
"""

from __future__ import annotations

from undertow.sim.evaluate.ablation import (
    ABLATION_VARIANTS,
    AblationResult,
    RegimeMatrix,
    regime_matrix_wide,
    run_ablations,
    run_regime_evaluation,
)
from undertow.sim.evaluate.gap import GapReport, run_gap_analysis
from undertow.sim.evaluate.protocol import (
    METRIC_NAMES,
    REGIME_LABELS,
    EpisodeRecord,
    EvaluationError,
    aggregate_metrics,
    build_replay_env,
    dominant_regime,
    episode_metrics,
    evaluate_episodes,
    evaluate_policy,
    observation_from_vector,
    per_regime_frame,
    walk_forward_windows,
)
from undertow.sim.evaluate.report import write_results

__all__ = [
    "ABLATION_VARIANTS",
    "METRIC_NAMES",
    "REGIME_LABELS",
    "AblationResult",
    "EpisodeRecord",
    "EvaluationError",
    "GapReport",
    "RegimeMatrix",
    "aggregate_metrics",
    "build_replay_env",
    "dominant_regime",
    "episode_metrics",
    "evaluate_episodes",
    "evaluate_policy",
    "observation_from_vector",
    "per_regime_frame",
    "regime_matrix_wide",
    "run_ablations",
    "run_gap_analysis",
    "run_regime_evaluation",
    "walk_forward_windows",
    "write_results",
]
