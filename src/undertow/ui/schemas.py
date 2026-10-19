"""Pydantic v2 wire models for ``undertow.ui`` (CONTRACTS.md §4).

These models are the OpenAPI source and the generated-TypeScript source for the
dashboard.  They are **mirrored**, not imported, from ``undertow.sim``: the UI is
an artifact-only consumer (`docs/plans/ui/PLAN.md` §0.6), so importing the
training stack (and its ``stable-baselines3`` dependency) here is forbidden.
:class:`RunStatusModel` therefore mirrors ``undertow.sim.train.telemetry.RunStatus``
field-for-field and is parsed directly from ``status.json``.

Forward compatibility is handled per contract: telemetry metrics allow unknown
keys, and pydantic's default ``extra="ignore"`` tolerates (but drops) unknown
fields on every other wire object.  Non-telemetry models deliberately keep
``extra="ignore"`` (never ``forbid``): CONTRACTS §0 requires readers to tolerate
unknown fields, overriding the U02 brief's "otherwise forbid extras" wording.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "EvalMetrics",
    "LearningCurve",
    "RolloutSeries",
    "RolloutSummary",
    "RunManifestSummary",
    "RunRecord",
    "RunStatusModel",
    "TelemetryPoint",
    "TimeMetrics",
    "TrainMetrics",
]


class _TelemetryMetrics(BaseModel):
    """Base for SB3 metric sub-objects: unknown keys survive (CONTRACTS §4)."""

    model_config = ConfigDict(extra="allow")


class TrainMetrics(_TelemetryMetrics):
    """``train/`` metrics from SB3's logger; every field is nullable."""

    policy_loss: float | None = None
    value_loss: float | None = None
    entropy_loss: float | None = None
    approx_kl: float | None = None
    clip_fraction: float | None = None
    explained_variance: float | None = None
    learning_rate: float | None = None
    std: float | None = None
    n_updates: float | None = None


class TimeMetrics(_TelemetryMetrics):
    """``time/`` metrics from SB3's logger; every field is nullable."""

    fps: float | None = None
    iterations: float | None = None
    time_elapsed: float | None = None


class TelemetryPoint(BaseModel):
    """One line of ``telemetry.jsonl`` (CONTRACTS §1.1)."""

    schema_version: int = 1
    run_id: str
    seed: int
    timestep: int
    wall_time_s: float | None = None
    ep_rew_mean: float | None = None
    ep_len_mean: float | None = None
    episodes: int | None = None
    train: TrainMetrics = Field(default_factory=TrainMetrics)
    time: TimeMetrics = Field(default_factory=TimeMetrics)


class RunStatusModel(BaseModel):
    """Mirrors ``undertow.sim.train.telemetry.RunStatus`` (CONTRACTS §1); not imported."""

    schema_version: int = 1
    run_id: str
    seed: int
    state: Literal["running", "done", "failed"]
    timestep: int
    total_timesteps: int
    updated_at: datetime
    pid: int
    error: str | None = None


class EvalMetrics(BaseModel):
    """Parsed from a ``results/`` directory when present (CONTRACTS §4)."""

    sharpe: float | None = None
    total_pnl: float | None = None
    annualized_return: float | None = None
    max_drawdown: float | None = None
    worst_regime_pnl: float | None = None
    per_regime: dict[str, float] = Field(default_factory=dict)
    source: str | None = None


class RunManifestSummary(BaseModel):
    """The dashboard-relevant subset of ``manifest.json`` (CONTRACTS §4).

    Unknown on-disk keys (``learning_curve``, future fields) are tolerated and
    ignored by pydantic's default ``extra="ignore"`` so the full curve is not
    echoed inside every serialised :class:`RunRecord`.
    """

    run_id: str
    seed: int
    config_hash: str
    git_commit: str
    algorithm: str
    total_timesteps: int
    best_reward: float
    checkpoint_path: str
    library_versions: dict[str, str] = Field(default_factory=dict)


class RunRecord(BaseModel):
    """A single discovered run as exposed to the client (CONTRACTS §4)."""

    run_id: str
    experiment: str
    seed: int
    state: Literal["running", "done", "failed", "unknown"]
    updated_at: datetime
    total_timesteps: int
    latest_timestep: int
    best_reward: float | None = None
    config_hash: str | None = None
    git_commit: str | None = None
    manifest: RunManifestSummary | None = None
    eval: EvalMetrics | None = None
    has_telemetry: bool = False
    has_rollouts: bool = False
    telemetry_points: int = 0


class LearningCurve(BaseModel):
    """Per-seed learning curves plus their mean/std (CONTRACTS §4)."""

    run_id: str
    timesteps: list[int]
    mean: list[float]
    std: list[float]
    per_seed: dict[str, list[float]]
    downsampled: bool = False


class RolloutSummary(BaseModel):
    """Descriptive metadata for one recorded evaluation episode (CONTRACTS §4)."""

    seed: int
    episode: int
    n_steps: int
    regime: str
    start: datetime
    end: datetime


class RolloutSeries(BaseModel):
    """A downsampled rollout episode (CONTRACTS §4)."""

    run_id: str
    seed: int
    episode: int
    columns: list[str]
    rows: list[dict[str, object]]
    downsampled: bool = False
