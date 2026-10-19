"""In-memory registry over a ``runs/`` tree (U02).

:class:`RunRegistry` is the backend's source of truth: it scans run directories
into :class:`~undertow.ui.schemas.RunRecord` objects, aggregates per-seed
learning curves and ranks runs for the leaderboard.  It is strictly read-only
and never writes into a run directory (`docs/plans/ui/PLAN.md` §4).

Robustness is a contract: a corrupt or partial ``manifest.json``/``status.json``
never raises out of :meth:`RunRegistry.scan`.  A bad manifest degrades the record
to ``state="unknown"`` with null provenance and a logged warning; a bad status
sidecar merely yields ``state="unknown"``/inferred timestamps.

Determinism is a contract too (`docs/plans/ui/CONTRACTS.md` §0): scans are sorted
by ``updated_at`` descending then ``run_id`` ascending, and series downsampling
uses a uniform stride that always keeps the first and last point.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Literal

from pydantic import ValidationError

from undertow.ui.paths import RunDir, discover_run_dirs
from undertow.ui.results import load_eval_metrics
from undertow.ui.schemas import (
    EvalMetrics,
    LearningCurve,
    RunManifestSummary,
    RunRecord,
    RunStatusModel,
)

__all__ = ["RunRegistry", "downsample_indices", "rank"]

logger = logging.getLogger("undertow.ui.registry")


def _read_json(path: Path | None) -> dict[str, Any] | None:
    """Read a JSON object, logging (never raising) on failure."""
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("unreadable JSON artifact %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("JSON artifact is not an object: %s", path)
        return None
    return payload


def _as_utc(value: datetime) -> datetime:
    """Normalise any datetime to UTC (CONTRACTS §0: timestamps are UTC)."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _infer_updated_at(run_dir: RunDir) -> datetime:
    """Newest artifact mtime, used when ``status.json`` is missing/malformed."""
    mtimes: list[float] = []
    for path in (
        run_dir.manifest_path,
        run_dir.telemetry_path,
        run_dir.status_path,
    ):
        if path is not None:
            try:
                mtimes.append(path.stat().st_mtime)
            except OSError:  # pragma: no cover - raced deletion
                continue
    if not mtimes:
        return datetime.fromtimestamp(0, tz=UTC)
    return datetime.fromtimestamp(max(mtimes), tz=UTC)


def _read_telemetry(path: Path | None) -> tuple[int, list[int]]:
    """Return ``(valid_point_count, timesteps)`` tolerating torn/bad lines."""
    if path is None or not path.is_file():
        return 0, []
    count = 0
    timesteps: list[int] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    point = json.loads(stripped)
                except ValueError:
                    continue
                if not isinstance(point, dict):
                    continue
                count += 1
                timestep = point.get("timestep")
                if isinstance(timestep, int) and not isinstance(timestep, bool):
                    timesteps.append(timestep)
    except OSError as exc:
        logger.warning("unreadable telemetry %s: %s", path, exc)
        return 0, []
    except ValueError as exc:
        logger.warning("undecodable telemetry %s: %s", path, exc)
        return 0, []
    return count, timesteps


def _has_parquet(rollout_dir: Path | None) -> bool:
    return rollout_dir is not None and any(rollout_dir.glob("*.parquet"))


def _finite(value: object) -> float | None:
    """Return ``value`` as a finite float, or ``None`` for null/NaN/non-numeric."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _aligned_timesteps(timesteps: Sequence[int], length: int) -> list[int]:
    """Return ``length`` timesteps aligned to a manifest curve.

    Telemetry is sampled at the same cadence as the learning curve, so its
    timesteps are the correct x-axis (sorted ascending, per CONTRACTS §0).  If
    telemetry is missing we fall back to the evaluation index; if it is merely
    short (a run read mid-write) we extrapolate the observed positive stride
    rather than mislabel the axis.
    """
    if not timesteps:
        return list(range(length))
    ordered = sorted(timesteps)
    aligned = ordered[:length]
    step = max(1, ordered[1] - ordered[0]) if len(ordered) > 1 else max(1, ordered[0])
    while len(aligned) < length:
        aligned.append(aligned[-1] + step)
    return aligned


def downsample_indices(n: int, max_points: int) -> list[int]:
    """Uniform-stride indices over ``range(n)`` keeping the first and last point.

    The result has at most ``max_points`` entries (fewer only when ``n`` itself
    is smaller).  This is the deterministic downsampling documented in
    CONTRACTS §0 — no LTTB randomness, so payloads diff cleanly.
    """
    if n <= 0:
        return []
    if max_points < 2:
        return [0]
    if n <= max_points:
        return list(range(n))

    step = (n - 1) // (max_points - 1)
    indices = list(range(0, n, max(1, step)))
    if indices[-1] != n - 1:
        indices.append(n - 1)
    if len(indices) > max_points:
        indices = indices[: max_points - 1] + [n - 1]
    return indices


def _descending(value: float | None) -> float:
    """Ascending sort key ordering *finite* values first and missing last.

    ``sorted`` is ascending, so a finite score is negated.  ``None``, ``NaN`` and
    ``±inf`` all map to ``+inf`` so they sort after every real value; ties on
    ``+inf`` are then broken by the next tiebreaker in the tuple.
    """
    finite = _finite(value)
    return math.inf if finite is None else -finite


def rank(
    records: Sequence[RunRecord],
    key: str | Callable[[RunRecord], float | None] = "sharpe",
) -> list[RunRecord]:
    """Rank runs best-first with the pinned leaderboard rule (U10, PLAN §7).

    Every run that has eval artifacts (``eval is not None``) sorts before every
    train-only run.  Within the evaluated group the order is ``key`` descending
    (``"sharpe"`` by default, or an arbitrary accessor), then
    ``worst_regime_pnl`` descending, then ``best_reward`` descending, then
    ``run_id`` ascending.  Train-only runs order by ``best_reward`` descending,
    then ``run_id``.  Missing/NaN/``±inf`` values sort last *within* their group;
    the tuple is a total order.  A string ``key`` that is not an
    :class:`EvalMetrics` field raises :class:`ValueError` rather than silently
    degrading.
    """
    if isinstance(key, str) and key not in EvalMetrics.model_fields:
        raise ValueError(
            f"unknown rank key {key!r}; expected one of {sorted(EvalMetrics.model_fields)}"
        )

    def primary(record: RunRecord) -> float | None:
        if callable(key):
            return key(record)
        assert record.eval is not None  # only called for evaluated runs
        return getattr(record.eval, key, None)

    def sort_key(record: RunRecord) -> tuple[int, float, float, float, str]:
        if record.eval is None:
            # Group 1: train-only runs always trail the evaluated group.
            return (1, math.inf, math.inf, _descending(record.best_reward), record.run_id)
        return (
            0,
            _descending(primary(record)),
            _descending(record.eval.worst_regime_pnl),
            _descending(record.best_reward),
            record.run_id,
        )

    return sorted(records, key=sort_key)


class RunRegistry:
    """Read-only index of the runs under ``runs_dir`` (CONTRACTS §4/§7)."""

    def __init__(
        self,
        runs_dir: Path | str,
        results_dir: Path | str | None = None,
        max_points: int = 2000,
    ) -> None:
        self.runs_dir = Path(runs_dir)
        self.results_dir = Path(results_dir) if results_dir is not None else None
        self.max_points = int(max_points)
        self._records: dict[str, RunRecord] | None = None
        self._curves: dict[str, list[float]] = {}
        self._timesteps: dict[str, list[int]] = {}
        self._eval_cache: dict[tuple[Path, str | None], EvalMetrics | None] = {}

    # -- scanning ---------------------------------------------------------
    def scan(self) -> list[RunRecord]:
        """Rebuild the registry from disk and return the sorted records."""
        self._curves = {}
        self._timesteps = {}
        self._eval_cache = {}
        run_dirs = discover_run_dirs(self.runs_dir)
        # A root-level results/ dir describes one policy; only trust it when the
        # whole tree is a single run, otherwise every record would share it.
        allow_shared_root = len(run_dirs) == 1
        records = [
            self._build_record(run_dir, allow_shared_root=allow_shared_root)
            for run_dir in run_dirs
        ]
        records.sort(key=lambda record: (-_as_utc(record.updated_at).timestamp(), record.run_id))
        by_id: dict[str, RunRecord] = {}
        for record in records:
            if record.run_id in by_id:
                logger.warning("duplicate run_id %r; keeping the first record", record.run_id)
                continue
            by_id[record.run_id] = record
        self._records = by_id
        return list(by_id.values())

    def refresh(self) -> list[RunRecord]:
        """Re-scan the tree (alias for :meth:`scan`)."""
        return self.scan()

    def _ensure_scanned(self) -> None:
        if self._records is None:
            self.scan()

    def _build_record(self, run_dir: RunDir, *, allow_shared_root: bool) -> RunRecord:
        manifest_data = _read_json(run_dir.manifest_path) if run_dir.manifest_path else None
        manifest_bad = run_dir.manifest_path is not None and manifest_data is None

        manifest: RunManifestSummary | None = None
        if manifest_data is not None:
            try:
                manifest = RunManifestSummary.model_validate(manifest_data)
            except ValidationError as exc:
                manifest_bad = True
                logger.warning("invalid manifest %s: %s", run_dir.manifest_path, exc)
                manifest = None

        status: RunStatusModel | None = None
        status_data = _read_json(run_dir.status_path) if run_dir.status_path else None
        if status_data is not None:
            try:
                status = RunStatusModel.model_validate(status_data)
            except ValidationError as exc:
                logger.warning("invalid status %s: %s", run_dir.status_path, exc)

        if manifest_bad:
            state: Literal["running", "done", "failed", "unknown"] = "unknown"
        elif status is not None:
            state = status.state
        else:
            state = "unknown"

        updated_at = (
            _as_utc(status.updated_at) if status is not None else _infer_updated_at(run_dir)
        )

        telemetry_points, telemetry_timesteps = _read_telemetry(run_dir.telemetry_path)
        if telemetry_timesteps:
            self._timesteps[run_dir.run_id] = telemetry_timesteps

        total_timesteps = (
            manifest.total_timesteps
            if manifest is not None
            else (status.total_timesteps if status is not None else 0)
        )
        latest_timestep = (
            status.timestep
            if status is not None
            else (max(telemetry_timesteps) if telemetry_timesteps else 0)
        )

        if not manifest_bad and manifest_data is not None:
            curve = manifest_data.get("learning_curve")
            if isinstance(curve, list):
                values = [
                    float(value)
                    for value in curve
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                ]
                if values:
                    self._curves[run_dir.run_id] = values

        return RunRecord(
            run_id=run_dir.run_id,
            experiment=run_dir.experiment,
            seed=run_dir.seed,
            state=state,
            updated_at=updated_at,
            total_timesteps=total_timesteps,
            latest_timestep=latest_timestep,
            best_reward=manifest.best_reward if manifest is not None else None,
            config_hash=manifest.config_hash if manifest is not None else None,
            git_commit=manifest.git_commit if manifest is not None else None,
            manifest=manifest,
            eval=self._find_eval(
                run_dir,
                policy=manifest.algorithm if manifest is not None else None,
                allow_shared_root=allow_shared_root,
            ),
            has_telemetry=run_dir.telemetry_path is not None,
            has_rollouts=_has_parquet(run_dir.rollout_dir),
            telemetry_points=telemetry_points,
        )

    def _find_eval(
        self, run_dir: RunDir, *, policy: str | None, allow_shared_root: bool
    ) -> EvalMetrics | None:
        """Match a run to its ``results/`` artifacts, if any exist."""
        results_root = self.results_dir
        if results_root is None or not results_root.is_dir():
            return None

        candidates: list[Path] = [results_root / run_dir.run_id]
        if run_dir.experiment:
            candidates.append(results_root / run_dir.experiment / f"seed_{run_dir.seed}")
        candidates.append(results_root / f"seed_{run_dir.seed}")
        if allow_shared_root:
            candidates.append(results_root)

        for candidate in candidates:
            if not candidate.is_dir():
                continue
            try:
                resolved = candidate.resolve()
                root = results_root.resolve()
            except OSError:  # pragma: no cover - raced deletion
                continue
            if not resolved.is_relative_to(root):
                logger.warning("ignoring results candidate outside %s: %s", root, candidate)
                continue
            cache_key = (candidate, policy)
            if cache_key in self._eval_cache:
                metrics = self._eval_cache[cache_key]
            else:
                try:
                    relative = candidate.relative_to(results_root).as_posix()
                except ValueError:  # pragma: no cover - candidates are under the root
                    relative = ""
                prefix = None if relative in ("", ".") else relative
                metrics = load_eval_metrics(candidate, source_prefix=prefix, policy=policy)
                self._eval_cache[cache_key] = metrics
            if metrics is not None:
                return metrics
        return None

    # -- lookups ----------------------------------------------------------
    def get(self, run_id: str) -> RunRecord | None:
        """Return the record for ``run_id`` (or ``None``)."""
        self._ensure_scanned()
        assert self._records is not None
        return self._records.get(run_id)

    def curve(self, run_id: str) -> LearningCurve | None:
        """Aggregate the per-seed learning curves of ``run_id``'s experiment.

        Curves are truncated to their common length and aligned on the reference
        run's telemetry timesteps (falling back to the evaluation index when no
        telemetry exists).  Mean/std are computed across seeds and the series is
        downsampled deterministically to ``max_points``.
        """
        self._ensure_scanned()
        assert self._records is not None
        record = self._records.get(run_id)
        if record is None:
            return None

        group = [
            candidate
            for candidate in self._records.values()
            if candidate.experiment == record.experiment and self._curves.get(candidate.run_id)
        ]
        if not group:
            return None

        length = min(len(self._curves[candidate.run_id]) for candidate in group)
        if length == 0:
            return None

        matrix = [self._curves[candidate.run_id][:length] for candidate in group]
        mean = [fmean(row[position] for row in matrix) for position in range(length)]
        std = [pstdev(row[position] for row in matrix) for position in range(length)]

        timesteps = _aligned_timesteps(self._timesteps.get(record.run_id, []), length)

        per_seed = {
            str(candidate.seed): list(self._curves[candidate.run_id][:length])
            for candidate in group
        }

        indices = downsample_indices(length, self.max_points)
        return LearningCurve(
            run_id=run_id,
            timesteps=[timesteps[index] for index in indices],
            mean=[mean[index] for index in indices],
            std=[std[index] for index in indices],
            per_seed={
                seed: [values[index] for index in indices] for seed, values in per_seed.items()
            },
            downsampled=len(indices) < length,
        )

    def rank(
        self,
        records: Sequence[RunRecord],
        key: str | Callable[[RunRecord], float | None] = "sharpe",
    ) -> list[RunRecord]:
        """Delegate to the module-level :func:`rank` helper."""
        return rank(records, key)
