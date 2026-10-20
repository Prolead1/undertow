"""Plain-data assembly for the REST figure/compare routes (U05).

The REST layer never builds Plotly specs itself (PLAN §0.6): this module turns
the registry and artifact readers into the plain dict/list inputs the pure
builders in :mod:`undertow.ui.figures` expect, then calls them.  Keeping the
assembly here leaves :mod:`undertow.ui.app` as route wiring only and makes the
data-shaping rules unit-testable without a web client.

Errors are domain errors (:class:`UnknownMetricError`), not HTTP errors; the app
translates them into ``422`` responses.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from undertow.ui.figures.actions import action_heatmap_figure
from undertow.ui.figures.compare import compare_equity_figure
from undertow.ui.figures.diagnostics import DIAGNOSTIC_METRICS, diagnostics_figure
from undertow.ui.figures.leaderboard import leaderboard_figure
from undertow.ui.figures.learning import learning_curve_figure
from undertow.ui.figures.ribbon import ribbon_figure
from undertow.ui.figures.theme import FigureSpec, empty_figure
from undertow.ui.paths import RunDir
from undertow.ui.registry import RunRegistry
from undertow.ui.rollouts import read_rollout
from undertow.ui.schemas import (
    RolloutSeries,
    RunRecord,
    SchemaVersionError,
    TelemetryPoint,
    check_schema_version,
)
from undertow.ui.telemetry import count_lines, read_points

__all__ = [
    "NUMERIC_METRICS",
    "UnknownMetricError",
    "build_compare_figure",
    "build_figure",
    "compare_series",
    "rank_compare",
    "read_telemetry_points",
    "resolve_figure_metric",
]

logger = logging.getLogger("undertow.ui.assembly")

#: Metrics accepted by ``POST /api/compare`` (mirrored from :class:`EvalMetrics`
#: plus the train-only ``best_reward`` fallback; PLAN §7 ranking rule).
NUMERIC_METRICS: frozenset[str] = frozenset(
    {"sharpe", "total_pnl", "annualized_return", "max_drawdown", "worst_regime_pnl", "best_reward"}
)


class UnknownMetricError(ValueError):
    """A query/body metric is not one the selected builder can plot or rank."""


def resolve_figure_metric(name: str, metric: str | None) -> str:
    """Validate the optional ``metric`` query for figure ``name``.

    ``diagnostics`` only knows the SB3 ``train/`` keys and its builder raises
    :class:`KeyError` for anything else, so the route validates up front and
    returns a helpful ``422``.  ``leaderboard`` tolerates any metric key and
    renders the empty state when no record carries it.
    """
    if name == "diagnostics":
        chosen = metric or "policy_loss"
        if chosen not in DIAGNOSTIC_METRICS:
            raise UnknownMetricError(
                f"unknown diagnostic metric {chosen!r}; "
                f"expected one of {sorted(DIAGNOSTIC_METRICS)}"
            )
        return chosen
    if name == "leaderboard":
        return metric or "sharpe"
    return metric or ""


def read_telemetry_points(run_dir: RunDir, *, max_points: int) -> list[TelemetryPoint]:
    """Read at most ``max_points`` recent telemetry points (bounded read).

    The diagnostics figure only needs the most recent window, so counting the
    complete lines first avoids parsing an unbounded live ``telemetry.jsonl``.
    """
    if run_dir.telemetry_path is None:
        return []
    if max_points <= 0:
        return []
    total = count_lines(run_dir.telemetry_path)
    since = max(0, total - max_points)
    points, _ = read_points(run_dir.telemetry_path, since=since, limit=max_points)
    return points


def _select_rollout_path(
    run_dir: RunDir, *, seed: int | None, episode: int | None
) -> Path | None:
    """Pick the rollout parquet matching the optional ``(seed, episode)`` filter.

    Selection reads only the identity columns directly (not via
    :func:`list_rollouts`, which intentionally skips incompatible files) so a
    matching rollout carrying a newer major ``schema_version`` raises
    :class:`SchemaVersionError` (→ HTTP 422) rather than being silently ignored
    and rendering an empty figure (CONTRACTS §0).
    """
    if run_dir.rollout_dir is None or not run_dir.rollout_dir.is_dir():
        return None
    for path in sorted(run_dir.rollout_dir.glob("*.parquet")):
        try:
            head = pl.read_parquet(
                path, columns=["seed", "episode", "schema_version"]
            )
        except Exception:  # noqa: BLE001 - an unreadable file is handled by read_rollout
            continue
        if head.height == 0:
            continue
        row = head.row(0, named=True)
        if seed is not None and row.get("seed") != seed:
            continue
        if episode is not None and row.get("episode") != episode:
            continue
        check_schema_version(row.get("schema_version"), source=str(path))
        return path
    return None


def _load_rollout_series(
    run_dir: RunDir, *, seed: int | None, episode: int | None, max_points: int
) -> RolloutSeries | None:
    path = _select_rollout_path(run_dir, seed=seed, episode=episode)
    if path is None:
        return None
    try:
        return read_rollout(path, max_points=max_points)
    except SchemaVersionError as exc:
        # A newer major is a schema incompatibility, not a corrupt file: re-raise
        # so the figure route answers 422 (CONTRACTS §0) instead of silently
        # rendering the empty state.
        logger.warning("incompatible rollout schema %s: %s", path, exc)
        raise
    except ValueError as exc:
        logger.warning("skipping unreadable rollout %s: %s", path, exc)
        return None


def _actions_figure(
    run_dir: RunDir, *, seed: int | None, episode: int | None, max_points: int
) -> FigureSpec:
    series = _load_rollout_series(run_dir, seed=seed, episode=episode, max_points=max_points)
    if series is None:
        return empty_figure()
    counts: dict[tuple[int, int], float] = {}
    offsets: set[int] = set()
    widths: set[int] = set()
    for row in series.rows:
        offset = row.get("center_offset")
        width = row.get("width")
        if isinstance(offset, bool) or not isinstance(offset, (int, float)):
            continue
        if isinstance(width, bool) or not isinstance(width, (int, float)):
            continue
        key = (int(offset), int(width))
        counts[key] = counts.get(key, 0.0) + 1.0
        offsets.add(key[0])
        widths.add(key[1])
    if not offsets or not widths:
        return empty_figure()
    return action_heatmap_figure(
        counts,
        offsets=sorted(offsets),
        widths=sorted(widths),
        title=f"Action distribution \u2014 {series.run_id}",
    )


def _ribbon_figure(
    run_dir: RunDir, *, seed: int | None, episode: int | None, max_points: int
) -> FigureSpec:
    series = _load_rollout_series(run_dir, seed=seed, episode=episode, max_points=max_points)
    if series is None:
        return empty_figure()
    return ribbon_figure(series.rows)


def _leaderboard_records(
    resolved: RunRecord, records: Sequence[RunRecord]
) -> list[RunRecord]:
    """Ranking set for a leaderboard request, resolving the alias/canonical question.

    The leaderboard is a global view, but the route's ``run_id`` may resolve
    through the CONTRACTS §7 ``<experiment>/seed_<n>`` alias to a directory whose
    record lost the duplicate-``run_id`` dedup.  The resolved record therefore
    supersedes its canonical namesake for that request, so the requested
    directory is always represented exactly once (one bar per ``run_id``).
    """
    if not any(item.run_id == resolved.run_id for item in records):
        return [*records, resolved]
    return [resolved if item.run_id == resolved.run_id else item for item in records]


def build_figure(
    name: str,
    *,
    run_id: str,
    record: RunRecord,
    run_dir: RunDir,
    registry: RunRegistry,
    records: Sequence[RunRecord],
    seed: int | None,
    episode: int | None,
    log_y: bool,
    metric: str | None,
    max_points: int,
) -> FigureSpec:
    """Assemble the plain-data input for ``name`` and call its pure builder.

    ``run_id`` is the id as requested by the client (not necessarily
    ``record.run_id``): the learning figure must aggregate the *resolved*
    directory's experiment so an alias never plots its canonical namesake's
    curve.
    """
    chosen_metric = resolve_figure_metric(name, metric)
    if name == "learning":
        curve = registry.curve(run_id)
        if curve is None:
            return empty_figure()
        return learning_curve_figure(curve.model_dump(), log_y=log_y)
    if name == "diagnostics":
        points = read_telemetry_points(run_dir, max_points=max_points)
        return diagnostics_figure(
            [point.model_dump() for point in points], chosen_metric
        )
    if name == "actions":
        return _actions_figure(run_dir, seed=seed, episode=episode, max_points=max_points)
    if name == "ribbon":
        return _ribbon_figure(run_dir, seed=seed, episode=episode, max_points=max_points)
    if name == "leaderboard":
        return leaderboard_figure(
            [
                item.model_dump(mode="json")
                for item in _leaderboard_records(record, records)
            ],
            metric=chosen_metric,
        )
    raise KeyError(name)  # pragma: no cover - guarded by FIGURE_REGISTRY membership


def rank_compare(
    registry: RunRegistry, records: list[RunRecord], metric: str
) -> list[RunRecord]:
    """Rank compare records with the pinned leaderboard rule (ADR-025)."""
    if metric not in NUMERIC_METRICS:
        raise UnknownMetricError(
            f"unknown compare metric {metric!r}; expected one of {sorted(NUMERIC_METRICS)}"
        )
    if metric == "best_reward":
        return registry.rank(records, lambda record: record.best_reward)
    return registry.rank(records, metric)


def _alias_label(run_dir: RunDir) -> str:
    """The CONTRACTS §7 ``[<experiment>/]seed_<n>`` label for a directory."""
    if run_dir.experiment:
        return f"{run_dir.experiment}/seed_{run_dir.seed}"
    return f"seed_{run_dir.seed}"


def compare_series(
    runs: Sequence[tuple[RunRecord, RunDir]], *, max_points: int
) -> dict[str, list[float]]:
    """Equity overlay: first recorded episode per run (skips runs without rollouts).

    Two requested ids can resolve to directories that share a manifest
    ``run_id``.  Series keys are legend labels, so when a ``run_id`` repeats the
    CONTRACTS §7 ``<experiment>/seed_<n>`` alias disambiguates the lines instead
    of one silently overwriting the other.
    """
    run_id_counts: dict[str, int] = {}
    for record, _ in runs:
        run_id_counts[record.run_id] = run_id_counts.get(record.run_id, 0) + 1

    series: dict[str, list[float]] = {}
    for record, run_dir in runs:
        path = _select_rollout_path(run_dir, seed=None, episode=None)
        if path is None:
            continue
        try:
            rollout = read_rollout(path, max_points=max_points)
        except SchemaVersionError as exc:
            # A schema incompatibility propagates (the API answers 422), unlike a
            # merely unreadable file which is skipped so the overlay still plots.
            logger.warning("incompatible rollout schema %s: %s", path, exc)
            raise
        except ValueError as exc:
            logger.warning("skipping unreadable rollout %s: %s", path, exc)
            continue
        values: list[float] = []
        for row in rollout.rows:
            equity = row.get("equity")
            if isinstance(equity, bool) or not isinstance(equity, (int, float)):
                continue
            values.append(float(equity))
        if values:
            label = (
                _alias_label(run_dir)
                if run_id_counts[record.run_id] > 1
                else record.run_id
            )
            series[label] = values
    return series


def build_compare_figure(
    runs: Sequence[tuple[RunRecord, RunDir]], *, max_points: int
) -> FigureSpec:
    """The themed compare overlay for ``runs`` (empty state when unplottable)."""
    return compare_equity_figure(compare_series(runs, max_points=max_points))
