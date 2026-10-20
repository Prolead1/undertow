"""Rollout parquet reader for the Policy Replay view (U06, ``CONTRACTS.md`` §2/§4).

This is an **artifact-only** reader: it loads the recording schema directly from
parquet and validates it into the U02 wire models.  It deliberately does not
import ``undertow.sim`` (PLAN.md §0.6) — the parquet schema is the contract, so
the ``ui`` extra never pulls in the training stack.

Two entry points back the REST endpoints:

* :func:`list_rollouts` — descriptive metadata per episode, sorted by
  ``(seed, episode)``; a missing directory is an empty list and a corrupt file is
  skipped with a warning.
* :func:`read_rollout` — one episode downsampled with the deterministic
  uniform-stride-keep-ends rule from :mod:`undertow.ui.registry`, serialised with
  ISO-8601 UTC timestamps.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import cast

import polars as pl

from undertow.ui.registry import downsample_indices
from undertow.ui.schemas import (
    SCHEMA_VERSION,
    RolloutSeries,
    RolloutSummary,
    SchemaVersionError,
    check_schema_version,
)

__all__ = ["list_rollouts", "read_rollout", "SCHEMA_VERSION"]

logger = logging.getLogger("undertow.ui.rollouts")

# ``SCHEMA_VERSION`` is re-exported from :mod:`undertow.ui.schemas` so the UI
# package has a single canonical major version (the value is mirrored from
# ``undertow.sim`` rather than imported — CONTRACTS §0/§2).

#: Glob matching :class:`undertow.sim.train.rollouts.RolloutRecorder` output.
_PARQUET_GLOB: str = "episode_*.parquet"

#: Columns required to build a :class:`RolloutSummary` without loading the whole
#: trajectory (the replay view loads the full frame only when asked).
_SUMMARY_COLUMNS: tuple[str, ...] = (
    "schema_version",
    "seed",
    "episode",
    "step",
    "timestamp",
    "regime",
)


def _iso_z(moment: datetime) -> str:
    """Render a timestamp as an ISO-8601 UTC string ending in ``Z``."""
    if moment.tzinfo is None:
        return moment.isoformat() + "Z"
    return moment.isoformat().replace("+00:00", "Z")


def _schema_version_of(frame: pl.DataFrame, source: str) -> int:
    """Major ``schema_version`` of a frame, tolerating a missing/odd value.

    CONTRACTS.md §0: a reader rejects a *higher* major version with a typed
    :class:`~undertow.ui.schemas.SchemaVersionError`.  A missing or unparseable
    version is treated as compatible so old/foreign artifacts do not take down
    the whole listing.
    """
    if "schema_version" not in frame.columns or frame.height == 0:
        return SCHEMA_VERSION
    return check_schema_version(frame["schema_version"][0], source=source)


def _wire(value: object) -> object:
    """Convert a parquet cell to its JSON-on-the-wire representation."""
    if isinstance(value, datetime):
        return _iso_z(value)
    return value


def _dominant_regime(labels: Sequence[str | None]) -> str:
    """Most frequent regime label; ties break alphabetically (deterministic)."""
    counts: dict[str, int] = {}
    for label in labels:
        key = "unknown" if label is None else str(label)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return "unknown"
    return min(counts, key=lambda label: (-counts[label], label))


def list_rollouts(rollout_dir: Path | str) -> list[RolloutSummary]:
    """Summarise every episode in ``rollout_dir``, sorted by ``(seed, episode)``.

    A missing or non-directory path returns ``[]``.  A parquet that cannot be
    read (or lacks a required column) is skipped with a warning rather than
    taking down the whole listing.
    """
    directory = Path(rollout_dir)
    if not directory.is_dir():
        return []

    summaries: list[RolloutSummary] = []
    for path in sorted(directory.glob(_PARQUET_GLOB)):
        try:
            frame = pl.read_parquet(path, columns=list(_SUMMARY_COLUMNS))
        except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
            logger.warning("skipping unreadable rollout %s: %s", path, exc)
            continue
        if frame.height == 0:
            logger.warning("skipping empty rollout %s", path)
            continue
        ordered = frame.sort("step")
        try:
            _schema_version_of(ordered, str(path))
        except SchemaVersionError as exc:
            # A newer major schema is rejected (CONTRACTS §0); the *listing*
            # still survives so one future artifact cannot hide every episode.
            logger.warning("skipping rollout %s: %s", path, exc)
            continue
        summaries.append(
            RolloutSummary(
                seed=int(ordered["seed"][0]),
                episode=int(ordered["episode"][0]),
                n_steps=ordered.height,
                regime=_dominant_regime(ordered["regime"].to_list()),
                start=cast(datetime, ordered["timestamp"][0]),
                end=cast(datetime, ordered["timestamp"][-1]),
            )
        )
    summaries.sort(key=lambda summary: (summary.seed, summary.episode))
    return summaries


def read_rollout(path: Path | str, *, max_points: int = 2000) -> RolloutSeries:
    """Load one ``episode_<k>.parquet`` as a downsampled :class:`RolloutSeries`.

    Timestamps are serialised as ISO-8601 UTC strings and the rows are reduced
    to at most ``max_points`` entries with the deterministic stride rule keeping
    the first and last decision.  Raises for an unreadable or empty file; the
    caller decides whether that is fatal (the listing path skips it instead).
    """
    source = Path(path)
    frame = pl.read_parquet(source)
    if frame.height == 0:
        raise ValueError(f"rollout {source} contains no rows")
    # Raises ``SchemaVersionError`` for a newer major (CONTRACTS §0).
    _schema_version_of(frame, str(source))

    columns = list(frame.columns)
    indices = downsample_indices(frame.height, int(max_points))
    records = frame.to_dicts()
    rows = [
        {column: _wire(records[index][column]) for column in columns}
        for index in indices
    ]
    first = records[0]
    return RolloutSeries(
        run_id=str(first["run_id"]),
        seed=int(first["seed"]),
        episode=int(first["episode"]),
        columns=columns,
        rows=rows,
        downsampled=len(indices) < frame.height,
    )
