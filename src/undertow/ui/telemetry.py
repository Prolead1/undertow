"""Safe readers for the training telemetry sidecars (U03, CONTRACTS §1/§5).

``undertow.sim`` writes ``telemetry.jsonl`` as one complete JSON object per
line, each with a trailing newline, and ``status.json`` atomically.  A reader can
therefore still observe a torn final line while a run is live.  This module
turns those artifacts into the mirrored wire models from
:mod:`undertow.ui.schemas` without ever raising on a missing or malformed file.

The important semantic is the **line offset**: it is a count of telemetry lines,
not a timestep, so an incomplete trailing line is left for the next poll.  A
complete line that parses but fails schema validation is permanently skipped and
the offset advances past it; a torn trailing line never advances the offset.

The UI is an artifact-only consumer (`docs/plans/ui/PLAN.md` §0.6), so nothing
here imports ``undertow.sim`` or ``undertow.data``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from undertow.ui.schemas import RunStatusModel, TelemetryPoint

__all__ = [
    "count_lines",
    "read_curve_points",
    "read_points",
    "read_status",
]

logger = logging.getLogger("undertow.ui.telemetry")

#: Read the file in reasonably sized chunks when only counting newlines.
_COUNT_CHUNK_BYTES = 64 * 1024


def _read_json_object(path: Path) -> dict[str, Any] | None:
    """Best-effort read of a JSON object, logging (never raising) on failure."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        # A missing sidecar is an expected transient state, not a warning.
        return None
    except (OSError, ValueError) as exc:
        logger.warning("unreadable JSON artifact %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("JSON artifact is not an object: %s", path)
        return None
    return payload


def read_status(path: Path | str) -> RunStatusModel | None:
    """Parse ``status.json`` into a :class:`RunStatusModel`.

    A missing file returns ``None`` silently (the run may not have written its
    first heartbeat yet); malformed JSON or a schema-invalid payload logs a
    warning and returns ``None``.  Readers never raise.
    """
    status_path = Path(path)
    payload = _read_json_object(status_path)
    if payload is None:
        return None
    try:
        return RunStatusModel.model_validate(payload)
    except ValidationError as exc:
        logger.warning("invalid status sidecar %s: %s", status_path, exc)
        return None


def _read_points(
    path: Path,
    *,
    since: int,
    limit: int | None,
) -> tuple[list[TelemetryPoint], int]:
    """Implementation behind :func:`read_points` / :func:`read_curve_points`."""
    if since < 0:
        raise ValueError(f"since must be non-negative, got {since}")
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be non-negative or None, got {limit}")
    if limit == 0:
        return [], since

    points: list[TelemetryPoint] = []
    next_offset = since
    try:
        with path.open("r", encoding="utf-8") as handle:
            for index, raw_line in enumerate(handle):
                if index < since:
                    continue
                if limit is not None and len(points) >= limit:
                    # Leave the cursor on the first line we have not consumed.
                    break

                complete = raw_line.endswith("\n")
                stripped = raw_line.strip()
                if not stripped:
                    if complete:
                        next_offset = index + 1
                        continue
                    # A blank, newline-less tail is not a complete record.
                    break

                try:
                    payload = json.loads(stripped)
                except ValueError:
                    if complete:
                        # A complete but syntactically invalid line is a
                        # permanent bad record: skip it and advance.
                        logger.warning(
                            "skipping invalid telemetry line %d in %s", index + 1, path
                        )
                        next_offset = index + 1
                        continue
                    # The trailing partial write: stop before it so the next
                    # poll re-reads it once the writer finishes the line.
                    logger.debug("ignoring torn telemetry line %d in %s", index + 1, path)
                    break

                try:
                    point = TelemetryPoint.model_validate(payload)
                except ValidationError as exc:
                    logger.warning(
                        "skipping schema-invalid telemetry line %d in %s: %s",
                        index + 1,
                        path,
                        exc,
                    )
                    if complete:
                        next_offset = index + 1
                        continue
                    break

                points.append(point)
                next_offset = index + 1
    except OSError as exc:
        logger.warning("unreadable telemetry %s: %s", path, exc)
        return [], since

    return points, next_offset


def read_points(
    path: Path | str,
    *,
    since: int = 0,
    limit: int | None = 1000,
) -> tuple[list[TelemetryPoint], int]:
    """Read up to ``limit`` telemetry points starting at line offset ``since``.

    Returns ``(points, next_offset)`` where ``next_offset`` is the line offset a
    client should send on its next poll.  ``limit=None`` means "all remaining
    lines".  A torn trailing line is deliberately *not* consumed, while a
    complete line that fails JSON/schema validation is skipped with the cursor
    advanced past it (CONTRACTS §5).

    A missing/unreadable file yields ``([], since)`` rather than raising.
    """
    return _read_points(Path(path), since=since, limit=limit)


def read_curve_points(path: Path | str) -> list[TelemetryPoint]:
    """Return every valid telemetry point, applying the same torn-write rules."""
    points, _ = _read_points(Path(path), since=0, limit=None)
    return points


def count_lines(path: Path | str) -> int:
    """Count complete newline-terminated lines without parsing them.

    This is the cheap count used when indexing a run: a torn trailing fragment
    (no newline yet) is not counted, matching :func:`read_points`'s offset
    semantics.  A missing/unreadable file counts as zero.
    """
    count = 0
    try:
        with Path(path).open("rb") as handle:
            while chunk := handle.read(_COUNT_CHUNK_BYTES):
                count += chunk.count(b"\n")
    except OSError:
        return 0
    return count
