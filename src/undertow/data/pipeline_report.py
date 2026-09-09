"""Report types and progress tracking for ``undertow.data`` pipeline (T14).

These dataclasses are the structured results every orchestration function returns.
They live in their own module so ``cli.py`` can import them without pulling in the
fetchers, storage, and transform dependencies that ``pipeline.py`` carries.

Progress tracking (``.pull_progress.json``) is also here — it is the resumability
state the orchestrator writes between stream pulls, and is read by the CLI only for
reporting, never for decision-making.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from undertow.data.storage.manifest import DatasetManifest
from undertow.data.transforms.align import Dataset
from undertow.data.types import BlockNumber, CheckResult, Route

LOGGER = logging.getLogger("undertow.data.pipeline")

# ---------------------------------------------------------------------------
# Progress tracking (resumability)
# ---------------------------------------------------------------------------

_PROGRESS_FILENAME: str = ".pull_progress.json"


def _progress_path(output_dir: Path) -> Path:
    return output_dir / _PROGRESS_FILENAME


def _read_progress(output_dir: Path) -> dict[str, object]:
    path = _progress_path(output_dir)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        LOGGER.warning("Corrupt progress file at %s; starting fresh", path)
        return {}


def _write_progress(output_dir: Path, data: dict[str, object]) -> None:
    path = _progress_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StreamPullResult:
    """Result of pulling a single stream."""

    stream: str
    route: Route
    row_count: int
    n_requests: int
    from_cache: bool
    warnings: tuple[str, ...] = ()
    content_hash: str = ""
    min_block: BlockNumber | None = None
    max_block: BlockNumber | None = None


@dataclass(frozen=True, slots=True)
class PullReport:
    """Aggregate report of a ``pull()`` call."""

    streams: dict[str, StreamPullResult] = field(default_factory=dict)
    n_total_requests: int = 0
    all_from_cache: bool = False
    started_at_utc: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at_utc: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class SnapshotReport:
    """Result of a ``snapshot()`` call — the full dataset + validation."""

    pull: PullReport
    dataset: Dataset
    checks: list[CheckResult]
    manifest_path: Path
    validation_report_path: Path


@dataclass(frozen=True, slots=True)
class InfoReport:
    """Summary of a written dataset for ``info()``."""

    manifest: DatasetManifest
    dataset_exists: bool
    output_dir: Path