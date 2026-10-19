"""Run-directory layout discovery for ``undertow.ui`` (U02).

The filesystem is the store (PLAN.md §1.2), so every view starts by turning a
``runs/`` tree into a deterministic list of :class:`RunDir` records describing
where a run's artifacts live.  Both documented layouts are supported::

    runs/<experiment>/seed_<n>/
    runs/seed_<n>/

A directory is a run directory when it directly contains ``manifest.json``,
``telemetry.jsonl`` or ``status.json`` (the U02 task brief widens U01's marker
set with ``status.json``; see `docs/plans/ui/tasks/U02_registry.md`).  Extra
files are ignored and a missing/non-directory root is not an error — a fresh
clone has no runs yet.

This module is deliberately pure filesystem code: it never imports ``schemas``
so the parser can be reused (and unit-tested) without pydantic in the stack.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "MANIFEST_FILENAME",
    "STATUS_FILENAME",
    "TELEMETRY_FILENAME",
    "RunDir",
    "discover_run_dirs",
    "parse_run_dir",
]

MANIFEST_FILENAME = "manifest.json"
TELEMETRY_FILENAME = "telemetry.jsonl"
STATUS_FILENAME = "status.json"
ROLLOUT_DIRNAME = "rollouts"

#: Markers that make a directory a run directory, in scan order.
_MARKER_FILENAMES: tuple[str, ...] = (
    MANIFEST_FILENAME,
    TELEMETRY_FILENAME,
    STATUS_FILENAME,
)

_SEED_RE = re.compile(r"^seed_(-?\d+)$")


@dataclass(frozen=True, slots=True)
class RunDir:
    """Resolved paths for one run directory.

    ``manifest_path``/``telemetry_path``/``status_path``/``rollout_dir`` are
    ``None`` when the corresponding artifact is absent, so callers never have to
    re-stat.  ``experiment`` is the parent path relative to the scan root (``""``
    for the bare ``runs/seed_<n>/`` layout).
    """

    path: Path
    experiment: str
    seed: int
    run_id: str
    manifest_path: Path | None
    telemetry_path: Path | None
    status_path: Path | None
    rollout_dir: Path | None


def _read_json_object(path: Path | None) -> dict[str, object] | None:
    """Best-effort read of a JSON object; ``None`` on any failure."""
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _marker(path: Path, filename: str) -> Path | None:
    candidate = path / filename
    return candidate if candidate.is_file() else None


def _experiment_name(path: Path, root: Path | None) -> str:
    """Return the run's parent path relative to ``root`` (``""`` for bare runs)."""
    if root is None:
        return path.parent.name
    try:
        relative = path.parent.relative_to(root)
    except ValueError:
        return path.parent.name
    return "" if relative == Path(".") else relative.as_posix()


def _seed_from_name(path: Path) -> int | None:
    match = _SEED_RE.match(path.name)
    return int(match.group(1)) if match is not None else None


def parse_run_dir(path: Path | str, *, root: Path | None = None) -> RunDir:
    """Resolve the artifacts of a single run directory.

    ``root`` (when supplied) is the scan root used to name the experiment; for
    the bare layout the parent equals ``root`` and the experiment is ``""``.
    ``run_id`` prefers the manifest's value, then ``status.json``'s, then the
    canonical ``[<experiment>/]seed_<n>`` path.  ``seed`` prefers the directory
    name and falls back to the manifest, then the status sidecar.
    """
    run_path = Path(path)
    manifest_path = _marker(run_path, MANIFEST_FILENAME)
    telemetry_path = _marker(run_path, TELEMETRY_FILENAME)
    status_path = _marker(run_path, STATUS_FILENAME)

    manifest = _read_json_object(manifest_path)
    status = _read_json_object(status_path)

    experiment = _experiment_name(run_path, root)

    seed = _seed_from_name(run_path)
    if seed is None:
        seed = _coerce_int((manifest or {}).get("seed"))
    if seed is None:
        seed = _coerce_int((status or {}).get("seed"))
    if seed is None:
        seed = 0

    run_id = (manifest or {}).get("run_id")
    if not isinstance(run_id, str) or not run_id:
        run_id = (status or {}).get("run_id")
    if not isinstance(run_id, str) or not run_id:
        run_id = f"{experiment}/seed_{seed}" if experiment else f"seed_{seed}"

    rollout_path = run_path / ROLLOUT_DIRNAME
    return RunDir(
        path=run_path,
        experiment=experiment,
        seed=seed,
        run_id=run_id,
        manifest_path=manifest_path,
        telemetry_path=telemetry_path,
        status_path=status_path,
        rollout_dir=rollout_path if rollout_path.is_dir() else None,
    )


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def discover_run_dirs(root: Path | str) -> list[RunDir]:
    """Return every run directory under ``root``, sorted by path.

    The scan is deterministic (lexicographic by absolute path) and tolerates a
    missing root, non-directory root and extra files inside run directories.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return []

    found: set[Path] = set()
    for filename in _MARKER_FILENAMES:
        for marker in root_path.rglob(filename):
            if marker.is_file():
                found.add(marker.parent)

    return [parse_run_dir(path, root=root_path) for path in sorted(found)]
