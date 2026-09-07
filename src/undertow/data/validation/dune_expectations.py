"""Expected-magnitude lookup for the Dune cross-check (T15).

T13 imports :func:`expected_counts` from here as library code — it must NOT live in a
test module, because the validation suite depends on it. The magnitudes themselves
live in the JSON fixture ``tests/data/fixtures/dune_expected_magnitudes.json``, which
may legitimately carry ``"status": "unavailable"`` (no Dune account at authoring time).
In that case every lookup returns ``None`` and T13 should report the Dune comparison
as *skipped* — never as passed.

Lookup semantics (the T13 contract):

* ``None`` — honestly *unknown*: fixture unavailable, stream not tracked (the fixture
  only tracks ``swap``/``mint``/``burn``/``collect``), or month not present.
* ``(count, tolerance_pct)`` — an expected magnitude for that month and stream.
* :class:`~undertow.data.types.ValidationError` — the fixture *claims* data but is
  malformed (missing count, non-positive count, bad tolerance). Loud is correct here:
  a malformed fixture that returns ``None`` would be mistaken by T13 for 'no data'.

The fixture path is resolved relative to this source file (repo layout is fixed for
this project); ``fixture_path`` is a keyword-only escape hatch for installed-package /
test scenarios. Deterministic: reads are cached by path.
"""

from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from undertow.data.types import ValidationError

__all__ = ["expected_counts", "status"]

# Arguments decoded event names — the four counts the fixture records per month
# (CONTRACTS.md §4.1-4.3). ``flash`` exists as a stream but is not tracked here
# (CONTRACTS.md §4.3.1, negligible volume); ``expected_counts("2023-03", "flash")``
# therefore returns None, i.e. "no expected magnitude recorded".
STREAMS: frozenset[str] = frozenset({"swap", "mint", "burn", "collect"})

# The fixture records counts under PLURAL keys (brief template: "swaps": 123456, …)
# while the contract / CONTRACTS.md stream names are singular — map one to the other.
_COUNT_KEYS: dict[str, str] = {
    "swap": "swaps",
    "mint": "mints",
    "burn": "burns",
    "collect": "collects",
}

DEFAULT_FIXTURE_PATH: Path = (
    Path(__file__).resolve().parents[4]
    / "tests"
    / "data"
    / "fixtures"
    / "dune_expected_magnitudes.json"
)
"""The committed fixture, resolved from the repo root (4 parents above this module)."""


@lru_cache(maxsize=8)
def _load_fixture(fixture_path: Path) -> dict[str, Any]:
    """Read + shallow-validate the fixture. Cached per path; raises on corruption."""
    try:
        text = fixture_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError(
            f"dune_expectations: cannot read magnitudes fixture {fixture_path}: {exc}"
        ) from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"dune_expectations: magnitudes fixture {fixture_path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise ValidationError(
            f"dune_expectations: magnitudes fixture {fixture_path} must be a JSON object, "
            f"got {type(raw).__name__}"
        )
    return raw


def _as_count(value: object) -> int | None:
    """A count is a non-negative int, never a bool (bool is an int in Python)."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _as_tolerance(value: object) -> float | None:
    """tolerance_pct is a positive number (elsewhere constrained to <= 10)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    tol = float(value)
    return tol if tol > 0 else None


def expected_counts(
    month: str, stream: str, *, fixture_path: Path | None = None
) -> tuple[int, float] | None:
    """Expected ``(event_count, tolerance_pct)`` for ``stream`` in ``month`` ("YYYY-MM").

    Returns ``None`` when the fixture is unavailable, the month is not recorded, or
    the stream is not tracked — callers must treat ``None`` as *skip* (T13 reports
    the Dune comparison as an ``info``-severity skip, never as a pass or a fail).
    Raises :class:`UndertowDataError` (``ValidationError``) when the fixture claims
    data but contradicts its own shape — a real data problem, not an unknown.
    """
    data = _load_fixture(fixture_path if fixture_path is not None else DEFAULT_FIXTURE_PATH)
    if data.get("status") == "unavailable":
        return None
    if stream not in STREAMS:
        return None
    if not _is_valid_month(month):
        return None
    monthly = data.get("monthly")
    if not isinstance(monthly, dict):
        return None
    if month not in monthly:
        return None  # recorded months exist, but not this one: genuinely unknown
    entry = monthly.get(month)
    if not isinstance(entry, dict):
        raise ValidationError(
            f"dune_expectations: monthly entry for {month!r} must be an object, "
            f"got {type(entry).__name__}"
        )
    count_key = _COUNT_KEYS[stream]
    if count_key not in entry:
        raise ValidationError(
            f"dune_expectations: monthly[{month!r}] is missing the {stream!r}/"
            f"{count_key!r} count "
            "(every month must carry all of swap/mint/burn/collect)"
        )
    count = _as_count(entry.get(count_key))
    if count is None:
        raise ValidationError(
            f"dune_expectations: monthly[{month!r}][{stream!r}] must be a "
            "non-negative integer"
        )
    tolerance = _as_tolerance(data.get("tolerance_pct"))
    if tolerance is None:
        raise ValidationError(
            f"dune_expectations: tolerance_pct must be a positive number for month "
            f"{month!r}"
        )
    return (count, tolerance)


def status(*, fixture_path: Path | None = None) -> str:
    """``"available"`` when the fixture records real Dune magnitudes, else
    ``"unavailable"``. Lets T13 render the skip reason instead of hardcoding one."""
    data = _load_fixture(fixture_path if fixture_path is not None else DEFAULT_FIXTURE_PATH)
    value = data.get("status")
    return value if value == "available" else "unavailable"


def _is_valid_month(month: str) -> bool:
    """True for a "YYYY-MM" key inside the pinned window (PLAN.md §7)."""
    if not isinstance(month, str) or len(month) != 7 or month[4] != "-":
        return False
    try:
        first = date(int(month[:4]), int(month[5:7]), 1)
    except ValueError:
        return False
    return date(2022, 1, 1) <= first <= date(2024, 12, 1)