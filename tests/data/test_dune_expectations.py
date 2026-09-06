"""Tests for ``undertow.data.validation.dune_expectations`` (T15).

The shipped fixture has ``"status": "unavailable"`` — this sandbox has no Dune
account, so no magnitudes are recorded (a fabricated magnitude would be worse than
none: T13 would trust it). The "present month" code paths are therefore exercised
with a **clearly synthetic** per-test fixture written to ``tmp_path``; those numbers
exist only to exercise the library and must never be confused with expected
magnitudes. All tests that need real per-month values skip cleanly when the fixture
is (still) unavailable, so the module is forward-compatible with a future
``status: "available"`` fixture.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import date
from pathlib import Path

import pytest

from undertow.data.config import default_pools
from undertow.data.types import ValidationError
from undertow.data.validation import dune_expectations
from undertow.data.validation.dune_expectations import expected_counts

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "fixtures" / "dune_expected_magnitudes.json"
)
SQL_DIR = Path(__file__).resolve().parents[2] / "sql" / "dune"

PINNED_MONTH_FIRST = date(2022, 1, 1)  # PLAN.md §7 pinned window: 2022-01..2024-12
PINNED_MONTH_LAST = date(2024, 12, 1)

REQUIRED_FIXTURE_KEYS = frozenset(
    {
        "source",
        "queried_at_utc",
        "dashboard_url",
        "pool",
        "status",
        "monthly",
        "tolerance_pct",
        "notes",
    }
)
STREAMS = ("swap", "mint", "burn", "collect")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _month_number(month: str) -> date:
    return date(int(month[:4]), int(month[5:7]), 1)


def _add_month(d: date) -> date:
    """Same day-of-month, one month later (month serial arithmetic)."""
    year = d.year + (d.month // 12)
    month = d.month % 12 + 1
    return date(year, month, 1)


def _synthetic_fixture(
    tmp_path: Path,
    *,
    monthly: dict[str, object] | None,
    tolerance_pct: object = 2.0,
    status: str = "available",
) -> Path:
    """Write a synthetic fixture (test-only numbers) and return its path."""
    available = status == "available"
    data: dict[str, object] = {
        "source": "Dune Analytics (synthetic, test-only — not real magnitudes)",
        "queried_at_utc": "2025-01-15T00:00:00Z" if available else None,
        "dashboard_url": "https://dune.com/project/example" if available else None,
        "pool": "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",
        "status": status,
        "monthly": monthly,
        "tolerance_pct": tolerance_pct,
        "notes": "synthetic fixture for unit-testing the lookup; NOT an expected magnitude",
    }
    # unique name per call: the library caches loaded fixtures by path, so reusing one
    # filename inside a single test would silently serve the first write from the cache.
    path = tmp_path / f"dune_expected_magnitudes_{uuid.uuid4().hex[:8]}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. fixture shape: parses, has the required keys, months carry all four counts
#    (or the fixture is honestly unavailable)
# ---------------------------------------------------------------------------

def test_fixture_exists_and_has_required_keys() -> None:
    data = _load(FIXTURE_PATH)
    assert isinstance(data, dict)
    assert set(data) >= REQUIRED_FIXTURE_KEYS, (
        f"fixture is missing required keys: {sorted(REQUIRED_FIXTURE_KEYS - set(data))}"
    )
    assert data["status"] in {"available", "unavailable"}
    assert data["pool"] == default_pools()["USDC_WETH_3000"].address
    assert isinstance(data["source"], str) and data["source"]
    assert isinstance(data["notes"], str) and data["notes"]


def test_unavailable_fixture_looks_unavailable() -> None:
    data = _load(FIXTURE_PATH)
    if data["status"] != "unavailable":
        pytest.skip("fixture was populated with real magnitudes; unavailable-shape check n/a")
    # Every count-bearing field must be explicitly null — no numbers parked in dead keys.
    assert data["monthly"] is None
    assert data["tolerance_pct"] is None
    assert data["dashboard_url"] is None
    assert data["queried_at_utc"] is None


def test_monthly_entries_carry_all_four_counts() -> None:
    data = _load(FIXTURE_PATH)
    monthly = data.get("monthly")
    if monthly is None:
        pytest.skip("status is unavailable; no monthly entries to validate")
    assert isinstance(monthly, dict)
    for month, entry in monthly.items():
        assert isinstance(entry, dict), f"{month}: entry is not an object"
        for stream in STREAMS:
            value = entry.get(stream)
            assert isinstance(value, int) and not isinstance(value, bool) and value > 0, (
                f"{month}.{stream}: expected a positive integer count, got {value!r}"
            )
        # extra streams (e.g. flash) are allowed but must not be negative ints/booleans
        for stream, value in entry.items():
            if stream in STREAMS:
                continue
            assert value is None or (
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
            ), f"{month}.{stream}: unexpected value {value!r}"


# ---------------------------------------------------------------------------
# 2. month keys: YYYY-MM, sorted, contiguous, inside the pinned window
# ---------------------------------------------------------------------------

def test_month_keys_are_sorted_contiguous_and_in_window() -> None:
    data = _load(FIXTURE_PATH)
    monthly = data.get("monthly")
    if monthly is None:
        pytest.skip("status is unavailable; no month keys to validate")
    assert isinstance(monthly, dict)
    keys = sorted(monthly)
    assert keys == sorted(set(monthly)), "duplicate or misordered month keys"
    previous: date | None = None
    for key in keys:
        assert len(key) == 7 and key[4] == "-", f"{key!r} is not a YYYY-MM key"
        first = _month_number(key)
        assert PINNED_MONTH_FIRST <= first <= PINNED_MONTH_LAST, (
            f"{key!r} falls outside the pinned window "
            f"{PINNED_MONTH_FIRST:%Y-%m}..{PINNED_MONTH_LAST:%Y-%m}"
        )
        if previous is not None:
            assert _add_month(previous) == first, (
                f"month keys are not contiguous: {previous:%Y-%m} -> {first:%Y-%m}"
            )
        previous = first


# ---------------------------------------------------------------------------
# 3. counts positive ints where present; tolerance_pct in (0, 10]
# ---------------------------------------------------------------------------

def test_tolerance_pct_is_in_allowed_range() -> None:
    data = _load(FIXTURE_PATH)
    if data["status"] != "available":
        pytest.skip("status is unavailable; no tolerance recorded")
    tolerance = data.get("tolerance_pct")
    assert isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool)
    assert 0 < tolerance <= 10, f"tolerance_pct must be in (0, 10], got {tolerance!r}"


# ---------------------------------------------------------------------------
# 4. expected_counts(month, stream) -> tuple[int, float] | None
# ---------------------------------------------------------------------------

def test_expected_counts_present_month(tmp_path: Path) -> None:
    monthly: dict[str, object] = {
        "2023-03": {"swaps": 123456, "mints": 4567, "burns": 4321, "collects": 3210},
        "2023-04": {"swaps": 131313, "mints": 4600, "burns": 4400, "collects": 3001},
    }
    fixture = _synthetic_fixture(tmp_path, monthly=monthly, tolerance_pct=2.0)
    assert expected_counts("2023-03", "swap", fixture_path=fixture) == (123456, 2.0)
    assert expected_counts("2023-03", "mint", fixture_path=fixture) == (4567, 2.0)
    assert expected_counts("2023-03", "burn", fixture_path=fixture) == (4321, 2.0)
    assert expected_counts("2023-03", "collect", fixture_path=fixture) == (3210, 2.0)


def test_expected_counts_absent_month_is_none(tmp_path: Path) -> None:
    monthly: dict[str, object] = {
        "2023-03": {"swaps": 123456, "mints": 4567, "burns": 4321, "collects": 3210}
    }
    fixture = _synthetic_fixture(tmp_path, monthly=monthly)
    assert expected_counts("2023-06", "swap", fixture_path=fixture) is None


def test_expected_counts_unknown_stream_is_none(tmp_path: Path) -> None:
    monthly: dict[str, object] = {
        "2023-03": {"swaps": 123456, "mints": 4567, "burns": 4321, "collects": 3210}
    }
    fixture = _synthetic_fixture(tmp_path, monthly=monthly)
    # flash is a real CONTRACTS stream but the fixture does not track it: None = skip.
    assert expected_counts("2023-03", "flash", fixture_path=fixture) is None


def test_expected_counts_invalid_month_is_none(tmp_path: Path) -> None:
    monthly: dict[str, object] = {
        "2023-03": {"swaps": 123456, "mints": 4567, "burns": 4321, "collects": 3210}
    }
    fixture = _synthetic_fixture(tmp_path, monthly=monthly)
    assert expected_counts("2023-13", "swap", fixture_path=fixture) is None
    assert expected_counts("2023/03", "swap", fixture_path=fixture) is None
    assert expected_counts("2021-12", "swap", fixture_path=fixture) is None  # pre-window


def test_expected_counts_unavailable_is_none() -> None:
    # The SHIPPED fixture is unavailable in this sandbox: every lookup must be None.
    assert expected_counts("2023-03", "swap") is None
    assert expected_counts("2022-01", "mint") is None
    assert expected_counts("2024-12", "collect") is None


def test_expected_counts_unavailable_via_synthetic_fixture(tmp_path: Path) -> None:
    fixture = _synthetic_fixture(tmp_path, monthly=None, tolerance_pct=None, status="unavailable")
    assert expected_counts("2023-03", "swap", fixture_path=fixture) is None
    assert dune_expectations.status(fixture_path=fixture) == "unavailable"


def test_expected_counts_status_helper() -> None:
    assert dune_expectations.status() == "unavailable"  # shipped fixture, this sandbox


def test_expected_counts_malformed_fixture_raises(tmp_path: Path) -> None:
    # Present-but-malformed data must be loud (ValidationError), not a silent None:
    # a silent None would read as "no data" to T13 and mask a real corruption.
    bad_counts = _synthetic_fixture(
        tmp_path,
        monthly={
            "2023-03": {"swaps": -5, "mints": 4567, "burns": 4321, "collects": 3210}
        },
    )
    with pytest.raises(ValidationError):
        expected_counts("2023-03", "swap", fixture_path=bad_counts)

    missing_count = _synthetic_fixture(
        tmp_path,
        monthly={
            "2023-03": {"swaps": 123, "mints": 45, "burns": 67, "collects": None}
        },
    )
    with pytest.raises(ValidationError):
        expected_counts("2023-03", "collect", fixture_path=missing_count)

    bad_tolerance = _synthetic_fixture(
        tmp_path,
        monthly={
            "2023-03": {"swaps": 123, "mints": 45, "burns": 67, "collects": 12}
        },
        tolerance_pct=0.0,  # 0 is not a valid tolerance
    )
    with pytest.raises(ValidationError):
        expected_counts("2023-03", "swap", fixture_path=bad_tolerance)


def test_missing_fixture_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(ValidationError):
        expected_counts("2023-03", "swap", fixture_path=missing)


# ---------------------------------------------------------------------------
# 5. sql/dune guard: non-empty, -- comment block, pool-parameter CTE present
# ---------------------------------------------------------------------------

def test_sql_dir_contains_exactly_the_six_queries() -> None:
    files = sorted(SQL_DIR.glob("*.sql"))
    assert len(files) == 6, f"expected exactly 6 queries, found {[f.name for f in files]}"
    for query in files:
        assert re.fullmatch(r"0[1-6]_.*\.sql", query.name), (
            f"{query.name} is not one of the six numbered T15 queries"
        )


def test_every_sql_query_is_complete() -> None:
    files = sorted(SQL_DIR.glob("*.sql"))
    assert files, "no .sql files under sql/dune/"
    for query in files:
        text = query.read_text(encoding="utf-8")
        assert text.strip(), f"{query.name} is empty"
        assert text.lstrip().startswith("--"), f"{query.name} must start with a -- comment block"
        assert re.search(
            r"WITH params AS \(\s*SELECT\s+0x[0-9a-fA-F]{40}\s+AS pool", text
        ), (
            f"{query.name} must parameterize the pool address via the params CTE "
            "(e.g. `WITH params AS (SELECT 0x… AS pool)`)"
        )


def test_dashboard_readme_exists() -> None:
    readme = SQL_DIR / "README.md"
    assert readme.is_file(), "sql/dune/README.md is missing"
    text = readme.read_text(encoding="utf-8")
    assert "Dashboard URL: none yet" in text  # the handoff slot exists and is honest