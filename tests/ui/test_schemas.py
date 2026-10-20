"""Schema-version guard (CONTRACTS.md §0) — typed rejection, tolerant defaults."""

from __future__ import annotations

import pytest

from undertow.ui.schemas import SCHEMA_VERSION, SchemaVersionError, check_schema_version


def test_supported_and_older_versions_are_accepted() -> None:
    assert SCHEMA_VERSION == 1
    assert check_schema_version(1) == 1
    assert check_schema_version(0) == 0
    # A float that is integral coerces cleanly.
    assert check_schema_version(1.0) == 1


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        "not-a-number",
        object(),
        [],
        {},
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_missing_or_non_integer_versions_are_tolerated(value: object) -> None:
    """Old/foreign artifacts must not take down a listing (CONTRACTS §0)."""
    assert check_schema_version(value) == SCHEMA_VERSION


def test_higher_major_raises_typed_error_with_context() -> None:
    with pytest.raises(SchemaVersionError) as info:
        check_schema_version(2, source="runs/exp/seed_0/status.json")

    error = info.value
    assert isinstance(error, ValueError)
    assert error.version == 2
    assert error.supported == 1
    assert error.source == "runs/exp/seed_0/status.json"
    assert "schema_version 2" in str(error)
    assert "runs/exp/seed_0/status.json" in str(error)


def test_explicit_supported_major_can_be_raised() -> None:
    assert check_schema_version(2, supported=2) == 2
    with pytest.raises(SchemaVersionError) as info:
        check_schema_version(3, supported=2)
    assert info.value.supported == 2
