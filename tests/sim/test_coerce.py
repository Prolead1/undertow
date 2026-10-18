"""Tests for tape-cell coercion helpers (:mod:`undertow.sim.coerce`).

These pin the contract every ``int()``/``float()`` replacement in ``undertow.sim``
now relies on: polars/pyarrow Python scalars (bool, int, float, numpy scalars,
decimal strings) coerce, and non-numeric text raises rather than silently
producing a value.
"""

from __future__ import annotations

import numpy as np
import pytest

from undertow.sim.coerce import as_float, as_int


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (123, 123),
        (True, 1),
        (False, 0),
        (1.9, 1),  # truncates toward zero, like int()
        (-2.9, -2),
        ("123", 123),
        ("-7", -7),
    ],
)
def test_as_int_coerces_tape_scalars(value: object, expected: int) -> None:
    assert as_int(value) == expected


def test_as_int_accepts_numpy_scalars() -> None:
    """numpy scalars expose ``__int__``/``__float__``; the tape can carry them."""
    assert as_int(np.float64(3.9)) == 3
    assert as_int(np.int64(4)) == 4


@pytest.mark.parametrize("value", ["not-a-number", "1.5", "", None])
def test_as_int_rejects_non_integer_text(value: object) -> None:
    with pytest.raises(ValueError):
        as_int(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (123, 123.0),
        (True, 1.0),
        (1.5, 1.5),
        ("1.25", 1.25),
        ("-3", -3.0),
    ],
)
def test_as_float_coerces_tape_scalars(value: object, expected: float) -> None:
    assert as_float(value) == expected


def test_as_float_accepts_numpy_scalars() -> None:
    assert as_float(np.int64(4)) == 4.0
    assert as_float(np.float32(1.5)) == 1.5


@pytest.mark.parametrize("value", ["nope", "", None])
def test_as_float_rejects_non_numeric_text(value: object) -> None:
    with pytest.raises(ValueError):
        as_float(value)
