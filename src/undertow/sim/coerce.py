"""Cell coercion helpers for tape/snapshot rows typed as ``object``.

Tape rows arrive as ``dict[str, object]`` (polars ``to_dicts``) or pyarrow
scalars, so every numeric read needs a conversion to the domain type.  Keeping
that in one place avoids scattering ``int(str(...))`` across modules and gives
mypy a typed boundary.  Domain aliases live in :mod:`undertow.sim.types`; these
are coercions, not types, so they live here.

Accepted inputs are the scalars polars/pyarrow produce: ``bool``, ``int``,
``float``, numpy scalars and decimal strings.  Objects with ``__int__`` /
``__float__`` convert directly; everything else falls back to
``int(str(value))`` / ``float(str(value))`` and raises ``ValueError`` when the
text is not numeric.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["as_float", "as_int"]


@runtime_checkable
class _SupportsInt(Protocol):
    """``__int__`` holder (``numbers.Real`` omits it from its interface)."""

    def __int__(self) -> int: ...


@runtime_checkable
class _SupportsFloat(Protocol):
    """``__float__`` holder (``numbers.Real`` omits it from its interface)."""

    def __float__(self) -> float: ...


def as_int(value: object) -> int:
    """Coerce a numeric cell to ``int`` (floats truncate, as ``int`` does)."""
    if isinstance(value, _SupportsInt):
        return int(value)
    return int(str(value))


def as_float(value: object) -> float:
    """Coerce a numeric cell to ``float``."""
    if isinstance(value, _SupportsFloat):
        return float(value)
    return float(str(value))
