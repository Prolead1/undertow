"""Shared scalar types, enums and the exception hierarchy for ``undertow.data``.

This module is the frozen cross-task contract (``CONTRACTS.md`` §1, T01). Nothing below may
rename, retype or reorder anything declared here; only extensions are permitted. It contains no
I/O and no math: every fetcher, transform and validator imports its primitives from here so the
modules stay decoupled.

``EventType``, ``FeeTier``, ``Regime`` and ``Route`` are ``str`` / ``int`` mixin enums
(`enum.StrEnum` / `int-Enum`) so their values serialize directly into Parquet string / int
columns.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Literal, NewType

Address = NewType("Address", str)
"""Lowercase ``0x``-prefixed, 42-char, checksum-stripped Ethereum address."""

BlockNumber = NewType("BlockNumber", int)
"""Ethereum block height."""

Tick = NewType("Tick", int)
"""Uniswap V3 tick index (raw; grid-alignment is applied by the caller)."""


class EventType(StrEnum):
    """The pool log streams. ``str`` mixin -> Parquet ``string`` column."""

    SWAP = "swap"
    MINT = "mint"
    BURN = "burn"
    COLLECT = "collect"
    FLASH = "flash"  # fetched; contributes fee growth, no liquidity change


class FeeTier(int, Enum):
    """Fee tiers in basis points; ``int`` mixin -> Parquet ``int`` column."""

    BPS_1 = 100  # 0.01%, tick spacing 1
    BPS_5 = 500  # 0.05%, tick spacing 10
    BPS_30 = 3000  # 0.30%, tick spacing 60
    BPS_100 = 10000  # 1.00%, tick spacing 200


TICK_SPACING: dict[FeeTier, int] = {
    FeeTier.BPS_1: 1,
    FeeTier.BPS_5: 10,
    FeeTier.BPS_30: 60,
    FeeTier.BPS_100: 200,
}


class Regime(StrEnum):
    """Regime labels produced by T12 from the reference feed (CONTRACTS.md §6.3)."""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"
    UNKNOWN = "unknown"  # window not yet full (first 30 days) — never silently dropped


class Route(StrEnum):
    """Data source routes. ``BIGQUERY`` is the escape hatch; not implemented in v1."""

    THEGRAPH = "thegraph"
    RPC = "rpc"
    BIGQUERY = "bigquery"
    REFERENCE = "reference"


Severity = Literal["critical", "warning", "info"]


@dataclass(frozen=True, slots=True)
class CheckResult:
    """A single validation/observation outcome.

    Defined HERE, in ``types.py``, not in ``validation/`` — on purpose: T08 (wave 3) returns one
    from ``compare_symbols``, but ``validation/`` is owned by T13 (wave 5). A result type shared
    across waves must be defined no later than the earliest wave that produces one. T13 imports
    it from here and must not redefine it.
    """

    name: str
    passed: bool
    severity: Severity
    detail: str
    metrics: Mapping[str, float | int | str] = field(default_factory=dict)


# Exception hierarchy — every fetcher and transform raises only these.
class UndertowDataError(Exception):
    """Base class for every error raised by ``undertow.data``."""


class ConfigError(UndertowDataError):
    """Invalid configuration. Carries the offending value in the ``args[0]`` message."""


class FetchError(UndertowDataError):
    """Base for all I/O fetch failures (transport, decoding, provider)."""


class RateLimitError(FetchError):
    """HTTP 429 / provider rate limit — retryable."""


class TransientNetworkError(FetchError):
    """Spurious transport failure — retryable."""


class PermanentFetchError(FetchError):
    """Not retryable (4xx other than 429, bad query)."""


class SchemaViolationError(UndertowDataError):
    """A table does not conform to its canonical schema. Carries the offending value."""


class ValidationError(UndertowDataError):
    """A data invariant failed."""


class ReorgDetectedError(UndertowDataError):
    """Block-chain reorganisation detected during a range fetch."""
