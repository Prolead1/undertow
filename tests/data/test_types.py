"""Tests for ``undertow.data.types`` (T01) — enums, mixins, exception hierarchy, CheckResult.

Covers everything ``CONTRACTS.md`` §1 pins: the ``str``/``int`` mixin enums that must serialize
to Parquet without converters, ``TICK_SPACING`` per the §5.4 table, the full exception hierarchy,
and ``CheckResult`` (frozen, severity-typed, non-shared ``metrics`` default).
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from undertow.data.types import (
    TICK_SPACING,
    Address,
    BlockNumber,
    CheckResult,
    ConfigError,
    EventType,
    FeeTier,
    FetchError,
    PermanentFetchError,
    RateLimitError,
    Regime,
    ReorgDetectedError,
    Route,
    SchemaViolationError,
    Tick,
    TransientNetworkError,
    UndertowDataError,
    ValidationError,
)


def test_fee_tier_spacing_table_matches_protocol() -> None:
    expected = {
        FeeTier.BPS_1: 1,
        FeeTier.BPS_5: 10,
        FeeTier.BPS_30: 60,
        FeeTier.BPS_100: 200,
    }
    assert expected == TICK_SPACING
    # Every FeeTier member has a spacing entry and vice versa (no silent gaps).
    assert set(TICK_SPACING) == set(FeeTier)


def test_event_type_str_mixin() -> None:
    assert isinstance(EventType.SWAP, str)
    assert EventType.SWAP == "swap"
    assert EventType.SWAP.value == "swap"
    assert {e.value for e in EventType} == {"swap", "mint", "burn", "collect", "flash"}


def test_fee_tier_int_mixin() -> None:
    assert isinstance(FeeTier.BPS_30, int)
    assert FeeTier.BPS_30 == 3000
    assert FeeTier.BPS_30.value == 3000
    assert {e.value for e in FeeTier} == {100, 500, 3000, 10000}


def test_regime_members_and_values() -> None:
    assert {r.value for r in Regime} == {"bull", "bear", "sideways", "high_vol", "unknown"}
    assert isinstance(Regime.UNKNOWN, str)
    assert Regime.UNKNOWN == "unknown"


def test_route_members_and_values() -> None:
    assert {r.value for r in Route} == {"thegraph", "rpc", "bigquery", "reference"}
    assert isinstance(Route.RPC, str)


def test_newtype_aliases_are_identity_functions() -> None:
    assert Address("0xabc") == "0xabc"
    assert BlockNumber(5) == 5
    assert Tick(-42) == -42


def test_exception_hierarchy() -> None:
    assert issubclass(ConfigError, UndertowDataError)
    for exc in (FetchError, SchemaViolationError, ValidationError, ReorgDetectedError):
        assert issubclass(exc, UndertowDataError)
    for exc in (RateLimitError, TransientNetworkError, PermanentFetchError):
        assert issubclass(exc, FetchError)
    # Retryability split the fetcher base relies on (T04): retry the two, never the third.
    assert not issubclass(PermanentFetchError, (RateLimitError, TransientNetworkError))


def test_check_result_constructible_with_each_severity() -> None:
    cr = CheckResult(name="a", passed=True, severity="critical", detail="d")
    assert cr.severity == "critical"
    cr = CheckResult(name="b", passed=False, severity="warning", detail="d")
    assert cr.severity == "warning"
    cr = CheckResult(name="c", passed=True, severity="info", detail="d")
    assert cr.severity == "info"


def test_check_result_defaults_metrics_to_empty_map() -> None:
    cr = CheckResult(name="a", passed=True, severity="info", detail="d")
    assert cr.metrics == {}


def test_check_result_metrics_not_shared_between_instances() -> None:
    a = CheckResult(name="a", passed=True, severity="info", detail="d")
    b = CheckResult(name="b", passed=True, severity="info", detail="d")
    assert a.metrics is not b.metrics
    # default_factory gives each instance its own dict: mutating one must not leak into the other
    # (a shared literal default would make `a.metrics is b.metrics` True and both would change).
    a_metrics = cast(MutableMapping[str, object], a.metrics)
    a_metrics["rows"] = 3
    assert "rows" not in b.metrics
    assert a.metrics["rows"] == 3


def test_check_result_is_frozen() -> None:
    cr = CheckResult(name="a", passed=True, severity="info", detail="d")
    with pytest.raises(FrozenInstanceError):
        cr.name = "mutated"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        cr.passed = False  # type: ignore[misc]
