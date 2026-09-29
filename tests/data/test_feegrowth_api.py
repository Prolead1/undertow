"""Tests for the ADR-013 :class:`FeeGrowthReplay` public facade.

The facade is a thin, documented wrapper over the internal
``FeeGrowthTracker``.  These tests pin that it is exactly that — the same
arithmetic, the same provenance and the same checkpoint round-trip — and that
its public surface stays integer-only on the accrual path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from undertow.data import FeeGrowthReplay
from undertow.data.config import default_pools
from undertow.data.fixedpoint import get_amount1_delta, tick_to_sqrt_price_x96
from undertow.data.transforms.feegrowth import (
    FeeGrowthApproximation,
    FeeGrowthState,
    FeeGrowthTracker,
    PositionKey,
    TickState,
)
from undertow.data.types import Address, BlockNumber

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "feegrowth_replay.json"
POOL = default_pools()["USDC_WETH_3000"]  # fee 3000, spacing 60
Q96 = 2**96
_OWNER = Address("0x" + "aa" * 20)


def _load_fixture() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _initial_state() -> FeeGrowthState:
    init = _load_fixture()["initial_state"]
    assert isinstance(init, dict)
    return FeeGrowthState(
        block_number=BlockNumber(int(init["block_number"])),
        fee_growth_global_0_x128=int(init["fee_growth_global_0_x128"]),
        fee_growth_global_1_x128=int(init["fee_growth_global_1_x128"]),
        current_tick=int(init["current_tick"]),
        current_liquidity=int(init["current_liquidity"]),
        ticks={},
        current_sqrt_price_x96=int(init["current_sqrt_price_x96"]),
    )


def _internal_replay(tracker: FeeGrowthTracker, events: list[dict[str, object]]) -> None:
    """Drive the internal tracker exactly as ``FeeGrowthReplay.apply_event`` does."""
    for event in events:
        event_type = event["event_type"]
        if event_type == "swap":
            tracker.apply_swap(event)
        elif event_type in ("mint", "burn"):
            tracker.apply_liquidity_event(event)
        elif event_type in ("collect", "flash"):
            tracker.block_number = BlockNumber(int(event["block_number"]))
        else:  # pragma: no cover - fixture has only known event types
            raise AssertionError(event_type)


def test_public_facade_is_importable_from_top_level() -> None:
    import undertow.data as data

    assert data.FeeGrowthReplay is FeeGrowthReplay
    assert "FeeGrowthReplay" in data.__all__


def test_replay_of_fixture_matches_internal_tracker_exactly() -> None:
    """The facade and the internal tracker end the fixture replay byte-identical."""
    state = _initial_state()
    facade = FeeGrowthReplay.from_state(POOL, state)
    tracker = FeeGrowthTracker(POOL, state)

    events = _load_fixture()["events"]
    assert isinstance(events, list)
    for event in events:
        facade.apply_event(event)
    _internal_replay(tracker, events)

    assert facade.snapshot() == tracker.snapshot()


def test_accrue_matches_internal_tracker_and_returns_raw_q128_ints() -> None:
    """``accrue`` returns the tracker's raw Q128 integers, tuple-for-tuple."""
    state = _initial_state()
    facade = FeeGrowthReplay.from_state(POOL, state)
    tracker = FeeGrowthTracker(POOL, state)

    events = _load_fixture()["events"]
    assert isinstance(events, list)
    for event in events:
        facade.apply_event(event)
    _internal_replay(tracker, events)

    liquidity = 10**18
    fees0, fees1, g0, g1 = facade.accrue(
        tick_lower=-120,
        tick_upper=360,
        liquidity=liquidity,
        g_inside_last_0=0,
        g_inside_last_1=0,
    )
    expected = tracker.accrue(PositionKey(_OWNER, -120, 360), liquidity, 0, 0)

    assert (fees0, fees1, g0, g1) == (
        expected.fees0,
        expected.fees1,
        expected.g_inside_0_last,
        expected.g_inside_1_last,
    )
    # No float ever enters the accrual path.
    assert all(isinstance(value, int) for value in (fees0, fees1, g0, g1))
    assert not any(isinstance(value, float) for value in (fees0, fees1, g0, g1))


def test_snapshot_restore_roundtrip_via_facade() -> None:
    """A facade checkpoint resumes a second facade to the same final state."""
    events = _load_fixture()["events"]
    assert isinstance(events, list)
    state = _initial_state()

    full = FeeGrowthReplay.from_state(POOL, state)
    for event in events:
        full.apply_event(event)

    resumed = FeeGrowthReplay.from_state(POOL, state)
    for event in events[:10]:
        resumed.apply_event(event)
    checkpoint = resumed.snapshot()
    resumed.restore(checkpoint)
    for event in events[10:]:
        resumed.apply_event(event)

    assert resumed.snapshot() == full.snapshot()


def test_exact_flag_defaults_true_and_tracks_approximation() -> None:
    """``exact`` is the tracker's provenance flag, surfaced verbatim."""
    clean = FeeGrowthReplay.from_state(POOL, _initial_state())
    assert clean.exact is True

    # A mint spanning the current tick gives the crossing swap active liquidity.
    replay = FeeGrowthReplay(POOL, current_tick=0, current_liquidity=0)
    replay.apply_event(
        {
            "event_type": "mint",
            "block_number": 1,
            "tick_lower": 0,
            "tick_upper": 120,
            "liquidity_amount": str(3 * 10**18),
        }
    )
    net1 = get_amount1_delta(Q96, tick_to_sqrt_price_x96(120), 3 * 10**18, True)
    crossing_row = {
        "event_type": "swap",
        "block_number": 2,
        "amount0": "-1",
        "amount1": str(net1),
        "sqrt_price_x96": str(tick_to_sqrt_price_x96(120) + 1),
        "tick": 120,
    }
    # No tracked pre-swap price at replay start -> the crossing is approximated.
    with pytest.warns(FeeGrowthApproximation):
        replay.apply_event(crossing_row)
    assert replay.exact is False
    assert replay.snapshot().exact is False


def test_fee_pips_override_changes_the_fee_tier_exactly() -> None:
    """The override honours a sim ablation fee tier (e.g. 0) with the same pool."""
    row = {
        "event_type": "swap",
        "block_number": 1,
        "amount0": "1000000",
        "amount1": "-1",
        "sqrt_price_x96": str(Q96 - 1),
        "tick": 0,
    }
    zero_fee = FeeGrowthReplay(
        POOL, fee_pips=0, current_tick=0, current_liquidity=3 * 10**18
    )
    zero_fee.apply_event(row)
    assert zero_fee.snapshot().fee_growth_global_0_x128 == 0
    assert zero_fee.snapshot().fee_pips == 0

    # Round-trips through a checkpoint.
    resumed = FeeGrowthReplay.from_state(POOL, zero_fee.snapshot())
    assert resumed.snapshot().fee_pips == 0
    assert resumed.exact is True


def test_apply_event_unknown_type_raises_and_collect_is_a_noop() -> None:
    replay = FeeGrowthReplay(POOL)
    replay.apply_event({"event_type": "collect", "block_number": 7})
    assert replay.snapshot().block_number == 7
    assert replay.exact is True
    with pytest.raises(ValueError, match="unknown event_type"):
        replay.apply_event({"event_type": "teleport", "block_number": 8})


def test_facade_does_not_expose_the_mutable_tracker() -> None:
    """The facade never leaks ``FeeGrowthTracker``/``PositionKey`` attributes."""
    replay = FeeGrowthReplay(POOL)
    assert not hasattr(replay, "ticks")
    assert not hasattr(replay, "fee_growth_global_0_x128")
    assert isinstance(replay.snapshot(), FeeGrowthState)
    # A snapshot is a frozen value object, not a live tracker.
    assert replay.snapshot().ticks == {}


def test_register_position_seeds_boundaries_without_liquidity() -> None:
    """``register_position`` seeds absent boundaries and leaves existing ones alone."""
    existing = TickState(11, 13, 5, 2, True)
    state = FeeGrowthState(
        block_number=BlockNumber(1),
        fee_growth_global_0_x128=7,
        fee_growth_global_1_x128=9,
        current_tick=0,
        current_liquidity=0,
        ticks={60: existing},
    )
    replay = FeeGrowthReplay.from_state(POOL, state)
    replay.register_position(tick_lower=-120, tick_upper=60)

    snap = replay.snapshot()
    # at/below the current tick -> seeded from the current global; above -> 0.
    assert snap.ticks[-120] == TickState(7, 9, 1, 0, True)
    # an already-initialised tick is left untouched (no re-seeding).
    assert snap.ticks[60] == existing
    # the sentinel contributes no active liquidity and does not move the global.
    assert snap.current_liquidity == 0
    assert snap.fee_growth_global_0_x128 == 7

    # A fresh tick strictly above the current tick seeds a zero outside.
    replay.register_position(tick_lower=120, tick_upper=240)
    assert replay.snapshot().ticks[120] == TickState(0, 0, 1, 0, True)


def test_cross_tick_state_is_preserved_in_snapshot() -> None:
    """A tick seeded by a mint survives the public snapshot round-trip."""
    replay = FeeGrowthReplay(POOL, current_tick=0)
    replay.apply_event(
        {
            "event_type": "mint",
            "block_number": 1,
            "tick_lower": -120,
            "tick_upper": 120,
            "liquidity_amount": str(10**18),
        }
    )
    snap = replay.snapshot()
    assert snap.ticks[-120] == TickState(0, 0, 10**18, 10**18, True)
    resumed = FeeGrowthReplay.from_state(POOL, snap)
    assert resumed.snapshot() == snap
