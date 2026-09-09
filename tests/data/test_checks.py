"""Tests for ``undertow.data.validation.checks`` (T13) — single-stream invariants.

Every check is exercised **both ways**: a passing case on clean data and a
failing case on deliberately corrupted data with the correct severity and a
detail that names the offending row/column — a check that has never been seen to
fire is not known to work (T13 brief §Tests).

The dirty cases are hand-built tables, not mutations of shared fixtures: minimal
repro does not risk the fixture's other consumers. The shared ``tiny_dataset``
is used raw only where it is clean (its out-of-range position is grid-aligned by
this task's conftest fix; its reference deliberately carries a >2% gap-fill
share, so that check's passing case uses a de-filled copy).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pyarrow as pa  # type: ignore[import-untyped]

from tests.data.conftest import POOL, build_tiny_dataset
from undertow.data.config import WindowConfig
from undertow.data.schemas import (
    EVENT_TAPE_SCHEMA,
    FEE_GROWTH_SCHEMA,
    MINT_SCHEMA,
    REFERENCE_SCHEMA,
    REGIME_SCHEMA,
    SWAP_SCHEMA,
)
from undertow.data.transforms.align import build_event_tape
from undertow.data.types import BlockNumber
from undertow.data.validation.checks import (
    CHECK_SEVERITIES,
    check_block_ordering,
    check_key_uniqueness,
    check_liquidity_conservation,
    check_monotonic_timestamps,
    check_no_block_gaps,
    check_no_future_data,
    check_reference_coverage,
    check_regime_labels_complete,
    check_swap_sign_convention,
    check_tick_price_consistency,
    check_ticks_on_spacing_grid,
)

WINDOW = WindowConfig(start_block=BlockNumber(17_000_000), end_block=BlockNumber(17_000_062))
_BASE_TS = datetime(2023, 5, 31, 23, 58, 0, tzinfo=UTC)


def _table(schema: pa.Schema, rows: list[dict[str, object]]) -> pa.Table:
    """Schema-valid table from partial row dicts (missing keys -> null), the same
    trick the shared conftest uses."""
    arrays = {f.name: pa.array([r.get(f.name) for r in rows], type=f.type) for f in schema}
    return pa.table(arrays, schema=schema)


def _rows(table: pa.Table) -> list[dict[str, object]]:
    cols = table.column_names
    return [{c: table.column(c)[i].as_py() for c in cols} for i in range(table.num_rows)]


def _tiny_tape() -> pa.Table:
    data = build_tiny_dataset()
    return build_event_tape(
        data.streams,
        pool=data.pool,
        gas=data.gas,
        reference=data.reference,
        regime=data.regime,
        fee_growth=data.fee_growth,
    )


def _swap_row(block: int, log: int, a0: int, a1: int, tick: int) -> dict[str, object]:
    return {
        "block_number": block,
        "log_index": log,
        "block_timestamp": _BASE_TS,
        "tx_hash": f"0x{block:064x}",
        "pool_address": POOL.address,
        "event_type": "swap",
        "amount0": str(a0),
        "amount1": str(a1),
        "sqrt_price_x96": str(2**96),
        "liquidity": "1000000000000000000",
        "tick": tick,
        "sender": "0x" + "e" * 40,
        "recipient": "0x" + "f" * 40,
    }


def _fg_global(block: int, liq: str, current_tick: int) -> dict[str, object]:
    return {
        "block_number": block,
        "tick": -(2**31),
        "fee_growth_outside_0_x128": None,
        "fee_growth_outside_1_x128": None,
        "liquidity_gross": liq,
        "liquidity_net": "0",
        "initialized": True,
        "fee_growth_global_0_x128": "0",
        "fee_growth_global_1_x128": "0",
        "current_tick": current_tick,
        "current_liquidity": liq,
        "source": "rpc_call",
        "pool_address": POOL.address,
        "fee_protocol": 0,
    }


def _mint_row(block: int, owner: str, lower: int, upper: int, liq: str) -> dict[str, object]:
    return {
        "block_number": block,
        "log_index": 0,
        "block_timestamp": _BASE_TS,
        "tx_hash": f"0x{block:064x}",
        "pool_address": POOL.address,
        "event_type": "mint",
        "owner": owner,
        "tick_lower": lower,
        "tick_upper": upper,
        "liquidity_amount": liq,
        "amount0": "0",
        "amount1": "0",
        "sender": None,
    }


def _swap_liq_row(block: int, liq: str, tick: int) -> dict[str, object]:
    return {
        **_swap_row(block, 0, -1, 1, tick),
        "liquidity": liq,
    }


# ---------------------------------------------------------------------------
# severity pin (the quality gate: a downgrade breaks this test)
# ---------------------------------------------------------------------------


def test_check_severities_are_pinned() -> None:
    """The severity registry is frozen; a later edit that quietly downgrades a
    Critical to a warning breaks this test."""
    assert CHECK_SEVERITIES == {
        "check_block_ordering": "critical",
        "check_key_uniqueness": "critical",
        "check_monotonic_timestamps": "critical",
        "check_no_block_gaps": "critical",
        "check_no_future_data": "warning",
        "check_swap_sign_convention": "critical",
        "check_tick_price_consistency": "critical",
        "check_ticks_on_spacing_grid": "critical",
        "check_liquidity_conservation": "warning",
        "check_reference_coverage": "warning",
        "check_regime_labels_complete": "warning",
    }


# ---------------------------------------------------------------------------
# tape-level invariants
# ---------------------------------------------------------------------------


def test_check_block_ordering_passes_on_clean_and_fails_out_of_order() -> None:
    assert check_block_ordering(_tiny_tape()).passed

    bad = _table(
        EVENT_TAPE_SCHEMA,
        [
            _swap_row(3, 1, 1, -1, 0),
            _swap_row(3, 0, 1, -1, 0),  # not strictly increasing
        ],
    )
    res = check_block_ordering(bad)
    assert not res.passed
    assert res.severity == "critical"
    assert res.metrics["violations"] == 1
    assert "(3, 0)" in res.detail


def test_check_key_uniqueness_passes_and_fails_on_duplicate_key() -> None:
    assert check_key_uniqueness(_tiny_tape()).passed

    bad = _table(
        EVENT_TAPE_SCHEMA,
        [
            _swap_row(7, 3, 5, -5, 10),
            _swap_row(7, 3, 5, -5, 10),  # duplicate key
        ],
    )
    res = check_key_uniqueness(bad)
    assert not res.passed
    assert res.severity == "critical"
    assert res.metrics["duplicates"] == 1
    assert "must not be silently dropped" in res.detail


def test_check_monotonic_timestamps_passes_and_fails_on_regression() -> None:
    assert check_monotonic_timestamps(_tiny_tape()).passed

    rows = [_swap_row(b, 0, 1, -1, 0) for b in (10, 11, 12)]
    rows[2]["block_timestamp"] = datetime(2020, 1, 1, tzinfo=UTC)  # older than block 11
    res = check_monotonic_timestamps(_table(EVENT_TAPE_SCHEMA, rows))
    assert not res.passed
    assert res.severity == "critical"
    assert res.metrics["violations"] == 1


def test_check_no_future_data_passes_and_fails_beyond_end_block() -> None:
    ok = check_no_future_data(_tiny_tape(), WINDOW)
    assert ok.passed
    assert ok.severity == "warning"

    res = check_no_future_data(
        _table(EVENT_TAPE_SCHEMA, [_swap_row(17_000_999, 0, 1, -1, 0)]), WINDOW
    )
    assert not res.passed
    assert res.severity == "warning"
    assert res.metrics["rows_beyond"] == 1


# ---------------------------------------------------------------------------
# stream-content invariants
# ---------------------------------------------------------------------------


def test_check_no_block_gaps_passes_and_fails_on_hole() -> None:
    data = build_tiny_dataset()
    assert check_no_block_gaps(data.gas, WINDOW).passed

    missing = _table(
        data.gas.schema,
        [r for r in _rows(data.gas) if r["block_number"] != 17_000_037],
    )
    res = check_no_block_gaps(missing, WINDOW)
    assert not res.passed
    assert res.severity == "critical"
    assert res.metrics["gaps"] == 1
    assert "silently dropped" in res.detail


def test_check_swap_sign_convention_passes_and_fails_same_sign() -> None:
    rows = [r for r in _rows(_tiny_tape()) if r["event_type"] == "swap"]
    swaps = _table(SWAP_SCHEMA, rows)
    ok = check_swap_sign_convention(swaps)
    assert ok.passed
    assert ok.severity == "critical"

    broken = _table(
        SWAP_SCHEMA, [dict(r, amount0="123456789", amount1="987654321") for r in rows[:2]]
    )
    res = check_swap_sign_convention(broken)
    assert not res.passed
    assert res.severity == "critical"
    assert res.metrics["violations"] == 2
    assert "tx_hash" in res.detail


def test_check_tick_price_consistency_passes_and_fails_beyond_tolerance() -> None:
    rows = [r for r in _rows(_tiny_tape()) if r["event_type"] == "swap"]
    ok = check_tick_price_consistency(_table(SWAP_SCHEMA, rows), POOL)
    assert ok.passed, ok.detail
    assert ok.severity == "critical"
    assert "distribution" in ok.detail

    # ±1 is the documented protocol allowance (a downward crossing writes
    # slot0.tick = tickNext - 1): perturbing by exactly 1 must still pass AND be
    # visible in the reported distribution (the tolerance cannot hide bugs).
    shifted = _table(
        SWAP_SCHEMA,
        [dict(r, tick=int(r["tick"]) + 1) for r in rows[:5]] + rows[5:],
    )
    res1 = check_tick_price_consistency(shifted, POOL)
    assert res1.passed, res1.detail
    assert res1.metrics.get("dev1", 0) == 5

    # |deviation| > 1 is Critical
    bad = _table(
        SWAP_SCHEMA,
        [dict(r, tick=int(r["tick"]) + 2) for r in rows[:3]] + rows[3:],
    )
    res = check_tick_price_consistency(bad, POOL)
    assert not res.passed
    assert res.severity == "critical"
    assert res.metrics["violating_rows"] == 3
    assert "distribution" in res.detail


def test_check_ticks_on_spacing_grid_passes_and_fails_off_grid() -> None:
    data = build_tiny_dataset()
    ok = check_ticks_on_spacing_grid(data.streams["mint"], data.streams["burn"], POOL)
    assert ok.passed, ok.detail
    assert ok.severity == "critical"

    # the brief's own failing case: tick_lower=61 on a spacing-60 pool
    bad = _table(
        MINT_SCHEMA,
        [dict(_rows(data.streams["mint"])[0], tick_lower=61)] + _rows(data.streams["mint"])[1:],
    )
    res = check_ticks_on_spacing_grid(bad, None, POOL)
    assert not res.passed
    assert res.severity == "critical"
    assert "tick_lower=61" in res.detail


# ---------------------------------------------------------------------------
# liquidity conservation (delta-based; pass = exact)
# ---------------------------------------------------------------------------


def test_check_liquidity_conservation_exact_on_delta_only_data() -> None:
    """Seed 5e18 at tick 0; +2e18 minted in range; a no-crossing swap; +1e18 more;
    a crossing swap that removes all our L. Deviation is exactly 0: the check
    proves T06-style decoding and the crossing bookkeeping agree."""
    fg = _table(FEE_GROWTH_SCHEMA, [_fg_global(1000, "5000000000000000000", 0)])
    mints = _table(
        MINT_SCHEMA,
        [
            _mint_row(1001, "0x" + "1" * 40, -60, 60, "2000000000000000000"),
            _mint_row(1003, "0x" + "1" * 40, -60, 60, "1000000000000000000"),
        ],
    )
    swaps = _table(
        SWAP_SCHEMA,
        [
            _swap_liq_row(1002, "7000000000000000000", -30),  # 5 + 2
            _swap_liq_row(1004, "5000000000000000000", -60),  # crossed -60: our 3e18 leaves
        ],
    )
    res = check_liquidity_conservation(swaps, mints, None, fg, start_block=1000)
    assert res.passed, res.detail
    assert res.severity == "warning"
    assert res.metrics["max_abs_deviation"] == 0


def test_check_liquidity_conservation_flags_drift_at_its_block() -> None:
    """Same shape with ONE corrupted reported liquidity (6e18 instead of 5e18)
    reports warning, the drift block and the exact deviation — the diagnostic the
    brief demands."""
    fg = _table(FEE_GROWTH_SCHEMA, [_fg_global(1000, "5000000000000000000", 0)])
    mints = _table(
        MINT_SCHEMA,
        [_mint_row(1001, "0x" + "1" * 40, -60, 60, "2000000000000000000")],
    )
    swaps = _table(
        SWAP_SCHEMA,
        [_swap_liq_row(1004, "6000000000000000000", -60)],  # corrupted: should be 5e18
    )
    res = check_liquidity_conservation(swaps, mints, None, fg, start_block=1000)
    assert not res.passed
    assert res.severity == "warning"
    assert res.metrics["first_drift_block"] == 1004
    assert res.metrics["max_abs_deviation"] == 10**18


def test_check_liquidity_conservation_skips_without_snapshot() -> None:
    empty = _table(FEE_GROWTH_SCHEMA, [])
    res = check_liquidity_conservation(_tiny_tape().slice(0, 0), None, None, empty, 1000)
    assert res.passed and res.severity == "info"
    assert "skipped" in res.detail


# ---------------------------------------------------------------------------
# coverage-quality checks
# ---------------------------------------------------------------------------


def test_check_reference_coverage_passes_and_fails_on_gap_share() -> None:
    data = build_tiny_dataset()
    # the shared fixture's reference is >2% gap-filled BY DESIGN (an exchange
    # outage) — that reports the warning; the PASSING case uses a de-filled copy.
    de_filled = _table(
        REFERENCE_SCHEMA,
        [dict(r, is_gap_filled=False) for r in _rows(data.reference)],
    )
    ok = check_reference_coverage(de_filled, WINDOW)
    assert ok.passed
    assert ok.severity == "warning"

    res = check_reference_coverage(data.reference, WINDOW)
    assert not res.passed
    assert res.severity == "warning"
    assert res.metrics["gap_filled"] == 2
    assert "gap-fill share" in res.detail


def test_check_regime_labels_complete_passes_and_fails() -> None:
    data = build_tiny_dataset()
    ok = check_regime_labels_complete(data.regime)
    assert ok.passed
    assert ok.severity == "warning"

    broken = _table(
        REGIME_SCHEMA,
        [dict(r, regime="unknown", window_complete=True) for r in _rows(data.regime)[3:]],
    )
    res = check_regime_labels_complete(broken)
    assert not res.passed
    assert res.severity == "warning"
    assert "outside the warmup" in res.detail


def test_check_regime_labels_complete_warns_on_empty_bin() -> None:
    data = build_tiny_dataset()
    # force every real label to "bull": three bins never appear
    squeezed = _table(
        REGIME_SCHEMA,
        [
            dict(r, regime=("bull" if r["regime"] != "unknown" else "unknown"))
            for r in _rows(data.regime)
        ],
    )
    res = check_regime_labels_complete(squeezed)
    assert not res.passed
    assert "never observed" in res.detail
