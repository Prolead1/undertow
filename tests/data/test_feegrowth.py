"""Tests for ``undertow.data.transforms.feegrowth`` (T10).

Covers the roadmap §10.2.4 equations (3)–(6) as exact integers, the 12
invariants of the T10 brief (worked example, below/above/full-range/branch-flip/
half-open/wrapping/cross-tick-involution/seeding/liquidity-bookkeeping/replay/
no-float), the ``FeeGrowthTracker`` state machine (single-segment exactness,
multi-tick re-simulation + apportionment, update-then-cross order, seeding,
uninitialization, snapshot/restore), and ``reconcile`` against synthetic
``FEE_GROWTH_SCHEMA`` rows.

Fixture provenance (``tests/data/fixtures/feegrowth_replay.json``): fully
synthetic — wave 2 has no real on-chain captures (T06, wave 3, supplies those).
Every amount/price in the fixture is mutually consistent with the price movement
and active liquidity, computed from the T02 fixedpoint ports. The expected final
state below is hand-computed: the global accumulators and per-tick outsides are
embedded as literal integers (computed from the documented fee formula and
crossing rules), and the liquidity/tick bookkeeping is asserted directly.
"""

from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path

import pyarrow as pa
import pytest

from undertow.data.config import GLOBAL_TICK_SENTINEL, default_pools
from undertow.data.fixedpoint import (
    MAX_TICK,
    MIN_TICK,
    get_amount0_delta,
    get_amount1_delta,
    tick_to_sqrt_price_x96,
    wrapping_sub_256,
)
from undertow.data.transforms.feegrowth import (
    FeeAccrual,
    FeeGrowthApproximation,
    FeeGrowthState,
    FeeGrowthTracker,
    Mismatch,
    PositionKey,
    ReconciliationReport,
    TickState,
    fee_growth_above,
    fee_growth_below,
    fee_growth_inside,
    uncollected_fees,
)

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "feegrowth_replay.json"
_MODULE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "undertow"
    / "data"
    / "transforms"
    / "feegrowth.py"
)

POOL = default_pools()["USDC_WETH_3000"]  # fee 3000, spacing 60
FEE_PIPS = 3000
Q96 = 2**96
R = {t: tick_to_sqrt_price_x96(t) for t in (-240, -120, -60, -30, 0, 60, 120, 180, 240, 360)}

# ---------------------------------------------------------------------------
# Hand-computed expected final state of the fixture replay (block 1020).
# The globals and outsides are literal integers derived from the documented fee
# formula (fee = gross * fee_pips // 1e6, apportioned per segment) and the
# crossing rules (outside = wrapping_sub(global, outside), update-then-cross).
# ---------------------------------------------------------------------------
FINAL_G0 = 15307496971940537013246697944853406
FINAL_G1 = 21570527055680178353739036062751114
FINAL_TICK_120_OUT0 = 12258842540418539868091613350300115
FINAL_TICK_120_OUT1 = 18475800433634734274731981885012551
FINAL_TICK_D_OUT0 = 12249683303091343464669235830341383  # ticks -240 and -120 (seeded at mint D)
FINAL_TICK_D_OUT1 = 18485070238359306158360069902313151
FINAL_TICK_360_OUT0 = 0
FINAL_TICK_360_OUT1 = 0


def _mul_div_rounding_up(a: int, b: int, d: int) -> int:
    """The protocol's per-step fee rounding (used only to build consistent rows)."""
    return (a * b + d - 1) // d


def _gross_for(net: int) -> int:
    """Gross input consistent with a net input: net + protocol fee (1e6 - fee denominator)."""
    return net + _mul_div_rounding_up(net, FEE_PIPS, 1_000_000 - FEE_PIPS)


def _tracker(
    *,
    current_tick: int = 0,
    current_liquidity: int = 0,
    g0: int = 0,
    g1: int = 0,
    ticks: dict[int, TickState] | None = None,
    price: int | None = Q96,
    block: int = 1000,
    exact: bool = True,
) -> FeeGrowthTracker:
    return FeeGrowthTracker(
        POOL,
        FeeGrowthState(
            block_number=block,
            fee_growth_global_0_x128=g0,
            fee_growth_global_1_x128=g1,
            current_tick=current_tick,
            current_liquidity=current_liquidity,
            ticks=ticks or {},
            current_sqrt_price_x96=price,
            exact=exact,
        ),
    )


def _swap_row(
    *,
    block: int,
    input_token: int,
    gross: int,
    out: int,
    post_tick: int,
    post_price: int,
) -> dict[str, object]:
    if input_token == 0:
        amount0, amount1 = gross, -out
    else:
        amount0, amount1 = -out, gross
    return {
        "block_number": block,
        "event_type": "swap",
        "amount0": str(amount0),
        "amount1": str(amount1),
        "sqrt_price_x96": str(post_price),
        "tick": post_tick,
    }


def _mint_row(*, block: int, tick_lower: int, tick_upper: int, liquidity: int) -> dict[str, object]:
    return {
        "block_number": block,
        "event_type": "mint",
        "tick_lower": tick_lower,
        "tick_upper": tick_upper,
        "liquidity_amount": str(liquidity),
    }


def _burn_row(*, block: int, tick_lower: int, tick_upper: int, liquidity: int) -> dict[str, object]:
    return {
        "block_number": block,
        "event_type": "burn",
        "tick_lower": tick_lower,
        "tick_upper": tick_upper,
        "liquidity_amount": str(liquidity),
    }


def _load_fixture() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _fixture_tracker() -> FeeGrowthTracker:
    fx = _load_fixture()
    init = fx["initial_state"]
    return _tracker(
        current_tick=init["current_tick"],
        current_liquidity=int(init["current_liquidity"]),
        g0=int(init["fee_growth_global_0_x128"]),
        g1=int(init["fee_growth_global_1_x128"]),
        price=int(init["current_sqrt_price_x96"]),
        block=init["block_number"],
    )


def _replay(tracker: FeeGrowthTracker, events: list[dict[str, object]]) -> None:
    for e in events:
        etype = e["event_type"]
        if etype == "swap":
            tracker.apply_swap(e)
        elif etype in ("mint", "burn"):
            tracker.apply_liquidity_event(e)
        # collect: no pool-state effect — the tracker has no method for it by design


# ---------------------------------------------------------------------------
# 1. The roadmap §10.2.4 worked example — exact integers.
# ---------------------------------------------------------------------------


def test_worked_example_exact_integers() -> None:
    """§10.2.4's printed example: g_global=1e36, g_out_low=0.4e36, g_out_up=0.2e36,
    L=1e20, g_last=0.1e36, ic=50, range [-200, 200]."""
    g_global, g_out_low, g_out_up = 10**36, 4 * 10**35, 2 * 10**35
    L, g_last = 10**20, 10**35
    ic, i_low, i_up = 50, -200, 200

    assert fee_growth_below(ic, i_low, g_global, g_out_low) == 4 * 10**35
    assert fee_growth_above(ic, i_up, g_global, g_out_up) == 2 * 10**35
    g_inside = fee_growth_inside(ic, i_low, i_up, g_global, g_out_low, g_out_up)
    assert g_inside == 4 * 10**35  # 1e36 - 0.4e36 - 0.2e36
    # eq (6): L * (0.4e36 - 0.1e36) >> 128 = 3e55 >> 128, exactly.
    assert uncollected_fees(L, g_inside, g_last) == (3 * 10**55) >> 128
    assert uncollected_fees(L, g_inside, g_last) == 88162076311671563


# ---------------------------------------------------------------------------
# 2–4. Below / above / full-range.
# ---------------------------------------------------------------------------


def test_position_below_current_tick_accrues_nothing_new() -> None:
    """Roadmap Q3: a range entirely below the current tick (tick_upper < ic) gets
    zero NEW interior growth as the global accumulator grows."""
    out_low, out_up = 3 * 10**35, 5 * 10**35
    before = fee_growth_inside(100, -200, -100, 10**36, out_low, out_up)
    after = fee_growth_inside(100, -200, -100, 2 * 10**36, out_low, out_up)
    assert before == after == 2 * 10**35  # out_up - out_low, independent of g_global


def test_position_above_current_tick_accrues_nothing_new() -> None:
    """Symmetric: a range entirely above the current tick (tick_lower > ic) also
    accrues nothing new as the global accumulator grows."""
    out_low, out_up = 7 * 10**35, 2 * 10**35
    before = fee_growth_inside(100, 200, 300, 10**36, out_low, out_up)
    after = fee_growth_inside(100, 200, 300, 3 * 10**36, out_low, out_up)
    assert before == after == 5 * 10**35  # out_low - out_up, independent of g_global


def test_full_range_inside_equals_global() -> None:
    """fee_growth_inside(MIN_TICK, MAX_TICK) with g_out = 0 on both sides equals
    fee_growth_global — including a wrapped global."""
    assert fee_growth_inside(50, MIN_TICK, MAX_TICK, 10**36, 0, 0) == 10**36
    wrapped = 2**256 - 5
    assert fee_growth_inside(50, MIN_TICK, MAX_TICK, wrapped, 0, 0) == wrapped


# ---------------------------------------------------------------------------
# 5–6. Branch flip and the half-open range convention.
# ---------------------------------------------------------------------------


def test_branch_flip_values() -> None:
    """With fixed g_out_*, each current-tick position takes the intended branch of
    eqs (4)/(5); assert the hand-computed value, not just that branches differ."""
    g, out_low, out_up = 10**36, 4 * 10**35, 2 * 10**35
    i_low, i_up = -200, 200
    # just below the lower bound: below = g - out_low, above = out_up
    assert fee_growth_inside(-201, i_low, i_up, g, out_low, out_up) == 2 * 10**35
    # at the lower bound (in range): below = out_low, above = out_up
    assert fee_growth_inside(-200, i_low, i_up, g, out_low, out_up) == 4 * 10**35
    # inside the range: same branches
    assert fee_growth_inside(0, i_low, i_up, g, out_low, out_up) == 4 * 10**35
    assert fee_growth_inside(199, i_low, i_up, g, out_low, out_up) == 4 * 10**35
    # at/above the upper bound: below = out_low, above = g - out_up; the interior
    # value wraps negative (out_up - out_low = -0.2e36) — the protocol's unchecked
    # arithmetic, asserted exactly.
    assert fee_growth_inside(200, i_low, i_up, g, out_low, out_up) == 2**256 - 2 * 10**35
    assert fee_growth_inside(201, i_low, i_up, g, out_low, out_up) == 2**256 - 2 * 10**35


def test_half_open_range_convention() -> None:
    """current_tick == tick_lower is in range; current_tick == tick_upper is out."""
    g, out_low, out_up = 10**36, 4 * 10**35, 2 * 10**35
    # at tick_lower: the below branch is the "ic >= i_low" side (out_low)
    assert fee_growth_below(-200, -200, g, out_low) == out_low
    assert fee_growth_inside(-200, -200, 200, g, out_low, out_up) == 4 * 10**35
    # at tick_upper: the above branch is the "ic >= i_up" side (g - out_up)
    assert fee_growth_above(200, 200, g, out_up) == g - out_up
    assert fee_growth_inside(200, -200, 200, g, out_low, out_up) == 2**256 - 2 * 10**35


# ---------------------------------------------------------------------------
# 7. Wrapping.
# ---------------------------------------------------------------------------


def test_uncollected_fees_real_wrap() -> None:
    """A genuinely wrapped accumulator: g_inside_last near 2**256 - k, g_inside_now
    a small value reached after the accumulator wrapped past 2**256. The true
    growth is (2**256 - last) + now = 1500, asserted exactly."""
    last = 2**256 - 1000
    now = 500
    L = 10**40
    assert wrapping_sub_256(now, last) == 1500
    assert uncollected_fees(L, now, last) == (L * 1500) >> 128
    # the naive reading "last > now implies wrap" is not what this tests: any
    # inverted pair wraps; this one is built from a real wrap past 2**256.
    assert (2**256 - last) + now == 1500


def test_pure_functions_wrap() -> None:
    """The eq (4)/(5) branches wrap through wrapping_sub_256 like the contract."""
    # ic < i_low: below = g - out_low, wrapping
    assert fee_growth_below(0, 100, 5, 2**256 - 3) == 8
    # ic >= i_up: above = g - out_up, wrapping
    assert fee_growth_above(200, 100, 5, 2**256 - 3) == 8
    # inside composes the two wraps
    assert fee_growth_inside(0, 100, 200, 5, 2**256 - 3, 2**256 - 7) == 4


# ---------------------------------------------------------------------------
# 8. Cross-tick involution.
# ---------------------------------------------------------------------------


def test_cross_tick_involution_no_growth() -> None:
    """Crossing a tick up then down with no growth in between returns
    fee_growth_outside to its original value and restores current_liquidity."""
    t = _tracker(
        current_liquidity=2 * 10**18, ticks={60: TickState(0, 0, 2 * 10**18, -10**18, True)}
    )
    t.cross_tick(60, True)
    assert t.ticks[60].fee_growth_outside_0_x128 == 0
    assert t.current_liquidity == 10**18
    t.cross_tick(60, False)
    assert t.ticks[60].fee_growth_outside_0_x128 == 0  # involution
    assert t.current_liquidity == 2 * 10**18


def test_cross_tick_not_involution_with_growth() -> None:
    """With growth in between, crossing up then down does NOT restore the outside."""
    t = _tracker(
        current_liquidity=2 * 10**18, ticks={60: TickState(0, 0, 2 * 10**18, -10**18, True)}
    )
    t.cross_tick(60, True)  # outside_0 = 0, liquidity = 1e18
    # grow the global with a single-segment swap (token0 in, price down to tick -30)
    net = get_amount0_delta(Q96, R[-30] + 1, 10**18, True)
    gross = _gross_for(net)
    t.apply_swap(
        _swap_row(
            block=1001, input_token=0, gross=gross, out=0, post_tick=-30, post_price=R[-30] + 1
        )
    )
    inc = ((gross * FEE_PIPS // 1_000_000) << 128) // 10**18
    assert t.fee_growth_global_0_x128 == inc
    t.cross_tick(60, False)  # outside_0 = inc - 0 = inc != original 0
    assert t.ticks[60].fee_growth_outside_0_x128 == inc
    assert t.ticks[60].fee_growth_outside_0_x128 != 0


def test_cross_tick_uninitialized_raises() -> None:
    t = _tracker()
    with pytest.raises(KeyError):
        t.cross_tick(60, True)


# ---------------------------------------------------------------------------
# 9. Tick seeding on Mint.
# ---------------------------------------------------------------------------


def test_tick_seeding_below_gets_global_above_gets_zero() -> None:
    """A tick initialized at/below current_tick gets g_out = g_global; above gets 0
    (whitepaper eq. 6.21). Then accruals from the position are correct from the
    next swap onward."""
    t = _tracker(g0=5 * 10**36, g1=7 * 10**36)
    t.apply_liquidity_event(
        _mint_row(block=1001, tick_lower=-120, tick_upper=120, liquidity=10**18)
    )
    assert t.ticks[-120] == TickState(5 * 10**36, 7 * 10**36, 10**18, 10**18, True)
    assert t.ticks[120] == TickState(0, 0, 10**18, -10**18, True)
    assert t.current_liquidity == 10**18

    key = PositionKey(owner="0x" + "aa" * 20, tick_lower=-120, tick_upper=120)
    # before any swap: the seeded outside cancels the pre-existing global, so the
    # position accrues nothing for growth that happened before it was minted.
    a0 = t.accrue(key, 10**18, 0, 0)
    assert a0.fees0 == 0 and a0.fees1 == 0
    assert a0.g_inside_0_last == 0 and a0.g_inside_1_last == 0

    # one swap later: the position accrues exactly the new growth.
    net = get_amount0_delta(Q96, R[-30] + 1, 10**18, True)
    gross = _gross_for(net)
    t.apply_swap(
        _swap_row(
            block=1002, input_token=0, gross=gross, out=0, post_tick=-30, post_price=R[-30] + 1
        )
    )
    inc = ((gross * FEE_PIPS // 1_000_000) << 128) // 10**18
    a1 = t.accrue(key, 10**18, 0, 0)
    assert a1.fees0 == (10**18 * inc) >> 128  # L * (global_after - 5e36) >> 128
    assert a1.fees1 == 0
    assert a1.g_inside_0_last == inc
    assert a1.exact is True


# ---------------------------------------------------------------------------
# 10. Liquidity bookkeeping.
# ---------------------------------------------------------------------------


def test_liquidity_bookkeeping_mint_burn() -> None:
    t = _tracker()
    t.apply_liquidity_event(
        _mint_row(block=1001, tick_lower=-120, tick_upper=120, liquidity=10**18)
    )
    assert t.current_liquidity == 10**18
    # a position that does not span the current tick changes nothing
    t.apply_liquidity_event(
        _mint_row(block=1002, tick_lower=120, tick_upper=240, liquidity=2 * 10**18)
    )
    assert t.current_liquidity == 10**18
    # the matching burn returns exactly to the prior value
    t.apply_liquidity_event(
        _burn_row(block=1003, tick_lower=-120, tick_upper=120, liquidity=10**18)
    )
    assert t.current_liquidity == 0
    # half-open: tick_lower == current_tick spans; tick_upper == current_tick does not
    t.apply_liquidity_event(
        _mint_row(block=1004, tick_lower=0, tick_upper=120, liquidity=3 * 10**18)
    )
    assert t.current_liquidity == 3 * 10**18
    t.apply_liquidity_event(
        _mint_row(block=1005, tick_lower=-120, tick_upper=0, liquidity=4 * 10**18)
    )
    assert t.current_liquidity == 3 * 10**18


def test_burn_uninitializes_tick() -> None:
    """A tick whose liquidity_gross returns to 0 is removed (Tick.update's clear)."""
    t = _tracker()
    t.apply_liquidity_event(
        _mint_row(block=1001, tick_lower=-120, tick_upper=120, liquidity=10**18)
    )
    assert set(t.ticks) == {-120, 120}
    t.apply_liquidity_event(
        _burn_row(block=1002, tick_lower=-120, tick_upper=120, liquidity=10**18)
    )
    assert t.ticks == {}
    # re-minting re-seeds the tick from the current global
    t.apply_liquidity_event(
        _mint_row(block=1003, tick_lower=-120, tick_upper=120, liquidity=10**18)
    )
    assert t.ticks[-120].fee_growth_outside_0_x128 == 0
    assert t.ticks[-120].liquidity_gross == 10**18


# ---------------------------------------------------------------------------
# Single-segment exactness and multi-tick apportionment.
# ---------------------------------------------------------------------------


def test_single_segment_swap_is_exact() -> None:
    """A swap crossing no initialized tick accrues (G * fee_pips // 1e6) << 128 // L
    on the input token, exactly — the pre-swap price does not enter."""
    t = _tracker(current_liquidity=3 * 10**18)
    net = get_amount0_delta(Q96, R[-30] + 1, 3 * 10**18, True)
    gross = _gross_for(net)
    t.apply_swap(
        _swap_row(
            block=1001, input_token=0, gross=gross, out=0, post_tick=-30, post_price=R[-30] + 1
        )
    )
    inc = ((gross * FEE_PIPS // 1_000_000) << 128) // (3 * 10**18)
    assert t.fee_growth_global_0_x128 == inc
    assert t.fee_growth_global_1_x128 == 0
    assert t.current_tick == -30
    assert t.current_liquidity == 3 * 10**18


def test_multi_tick_swap_apportionment_and_cross_order() -> None:
    """A swap crossing an initialized tick apportions the gross input across the
    two segments by their re-simulated net inputs, accrues per segment with the
    segment's liquidity, and crosses the tick AFTER its segment's fee — so the
    tick's outside sees the global including segment 1's fee but not segment 2's."""
    t = _tracker(
        current_liquidity=3 * 10**18,
        ticks={60: TickState(0, 0, 10**18, -10**18, True)},
    )
    # token1 in, 0 -> 120: segments [Q96, R60] L=3e18 and [R60, R120+1] L=2e18
    net1 = get_amount1_delta(Q96, R[60], 3 * 10**18, True)
    net2 = get_amount1_delta(R[60], R[120] + 1, 2 * 10**18, True)
    net_total = net1 + net2
    gross = _gross_for(net_total)
    out = get_amount0_delta(Q96, R[60], 3 * 10**18, False) + get_amount0_delta(
        R[60], R[120] + 1, 2 * 10**18, False
    )
    t.apply_swap(
        _swap_row(
            block=1001, input_token=1, gross=gross, out=out, post_tick=120, post_price=R[120] + 1
        )
    )

    gross1 = gross * net1 // net_total
    gross2 = gross - gross1
    fee1 = gross1 * FEE_PIPS // 1_000_000
    fee2 = gross2 * FEE_PIPS // 1_000_000
    inc1 = (fee1 << 128) // (3 * 10**18)
    inc2 = (fee2 << 128) // (2 * 10**18)
    assert t.fee_growth_global_1_x128 == inc1 + inc2
    assert t.fee_growth_global_0_x128 == 0
    # update-then-cross: the tick's outside is the global after segment 1 only
    assert t.ticks[60].fee_growth_outside_1_x128 == inc1
    assert t.ticks[60].fee_growth_outside_1_x128 != inc1 + inc2
    assert t.current_liquidity == 2 * 10**18
    assert t.current_tick == 120


# ---------------------------------------------------------------------------
# FeeGrowthApproximation provenance.
# ---------------------------------------------------------------------------


def test_crossing_swap_with_unknown_price_warns_and_flags_inexact() -> None:
    """A crossing swap at replay start (no tracked pre-swap price) raises the
    FeeGrowthApproximation warning and permanently sets exact=False."""
    t = _tracker(
        current_liquidity=3 * 10**18,
        ticks={60: TickState(0, 0, 10**18, -10**18, True)},
        price=None,
    )
    net1 = get_amount1_delta(Q96, R[60], 3 * 10**18, True)
    net2 = get_amount1_delta(R[60], R[120] + 1, 2 * 10**18, True)
    gross = _gross_for(net1 + net2)
    out = get_amount0_delta(Q96, R[60], 3 * 10**18, False) + get_amount0_delta(
        R[60], R[120] + 1, 2 * 10**18, False
    )
    row = _swap_row(
        block=1001, input_token=1, gross=gross, out=out, post_tick=120, post_price=R[120] + 1
    )
    with pytest.warns(FeeGrowthApproximation):
        t.apply_swap(row)
    assert t.snapshot().exact is False
    # every subsequent accrual reports exact=False — no path lies about provenance
    key = PositionKey(owner="0x" + "aa" * 20, tick_lower=-120, tick_upper=120)
    assert t.accrue(key, 10**18, 0, 0).exact is False


def test_single_segment_unknown_price_does_not_warn() -> None:
    """Single-segment swaps are exact even with an unknown pre-swap price."""
    t = _tracker(current_liquidity=3 * 10**18, price=None)
    net = get_amount0_delta(Q96, R[-30] + 1, 3 * 10**18, True)
    gross = _gross_for(net)
    row = _swap_row(
        block=1001, input_token=0, gross=gross, out=0, post_tick=-30, post_price=R[-30] + 1
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", FeeGrowthApproximation)
        t.apply_swap(row)
    assert t.snapshot().exact is True


def test_crossing_swap_with_known_price_does_not_warn() -> None:
    """With the pre-swap price tracked, a crossing swap is exact and silent."""
    t = _tracker(
        current_liquidity=3 * 10**18,
        ticks={60: TickState(0, 0, 10**18, -10**18, True)},
    )
    net1 = get_amount1_delta(Q96, R[60], 3 * 10**18, True)
    net2 = get_amount1_delta(R[60], R[120] + 1, 2 * 10**18, True)
    gross = _gross_for(net1 + net2)
    out = get_amount0_delta(Q96, R[60], 3 * 10**18, False) + get_amount0_delta(
        R[60], R[120] + 1, 2 * 10**18, False
    )
    row = _swap_row(
        block=1001, input_token=1, gross=gross, out=out, post_tick=120, post_price=R[120] + 1
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", FeeGrowthApproximation)
        t.apply_swap(row)
    assert t.snapshot().exact is True


# ---------------------------------------------------------------------------
# accrue returns the new snapshots; error paths.
# ---------------------------------------------------------------------------


def test_accrue_returns_new_snapshots() -> None:
    t = _tracker(current_liquidity=10**18)
    t.apply_liquidity_event(
        _mint_row(block=1001, tick_lower=-120, tick_upper=120, liquidity=10**18)
    )
    key = PositionKey(owner="0x" + "aa" * 20, tick_lower=-120, tick_upper=120)
    a = t.accrue(key, 10**18, 0, 0)
    assert isinstance(a, FeeAccrual)
    assert a.block_number == 1001
    assert a.g_inside_0_last == 0 and a.g_inside_1_last == 0
    # a second accrual with the returned snapshots yields zero new fees
    a2 = t.accrue(key, 10**18, a.g_inside_0_last, a.g_inside_1_last)
    assert a2.fees0 == 0 and a2.fees1 == 0


def test_apply_swap_rejects_inconsistent_rows() -> None:
    t = _tracker(current_liquidity=3 * 10**18)
    with pytest.raises(ValueError):
        t.apply_swap({"block_number": 1, "event_type": "swap", "amount0": "5", "amount1": "7",
                      "sqrt_price_x96": str(R[60] + 1), "tick": 60})  # same sign
    with pytest.raises(ValueError):
        t.apply_swap({"block_number": 1, "event_type": "swap", "amount0": "5", "amount1": "-7",
                      "sqrt_price_x96": str(R[60] + 1), "tick": 60})  # token0 in but price up


def test_apply_liquidity_event_rejects_non_mint_burn() -> None:
    t = _tracker()
    with pytest.raises(ValueError):
        t.apply_liquidity_event({"block_number": 1, "event_type": "collect"})


def test_zero_amount_burn_is_noop_even_on_uninitialized_ticks() -> None:
    """Real-chain finding (T13 real-data capture): the pool emits burn(0) events
    (fee collection) over arbitrarily wide tick ranges whose boundaries are often
    uninitialized. They move no liquidity, so they must be a no-op — a previous
    version crashed with KeyError deleting a tick that was never added."""
    t = _tracker(current_tick=198120, current_liquidity=3 * 10**18)
    before_ticks = dict(t.ticks)
    before_liq = t.current_liquidity
    t.apply_liquidity_event({
        "block_number": 1, "event_type": "burn",
        "tick_lower": 198480, "tick_upper": 212340,  # upper bound far from the
        "liquidity_amount": "0",                    # path, uninitialized in the tracker
    })
    assert t.current_liquidity == before_liq
    assert t.ticks == before_ticks
    assert t.block_number == 1


def test_apply_swap_allows_same_tick_no_direction_signal() -> None:
    """Real-chain finding (T13 real-data capture): a swap whose price moves but
    stays within the same tick (post_tick == pre_tick) is legitimate on-chain —
    it still accrues fees — and carries no tick-direction signal. It must be
    accepted for BOTH input tokens, not raise. Regression for the find that made
    reconcile_fees_against_collect fail on real data."""
    t = _tracker(current_tick=198000, current_liquidity=3 * 10**18)
    # a token0-IN swap that leaves the tick unchanged (price fell within the tick)
    t.apply_swap({
        "block_number": 1, "event_type": "swap", "amount0": "5", "amount1": "-7",
        "sqrt_price_x96": str(R[0] * 99 // 100), "tick": 198000,
    })
    # a token0-OUT swap that leaves the tick unchanged (price rose within the tick)
    t.apply_swap({
        "block_number": 2, "event_type": "swap", "amount0": "-5", "amount1": "7",
        "sqrt_price_x96": str(R[0] * 101 // 100), "tick": 198000,
    })


# ---------------------------------------------------------------------------
# 11. The ~20-event replay fixture — hand-computed expected state.
# ---------------------------------------------------------------------------


def test_replay_fixture_final_state() -> None:
    """Replaying the fixture's 20 events ends with the hand-computed expected
    state: literal global accumulators and per-tick outsides, and directly
    asserted liquidity/tick bookkeeping."""
    t = _fixture_tracker()
    _replay(t, _load_fixture()["events"])

    s = t.snapshot()
    assert s.block_number == 1020
    assert s.fee_growth_global_0_x128 == FINAL_G0
    assert s.fee_growth_global_1_x128 == FINAL_G1
    assert s.current_tick == 120
    assert s.current_liquidity == 3 * 10**18
    assert s.exact is True

    # tick bookkeeping: gross/net are directly hand-computable from the mints/burns
    assert s.ticks[-240] == TickState(
        FINAL_TICK_D_OUT0, FINAL_TICK_D_OUT1, 4 * 10**18, 4 * 10**18, True
    )
    assert s.ticks[-120] == TickState(
        FINAL_TICK_D_OUT0, FINAL_TICK_D_OUT1, 4 * 10**18, -4 * 10**18, True
    )
    assert s.ticks[120] == TickState(
        FINAL_TICK_120_OUT0, FINAL_TICK_120_OUT1, 3 * 10**18, 3 * 10**18, True
    )
    assert s.ticks[360] == TickState(
        FINAL_TICK_360_OUT0, FINAL_TICK_360_OUT1, 3 * 10**18, -3 * 10**18, True
    )
    # burned positions' ticks are gone; only the four live boundaries remain
    assert set(s.ticks) == {-240, -120, 120, 360}


def test_replay_fixture_intermediate_state_after_swap2() -> None:
    """After swap 2 (the first crossing swap) the global and the crossed tick's
    outside are the hand-computed values — this pins the apportionment and the
    update-then-cross order inside the replay."""
    t = _fixture_tracker()
    events = _load_fixture()["events"]
    _replay(t, events[:4])  # mints A/B/C + swap 1

    # swap 1: token1 in, single segment, L=3e18
    e1 = events[3]
    g1_swap1 = ((int(e1["amount1"]) * FEE_PIPS // 1_000_000) << 128) // (3 * 10**18)
    assert t.fee_growth_global_1_x128 == g1_swap1
    assert t.fee_growth_global_0_x128 == 0

    # swap 2: token1 in, 60 -> 120, crosses tick 120; segments L=3e18 then L=5e18
    t.apply_swap(events[4])
    e2 = events[4]
    pre_price = int(events[3]["sqrt_price_x96"])
    net1 = get_amount1_delta(pre_price, R[120], 3 * 10**18, True)
    net2 = get_amount1_delta(R[120], int(e2["sqrt_price_x96"]), 5 * 10**18, True)
    net_total = net1 + net2
    gross = int(e2["amount1"])
    gross1 = gross * net1 // net_total
    gross2 = gross - gross1
    inc1 = ((gross1 * FEE_PIPS // 1_000_000) << 128) // (3 * 10**18)
    inc2 = ((gross2 * FEE_PIPS // 1_000_000) << 128) // (5 * 10**18)
    assert t.fee_growth_global_1_x128 == g1_swap1 + inc1 + inc2
    assert t.ticks[120].fee_growth_outside_1_x128 == g1_swap1 + inc1  # after segment 1 only
    assert t.current_liquidity == 5 * 10**18
    assert t.current_tick == 120


def test_reconcile_exact_against_fixture_observed() -> None:
    """reconcile against the fixture's synthetic FEE_GROWTH_SCHEMA rows (filled with
    the hand-computed final values) reports all-exact: 4 global fields + 4 ticks x
    5 fields = 24 compared."""
    t = _fixture_tracker()
    _replay(t, _load_fixture()["events"])
    observed = _observed_table(_final_observed_rows())
    report = t.reconcile(observed)
    assert isinstance(report, ReconciliationReport)
    assert report.n_compared == 24
    assert report.n_exact == 24
    assert report.max_abs_delta_g0 == 0
    assert report.max_abs_delta_g1 == 0
    assert report.max_rel_delta == 0.0
    assert report.mismatches == ()


def test_reconcile_reports_mismatches() -> None:
    """A wrong observed value is reported (never raised) with exact abs/rel deltas."""
    t = _fixture_tracker()
    _replay(t, _load_fixture()["events"])
    rows = _final_observed_rows()
    # corrupt tick 120's outside_0 by +1
    for r in rows:
        if r["tick"] == 120:
            r["fee_growth_outside_0_x128"] = str(FINAL_TICK_120_OUT0 + 1)
    report = t.reconcile(_observed_table(rows))
    assert report.n_compared == 24
    assert report.n_exact == 23
    assert report.max_abs_delta_g0 == 1
    assert report.max_abs_delta_g1 == 0
    assert report.max_rel_delta == pytest.approx(1 / FINAL_TICK_120_OUT0)
    assert len(report.mismatches) == 1
    m = report.mismatches[0]
    assert isinstance(m, Mismatch)
    assert m.block_number == 1020
    assert m.tick == 120
    assert m.field_name == "fee_growth_outside_0_x128"
    assert m.replayed == FINAL_TICK_120_OUT0
    assert m.observed == FINAL_TICK_120_OUT0 + 1
    assert m.abs_delta == 1


def test_reconcile_rejects_wrong_block() -> None:
    """A caller that passes observed rows for a block the tracker is not at gets a
    clear error, never a silently inflated comparison (reviewer finding)."""
    t = _fixture_tracker()
    _replay(t, _load_fixture()["events"])
    rows = _final_observed_rows()
    for r in rows:
        r["block_number"] = 999  # wrong block
    with pytest.raises(ValueError, match="position the tracker"):
        t.reconcile(_observed_table(rows))


def test_reconcile_reports_global_mismatch() -> None:
    """A global-accumulator mismatch is reported with tick=None."""
    t = _fixture_tracker()
    _replay(t, _load_fixture()["events"])
    rows = _final_observed_rows()
    for r in rows:
        if r["tick"] == GLOBAL_TICK_SENTINEL:
            r["fee_growth_global_0_x128"] = str(FINAL_G0 + 7)
    report = t.reconcile(_observed_table(rows))
    assert report.n_exact == 23
    assert report.max_abs_delta_g0 == 7
    assert report.mismatches[0].tick is None
    assert report.mismatches[0].field_name == "fee_growth_global_0_x128"


# ---------------------------------------------------------------------------
# snapshot / restore.
# ---------------------------------------------------------------------------


def test_snapshot_restore_roundtrip() -> None:
    """A checkpoint taken mid-replay restores into a fresh tracker that finishes
    the replay identically to the uninterrupted one."""
    events = _load_fixture()["events"]
    full = _fixture_tracker()
    _replay(full, events)

    resumed = _fixture_tracker()
    _replay(resumed, events[:10])
    checkpoint = resumed.snapshot()
    fresh = FeeGrowthTracker(POOL, checkpoint)
    _replay(fresh, events[10:])

    assert fresh.snapshot() == full.snapshot()
    # the checkpoint carries the price and the provenance flag
    assert checkpoint.current_sqrt_price_x96 is not None
    assert checkpoint.exact is True


def test_restore_method() -> None:
    t = _fixture_tracker()
    _replay(t, _load_fixture()["events"][:5])
    checkpoint = t.snapshot()
    t.restore(checkpoint)
    assert t.snapshot() == checkpoint


# ---------------------------------------------------------------------------
# 12. No float anywhere on the accrual path.
# ---------------------------------------------------------------------------


def test_no_float_guard_ast() -> None:
    """Blunt invariant: feegrowth.py contains no float literal and no float() call
    (same guard style as T02). The only floats in the module are the relative-
    delta diagnostics of ReconciliationReport, which are not on the accrual path."""
    tree = ast.parse(_MODULE.read_text(encoding="utf-8"))
    float_literal: list[ast.AST] = []
    float_calls: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            float_literal.append(node)
        if isinstance(node, ast.Call):
            fn = node.func
            if (isinstance(fn, ast.Name) and fn.id == "float") or (
                isinstance(fn, ast.Attribute) and "float" in fn.attr
            ):
                float_calls.append(node)
    assert not float_literal, f"float literals present: {len(float_literal)}"
    assert not float_calls, f"float(...) calls present: {len(float_calls)}"


# ---------------------------------------------------------------------------
# Helpers for the reconcile observed table (FEE_GROWTH_SCHEMA shape, §4.4).
# ---------------------------------------------------------------------------


def _final_observed_rows() -> list[dict[str, object]]:
    """The fixture's reconcile_observed rows with the hand-computed final values
    filled into the PLACEHOLDER fields."""
    fx = _load_fixture()
    rows = list(fx["reconcile_observed"])
    for r in rows:
        r["fee_growth_global_0_x128"] = str(FINAL_G0)
        r["fee_growth_global_1_x128"] = str(FINAL_G1)
        if r["tick"] != GLOBAL_TICK_SENTINEL:
            if r["tick"] in (-240, -120):
                r["fee_growth_outside_0_x128"] = str(FINAL_TICK_D_OUT0)
                r["fee_growth_outside_1_x128"] = str(FINAL_TICK_D_OUT1)
            elif r["tick"] == 120:
                r["fee_growth_outside_0_x128"] = str(FINAL_TICK_120_OUT0)
                r["fee_growth_outside_1_x128"] = str(FINAL_TICK_120_OUT1)
            elif r["tick"] == 360:
                r["fee_growth_outside_0_x128"] = str(FINAL_TICK_360_OUT0)
                r["fee_growth_outside_1_x128"] = str(FINAL_TICK_360_OUT1)
    return rows


def _observed_table(rows: list[dict[str, object]]) -> pa.Table:
    """Build a FEE_GROWTH_SCHEMA-shaped pyarrow table from row dicts."""
    cols = {
        "block_number": pa.array([int(r["block_number"]) for r in rows], pa.int64()),
        "tick": pa.array([int(r["tick"]) for r in rows], pa.int32()),
        "fee_growth_outside_0_x128": pa.array(
            [
                None
                if r["fee_growth_outside_0_x128"] is None
                else str(r["fee_growth_outside_0_x128"])
                for r in rows
            ],
            pa.string(),
        ),
        "fee_growth_outside_1_x128": pa.array(
            [
                None
                if r["fee_growth_outside_1_x128"] is None
                else str(r["fee_growth_outside_1_x128"])
                for r in rows
            ],
            pa.string(),
        ),
        "liquidity_gross": pa.array(
            [None if r["liquidity_gross"] is None else str(r["liquidity_gross"]) for r in rows],
            pa.string(),
        ),
        "liquidity_net": pa.array(
            [None if r["liquidity_net"] is None else str(r["liquidity_net"]) for r in rows],
            pa.string(),
        ),
        "initialized": pa.array(
            [None if r["initialized"] is None else bool(r["initialized"]) for r in rows], pa.bool_()
        ),
        "fee_growth_global_0_x128": pa.array(
            [str(r["fee_growth_global_0_x128"]) for r in rows], pa.string()
        ),
        "fee_growth_global_1_x128": pa.array(
            [str(r["fee_growth_global_1_x128"]) for r in rows], pa.string()
        ),
        "current_tick": pa.array([int(r["current_tick"]) for r in rows], pa.int32()),
        "current_liquidity": pa.array([str(r["current_liquidity"]) for r in rows], pa.string()),
        "source": pa.array([str(r["source"]) for r in rows], pa.string()),
        "pool_address": pa.array([str(r["pool_address"]) for r in rows], pa.string()),
    }
    return pa.table(cols)
