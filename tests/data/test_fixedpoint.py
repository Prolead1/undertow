"""Tests for ``undertow.data.fixedpoint`` (T02).

Covers the exact integer ports of Uniswap V3 ``TickMath`` / ``SqrtPriceMath`` /
``LiquidityAmounts``, the decimal price orientation of CONTRACTS.md §3.0, the
wrapping arithmetic T10 depends on, and the no-float invariant.

Fixture provenance (``tests/data/fixtures/tickmath_vectors.json``):
- route 1 (v3-core ``test/TickMath.spec.ts``, bytes-verified): ``MAX_TICK-1``.
- route 2 (verbatim ``TickMath.sol`` constants / analytic): ``MIN_TICK``
  (``MIN_SQRT_RATIO``), ``MAX_TICK`` (``MAX_SQRT_RATIO``), ``0`` (``2**96``).
- route 3 (independent 60-digit ``Decimal`` ideal, to ≤2 units for |t| ≤ 200k,
  tight relative bound otherwise): all remaining mid-range vectors.
"""

from __future__ import annotations

import ast
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from undertow.data.fixedpoint import (
    MAX_SQRT_RATIO,
    MAX_TICK,
    MAX_UINT256,
    MIN_SQRT_RATIO,
    MIN_TICK,
    Q128,
    align_tick_down,
    align_tick_up,
    amounts_for_liquidity,
    get_amount0_delta,
    get_amount1_delta,
    liquidity_for_amounts,
    price_to_sqrt_price_x96,
    price_to_tick,
    q128_to_decimal,
    sqrt_price_x96_to_price,
    sqrt_price_x96_to_tick,
    tick_to_price,
    tick_to_sqrt_price_x96,
    wrapping_add_256,
    wrapping_sub_256,
)

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "tickmath_vectors.json"
_MODULE = Path(__file__).resolve().parents[2] / "src" / "undertow" / "data" / "fixedpoint.py"

# The CONTRACTS.md §3.0 worked-check anchor: sqrt_price_x96 giving ETH == $3000 exactly.
ANCHOR_3000 = 1446501726624926496477173928747177


def _vectors() -> list[dict[str, object]]:
    import json

    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _fixture_ticks() -> list[int]:
    return [int(v["tick"]) for v in _vectors()]


# A 27-element sweep spanning the full domain plus fine mid-range ticks.
SWEEP = [
    MIN_TICK,
    -800000,
    -600000,
    -400000,
    -200000,
    -100000,
    -50000,
    -10000,
    -5000,
    -1000,
    -100,
    -10,
    -1,
    0,
    1,
    10,
    100,
    1000,
    5000,
    10000,
    50000,
    100000,
    200000,
    400000,
    600000,
    800000,
    MAX_TICK,
]


# ---------------------------------------------------------------------------
def test_roundtrip_fixture_ticks() -> None:
    """tick -> sqrt -> tick round-trips for every fixture tick.

    ``MAX_TICK`` is intentionally excluded: its ratio equals ``MAX_SQRT_RATIO``,
    the domain's exclusive upper bound (the protocol's ``<`` inequality), so the
    inverse raises by design — asserted in ``test_out_of_domain_raises``.
    """
    for t in _fixture_ticks():
        if t == MAX_TICK:
            continue
        assert sqrt_price_x96_to_tick(tick_to_sqrt_price_x96(t)) == t


def test_roundtrip_boundary_and_sweep_ticks() -> None:
    """Round-trip holds across the whole lattice incl. the §3.0 boundary set."""
    coverage = [
        MIN_TICK,
        MIN_TICK + 1,
        -887000,
        -100000,
        -60,
        -1,
        0,
        1,
        60,
        100000,
        887000,
        MAX_TICK - 1,
        MAX_TICK,
    ] + SWEEP
    for t in set(coverage):
        if t == MAX_TICK:
            continue
        assert sqrt_price_x96_to_tick(tick_to_sqrt_price_x96(t)) == t


def test_tick_to_sqrt_price_matches_fixture_exactly() -> None:
    """``tick_to_sqrt_price_x96`` equals every fixture vector bit-for-bit."""
    for v in _vectors():
        assert tick_to_sqrt_price_x96(int(v["tick"])) == int(v["sqrt_price_x96"])


def test_boundary_constants_are_anchored() -> None:
    """The three independent anchors hold exactly (MIN_SQRT_RATIO / MAX_SQRT_RATIO / 2^96)."""
    assert tick_to_sqrt_price_x96(MIN_TICK) == MIN_SQRT_RATIO
    assert tick_to_sqrt_price_x96(MAX_TICK) == MAX_SQRT_RATIO
    assert tick_to_sqrt_price_x96(0) == 2**96
    assert sqrt_price_x96_to_tick(MIN_SQRT_RATIO) == MIN_TICK
    assert sqrt_price_x96_to_tick(MAX_SQRT_RATIO - 1) == MAX_TICK - 1


# ---------------------------------------------------------------------------
def test_tick_to_sqrt_price_strictly_increasing() -> None:
    """Monotonicity: strictly increasing across the sweep."""
    prev_tick, prev = SWEEP[0], tick_to_sqrt_price_x96(SWEEP[0])
    for t in SWEEP[1:]:
        cur = tick_to_sqrt_price_x96(t)
        assert cur > prev, f"not strictly increasing between {prev_tick} and {t}"
        prev, prev_tick = cur, t


def test_floor_semantics() -> None:
    """getTickAtSqrtRatio floors: t is returned for ratios in [ratio(t), ratio(t+1))."""
    for t in SWEEP[1:-1]:  # avoid the domain boundaries
        rt = tick_to_sqrt_price_x96(t)
        assert sqrt_price_x96_to_tick(rt) == t, "exact ratio maps to its own tick"
        assert sqrt_price_x96_to_tick(rt + 1) == t, "slightly-below-next maps to t"
        assert sqrt_price_x96_to_tick(rt - 1) == t - 1, "slightly-above-prev maps to t-1"


# ---------------------------------------------------------------------------
def test_out_of_domain_raises() -> None:
    with pytest.raises(ValueError):
        tick_to_sqrt_price_x96(MAX_TICK + 1)
    with pytest.raises(ValueError):
        tick_to_sqrt_price_x96(MIN_TICK - 1)
    with pytest.raises(ValueError):
        sqrt_price_x96_to_tick(MIN_SQRT_RATIO - 1)
    # domain is [MIN_SQRT_RATIO, MAX_SQRT_RATIO): the upper bound is excluded.
    with pytest.raises(ValueError):
        sqrt_price_x96_to_tick(MAX_SQRT_RATIO)


# ---------------------------------------------------------------------------
def test_price_3000_worked_check() -> None:
    """§3.0 worked check: anchor gives 3000.000… to far more than 20 sig digits."""
    price = sqrt_price_x96_to_price(ANCHOR_3000, 6, 18)
    assert abs(price - Decimal("3000")) < Decimal(3000) * Decimal(10) ** -20
    assert price > 1, "price must be USDC-per-WETH (~3000), not the ~3e-4 reciprocal"


def test_price_orientation_swapped_decimals_is_inverted() -> None:
    """Swapping dec0/dec1 massively inverts the price — the orientation trap.

    ``(6, 18)`` is USDC-per-WETH (~3000); ``(18, 6)`` (same raw sqrt) is a tiny
    number, never the familiar dollar price. This is the silent failure CONTRACTS
    §3.0 warns about.
    """
    usdc_per_weth = sqrt_price_x96_to_price(ANCHOR_3000, 6, 18)
    swapped = sqrt_price_x96_to_price(ANCHOR_3000, 18, 6)
    assert usdc_per_weth > Decimal("2000")
    assert abs(swapped) < Decimal("1e-6")


def test_anchor_maps_to_tick_196256_under_floor() -> None:
    """The §3.0 anchor sqrt value lies in [ratio(196256), ratio(196257))."""
    anchor = 1446501726624926496477173928747177
    assert sqrt_price_x96_to_tick(anchor) == 196256
    assert tick_to_sqrt_price_x96(196256) < anchor < tick_to_sqrt_price_x96(196257)


# ---------------------------------------------------------------------------
def test_tick_to_price_five_row_anchor_table() -> None:
    """§3.0's five-row reference table is reproduced.

    The table is rounded to the nearest dollar; assert nearest-dollar
    reproduction plus a tight relative bound on the two rows whose tabulated
    value falls within a basis point.
    """
    table = [
        (190000, Decimal("5608")),
        (194000, Decimal("3759")),
        (196242, Decimal("3004")),
        (200000, Decimal("2063")),
        (207000, Decimal("1025")),
    ]
    for t, expected in table:
        v = tick_to_price(t, 6, 18)
        # nearest-dollar reproduction (the table is dollar-rounded)
        assert abs(v - expected) <= Decimal("0.5"), t
        # ordering sanity: all in the familiar USDC-per-WETH band
        assert Decimal("1000") < v < Decimal("6000")
    # basis-point fidelity on the two rows whose rounding allows it
    for t, expected in ((190000, Decimal("5608")), (194000, Decimal("3759"))):
        assert abs(tick_to_price(t, 6, 18) - expected) / expected < Decimal("1e-4")


def test_tick_to_price_decreasing_in_tick() -> None:
    """The pool's human price DECREASES as tick rises (higher raw token1/token0)."""
    for t1, t2 in zip(SWEEP[:-1], SWEEP[1:], strict=True):
        assert tick_to_price(t1, 6, 18) > tick_to_price(t2, 6, 18)
    # the pinned pool's tick is large and positive
    assert price_to_tick(Decimal("3000"), 6, 18) > 190000
    assert price_to_tick(Decimal("3000"), 6, 18) < 210000


# ---------------------------------------------------------------------------
def test_price_sqrt_price_roundtrip_within_one_unit() -> None:
    """price_to_sqrt_price_x96 inverts sqrt_price_x96_to_price within 1 unit."""
    for s in (MIN_SQRT_RATIO + 5, tick_to_sqrt_price_x96(100000), 2**96,
              tick_to_sqrt_price_x96(196256), MAX_SQRT_RATIO - 5):
        p = sqrt_price_x96_to_price(s, 6, 18)
        assert abs(price_to_sqrt_price_x96(p, 6, 18) - s) <= 1


def test_price_sqrt_price_roundtrip_negative_delta() -> None:
    """The dec1 < dec0 branch of price_to_sqrt_price_x96 round-trips too."""
    for s in (MIN_SQRT_RATIO + 5, tick_to_sqrt_price_x96(100000), 2**96,
              tick_to_sqrt_price_x96(196256), MAX_SQRT_RATIO - 5):
        p = sqrt_price_x96_to_price(s, 18, 6)  # negative decimal exponent path
        assert abs(price_to_sqrt_price_x96(p, 18, 6) - s) <= 1


def test_tick_to_price_negative_delta() -> None:
    """tick_to_price stays Decimal and decreasing even when dec1 < dec0."""
    lo = tick_to_price(190000, 18, 6)
    hi = tick_to_price(207000, 18, 6)
    assert isinstance(lo, Decimal) and isinstance(hi, Decimal)
    assert lo > hi > 0       # still decreasing in tick
    assert hi < Decimal("1e-2")  # and inverts to a tiny number


def test_price_zero_input_raises() -> None:
    with pytest.raises(ValueError):
        sqrt_price_x96_to_price(0, 6, 18)
    with pytest.raises(ValueError):
        price_to_sqrt_price_x96(Decimal("0"), 6, 18)


# ---------------------------------------------------------------------------
def test_wrapping_arithmetic() -> None:
    assert wrapping_sub_256(0, 1) == 2**256 - 1
    assert wrapping_add_256(2**256 - 1, 1) == 0
    assert wrapping_sub_256(5, 10) == MAX_UINT256 - 4
    assert wrapping_add_256(MAX_UINT256, MAX_UINT256) == MAX_UINT256 - 1


# ---------------------------------------------------------------------------
def test_negative_tick_grid_alignment() -> None:
    assert align_tick_down(-61, 60) == -120
    assert align_tick_up(-61, 60) == -60
    # idempotent on aligned ticks, and correct for positives too
    assert align_tick_down(-120, 60) == -120
    assert align_tick_up(-60, 60) == -60
    assert align_tick_down(61, 60) == 60
    assert align_tick_up(61, 60) == 120
    with pytest.raises(ValueError):
        align_tick_down(1, 0)


# ---------------------------------------------------------------------------
def _position_edges() -> tuple[int, int]:
    return tick_to_sqrt_price_x96(195000), tick_to_sqrt_price_x96(197000)


def test_amounts_liquidity_roundtrip_all_branches() -> None:
    """amounts_for_liquidity / liquidity_for_amounts round-trip in all three
    branches, preserving the protocol's floor direction."""
    a, b = _position_edges()
    liq = 10**18
    for _label, p in (
        ("below", tick_to_sqrt_price_x96(194000)),
        ("inside", tick_to_sqrt_price_x96(196000)),
        ("above", tick_to_sqrt_price_x96(198000)),
    ):
        amt0, amt1 = amounts_for_liquidity(p, a, b, liq)
        liq2 = liquidity_for_amounts(p, a, b, amt0, amt1)
        # floor both ways: incurred liquidity never exceeds the input
        assert 0 <= liq - liq2 <= max(4, liq // 10**6), _label
        # token0 comes back within a wei; token1 within 1e-6 relative
        a2, b2 = amounts_for_liquidity(p, a, b, liq2)
        assert 0 <= amt0 - a2 <= 1, _label
        assert 0 <= amt1 - b2 <= max(2, amt1 // 10**6), _label


def test_amount_delta_rounding_direction() -> None:
    """get_amount{0,1}_delta ceil/floor round to at most 2 apart, ordered correctly."""
    a, b = _position_edges()
    liq = 10**18
    up0, dn0 = get_amount0_delta(a, b, liq, True), get_amount0_delta(a, b, liq, False)
    assert dn0 <= up0 <= dn0 + 2
    up1, dn1 = get_amount1_delta(a, b, liq, True), get_amount1_delta(a, b, liq, False)
    assert dn1 <= up1 <= dn1 + 2


def test_amounts_equal_floor_deltas() -> None:
    """The all-amount0 branch equals get_amount0_delta(..., False); likewise token1."""
    a, b = _position_edges()
    liq = 10**18
    below = amounts_for_liquidity(tick_to_sqrt_price_x96(194000), a, b, liq)
    assert below[0] == get_amount0_delta(a, b, liq, False)
    assert below[1] == 0
    above = amounts_for_liquidity(tick_to_sqrt_price_x96(198000), a, b, liq)
    assert above[0] == 0
    assert above[1] == get_amount1_delta(a, b, liq, False)


# ---------------------------------------------------------------------------
def test_q128_to_decimal_is_exact() -> None:
    assert q128_to_decimal(Q128) == Decimal(1)
    assert q128_to_decimal(0) == Decimal(0)
    # exact fraction; the round-trip back to an int needs a high-precision context
    with localcontext() as ctx:
        ctx.prec = 300
        assert q128_to_decimal(1) * Decimal(Q128) == Decimal(1)
        assert q128_to_decimal(MAX_UINT256) * Decimal(Q128) == Decimal(MAX_UINT256)
    with pytest.raises(TypeError):
        q128_to_decimal("not an int")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
def test_no_float_guard_ast() -> None:
    """Blunt invariant: fixedpoint.py contains no ``float`` call and no float literal.

    This keeps the module float-free after any future edit, even one that would
    slip past review otherwise. Docstring prose containing the word 'float' is
    allowed (raw strings); only code-level float constants and ``float(...)``
    calls are rejected.
    """
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