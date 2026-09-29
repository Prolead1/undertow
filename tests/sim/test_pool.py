"""S07 — pool engine: tick lattice + agent fee accrual (`CONTRACTS.md` §6).

ADR-009 orientation: the lattice is **raw** — ``sqrt_price`` is the raw
Uniswap sqrt price ``sqrt(1.0001**tick)``, ``tick`` the protocol tick and
``liquidity`` the raw on-chain ``L``.  Fee accrual is
``fees_raw = L * Δfee_growth`` with no unit conversion; humanising is the
reporting boundary's job (``Position.uncollected_fees``).  The golden
``sqrt_price_x96`` anchor from §19 is checked here against the float64 fast
path, with the exact-vs-float representation error recorded for S14.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from undertow.data import Q96, FeeTier, sqrt_price_x96_to_price
from undertow.sim.core.pool import PoolEngine, PoolState, TickState
from undertow.sim.core.position import calc_sqrt_price_a
from undertow.sim.types import PositionError, Tick, TickSpacing

ENTRY_TICK = Tick(196242)
SPACING = 60
FEE_TIER_BPS = 3000
# Uniswap fee tiers are pips (1e-6); the data reference divides by 1e6.
FEE_DENOMINATOR = 1_000_000.0

# CONTRACTS.md §19 / data CONTRACTS.md §3.0: sqrt_price_x96 at ETH = $3000.
ANCHOR_3000_X96 = 1446501726624926496477173928747177


def _base_state(
    *,
    tick: Tick = ENTRY_TICK,
    liquidity: float = 0.0,
    fee_growth_global_0: float = 0.0,
    fee_growth_global_1: float = 0.0,
) -> PoolState:
    """A raw-orientation ``PoolState`` at ``tick`` with the 0.30% tier/spacing."""
    return PoolState(
        sqrt_price=calc_sqrt_price_a(tick),
        tick=tick,
        liquidity=liquidity,
        fee_growth_global_0=fee_growth_global_0,
        fee_growth_global_1=fee_growth_global_1,
        fee_tier_bps=FEE_TIER_BPS,
        tick_spacing=SPACING,
    )


def _engine(*, liquidity: float = 0.0, tick: Tick = ENTRY_TICK) -> PoolEngine:
    return PoolEngine(state=_base_state(tick=tick, liquidity=liquidity), ticks={})


# ---------------------------------------------------------------------------
# 1. TickState defaults
# ---------------------------------------------------------------------------
def test_tick_state_defaults() -> None:
    tick = TickState()
    assert tick.fee_growth_outside_0 == 0.0
    assert tick.fee_growth_outside_1 == 0.0
    assert tick.liquidity_gross == 0.0
    assert tick.liquidity_net == 0.0
    assert tick.initialized is False


# ---------------------------------------------------------------------------
# 2. Opening a position initializes its lower and upper ticks
# ---------------------------------------------------------------------------
def test_open_position_initializes_both_ticks() -> None:
    engine = _engine()
    lower = Tick(ENTRY_TICK - 120)
    upper = Tick(ENTRY_TICK + 120)
    engine.open_position(lower, upper, 1_000.0)

    assert set(engine.ticks) == {engine.positions[0].tick_lower, engine.positions[0].tick_upper}
    assert engine.ticks[engine.positions[0].tick_lower].initialized is True
    assert engine.ticks[engine.positions[0].tick_upper].initialized is True
    assert engine.ticks[engine.positions[0].tick_lower].liquidity_gross == 1_000.0
    assert engine.ticks[engine.positions[0].tick_lower].liquidity_net == 1_000.0
    assert engine.ticks[engine.positions[0].tick_upper].liquidity_gross == 1_000.0
    assert engine.ticks[engine.positions[0].tick_upper].liquidity_net == -1_000.0


# ---------------------------------------------------------------------------
# 3. In-range position adds to active liquidity
# ---------------------------------------------------------------------------
def test_open_in_range_adds_active_liquidity() -> None:
    engine = _engine(liquidity=500.0)
    engine.open_position(Tick(ENTRY_TICK - 120), Tick(ENTRY_TICK + 120), 1_000.0)
    assert engine.state.liquidity == pytest.approx(1_500.0)


# ---------------------------------------------------------------------------
# 4. Out-of-range position does not add to active liquidity
# ---------------------------------------------------------------------------
def test_open_out_of_range_does_not_add_liquidity() -> None:
    above = _engine(liquidity=500.0)
    above.open_position(Tick(ENTRY_TICK + 120), Tick(ENTRY_TICK + 240), 1_000.0)
    assert above.state.liquidity == pytest.approx(500.0)

    below = _engine(liquidity=500.0)
    below.open_position(Tick(ENTRY_TICK - 240), Tick(ENTRY_TICK - 120), 1_000.0)
    assert below.state.liquidity == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# 5-7. fee_growth_below / fee_growth_above sign rules
# ---------------------------------------------------------------------------
def test_fee_growth_below_uses_outside_when_current_at_or_above() -> None:
    state = _base_state(tick=Tick(100))
    assert state.fee_growth_below(Tick(50), 0.7) == 0.7
    assert state.fee_growth_below(Tick(100), 0.7) == 0.7


def test_fee_growth_below_uses_global_minus_outside_when_current_below() -> None:
    state = _base_state(tick=Tick(100), fee_growth_global_0=5.0)
    assert state.fee_growth_below(Tick(150), 0.7) == pytest.approx(4.3)
    # Explicit token-1 global (the frozen signature has no token selector).
    assert state.fee_growth_below(Tick(150), 0.7, fee_growth_global=2.0) == pytest.approx(1.3)


def test_fee_growth_above_uses_outside_when_current_below() -> None:
    state = _base_state(tick=Tick(100))
    assert state.fee_growth_above(Tick(150), 0.7) == 0.7
    # Symmetric branch: current >= boundary -> global - outside.
    state_above = _base_state(tick=Tick(150), fee_growth_global_0=5.0)
    assert state_above.fee_growth_above(Tick(100), 0.7) == pytest.approx(4.3)


# ---------------------------------------------------------------------------
# 8. fg_inside matches the manual fee-share computation
# ---------------------------------------------------------------------------
def test_fee_growth_inside_matches_manual_volume_share() -> None:
    engine = _engine(liquidity=0.0)
    l1, l2 = 3.0e6, 5.0e6
    pid1 = engine.open_position(Tick(ENTRY_TICK - 120), Tick(ENTRY_TICK + 120), l1)
    pid2 = engine.open_position(Tick(ENTRY_TICK - 60), Tick(ENTRY_TICK + 60), l2)
    total = l1 + l2
    assert engine.state.liquidity == pytest.approx(total)

    result = engine.step(engine.state.sqrt_price, ENTRY_TICK, 10_000.0, 4_000.0)
    fee0 = 10_000.0 * FEE_TIER_BPS / FEE_DENOMINATOR
    fee1 = 4_000.0 * FEE_TIER_BPS / FEE_DENOMINATOR

    assert result["fees_accrued"][pid1][0] == pytest.approx(fee0 * l1 / total)
    assert result["fees_accrued"][pid2][0] == pytest.approx(fee0 * l2 / total)
    assert result["fees_accrued"][pid1][1] == pytest.approx(fee1 * l1 / total)
    assert result["fees_accrued"][pid2][1] == pytest.approx(fee1 * l2 / total)
    # position_fees and the returned mapping agree.
    assert engine.position_fees(pid1) == pytest.approx(result["fees_accrued"][pid1])


# ---------------------------------------------------------------------------
# 9-10. step: zero vs non-zero volume
# ---------------------------------------------------------------------------
def test_step_zero_volume_no_price_change_accrues_nothing() -> None:
    engine = _engine(liquidity=0.0)
    pid = engine.open_position(Tick(ENTRY_TICK - 120), Tick(ENTRY_TICK + 120), 2_000.0)
    before = (engine.state.fee_growth_global_0, engine.state.fee_growth_global_1)

    result = engine.step(engine.state.sqrt_price, ENTRY_TICK, 0.0, 0.0)

    assert result["ticks_crossed"] == []
    assert result["fees_accrued"][pid] == (0.0, 0.0)
    assert result["pool_fee_earned_0"] == 0.0
    assert result["pool_fee_earned_1"] == 0.0
    assert (engine.state.fee_growth_global_0, engine.state.fee_growth_global_1) == before


def test_step_volume_no_price_change_accrues_to_in_range_position() -> None:
    engine = _engine(liquidity=0.0)
    liquidity = 2_000.0
    pid = engine.open_position(Tick(ENTRY_TICK - 120), Tick(ENTRY_TICK + 120), liquidity)

    result = engine.step(engine.state.sqrt_price, ENTRY_TICK, 1_000.0, 500.0)

    fee0 = 1_000.0 * FEE_TIER_BPS / FEE_DENOMINATOR
    fee1 = 500.0 * FEE_TIER_BPS / FEE_DENOMINATOR
    assert engine.state.fee_growth_global_0 == pytest.approx(fee0 / liquidity)
    assert engine.state.fee_growth_global_1 == pytest.approx(fee1 / liquidity)
    assert result["fees_accrued"][pid][0] == pytest.approx(fee0)
    assert result["fees_accrued"][pid][1] == pytest.approx(fee1)
    assert result["pool_fee_earned_0"] == pytest.approx(fee0)
    assert result["pool_fee_earned_1"] == pytest.approx(fee1)


# ---------------------------------------------------------------------------
# 11. Crossing one tick changes liquidity
# ---------------------------------------------------------------------------
def test_step_crossing_one_tick_deactivates_position() -> None:
    engine = _engine(liquidity=1_000.0)
    pid = engine.open_position(Tick(ENTRY_TICK - 60), Tick(ENTRY_TICK + 60), 500.0)
    upper = engine.positions[pid].tick_upper
    assert engine.state.liquidity == pytest.approx(1_500.0)

    new_tick = Tick(ENTRY_TICK + 120)
    result = engine.step(calc_sqrt_price_a(new_tick), new_tick, 0.0, 0.0)

    assert result["ticks_crossed"] == [upper]
    assert result["active_liquidity"] == pytest.approx(1_000.0)
    assert engine.state.liquidity == pytest.approx(1_000.0)
    assert engine.state.tick == new_tick


# ---------------------------------------------------------------------------
# 12. Crossing multiple ticks with two positions (both directions)
# ---------------------------------------------------------------------------
def test_step_crosses_multiple_ticks_and_reallocates_liquidity() -> None:
    engine = _engine(liquidity=1.0e6)
    l_a, l_b = 2.0e6, 3.0e6
    pid_a = engine.open_position(Tick(ENTRY_TICK - 120), Tick(ENTRY_TICK + 120), l_a)
    pid_b = engine.open_position(Tick(ENTRY_TICK + 180), Tick(ENTRY_TICK + 360), l_b)
    upper_a = engine.positions[pid_a].tick_upper
    lower_b = engine.positions[pid_b].tick_lower

    assert engine.state.liquidity == pytest.approx(1.0e6 + l_a)

    # Right: cross A's upper (A off) then B's lower (B on).
    new_tick = Tick(ENTRY_TICK + 300)
    result = engine.step(calc_sqrt_price_a(new_tick), new_tick, 9_000.0, 0.0)

    assert result["ticks_crossed"] == [upper_a, lower_b]
    assert engine.state.liquidity == pytest.approx(1.0e6 + l_b)
    fee0 = 9_000.0 * FEE_TIER_BPS / FEE_DENOMINATOR
    # A is out of range after the crossing: it earns nothing this step.
    assert result["fees_accrued"][pid_a][0] == pytest.approx(0.0)
    # B is active at the end: its share is L_B / (base + L_B).
    assert result["fees_accrued"][pid_b][0] == pytest.approx(
        fee0 * l_b / (1.0e6 + l_b)
    )

    # Left: cross B's lower (B off) then A's upper (A on), zero volume so fees persist.
    back = Tick(ENTRY_TICK)
    result_back = engine.step(calc_sqrt_price_a(back), back, 0.0, 0.0)

    assert result_back["ticks_crossed"] == [upper_a, lower_b]
    assert engine.state.liquidity == pytest.approx(1.0e6 + l_a)
    # Fees already earned are not lost when a position exits and re-enters.
    assert result_back["fees_accrued"][pid_b][0] == pytest.approx(
        fee0 * l_b / (1.0e6 + l_b)
    )
    assert result_back["fees_accrued"][pid_a][0] == pytest.approx(0.0)


def test_tick_cross_flips_outside_both_directions() -> None:
    engine = _engine(liquidity=0.0)
    pid = engine.open_position(Tick(ENTRY_TICK - 60), Tick(ENTRY_TICK + 60), 1.0)
    upper = engine.positions[pid].tick_upper

    # Earn 0.3 raw fee while in range (global 0 -> 0.3).
    engine.step(engine.state.sqrt_price, ENTRY_TICK, 100.0, 0.0)
    assert engine.ticks[upper].fee_growth_outside_0 == 0.0

    # Cross right: flip outside to the pre-cross global (0.3).
    right = Tick(ENTRY_TICK + 120)
    engine.step(calc_sqrt_price_a(right), right, 0.0, 0.0)
    assert engine.ticks[upper].fee_growth_outside_0 == pytest.approx(0.3)
    assert engine.state.liquidity == pytest.approx(0.0)

    # Cross left: flip back to global - outside = 0.
    engine.step(calc_sqrt_price_a(ENTRY_TICK), ENTRY_TICK, 0.0, 0.0)
    assert engine.ticks[upper].fee_growth_outside_0 == pytest.approx(0.0)
    assert engine.state.liquidity == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Out-of-range positions earn nothing
# ---------------------------------------------------------------------------
def test_out_of_range_positions_earn_nothing() -> None:
    engine = _engine(liquidity=1.0e6)
    pid_above = engine.open_position(
        Tick(ENTRY_TICK + 120), Tick(ENTRY_TICK + 240), 5_000.0
    )
    pid_below = engine.open_position(
        Tick(ENTRY_TICK - 240), Tick(ENTRY_TICK - 120), 5_000.0
    )

    result = engine.step(engine.state.sqrt_price, ENTRY_TICK, 1_000.0, 1_000.0)

    assert engine.state.fee_growth_global_0 > 0.0
    assert result["fees_accrued"][pid_above] == (0.0, 0.0)
    assert result["fees_accrued"][pid_below] == (0.0, 0.0)


# ---------------------------------------------------------------------------
# 13-15. close_position
# ---------------------------------------------------------------------------
def test_close_position_returns_accrued_fees() -> None:
    engine = _engine(liquidity=0.0)
    pid = engine.open_position(Tick(ENTRY_TICK - 120), Tick(ENTRY_TICK + 120), 1.0)
    engine.step(engine.state.sqrt_price, ENTRY_TICK, 100.0, 0.0)

    accrued = engine.position_fees(pid)
    assert accrued[0] == pytest.approx(0.3)
    returned = engine.close_position(pid)
    assert returned == pytest.approx(accrued)
    assert returned[0] == pytest.approx(0.3)
    assert returned[0] > 0.0


def test_close_position_removes_position_and_cleans_lattice() -> None:
    engine = _engine(liquidity=0.0)
    pid = engine.open_position(Tick(ENTRY_TICK - 120), Tick(ENTRY_TICK + 120), 750.0)
    lower = engine.positions[pid].tick_lower
    upper = engine.positions[pid].tick_upper

    engine.close_position(pid)

    assert pid not in engine.positions
    assert pid not in engine.position_fee_snapshots
    assert engine.ticks[lower].liquidity_gross == 0.0
    assert engine.ticks[lower].liquidity_net == 0.0
    assert engine.ticks[lower].initialized is False
    assert engine.ticks[upper].liquidity_gross == 0.0
    assert engine.ticks[upper].liquidity_net == 0.0
    assert engine.ticks[upper].initialized is False
    assert engine.state.liquidity == pytest.approx(0.0)


def test_close_unknown_position_raises() -> None:
    engine = _engine()
    with pytest.raises(PositionError):
        engine.close_position(999)


# ---------------------------------------------------------------------------
# 16. position_fees is zero immediately after opening
# ---------------------------------------------------------------------------
def test_position_fees_zero_immediately_after_open() -> None:
    engine = _engine(liquidity=0.0)
    pid = engine.open_position(Tick(ENTRY_TICK - 120), Tick(ENTRY_TICK + 120), 1_234.0)
    assert engine.position_fees(pid) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# 17-18. shared ticks
# ---------------------------------------------------------------------------
def test_open_shared_ticks_increments_gross() -> None:
    engine = _engine(liquidity=0.0)
    lower = Tick(ENTRY_TICK - 120)
    upper = Tick(ENTRY_TICK + 120)
    pid1 = engine.open_position(lower, upper, 100.0)
    pid2 = engine.open_position(lower, upper, 250.0)
    snapped_lower = engine.positions[pid1].tick_lower
    snapped_upper = engine.positions[pid1].tick_upper
    assert engine.positions[pid2].tick_lower == snapped_lower
    assert engine.positions[pid2].tick_upper == snapped_upper

    assert engine.ticks[snapped_lower].liquidity_gross == 350.0
    assert engine.ticks[snapped_lower].liquidity_net == 350.0
    assert engine.ticks[snapped_upper].liquidity_gross == 350.0
    assert engine.ticks[snapped_upper].liquidity_net == -350.0
    assert engine.state.liquidity == pytest.approx(350.0)


def test_closing_all_shared_positions_uninitializes_tick() -> None:
    engine = _engine(liquidity=0.0)
    lower = Tick(ENTRY_TICK - 120)
    upper = Tick(ENTRY_TICK + 120)
    pid1 = engine.open_position(lower, upper, 100.0)
    pid2 = engine.open_position(lower, upper, 250.0)
    snapped_lower = engine.positions[pid1].tick_lower
    snapped_upper = engine.positions[pid1].tick_upper

    engine.close_position(pid1)
    assert engine.ticks[snapped_lower].liquidity_gross == 250.0
    assert engine.ticks[snapped_lower].initialized is True

    engine.close_position(pid2)
    assert engine.ticks[snapped_lower].liquidity_gross == 0.0
    assert engine.ticks[snapped_lower].initialized is False
    assert engine.ticks[snapped_upper].liquidity_gross == 0.0
    assert engine.ticks[snapped_upper].initialized is False


# ---------------------------------------------------------------------------
# Tick snapping
# ---------------------------------------------------------------------------
def test_open_position_snaps_ticks_to_spacing() -> None:
    engine = _engine(liquidity=0.0)
    pid = engine.open_position(Tick(ENTRY_TICK - 125), Tick(ENTRY_TICK + 125), 10.0)
    position = engine.positions[pid]
    assert position.tick_lower % SPACING == 0
    assert position.tick_upper % SPACING == 0
    assert position.tick_lower == Tick(196140)  # nearest multiple of 60 to 196117
    assert position.tick_upper == Tick(196380)  # nearest multiple of 60 to 196367


def test_open_position_rejects_inverted_or_zero_liquidity() -> None:
    engine = _engine()
    with pytest.raises(PositionError):
        engine.open_position(Tick(ENTRY_TICK + 120), Tick(ENTRY_TICK - 120), 10.0)
    with pytest.raises(PositionError):
        engine.open_position(Tick(ENTRY_TICK - 120), Tick(ENTRY_TICK + 120), 0.0)


# ---------------------------------------------------------------------------
# 19. Golden sqrt_price_x96 at ETH = $3000
# ---------------------------------------------------------------------------
def test_golden_sqrt_price_x96_at_3000() -> None:
    # The exact Q64.96 anchor from CONTRACTS §19.
    assert ANCHOR_3000_X96 == 1446501726624926496477173928747177
    # The sim stores the raw sqrt price as float64: anchor / Q96.
    raw = ANCHOR_3000_X96 / Q96
    # Value the pool engine would hold (recorded for S14's drift study):
    #   raw == 18257.418583505536
    #   Decimal.from_float(raw) - anchor/Q96 == -1.1804134...e-12
    assert raw == pytest.approx(18257.418583505536, rel=1e-15)
    exact = Decimal(ANCHOR_3000_X96) / Decimal(Q96)
    representation_error = Decimal.from_float(raw) - exact
    assert representation_error < Decimal("0")
    assert abs(representation_error) < Decimal("1e-9")

    # The data golden: this anchor converts back to $3000 exactly (USDC/WETH).
    price = sqrt_price_x96_to_price(ANCHOR_3000_X96, 6, 18)
    assert abs(price - Decimal("3000")) < Decimal(3000) * Decimal("1e-20")

    # The float64 raw price is accepted by the pool state unchanged.
    state = PoolState(
        sqrt_price=raw,
        tick=Tick(196256),
        liquidity=0.0,
        fee_growth_global_0=0.0,
        fee_growth_global_1=0.0,
        fee_tier_bps=FEE_TIER_BPS,
        tick_spacing=SPACING,
    )
    assert state.sqrt_price == raw


# ---------------------------------------------------------------------------
# 20. tiny_pool_engine fixture (CONTRACTS.md §18)
# ---------------------------------------------------------------------------
def test_tiny_pool_engine_fixture(tiny_pool_engine: PoolEngine) -> None:
    assert isinstance(tiny_pool_engine, PoolEngine)
    assert len(tiny_pool_engine.ticks) >= 20
    assert tiny_pool_engine.state.fee_tier_bps == 3000
    assert tiny_pool_engine.state.tick_spacing == SPACING
    assert tiny_pool_engine.state.tick == ENTRY_TICK
    assert len(tiny_pool_engine.positions) == 1
    # The entry position is in range and marginal against the base liquidity.
    assert tiny_pool_engine.state.liquidity > 1.0e15

    result = tiny_pool_engine.step(
        tiny_pool_engine.state.sqrt_price, ENTRY_TICK, 1_000.0, 0.0
    )
    (pid,) = tiny_pool_engine.positions
    assert result["fees_accrued"][pid][0] > 0.0
    assert result["fees_accrued"][pid][0] < 1_000.0 * FEE_TIER_BPS / FEE_DENOMINATOR


def test_pool_engine_exports_via_core_package() -> None:
    from undertow.sim.core import PoolEngine as ExportedEngine
    from undertow.sim.core import PoolState as ExportedState
    from undertow.sim.core import TickState as ExportedTick

    assert ExportedEngine is PoolEngine
    assert ExportedState is PoolState
    assert ExportedTick is TickState


def test_tick_spacing_newtype_import_used() -> None:
    # Guard the TickSpacing import so a future refactor keeps the public alias used.
    assert int(TickSpacing(SPACING)) == SPACING


# ---------------------------------------------------------------------------
# Reviewer follow-ups: fee denominator, boundary coherence, edge paths
# ---------------------------------------------------------------------------
def test_swap_fee_matches_data_pips_denominator() -> None:
    """The 0.30% tier is 3000 *pips*, so the data path divides by 1e6."""
    engine = _engine(liquidity=0.0)
    engine.open_position(Tick(ENTRY_TICK - 120), Tick(ENTRY_TICK + 120), 1.0)
    volume = 1_000_000.0

    result = engine.step(engine.state.sqrt_price, ENTRY_TICK, volume, 0.0)

    expected = volume * FeeTier.BPS_30.value / 1_000_000.0  # == 3000.0
    assert expected == pytest.approx(3_000.0)
    assert result["pool_fee_earned_0"] == pytest.approx(expected)
    # Not the 100x-inflated /10000 form, which would be 300_000.0.
    assert result["pool_fee_earned_0"] != pytest.approx(volume * 3000 / 10_000.0)


def test_downward_exact_boundary_is_consistent_with_range_predicate() -> None:
    engine = _engine(liquidity=0.0)
    pid = engine.open_position(Tick(ENTRY_TICK - 120), Tick(ENTRY_TICK + 120), 1_000.0)
    lower = engine.positions[pid].tick_lower
    upper = engine.positions[pid].tick_upper

    # Move to just above the upper bound: position inactive, no active liquidity.
    start = Tick(ENTRY_TICK + 240)
    engine.step(calc_sqrt_price_a(start), start, 0.0, 0.0)
    assert engine.state.liquidity == pytest.approx(0.0)

    # Land exactly on the upper bound: still out of range, liquidity stays 0.
    engine.step(calc_sqrt_price_a(upper), upper, 0.0, 0.0)
    assert engine.state.tick == upper
    assert engine.state.liquidity == pytest.approx(0.0)

    # Land exactly on the lower bound: in range, liquidity is restored.
    engine.step(calc_sqrt_price_a(lower), lower, 0.0, 0.0)
    assert engine.state.tick == lower
    assert engine.state.liquidity == pytest.approx(1_000.0)


def test_step_skips_uninitialized_ticks() -> None:
    engine = _engine(liquidity=1_000.0)
    pid = engine.open_position(Tick(ENTRY_TICK - 60), Tick(ENTRY_TICK + 60), 500.0)
    upper = engine.positions[pid].tick_upper
    engine.close_position(pid)
    assert engine.ticks[upper].initialized is False

    new_tick = Tick(ENTRY_TICK + 120)
    result = engine.step(calc_sqrt_price_a(new_tick), new_tick, 0.0, 0.0)

    assert result["ticks_crossed"] == []
    assert engine.state.liquidity == pytest.approx(1_000.0)


def test_open_position_snaps_negative_ticks() -> None:
    engine = _engine(liquidity=0.0)
    pid = engine.open_position(Tick(-125), Tick(125), 10.0)
    position = engine.positions[pid]
    assert position.tick_lower == Tick(-120)
    assert position.tick_upper == Tick(120)


def test_step_zero_liquidity_reports_nominal_fee_without_growth() -> None:
    engine = _engine(liquidity=0.0)
    result = engine.step(engine.state.sqrt_price, ENTRY_TICK, 1_000.0, 0.0)

    assert result["active_liquidity"] == 0.0
    assert result["fees_accrued"] == {}
    assert result["pool_fee_earned_0"] == pytest.approx(
        1_000.0 * FEE_TIER_BPS / FEE_DENOMINATOR
    )
    assert engine.state.fee_growth_global_0 == 0.0


def test_position_snapshot_mirrors_engine_and_humanises_fees() -> None:
    engine = _engine(liquidity=0.0)
    pid = engine.open_position(Tick(ENTRY_TICK - 120), Tick(ENTRY_TICK + 120), 1.0)
    engine.step(engine.state.sqrt_price, ENTRY_TICK, 100.0, 0.0)

    raw = engine.position_fees(pid)
    position = engine.positions[pid]
    lower_ts = engine.ticks[position.tick_lower]
    upper_ts = engine.ticks[position.tick_upper]
    inside = engine.state.fee_growth_inside(
        position.tick_lower,
        position.tick_upper,
        lower_ts.fee_growth_outside_0,
        lower_ts.fee_growth_outside_1,
        upper_ts.fee_growth_outside_0,
        upper_ts.fee_growth_outside_1,
    )
    human = position.uncollected_fees(inside[0], inside[1])

    assert raw[0] == pytest.approx(0.3)
    assert human[0] == pytest.approx(raw[0] / 10.0**position.dec0)
    assert human[0] == pytest.approx(3.0e-7)


def test_step_updates_sqrt_price_within_same_tick() -> None:
    """slot0 sqrt price tracks every step, even when the tick is unchanged."""
    engine = _engine(liquidity=1_000.0)
    engine.open_position(Tick(ENTRY_TICK - 120), Tick(ENTRY_TICK + 120), 500.0)

    # A new raw price that is still inside the same tick.
    moved_sqrt = engine.state.sqrt_price * 1.000_001
    result = engine.step(moved_sqrt, ENTRY_TICK, 0.0, 0.0)

    assert result["ticks_crossed"] == []
    assert engine.state.sqrt_price == moved_sqrt
    assert engine.state.tick == ENTRY_TICK


def test_tick_cross_requires_lattice_and_fallback_selection() -> None:
    # Missing lattice is a programming error.
    state = _base_state(tick=ENTRY_TICK)
    with pytest.raises(PositionError):
        state.tick_cross(Tick(ENTRY_TICK + 60))

    # Direct call (no precomputed `crossed` list) still selects and flips.
    engine = _engine(liquidity=1_000.0)
    pid = engine.open_position(Tick(ENTRY_TICK - 60), Tick(ENTRY_TICK + 60), 500.0)
    upper = engine.positions[pid].tick_upper
    new_tick = Tick(ENTRY_TICK + 120)
    engine.state.tick_cross(new_tick, engine.ticks)
    assert engine.state.liquidity == pytest.approx(1_000.0)
    assert engine.ticks[upper].fee_growth_outside_0 == pytest.approx(0.0)
    # tick_cross does not advance the current tick; step owns that.
    assert engine.state.tick == ENTRY_TICK


def test_pool_error_paths() -> None:
    # position_fees on an unknown id.
    engine = _engine()
    with pytest.raises(PositionError):
        engine.position_fees(999)

    # Non-positive tick spacing is rejected at open time.
    bad_state = PoolState(
        sqrt_price=calc_sqrt_price_a(ENTRY_TICK),
        tick=ENTRY_TICK,
        liquidity=0.0,
        fee_growth_global_0=0.0,
        fee_growth_global_1=0.0,
        fee_tier_bps=FEE_TIER_BPS,
        tick_spacing=0,
    )
    bad_engine = PoolEngine(state=bad_state, ticks={})
    with pytest.raises(PositionError):
        bad_engine.open_position(Tick(ENTRY_TICK - 60), Tick(ENTRY_TICK + 60), 1.0)


def test_tick_cross_noop_when_tick_unchanged() -> None:
    engine = _engine(liquidity=1_000.0)
    engine.open_position(Tick(ENTRY_TICK - 60), Tick(ENTRY_TICK + 60), 500.0)
    before = engine.state.liquidity
    engine.state.tick_cross(ENTRY_TICK, engine.ticks)
    assert engine.state.liquidity == before
