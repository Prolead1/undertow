"""S07 — pool engine: tick lattice + marginal-agent fee accrual (`CONTRACTS.md` §6).

ADR-019 orientation: the lattice is **raw** — ``sqrt_price`` is the raw
Uniswap sqrt price ``sqrt(1.0001**tick)``, ``tick`` the protocol tick and
``liquidity`` the raw on-chain ``L``.  The engine is the float64 replica of the
backtester's exact ``apply_swap``: swaps are apportioned *piecewise* across the
crossed-tick segments, and the agent's positions are *marginal* (their ``L`` is
not added to ``state.liquidity``; fee growth is distributed over the external
liquidity alone).  Fee accrual is ``fees_raw = L * Δfee_growth`` with no unit
conversion; humanising is the reporting boundary's job
(``Position.uncollected_fees``).  The golden ``sqrt_price_x96`` anchor from §19
is checked here against the float64 fast path.
"""

from __future__ import annotations

import math
from decimal import Decimal

import pytest

from undertow.data import (
    Q96,
    FeeTier,
    sqrt_price_x96_to_price,
    tick_to_sqrt_price_x96,
)
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


def _mint(engine: PoolEngine, lower: int, upper: int, liquidity: float) -> None:
    """Seed **external** liquidity via a real ``mint`` row (ADR-019)."""
    engine.apply_liquidity_event(
        {
            "event_type": "mint",
            "tick_lower": int(lower),
            "tick_upper": int(upper),
            "liquidity_amount": str(int(liquidity)),
        }
    )


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
# 2. Opening a position seeds its boundary ticks as marginal sentinels
# ---------------------------------------------------------------------------
def test_open_position_seeds_both_ticks_as_sentinels() -> None:
    engine = _engine()
    engine.open_position(Tick(ENTRY_TICK - 120), Tick(ENTRY_TICK + 120), 1_000.0)

    position = engine.positions[0]
    assert set(engine.ticks) == {position.tick_lower, position.tick_upper}
    lower_ts = engine.ticks[position.tick_lower]
    upper_ts = engine.ticks[position.tick_upper]
    assert lower_ts.initialized is True
    assert upper_ts.initialized is True
    # ``liquidity_gross=1, liquidity_net=0`` — the ``register_position`` sentinel.
    assert lower_ts.liquidity_gross == 1.0
    assert lower_ts.liquidity_net == 0.0
    assert upper_ts.liquidity_gross == 1.0
    assert upper_ts.liquidity_net == 0.0
    # The agent adds no external liquidity.
    assert engine.state.liquidity == 0.0


# ---------------------------------------------------------------------------
# 3. An in-range marginal position does NOT change active liquidity
# ---------------------------------------------------------------------------
def test_open_in_range_keeps_active_liquidity() -> None:
    engine = _engine(liquidity=500.0)
    engine.open_position(Tick(ENTRY_TICK - 120), Tick(ENTRY_TICK + 120), 1_000.0)
    assert engine.state.liquidity == pytest.approx(500.0)


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
# 8. fg_inside matches the manual external-liquidity fee share
# ---------------------------------------------------------------------------
def test_fee_growth_inside_matches_manual_volume_share() -> None:
    engine = _engine(liquidity=0.0)
    _mint(engine, ENTRY_TICK - 600, ENTRY_TICK + 600, 1.0e9)
    total = 1.0e9
    l1, l2 = 3.0e6, 5.0e6
    pid1 = engine.open_position(Tick(ENTRY_TICK - 120), Tick(ENTRY_TICK + 120), l1)
    pid2 = engine.open_position(Tick(ENTRY_TICK - 60), Tick(ENTRY_TICK + 60), l2)
    assert engine.state.liquidity == pytest.approx(total)

    post = Tick(ENTRY_TICK + 30)
    result = engine.step(calc_sqrt_price_a(post), post, 10_000.0, 0.0)
    fee0 = 10_000.0 * FEE_TIER_BPS / FEE_DENOMINATOR
    assert result["fees_accrued"][pid1][0] == pytest.approx(fee0 * l1 / total)
    assert result["fees_accrued"][pid2][0] == pytest.approx(fee0 * l2 / total)
    # position_fees and the returned mapping agree.
    assert engine.position_fees(pid1) == pytest.approx(result["fees_accrued"][pid1])

    back = Tick(ENTRY_TICK)
    result1 = engine.step(calc_sqrt_price_a(back), back, 0.0, 4_000.0)
    fee1 = 4_000.0 * FEE_TIER_BPS / FEE_DENOMINATOR
    assert result1["fees_accrued"][pid1][1] == pytest.approx(fee1 * l1 / total)
    assert result1["fees_accrued"][pid2][1] == pytest.approx(fee1 * l2 / total)


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


def test_step_volume_accrues_to_in_range_position() -> None:
    engine = _engine(liquidity=4_000.0)
    liquidity = 2_000.0
    pid = engine.open_position(Tick(ENTRY_TICK - 120), Tick(ENTRY_TICK + 120), liquidity)

    post = Tick(ENTRY_TICK + 30)
    result = engine.step(calc_sqrt_price_a(post), post, 1_000.0, 0.0)

    fee0 = 1_000.0 * FEE_TIER_BPS / FEE_DENOMINATOR
    assert engine.state.fee_growth_global_0 == pytest.approx(fee0 / 4_000.0)
    assert engine.state.fee_growth_global_1 == 0.0
    assert result["fees_accrued"][pid][0] == pytest.approx(fee0 * liquidity / 4_000.0)
    assert result["pool_fee_earned_0"] == pytest.approx(fee0)

    back = Tick(ENTRY_TICK)
    result1 = engine.step(calc_sqrt_price_a(back), back, 0.0, 500.0)
    fee1 = 500.0 * FEE_TIER_BPS / FEE_DENOMINATOR
    assert engine.state.fee_growth_global_1 == pytest.approx(fee1 / 4_000.0)
    assert result1["fees_accrued"][pid][1] == pytest.approx(fee1 * liquidity / 4_000.0)
    assert result1["pool_fee_earned_1"] == pytest.approx(fee1)


# ---------------------------------------------------------------------------
# 11. Crossing one tick does not change external liquidity (marginal sentinel)
# ---------------------------------------------------------------------------
def test_step_crossing_one_tick_keeps_external_liquidity() -> None:
    engine = _engine(liquidity=1_000.0)
    pid = engine.open_position(Tick(ENTRY_TICK - 60), Tick(ENTRY_TICK + 60), 500.0)
    upper = engine.positions[pid].tick_upper
    assert engine.state.liquidity == pytest.approx(1_000.0)

    new_tick = Tick(ENTRY_TICK + 120)
    result = engine.step(calc_sqrt_price_a(new_tick), new_tick, 0.0, 0.0)

    assert result["ticks_crossed"] == [upper]
    assert result["active_liquidity"] == pytest.approx(1_000.0)
    assert engine.state.liquidity == pytest.approx(1_000.0)
    assert engine.state.tick == new_tick


# ---------------------------------------------------------------------------
# 12. Crossing multiple ticks reallocates EXTERNAL liquidity (both directions)
# ---------------------------------------------------------------------------
def test_step_crosses_multiple_ticks_and_reallocates_liquidity() -> None:
    engine = _engine(liquidity=0.0)
    _mint(engine, 195600, 196800, 1.0e6)  # base
    _mint(engine, 196140, 196380, 2.0e6)  # A
    _mint(engine, 196440, 196620, 3.0e6)  # B
    assert engine.state.liquidity == pytest.approx(3.0e6)

    new_tick = Tick(ENTRY_TICK + 300)
    result = engine.step(calc_sqrt_price_a(new_tick), new_tick, 0.0, 0.0)
    assert result["ticks_crossed"] == [Tick(196380), Tick(196440)]
    assert engine.state.liquidity == pytest.approx(4.0e6)

    back = Tick(ENTRY_TICK)
    result_back = engine.step(calc_sqrt_price_a(back), back, 0.0, 0.0)
    assert result_back["ticks_crossed"] == [Tick(196380), Tick(196440)]
    assert engine.state.liquidity == pytest.approx(3.0e6)


def test_tick_cross_flips_outside_both_directions() -> None:
    engine = _engine(liquidity=0.0)
    _mint(engine, 195600, 196800, 1.0e9)
    pid = engine.open_position(Tick(ENTRY_TICK - 60), Tick(ENTRY_TICK + 60), 1.0)
    upper = engine.positions[pid].tick_upper

    # Earn a positive raw fee while in range: global = 0.3 / 1e9 (external L).
    in_tick = Tick(ENTRY_TICK + 30)
    engine.step(calc_sqrt_price_a(in_tick), in_tick, 100.0, 0.0)
    global_fee = (100.0 * FEE_TIER_BPS / FEE_DENOMINATOR) / 1.0e9
    assert engine.ticks[upper].fee_growth_outside_0 == 0.0

    # Cross right: flip outside to the pre-cross global.
    right = Tick(ENTRY_TICK + 120)
    engine.step(calc_sqrt_price_a(right), right, 0.0, 0.0)
    assert engine.ticks[upper].fee_growth_outside_0 == pytest.approx(global_fee)
    assert engine.state.liquidity == pytest.approx(1.0e9)

    # Cross left: flip back to global - outside = 0.
    engine.step(calc_sqrt_price_a(ENTRY_TICK), ENTRY_TICK, 0.0, 0.0)
    assert engine.ticks[upper].fee_growth_outside_0 == pytest.approx(0.0)
    assert engine.state.liquidity == pytest.approx(1.0e9)


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

    post = Tick(ENTRY_TICK + 30)
    result = engine.step(calc_sqrt_price_a(post), post, 1_000.0, 1_000.0)

    assert engine.state.fee_growth_global_0 > 0.0
    assert result["fees_accrued"][pid_above] == (0.0, 0.0)
    assert result["fees_accrued"][pid_below] == (0.0, 0.0)


# ---------------------------------------------------------------------------
# 13-15. close_position
# ---------------------------------------------------------------------------
def test_close_position_returns_accrued_fees() -> None:
    engine = _engine(liquidity=1.0)
    pid = engine.open_position(Tick(ENTRY_TICK - 120), Tick(ENTRY_TICK + 120), 1.0)
    in_tick = Tick(ENTRY_TICK + 30)
    engine.step(calc_sqrt_price_a(in_tick), in_tick, 100.0, 0.0)

    accrued = engine.position_fees(pid)
    assert accrued[0] == pytest.approx(0.3)
    returned = engine.close_position(pid)
    assert returned == pytest.approx(accrued)
    assert returned[0] == pytest.approx(0.3)
    assert returned[0] > 0.0


def test_close_position_keeps_sentinel_ticks() -> None:
    engine = _engine(liquidity=0.0)
    pid = engine.open_position(Tick(ENTRY_TICK - 120), Tick(ENTRY_TICK + 120), 750.0)
    lower = engine.positions[pid].tick_lower
    upper = engine.positions[pid].tick_upper

    engine.close_position(pid)

    assert pid not in engine.positions
    assert pid not in engine.position_fee_snapshots
    # The sentinel ticks are left initialized with no liquidity impact.
    for tick in (lower, upper):
        assert engine.ticks[tick].liquidity_gross == 1.0
        assert engine.ticks[tick].liquidity_net == 0.0
        assert engine.ticks[tick].initialized is True
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
# 17-18. shared ticks (marginal sentinels)
# ---------------------------------------------------------------------------
def test_open_shared_ticks_keep_sentinel() -> None:
    engine = _engine(liquidity=0.0)
    lower = Tick(ENTRY_TICK - 120)
    upper = Tick(ENTRY_TICK + 120)
    pid1 = engine.open_position(lower, upper, 100.0)
    pid2 = engine.open_position(lower, upper, 250.0)
    snapped_lower = engine.positions[pid1].tick_lower
    snapped_upper = engine.positions[pid1].tick_upper
    assert engine.positions[pid2].tick_lower == snapped_lower
    assert engine.positions[pid2].tick_upper == snapped_upper

    assert engine.ticks[snapped_lower].liquidity_gross == 1.0
    assert engine.ticks[snapped_lower].liquidity_net == 0.0
    assert engine.ticks[snapped_upper].liquidity_gross == 1.0
    assert engine.ticks[snapped_upper].liquidity_net == 0.0
    assert engine.state.liquidity == pytest.approx(0.0)


def test_closing_all_shared_positions_keeps_sentinel_ticks() -> None:
    engine = _engine(liquidity=0.0)
    lower = Tick(ENTRY_TICK - 120)
    upper = Tick(ENTRY_TICK + 120)
    pid1 = engine.open_position(lower, upper, 100.0)
    pid2 = engine.open_position(lower, upper, 250.0)
    snapped_lower = engine.positions[pid1].tick_lower
    snapped_upper = engine.positions[pid1].tick_upper

    engine.close_position(pid1)
    assert engine.ticks[snapped_lower].initialized is True
    assert engine.ticks[snapped_lower].liquidity_gross == 1.0

    engine.close_position(pid2)
    assert engine.ticks[snapped_lower].initialized is True
    assert engine.ticks[snapped_upper].initialized is True


# ---------------------------------------------------------------------------
# External mint/burn lattice
# ---------------------------------------------------------------------------
def test_apply_mint_and_burn_updates_external_liquidity() -> None:
    engine = _engine(liquidity=0.0)
    _mint(engine, 196140, 196380, 1_000.0)
    assert engine.state.liquidity == pytest.approx(1_000.0)
    assert engine.ticks[Tick(196140)].liquidity_net == 1_000.0
    assert engine.ticks[Tick(196380)].liquidity_net == -1_000.0

    engine.apply_liquidity_event(
        {
            "event_type": "burn",
            "tick_lower": 196140,
            "tick_upper": 196380,
            "liquidity_amount": "1000",
        }
    )
    assert engine.state.liquidity == pytest.approx(0.0)
    assert Tick(196140) not in engine.ticks
    assert Tick(196380) not in engine.ticks


def test_apply_burn_tolerates_float64_residue_above_2_pow_53() -> None:
    """ADR-020: an exact cancellation above 2**53 is not a stream inconsistency."""
    engine = _engine(liquidity=0.0)
    amount = 11_806_162_276_124_034  # > 2**53, where float64's ULP is 2
    # A float64 accumulator can land a few units below the integer burn amount
    # after a run of additions; this burn then cancels it to ~zero.
    gross = float(amount) - 2.0
    engine.ticks[Tick(ENTRY_TICK - 120)] = TickState(
        liquidity_gross=gross, liquidity_net=gross, initialized=True
    )
    engine.ticks[Tick(ENTRY_TICK + 120)] = TickState(
        liquidity_gross=gross, liquidity_net=-gross, initialized=True
    )
    engine.state.liquidity = gross

    engine.apply_liquidity_event(
        {
            "event_type": "burn",
            "tick_lower": ENTRY_TICK - 120,
            "tick_upper": ENTRY_TICK + 120,
            "liquidity_amount": str(amount),
        }
    )

    assert Tick(ENTRY_TICK - 120) not in engine.ticks
    assert Tick(ENTRY_TICK + 120) not in engine.ticks


def test_apply_burn_still_rejects_a_genuine_over_burn() -> None:
    """ADR-020: the tolerance absorbs float64 rounding only, not real gaps."""
    engine = _engine(liquidity=0.0)
    _mint(engine, ENTRY_TICK - 120, ENTRY_TICK + 120, 1_000.0)
    with pytest.raises(PositionError, match="would drive liquidity_gross negative"):
        engine.apply_liquidity_event(
            {
                "event_type": "burn",
                "tick_lower": ENTRY_TICK - 120,
                "tick_upper": ENTRY_TICK + 120,
                "liquidity_amount": "2000",
            }
        )


def test_apply_zero_amount_liquidity_is_noop() -> None:
    engine = _engine(liquidity=0.0)
    _mint(engine, ENTRY_TICK - 120, ENTRY_TICK + 120, 1_000.0)
    before = dict(engine.ticks)
    engine.apply_liquidity_event(
        {
            "event_type": "burn",
            "tick_lower": ENTRY_TICK - 120,
            "tick_upper": ENTRY_TICK + 120,
            "liquidity_amount": "0",
        }
    )
    assert engine.state.liquidity == pytest.approx(1_000.0)
    assert engine.ticks == before


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
    assert raw == pytest.approx(18257.418583505536, rel=1e-15)
    exact = Decimal(ANCHOR_3000_X96) / Decimal(Q96)
    representation_error = Decimal.from_float(raw) - exact
    assert representation_error < Decimal("0")
    assert abs(representation_error) < Decimal("1e-9")

    # The data golden: this anchor converts back to $3000 exactly (USDC/WETH).
    price = sqrt_price_x96_to_price(ANCHOR_3000_X96, 6, 18)
    assert abs(price - Decimal("3000")) < Decimal(3000) * Decimal("1e-20")

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
    # The agent position is marginal against the base liquidity.
    assert tiny_pool_engine.state.liquidity == pytest.approx(1.0e15)

    moved = Tick(ENTRY_TICK + 30)
    result = tiny_pool_engine.step(
        calc_sqrt_price_a(moved), moved, 1_000.0, 0.0
    )
    (pid,) = tiny_pool_engine.positions
    assert result["fees_accrued"][pid][0] > 0.0
    assert result["pool_fee_earned_0"] == pytest.approx(
        1_000.0 * FEE_TIER_BPS / FEE_DENOMINATOR
    )


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


def test_exact_boundary_crossing_uses_the_recorded_tick() -> None:
    """Crossing selection follows the protocol's ``slot0`` tick (ADR-021).

    A downward swap landing exactly on ``boundary(T)`` makes the contract record
    ``slot0.tick = T-1`` while the price stays at the boundary, so the next
    upward move must still cross ``T``.  A price-based rule would see the price
    start on the boundary and skip the crossing.
    """
    engine = _engine(liquidity=0.0)
    upper = Tick(196380)
    _mint(engine, 196140, upper, 1_000.0)
    assert engine.state.liquidity == pytest.approx(1_000.0)

    # Move above the upper bound: external position inactive.
    engine.step(calc_sqrt_price_a(Tick(196500)), Tick(196500), 0.0, 0.0)
    assert engine.state.liquidity == pytest.approx(0.0)

    # Land exactly on the upper bound from above: recorded tick is ``T-1`` and
    # the position is active again.
    engine.step(calc_sqrt_price_a(upper), Tick(int(upper) - 1), 0.0, 0.0)
    assert engine.state.tick == int(upper) - 1
    assert engine.state.liquidity == pytest.approx(1_000.0)

    # Reverse upward off the boundary: ``T`` must be crossed a second time.
    result = engine.step(calc_sqrt_price_a(upper), upper, 0.0, 0.0)
    assert result["ticks_crossed"] == [upper]
    assert engine.state.liquidity == pytest.approx(0.0)


def test_step_skips_uninitialized_ticks() -> None:
    engine = _engine(liquidity=1_000.0)
    # A mint outside the current tick, then a burn that removes both ticks.
    _mint(engine, ENTRY_TICK + 180, ENTRY_TICK + 360, 500.0)
    engine.apply_liquidity_event(
        {
            "event_type": "burn",
            "tick_lower": ENTRY_TICK + 180,
            "tick_upper": ENTRY_TICK + 360,
            "liquidity_amount": "500",
        }
    )

    new_tick = Tick(ENTRY_TICK + 480)
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
    engine = _engine(liquidity=1.0)
    pid = engine.open_position(Tick(ENTRY_TICK - 120), Tick(ENTRY_TICK + 120), 1.0)
    in_tick = Tick(ENTRY_TICK + 30)
    engine.step(calc_sqrt_price_a(in_tick), in_tick, 100.0, 0.0)

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


# ---------------------------------------------------------------------------
# Zero-liquidity final segment (reviewer finding 1, ADR-019)
# ---------------------------------------------------------------------------
def test_step_zero_liquidity_final_segment_does_not_divide_by_zero() -> None:
    """A swap exhausting liquidity past the last initialized tick must not crash.

    The final segment has ``L == 0`` while earlier segments have liquidity, so
    the float apportionment leaves a tiny (here negative) remainder for the
    last segment.  ``FeeGrowthTracker.apply_swap`` guards this with
    ``if L > 0 and fee_seg > 0``; the engine must mirror that guard rather than
    dividing by ``L == 0``.  The helper below picks a gross input whose
    remainder is genuinely non-zero (``-2.3e-10``), so this is a regression
    test for the guard, not merely a happy path.
    """
    engine = _engine(liquidity=0.0)
    # Segment 0 has L=1e6, segment 1 has L=3.7e6, segment 2 (the final one) L=0.
    _mint(engine, ENTRY_TICK - 600, ENTRY_TICK + 60, 1.0e6)
    _mint(engine, ENTRY_TICK + 60, ENTRY_TICK + 120, 3.7e6)
    assert engine.state.liquidity == pytest.approx(1.0e6)

    post = Tick(ENTRY_TICK + 600)
    result = engine.step(calc_sqrt_price_a(post), post, 0.0, 1_234_567.89)

    assert result["ticks_crossed"] == [Tick(ENTRY_TICK + 60), Tick(ENTRY_TICK + 120)]
    assert engine.state.liquidity == pytest.approx(0.0)
    assert engine.state.tick == post
    # The non-empty segments still accrued a finite, positive fee.
    assert math.isfinite(engine.state.fee_growth_global_1)
    assert engine.state.fee_growth_global_1 > 0.0


# ---------------------------------------------------------------------------
# Price-derived crossing direction (reviewer finding 2, ADR-019)
# ---------------------------------------------------------------------------
def test_step_crosses_tick_when_reversing_off_a_boundary() -> None:
    """Reversing off a boundary crosses it, matching the exact tracker (ADR-021).

    This is the real-lane regression (block 15,369,829): a downward swap ended
    exactly on ``boundary(T)`` and recorded ``slot0.tick = T-1``; the next
    upward move therefore had to cross ``T`` even though the price started on
    its boundary.  Keying off the price instead silently left the external
    position active and corrupted ``state.liquidity``.
    """
    boundary = tick_to_sqrt_price_x96(ENTRY_TICK)
    state = PoolState(
        sqrt_price=boundary / float(Q96),
        tick=Tick(ENTRY_TICK - 1),  # slot0 tick recorded by the down-landing
        liquidity=0.0,
        fee_growth_global_0=0.0,
        fee_growth_global_1=0.0,
        fee_tier_bps=FEE_TIER_BPS,
        tick_spacing=SPACING,
        sqrt_price_x96=boundary,
    )
    engine = PoolEngine(state=state, ticks={})
    # External liquidity begins AT the boundary tick; it is inactive while the
    # current tick (T-1) sits below it, so state.liquidity starts at 0.
    _mint(engine, ENTRY_TICK, ENTRY_TICK + 600, 5.0e6)
    assert engine.state.liquidity == 0.0

    post = Tick(ENTRY_TICK + 10)
    result = engine.step(
        calc_sqrt_price_a(post),
        post,
        0.0,
        1.0e6,
        new_sqrt_price_x96=tick_to_sqrt_price_x96(post),
    )

    assert result["ticks_crossed"] == [Tick(ENTRY_TICK)]
    assert engine.state.liquidity == pytest.approx(5.0e6)


# ---------------------------------------------------------------------------
# Protocol-fee split (reviewer finding 4, ADR-019)
# ---------------------------------------------------------------------------
def test_step_applies_protocol_fee_split_before_lp_distribution() -> None:
    """``fee_protocol`` follows the tracker's ``lp_fee = fee - fee // fp`` split."""
    engine = _engine(liquidity=4_000.0)
    pid = engine.open_position(Tick(ENTRY_TICK - 120), Tick(ENTRY_TICK + 120), 2_000.0)
    # fp0 = 4 -> the protocol keeps a quarter of the token0 fee, LP keeps 3/4.
    engine.state.fee_protocol = 0x04

    post = Tick(ENTRY_TICK + 30)
    result = engine.step(calc_sqrt_price_a(post), post, 1.0e9, 0.0)

    gross_fee = 1.0e9 * FEE_TIER_BPS / FEE_DENOMINATOR  # 3e6
    lp_fee = gross_fee - gross_fee // 4  # 2.25e6
    assert engine.state.fee_growth_global_0 == pytest.approx(lp_fee / 4_000.0)
    assert engine.state.fee_growth_global_0 < gross_fee / 4_000.0
    # The reported gross pool fee is unchanged; the cut is simply not LP income.
    assert result["pool_fee_earned_0"] == pytest.approx(gross_fee)
    assert result["fees_accrued"][pid][0] == pytest.approx(lp_fee * 2_000.0 / 4_000.0)
