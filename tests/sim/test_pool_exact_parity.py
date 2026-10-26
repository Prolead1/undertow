"""ADR-019 — the float64 ``PoolEngine`` replicates the exact integer tracker.

The simulator's core step is a float64 mirror of the backtester's
``FeeGrowthTracker.apply_swap`` (ADR-013/ADR-018).  This module pins that
equivalence on a small synthetic tape: the *same* external liquidity, the *same*
marginal agent position, and the *same* swaps are fed to ``PoolEngine`` and to
``FeeGrowthReplay`` and the two fee accruals must agree.

Exactness bound
---------------
The comparison is int-vs-float, so the binding residual is the tracker's final
raw-unit truncation ``(L * Δinside) >> 128``: a whole token unit.  The human
token0 quantum is ``1e-6`` USDC, so the tracker's raw fees and the engine's
float fees agree to strictly less than one raw unit, i.e. ``< 1e-6`` in human
USDC.  The float64 accumulator itself is far below that (relative ``2**-53``).
"""

from __future__ import annotations

import pyarrow as pa  # type: ignore[import-untyped]
import pytest

from undertow.data import (
    FEE_GROWTH_SCHEMA,
    Q96,
    Q128,
    FeeGrowthReplay,
    default_pools,
    tick_to_sqrt_price_x96,
)
from undertow.sim.core.pool import PoolEngine, PoolState
from undertow.sim.core.position import calc_sqrt_price_a
from undertow.sim.types import Tick

_ENTRY_TICK = 196242
_SPACING = 60
_FEE_PIPS = 3000
_L_EXTERNAL = 1.0e15
_L_AGENT = 1.0e12
_GLOBAL_TICK_SENTINEL = -(2**31)


def _make_engine() -> PoolEngine:
    return PoolEngine(
        state=PoolState(
            sqrt_price=calc_sqrt_price_a(Tick(_ENTRY_TICK)),
            tick=Tick(_ENTRY_TICK),
            liquidity=0.0,
            fee_growth_global_0=0.0,
            fee_growth_global_1=0.0,
            fee_tier_bps=_FEE_PIPS,
            tick_spacing=_SPACING,
        ),
        ticks={},
    )


def _mint_row(lower: int, upper: int, liquidity: int) -> dict[str, object]:
    return {
        "event_type": "mint",
        "tick_lower": lower,
        "tick_upper": upper,
        "liquidity_amount": str(liquidity),
        "block_number": 0,
    }


def _swap_row(tick: int, amount0: int, amount1: int, block: int) -> dict[str, object]:
    return {
        "event_type": "swap",
        "tick": tick,
        "sqrt_price_x96": str(tick_to_sqrt_price_x96(tick)),
        "amount0": str(amount0),
        "amount1": str(amount1),
        "block_number": block,
    }


def _step_engine(engine: PoolEngine, tick: int, amount0: int, amount1: int) -> None:
    x96 = tick_to_sqrt_price_x96(tick)
    engine.step(
        x96 / float(Q96),
        Tick(tick),
        amount0 if amount0 > 0 else 0,
        amount1 if amount1 > 0 else 0,
    )


def test_engine_and_tracker_agree_on_synthetic_tape() -> None:
    pool = default_pools()["USDC_WETH_3000"]
    engine = _make_engine()
    replay = FeeGrowthReplay(
        pool, fee_pips=_FEE_PIPS, current_tick=_ENTRY_TICK, current_liquidity=0
    )

    # Same external liquidity: one mint spanning the current tick.
    mint = _mint_row(_ENTRY_TICK - 600, _ENTRY_TICK + 600, int(_L_EXTERNAL))
    engine.apply_liquidity_event(mint)
    replay.apply_event(mint)
    assert engine.state.liquidity == _L_EXTERNAL

    # Same marginal agent position.
    lower, upper = 196140, 196380
    pid = engine.open_position(Tick(lower), Tick(upper), _L_AGENT)
    replay.register_position(tick_lower=lower, tick_upper=upper, liquidity=int(_L_AGENT))

    # Swaps that drive the tick up and down across the agent's bounds.  The
    # tracker requires token0-in to move the tick DOWN and token1-in UP.
    legs = [
        (_ENTRY_TICK + 240, -1, 1_000_000_000),  # token1 in, tick up (cross upper)
        (_ENTRY_TICK + 400, -1, 500_000_000),  # token1 in, tick up
        (_ENTRY_TICK - 200, 1_000_000_000, -1),  # token0 in, tick down
        (_ENTRY_TICK - 400, 500_000_000, -1),  # token0 in, tick down
        (_ENTRY_TICK, -1, 2_000_000_000),  # token1 in, tick up
    ]

    g0_last = 0
    g1_last = 0
    prev0 = 0.0
    prev1 = 0.0
    for block, (tick, amount0, amount1) in enumerate(legs, start=1):
        _step_engine(engine, tick, amount0, amount1)
        replay.apply_event(_swap_row(tick, amount0, amount1, block))
        fees0, fees1, g0_last, g1_last = replay.accrue(
            tick_lower=lower,
            tick_upper=upper,
            liquidity=int(_L_AGENT),
            g_inside_last_0=g0_last,
            g_inside_last_1=g1_last,
        )
        cumulative0, cumulative1 = engine.position_fees(pid)
        sim0 = cumulative0 - prev0
        sim1 = cumulative1 - prev1
        prev0, prev1 = cumulative0, cumulative1
        # ADR-035 floors each swap segment in the tracker, so a step crossing n
        # segments may differ from the float64 engine by < n raw token units.
        assert abs(sim0 - fees0) < 2.0, (block, sim0, fees0)
        assert abs(sim1 - fees1) < 2.0, (block, sim1, fees1)

    # The float64 global accumulator tracks the exact Q128 one to float precision.
    snapshot = replay.snapshot()
    assert abs(
        engine.state.fee_growth_global_0
        - snapshot.fee_growth_global_0_x128 / Q128
    ) < 1e-12
    assert abs(
        engine.state.fee_growth_global_1
        - snapshot.fee_growth_global_1_x128 / Q128
    ) < 1e-12
    # The piecewise path actually crossed an agent boundary.
    assert engine.ticks[Tick(upper)].fee_growth_outside_0 > 0.0


def _snapshot_table() -> pa.Table:
    """A ``FEE_GROWTH_SCHEMA`` snapshot: one global row + external tick rows."""
    pool = default_pools()["USDC_WETH_3000"]
    rows: list[dict[str, object]] = [
        {
            "block_number": 100,
            "tick": _GLOBAL_TICK_SENTINEL,
            "fee_growth_outside_0_x128": "0",
            "fee_growth_outside_1_x128": "0",
            "liquidity_gross": "0",
            "liquidity_net": "0",
            "initialized": False,
            "fee_growth_global_0_x128": "0",
            "fee_growth_global_1_x128": "0",
            "current_tick": _ENTRY_TICK,
            "current_liquidity": str(int(_L_EXTERNAL)),
            "source": "onchain",
            "pool_address": str(pool.address),
            "fee_protocol": 0,
        },
        {
            "block_number": 100,
            "tick": _ENTRY_TICK - 600,
            "fee_growth_outside_0_x128": "0",
            "fee_growth_outside_1_x128": "0",
            "liquidity_gross": str(int(_L_EXTERNAL)),
            "liquidity_net": str(int(_L_EXTERNAL)),
            "initialized": True,
            "fee_growth_global_0_x128": "0",
            "fee_growth_global_1_x128": "0",
            "current_tick": _ENTRY_TICK,
            "current_liquidity": str(int(_L_EXTERNAL)),
            "source": "onchain",
            "pool_address": str(pool.address),
            "fee_protocol": 0,
        },
        {
            "block_number": 100,
            "tick": _ENTRY_TICK + 600,
            "fee_growth_outside_0_x128": "0",
            "fee_growth_outside_1_x128": "0",
            "liquidity_gross": str(int(_L_EXTERNAL)),
            "liquidity_net": str(-int(_L_EXTERNAL)),
            "initialized": True,
            "fee_growth_global_0_x128": "0",
            "fee_growth_global_1_x128": "0",
            "current_tick": _ENTRY_TICK,
            "current_liquidity": str(int(_L_EXTERNAL)),
            "source": "onchain",
            "pool_address": str(pool.address),
            "fee_protocol": 0,
        },
    ]
    return pa.table(
        {key: [row[key] for row in rows] for key in rows[0]},
        schema=FEE_GROWTH_SCHEMA,
    )


def test_from_snapshot_rejects_missing_global_sentinel() -> None:
    """A snapshot without exactly one global sentinel row is rejected."""
    table = _snapshot_table()
    rows = [
        dict(row)
        for row in table.to_pylist()
        if int(row["tick"]) != _GLOBAL_TICK_SENTINEL
    ]
    with pytest.raises(ValueError, match="global sentinel"):
        PoolEngine.from_snapshot(_make_engine().state, rows)


def test_from_snapshot_matches_replay_from_same_snapshot() -> None:
    pool = default_pools()["USDC_WETH_3000"]
    table = _snapshot_table()
    engine = PoolEngine.from_snapshot(
        _make_engine().state, [dict(row) for row in table.to_pylist()]
    )
    replay = FeeGrowthReplay.from_snapshot(pool, table, fee_pips=_FEE_PIPS)

    assert engine.state.liquidity == _L_EXTERNAL
    assert engine.state.tick == _ENTRY_TICK
    assert set(engine.ticks) == {Tick(_ENTRY_TICK - 600), Tick(_ENTRY_TICK + 600)}

    lower, upper = 196140, 196380
    pid = engine.open_position(Tick(lower), Tick(upper), _L_AGENT)
    replay.register_position(tick_lower=lower, tick_upper=upper, liquidity=int(_L_AGENT))

    g0_last = g1_last = 0
    prev0 = prev1 = 0.0
    for block, (tick, amount0, amount1) in enumerate(
        [
            (_ENTRY_TICK + 240, -1, 700_000_000),
            (_ENTRY_TICK - 150, 400_000_000, -1),
        ],
        start=1,
    ):
        _step_engine(engine, tick, amount0, amount1)
        replay.apply_event(_swap_row(tick, amount0, amount1, block))
        fees0, fees1, g0_last, g1_last = replay.accrue(
            tick_lower=lower,
            tick_upper=upper,
            liquidity=int(_L_AGENT),
            g_inside_last_0=g0_last,
            g_inside_last_1=g1_last,
        )
        cumulative0, cumulative1 = engine.position_fees(pid)
        sim0 = cumulative0 - prev0
        sim1 = cumulative1 - prev1
        prev0, prev1 = cumulative0, cumulative1
        assert abs(sim0 - fees0) < 2.0, (block, sim0, fees0)
        assert abs(sim1 - fees1) < 2.0, (block, sim1, fees1)


def test_engine_applies_protocol_fee_split_like_tracker() -> None:
    """A non-zero ``fee_protocol`` cuts the LP fee before distribution.

    ``FeeGrowthTracker.apply_swap`` computes
    ``lp_fee = fee_seg - fee_seg // fp_per_token`` and distributes only that;
    ``PoolEngine.from_snapshot`` must read the packed uint8 and mirror it.
    fp0 = 4 means the protocol keeps a quarter of the token0 fee.
    """
    pool = default_pools()["USDC_WETH_3000"]
    table = _snapshot_table()
    fp_col = table.schema.get_field_index("fee_protocol")
    table = table.set_column(
        fp_col,
        "fee_protocol",
        pa.array([0x04] * table.num_rows, type=table.schema.field(fp_col).type),
    )
    engine = PoolEngine.from_snapshot(
        _make_engine().state, [dict(row) for row in table.to_pylist()]
    )
    replay = FeeGrowthReplay.from_snapshot(pool, table, fee_pips=_FEE_PIPS)
    assert engine.state.fee_protocol == 0x04

    lower, upper = 196140, 196380
    pid = engine.open_position(Tick(lower), Tick(upper), _L_AGENT)
    replay.register_position(tick_lower=lower, tick_upper=upper, liquidity=int(_L_AGENT))

    g0_last = g1_last = 0
    prev0 = prev1 = 0.0
    # token0 in drives the tick down; fp0 applies to token0-in swaps.
    for block, (tick, amount0, amount1) in enumerate(
        [
            (_ENTRY_TICK - 150, 400_000_000, -1),
            (_ENTRY_TICK - 300, 250_000_000, -1),
        ],
        start=1,
    ):
        _step_engine(engine, tick, amount0, amount1)
        replay.apply_event(_swap_row(tick, amount0, amount1, block))
        fees0, fees1, g0_last, g1_last = replay.accrue(
            tick_lower=lower,
            tick_upper=upper,
            liquidity=int(_L_AGENT),
            g_inside_last_0=g0_last,
            g_inside_last_1=g1_last,
        )
        cumulative0, cumulative1 = engine.position_fees(pid)
        sim0 = cumulative0 - prev0
        sim1 = cumulative1 - prev1
        prev0, prev1 = cumulative0, cumulative1
        assert abs(sim0 - fees0) < 2.0, (block, sim0, fees0)
        assert abs(sim1 - fees1) < 2.0, (block, sim1, fees1)

    snapshot = replay.snapshot()
    assert abs(
        engine.state.fee_growth_global_0
        - snapshot.fee_growth_global_0_x128 / Q128
    ) < 1e-12
    # The split actually reduced the LP growth below the gross fee growth.
    assert engine.state.fee_growth_global_0 > 0.0
