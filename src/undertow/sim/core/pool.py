"""Tick-lattice pool engine for the training simulator (`CONTRACTS.md` §6).

This module implements the Uniswap V3 fee-growth mechanism of the lesson plan
§5.4 / roadmap §10.2.4 in the simulator's **float64 fast path**: a sparse tick
lattice where each initialized tick stores ``fee_growth_outside`` snapshots,
global fee growth accumulates from swap volume, and per-position fees are
recovered through the ``fee_growth_inside`` formula.  The exact-integer Q128
re-implementation lives in the data module and is the backtester's (S11) path;
S14 quantifies the drift between the two worlds.

Orientation (ADR-009)
---------------------
Everything in :class:`TickState`, :class:`PoolState` and the engine is in
**raw** protocol units:

* ``sqrt_price`` is the raw Uniswap sqrt price ``sqrt(1.0001**tick)``
  (``sqrt_price_x96 / Q96``), monotonically increasing in tick;
* ``tick`` is the raw integer protocol tick;
* ``liquidity`` (``L``) is the raw on-chain liquidity, directly comparable to
  the tape's recorded active liquidity.

The step model is discrete: ticks are crossed first and swap fees are then
accrued over the resulting active liquidity, so a position crossed out during a
step earns nothing on that step's volume (matching the minute-bar simulator,
not the backtester's per-segment exact path; S14 quantifies the drift).

Fee accrual is therefore ``fees_raw = L * Δfee_growth`` with **no unit
conversion**.  Humanising the reported token fees is the job of the reporting
boundary (``Position.uncollected_fees`` divides by ``10**dec0``/``10**dec1``);
:meth:`PoolEngine.position_fees` deliberately returns the raw value so the
formula matches ``CONTRACTS.md`` §6 exactly.

Equation numbers in comments refer to roadmap §10.2.4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from undertow.sim.core.position import Position
from undertow.sim.types import (
    PositionError,
    SqrtPrice,
    Tick,
    TickSpacing,
)

__all__ = ["PoolEngine", "PoolState", "TickState"]

# Uniswap ``fee_tier`` values are pips (units of 1e-6), despite the
# ``fee_tier_bps`` field name: the 0.30% WETH/USDC tier is 3000 pips =
# 30 basis points.  This matches ``undertow.data.types.FeeTier.BPS_30`` and the
# exact reference path ``undertow.data.transforms.feegrowth`` which computes
# ``fee = amount_in * fee_pips // 1_000_000``.  (The S07 brief's ``/ 10000``
# would be 30%, a 100x overcharge.)
_FEE_DENOMINATOR: float = 1_000_000.0


def _snap_tick(tick: Tick, spacing: int) -> Tick:
    """Snap ``tick`` to the nearest multiple of ``spacing``.

    Uses half-up rounding (``floor(x + 0.5)``) rather than Python's banker's
    ``round`` so half-way cases are deterministic.  Position bounds in Uniswap
    V3 are constrained to multiples of the pool's tick spacing.
    """
    return Tick(int(math.floor(int(tick) / spacing + 0.5)) * spacing)


def _select_crossed_ticks(
    ticks: dict[Tick, TickState],
    current_tick: Tick,
    new_tick: Tick,
) -> list[Tick]:
    """Initialized ticks strictly between ``current_tick`` and ``new_tick``.

    Moving right (``new_tick > current_tick``) crosses ticks in
    ``(current_tick, new_tick]``; moving left crosses ticks in
    ``(new_tick, current_tick]``.  Both boundaries are chosen so the set stays
    consistent with the engine's half-open activity predicate
    ``lower <= current_tick < upper``:

    * right: a tick the price *reaches* is entered, so ``t <= new_tick``;
    * left: the price *lands on* ``new_tick`` (which is then the current tick),
      and at ``current_tick == t`` a position with lower bound ``t`` is still
      active, so ``t`` must NOT be crossed.  Hence ``t > new_tick``.

    The returned list is ascending.
    """
    crossed: list[Tick] = []
    for tick in sorted(ticks):
        if not ticks[tick].initialized:
            continue
        if new_tick > current_tick:
            if current_tick < tick <= new_tick:
                crossed.append(tick)
        elif new_tick < tick <= current_tick:
            crossed.append(tick)
    return crossed


@dataclass(slots=True)
class TickState:
    """Per-tick state in the sparse lattice (`CONTRACTS.md` §6)."""

    fee_growth_outside_0: float = 0.0  # Q128 as float64
    fee_growth_outside_1: float = 0.0
    liquidity_gross: float = 0.0  # total L locked at this tick
    liquidity_net: float = 0.0  # signed: + for lower ticks, - for upper ticks
    initialized: bool = False


@dataclass(slots=True)
class PoolState:
    """The pool's mutable state at a single point in time (`CONTRACTS.md` §6).

    All quantities are raw protocol units (ADR-009).
    """

    sqrt_price: SqrtPrice
    tick: Tick
    liquidity: float  # active L at the current tick
    fee_growth_global_0: float
    fee_growth_global_1: float
    fee_tier_bps: int  # e.g. 3000 for 0.30%
    tick_spacing: int

    # -- Fee-growth inside a range (eqs 4-6 of §10.2.4) -------------------
    def fee_growth_below(
        self,
        tick_boundary: Tick,
        fee_growth_outside: float,
        *,
        fee_growth_global: float | None = None,
    ) -> float:
        """Fee growth below ``tick_boundary`` for one token.

        ``fee_growth_outside`` is the outside accumulator of the relevant token.
        The global accumulator for the same token is supplied with the
        keyword-only ``fee_growth_global`` (defaulting to **token0**) because the
        frozen signature does not carry a token selector; ``fee_growth_inside``
        passes the correct global for each token.  A caller computing token1
        must pass ``fee_growth_global=self.fee_growth_global_1`` explicitly — the
        token0 default would otherwise be silently wrong.

        ``current_tick >= tick_boundary`` -> ``fee_growth_outside``; otherwise
        ``fee_growth_global - fee_growth_outside``.
        """
        if fee_growth_global is None:
            fee_growth_global = self.fee_growth_global_0
        if self.tick >= tick_boundary:
            return fee_growth_outside
        return fee_growth_global - fee_growth_outside

    def fee_growth_above(
        self,
        tick_boundary: Tick,
        fee_growth_outside: float,
        *,
        fee_growth_global: float | None = None,
    ) -> float:
        """Fee growth above ``tick_boundary`` for one token.

        ``current_tick < tick_boundary`` -> ``fee_growth_outside``; otherwise
        ``fee_growth_global - fee_growth_outside``.  See
        :meth:`fee_growth_below` for the token-selector convention.
        """
        if fee_growth_global is None:
            fee_growth_global = self.fee_growth_global_0
        if self.tick < tick_boundary:
            return fee_growth_outside
        return fee_growth_global - fee_growth_outside

    def fee_growth_inside(
        self,
        tick_lower: Tick,
        tick_upper: Tick,
        fg_outside_lower_0: float,
        fg_outside_lower_1: float,
        fg_outside_upper_0: float,
        fg_outside_upper_1: float,
    ) -> tuple[float, float]:
        """Fee growth inside ``[tick_lower, tick_upper)`` for both tokens.

        ``fg_inside = fg_global - fg_below(lower) - fg_above(upper)`` per token
        (eqs 4-6 of §10.2.4).
        """
        below_0 = self.fee_growth_below(
            tick_lower, fg_outside_lower_0, fee_growth_global=self.fee_growth_global_0
        )
        below_1 = self.fee_growth_below(
            tick_lower, fg_outside_lower_1, fee_growth_global=self.fee_growth_global_1
        )
        above_0 = self.fee_growth_above(
            tick_upper, fg_outside_upper_0, fee_growth_global=self.fee_growth_global_0
        )
        above_1 = self.fee_growth_above(
            tick_upper, fg_outside_upper_1, fee_growth_global=self.fee_growth_global_1
        )
        inside_0 = self.fee_growth_global_0 - below_0 - above_0
        inside_1 = self.fee_growth_global_1 - below_1 - above_1
        return (inside_0, inside_1)

    def tick_cross(
        self,
        new_tick: Tick,
        ticks: dict[Tick, TickState] | None = None,
        *,
        crossed: list[Tick] | None = None,
    ) -> None:
        """Cross every initialized tick between the current tick and ``new_tick``.

        For each crossed tick, active liquidity is adjusted by
        ``liquidity_net`` (added when moving right, subtracted when moving left)
        and both ``fee_growth_outside`` accumulators are flipped:
        ``outside = fee_growth_global - outside``.

        The lattice is owned by :class:`PoolEngine`, so ``ticks`` must be
        supplied; it is a keyword-defaulted extension of the frozen
        ``tick_cross(new_tick)`` signature (the frozen contract did not account
        for ``PoolState`` lacking the lattice reference).  ``crossed`` lets the
        caller pass the already-selected tick list (e.g. the one
        :meth:`PoolEngine.step` returns) to avoid a second lattice traversal.
        The current tick is **not** updated here — :meth:`PoolEngine.step` does
        that after crossing.
        """
        if ticks is None:
            raise PositionError("tick_cross requires the tick lattice")
        if new_tick == self.tick:
            return
        moving_right = new_tick > self.tick
        if crossed is None:
            crossed = _select_crossed_ticks(ticks, self.tick, new_tick)
        for tick in crossed:
            tick_state = ticks[tick]
            if moving_right:
                self.liquidity += tick_state.liquidity_net
            else:
                self.liquidity -= tick_state.liquidity_net
            tick_state.fee_growth_outside_0 = (
                self.fee_growth_global_0 - tick_state.fee_growth_outside_0
            )
            tick_state.fee_growth_outside_1 = (
                self.fee_growth_global_1 - tick_state.fee_growth_outside_1
            )


@dataclass(slots=True)
class PoolEngine:
    """Owns the tick lattice and provides the step interface (`CONTRACTS.md` §6).

    ``state.liquidity`` is the pool's **total** active liquidity: it starts at
    whatever base the caller supplies (the tape's recorded active liquidity in
    the marginal-agent model) and this engine adds/subtracts the liquidity of
    the agent-owned positions it manages.  Global fee growth is therefore
    distributed over ``base + sum(active positions)``, which yields the
    ``L_agent / L_pool_active`` fee share of the plan.
    """

    state: PoolState
    ticks: dict[Tick, TickState]
    positions: dict[int, Position] = field(default_factory=dict)
    position_fee_snapshots: dict[int, tuple[float, float]] = field(
        default_factory=dict
    )
    _next_position_id: int = 0

    # -- Internal helpers -------------------------------------------------
    def _adjust_tick(self, tick: Tick, liquidity: float, net_delta: float) -> None:
        """Add ``liquidity`` to gross and ``net_delta`` to net at ``tick``.

        A tick that does not exist (or was cleaned up) is (re)initialized with
        its ``fee_growth_outside`` set to ``fee_growth_global`` when the tick is
        at or below the current tick, else 0 — the Uniswap initialization rule.
        """
        tick_state = self.ticks.get(tick)
        if tick_state is None or not tick_state.initialized:
            outside_0 = (
                self.state.fee_growth_global_0 if self.state.tick >= tick else 0.0
            )
            outside_1 = (
                self.state.fee_growth_global_1 if self.state.tick >= tick else 0.0
            )
            tick_state = TickState(outside_0, outside_1, 0.0, 0.0, True)
            self.ticks[tick] = tick_state
        tick_state.liquidity_gross += liquidity
        tick_state.liquidity_net += net_delta

    def _crossed_ticks(self, old_tick: Tick, new_tick: Tick) -> list[Tick]:
        return _select_crossed_ticks(self.ticks, old_tick, new_tick)

    # -- Position management ----------------------------------------------
    def open_position(
        self,
        tick_lower: Tick,
        tick_upper: Tick,
        liquidity: float,
    ) -> int:
        """Register a new position; returns a fresh ``position_id``.

        Bounds are snapped to the nearest tick-spacing multiple before
        validation (``tick_lower < tick_upper``, ``liquidity > 0``).  The
        position's bounds become initialized lattice ticks and, when the range
        straddles the current tick, its ``L`` is added to ``state.liquidity``.
        """
        spacing = int(self.state.tick_spacing)
        if spacing <= 0:
            raise PositionError("tick_spacing must be > 0")
        if liquidity <= 0:
            raise PositionError("liquidity must be > 0")
        lower = _snap_tick(tick_lower, spacing)
        upper = _snap_tick(tick_upper, spacing)
        if lower >= upper:
            raise PositionError("tick_lower must be < tick_upper after snapping")

        amount = float(liquidity)
        self._adjust_tick(lower, amount, amount)
        self._adjust_tick(upper, amount, -amount)

        if lower <= self.state.tick < upper:
            self.state.liquidity += amount

        position = Position(lower, upper, amount, TickSpacing(spacing))
        position_id = self._next_position_id
        self._next_position_id += 1
        self.positions[position_id] = position

        lower_ts = self.ticks[lower]
        upper_ts = self.ticks[upper]
        snapshot = self.state.fee_growth_inside(
            lower,
            upper,
            lower_ts.fee_growth_outside_0,
            lower_ts.fee_growth_outside_1,
            upper_ts.fee_growth_outside_0,
            upper_ts.fee_growth_outside_1,
        )
        # Keep the engine's snapshot and the Position's own mirror in sync so
        # ``Position.uncollected_fees`` (human units) agrees with
        # ``position_fees`` (raw units) for the same interval.
        self.position_fee_snapshots[position_id] = snapshot
        position.snapshot_fees(snapshot[0], snapshot[1])
        return position_id

    def close_position(self, position_id: int) -> tuple[float, float]:
        """Remove ``position_id`` and return its accrued raw fees.

        The position's liquidity is removed from both lattice ticks; if that
        leaves a tick with no gross liquidity it is marked uninitialized.  The
        active-liquidity adjustment mirrors :meth:`open_position`.
        """
        position = self.positions.get(position_id)
        if position is None:
            raise PositionError(f"unknown position_id {position_id}")

        fees = self.position_fees(position_id)
        amount = position.liquidity
        lower = position.tick_lower
        upper = position.tick_upper

        lower_ts = self.ticks[lower]
        upper_ts = self.ticks[upper]
        lower_ts.liquidity_gross -= amount
        lower_ts.liquidity_net -= amount
        upper_ts.liquidity_gross -= amount
        upper_ts.liquidity_net += amount

        if lower <= self.state.tick < upper:
            self.state.liquidity -= amount

        for tick_state in (lower_ts, upper_ts):
            if tick_state.liquidity_gross <= 0.0:
                tick_state.liquidity_gross = 0.0
                tick_state.initialized = False

        del self.positions[position_id]
        del self.position_fee_snapshots[position_id]
        return fees

    def position_fees(self, position_id: int) -> tuple[float, float]:
        """Current uncollected **raw** fees for ``position_id``.

        ``fees = L * (fg_inside_current - fg_inside_snapshot)`` per token,
        where ``fg_inside`` is recomputed from the live lattice.  The result is
        **raw** token units and is deliberately *not* humanised here; the
        reporting accessor is ``Position.uncollected_fees``, which subtracts
        the Position's mirror snapshot (kept in sync at open) and divides by
        ``10**dec0``/``10**dec1`` (ADR-009).  Mixing the two accessors mixes
        raw and human units.
        """
        position = self.positions.get(position_id)
        if position is None:
            raise PositionError(f"unknown position_id {position_id}")
        lower_ts = self.ticks[position.tick_lower]
        upper_ts = self.ticks[position.tick_upper]
        inside_0, inside_1 = self.state.fee_growth_inside(
            position.tick_lower,
            position.tick_upper,
            lower_ts.fee_growth_outside_0,
            lower_ts.fee_growth_outside_1,
            upper_ts.fee_growth_outside_0,
            upper_ts.fee_growth_outside_1,
        )
        snapshot_0, snapshot_1 = self.position_fee_snapshots[position_id]
        return (
            position.liquidity * (inside_0 - snapshot_0),
            position.liquidity * (inside_1 - snapshot_1),
        )

    # -- Core step --------------------------------------------------------
    def step(
        self,
        new_sqrt_price: SqrtPrice,
        new_tick: Tick,
        swap_volume_0: float = 0.0,
        swap_volume_1: float = 0.0,
    ) -> dict[str, Any]:
        """Advance the pool by one step.

        Crosses ticks when the tick changes, then accrues the swap fee at the
        pool's fee tier over the resulting active liquidity and returns the
        documented dict: ``fees_accrued`` (per position, raw),
        ``ticks_crossed``, ``active_liquidity``, ``pool_fee_earned_0/1`` (raw).

        ``new_tick`` is expected to be slot0-consistent with ``new_sqrt_price``
        (the caller's price process is the source of truth), so the state's
        half-open activity predicate ``lower <= tick < upper`` stays coherent
        with the crossing set.  If ``state.liquidity == 0`` the nominal
        ``pool_fee_earned_*`` is still reported, but no fee growth can be
        attributed to any position (a zero-liquidity swap is not physically
        realizable; this is a defensive branch).
        """
        ticks_crossed: list[Tick] = []
        if new_tick != self.state.tick:
            ticks_crossed = self._crossed_ticks(self.state.tick, new_tick)
            self.state.tick_cross(new_tick, self.ticks, crossed=ticks_crossed)
            self.state.tick = new_tick
        # slot0 stores the actual sqrt price on every step, including moves that
        # stay within a tick, so downstream valuation never sees a stale price.
        self.state.sqrt_price = new_sqrt_price

        fee_0 = swap_volume_0 * self.state.fee_tier_bps / _FEE_DENOMINATOR
        fee_1 = swap_volume_1 * self.state.fee_tier_bps / _FEE_DENOMINATOR
        if self.state.liquidity > 0.0:
            self.state.fee_growth_global_0 += fee_0 / self.state.liquidity
            self.state.fee_growth_global_1 += fee_1 / self.state.liquidity

        fees_accrued = {
            position_id: self.position_fees(position_id)
            for position_id in self.positions
        }
        return {
            "fees_accrued": fees_accrued,
            "ticks_crossed": ticks_crossed,
            "active_liquidity": self.state.liquidity,
            "pool_fee_earned_0": fee_0,
            "pool_fee_earned_1": fee_1,
        }
