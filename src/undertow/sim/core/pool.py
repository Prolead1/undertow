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

The step model is **exact and piecewise** (ADR-019), a float64 replica of the
backtester's integer ``FeeGrowthTracker.apply_swap`` (ADR-013): the swap is
split into the segments bounded by the initialized ticks it crosses, the gross
input is apportioned across those segments proportionally to their net inputs
(the last segment absorbing the float remainder), each segment accrues
``fee_segment = gross_segment * fee_pips / 1_000_000`` over that segment's
active liquidity, and the boundary tick is crossed **after** its segment's fee
is applied (the protocol's update-then-cross order).  Ticks are crossed even on
a zero-volume alignment step, since the price move itself changes the active
liquidity.

The agent's positions are **marginal** (ADR-019): :meth:`PoolEngine.open_position`
seeds the position's boundary ticks with the ``liquidity_gross=1,
liquidity_net=0`` sentinel of ``FeeGrowthReplay.register_position`` and does
**not** add the agent's ``L`` to ``state.liquidity``.  ``state.liquidity`` is
therefore the *external* pool liquidity (seeded from the on-chain snapshot).

Fee accrual walks each swap segment and credits an active position
``fee_segment * L_agent / (L_agent + L_external_segment)`` (ADR-035; the
augmented marginal denominator), so the position can never be credited more
than the fee the pool actually took.  ``state.fee_growth_global_*`` still
accrues on the external ``L`` for protocol-level state.  Humanising the
reported token fees is the job of the reporting boundary
(``Position.uncollected_fees`` divides by ``10**dec0``/``10**dec1``);
:meth:`PoolEngine.position_fees` deliberately returns the raw ledger value so
``CONTRACTS.md`` §6 matches.

Equation numbers in comments refer to roadmap §10.2.4.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from undertow.data import (
    GLOBAL_TICK_SENTINEL,
    Q96,
    Q128,
    tick_to_sqrt_price_x96,
)
from undertow.sim.core.position import Position, calc_sqrt_price_a
from undertow.sim.types import (
    PositionError,
    SqrtPrice,
    Tick,
    TickSpacing,
)

__all__ = ["PoolEngine", "PoolState", "TickState"]

# Uniswap ``fee_tier`` values are pips (units of 1e-6), despite the
# ``fee_tier_bps`` field name: the primary 0.05% WETH/USDC tier is 500 pips =
# 5 basis points (the 0.30% contrast tier is 3000 pips).  This matches
# ``undertow.data.types.FeeTier.BPS_5`` and the exact reference path
# ``undertow.data.transforms.feegrowth`` which computes
# ``fee = amount_in * fee_pips // 1_000_000``.  (The S07 brief's ``/ 10000``
# would be 30%, a 100x overcharge.)
_FEE_DENOMINATOR: float = 1_000_000.0

# Liquidity on real pools routinely exceeds ``2**53``, where float64 can no
# longer represent the integers exactly.  The simulator's snapshot/tape path
# keeps ``liquidity_gross``/``liquidity_net`` as Python ints (ADR-032), so it
# does not need this allowance; it applies only to explicitly float-seeded
# states (hand-built engines and fixtures), where an exact cancellation
# (`gross + delta == 0`) can leave a few-ULP negative residue.  Treat a residue
# within this fraction of the applied delta as zero, and anything larger as a
# genuine stream inconsistency (ADR-020).
_LIQUIDITY_ROUNDING_TOLERANCE: float = 1e-12


def _as_int(value: object) -> int:
    """Coerce a snapshot/cell value (decimal string, int or ``None``) to ``int``."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return int(str(value))


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
    *,
    sorted_keys: list[Tick] | None = None,
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

    The returned list is ascending.  ``sorted_keys`` is an optional ascending
    key snapshot (ADR-033) so the engine can bisect instead of re-sorting the
    lattice on every swap; the default keeps the function self-contained.
    """
    keys = sorted_keys if sorted_keys is not None else sorted(ticks)
    if new_tick > current_tick:
        lo = bisect.bisect_right(keys, current_tick)
        hi = bisect.bisect_right(keys, new_tick)
    else:
        lo = bisect.bisect_right(keys, new_tick)
        hi = bisect.bisect_right(keys, current_tick)
    return [tick for tick in keys[lo:hi] if ticks[tick].initialized]


@dataclass(slots=True)
class TickState:
    """Per-tick state in the sparse lattice (`CONTRACTS.md` §6)."""

    fee_growth_outside_0: float = 0.0  # Q128 as float64
    fee_growth_outside_1: float = 0.0
    #: Exact ``int`` on the snapshot/tape path, where mint/burn accounting must
    #: cancel exactly above ``2**53`` (ADR-032); ``float`` only for explicitly
    #: float-seeded states (hand-built engines and fixtures).
    liquidity_gross: float | int = 0  # total L locked at this tick
    liquidity_net: float | int = 0  # signed: + for lower ticks, - for upper ticks
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
    fee_tier_bps: int  # e.g. 500 for the primary 0.05% pool
    tick_spacing: int
    #: Exact Q64.96 sqrt price of the last tape swap, retained as informational
    #: provenance for diagnostics.  No logic reads it: crossings are selected
    #: from the recorded ``tick`` (ADR-021) and valuation uses the float
    #: ``sqrt_price``.  ``None`` for synthetic/fixture states.
    sqrt_price_x96: int | None = None
    #: Packed uint8 protocol-fee split from slot0 (bits 0-3 = fp0 for token0-in
    #: swaps, bits 4-7 = fp1 for token1-in swaps); ``0`` means protocol fees are
    #: off and the LP keeps the whole segment fee.  Mirrors
    #: ``FeeGrowthTracker._fee_protocol`` and is applied *before* fee growth is
    #: distributed to the LP (ADR-019, reviewer finding 4).
    fee_protocol: int = 0

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

    ``state.liquidity`` is the pool's **external** active liquidity (ADR-019).
    Agent-owned positions are *marginal*: :meth:`open_position` never adds their
    ``L`` to ``state.liquidity``; it only seeds sentinel boundary ticks.  Fees
    are accrued per swap segment on the augmented denominator
    ``L_agent + L_external`` (ADR-035), exactly like the backtester's
    ``FeeGrowthReplay``.
    """

    state: PoolState
    ticks: dict[Tick, TickState]
    positions: dict[int, Position] = field(default_factory=dict)
    position_fee_snapshots: dict[int, tuple[float, float]] = field(
        default_factory=dict
    )
    #: Raw token fees accrued to each marginal position by ADR-035's exact
    #: per-segment share; :meth:`position_fees` returns this ledger.
    position_accrued: dict[int, tuple[float, float]] = field(default_factory=dict)
    _next_position_id: int = 0
    #: Memoised ``tick_to_sqrt_price_x96(tick)`` boundaries (the exact path calls
    #: it per tick per swap; the Decimal conversion is expensive).
    _tick_sqrt_x96: dict[Tick, int] = field(default_factory=dict)
    #: Ascending lattice keys, cached across swaps so crossing selection bisects
    #: instead of re-sorting the dict (ADR-033).  The engine's mutators clear it;
    #: ``_sorted_keys_len`` additionally catches direct mutations that change the
    #: key count (test/fixture setup bypasses the mutators).
    _sorted_keys: list[Tick] | None = field(default=None, init=False, repr=False)
    _sorted_keys_len: int = field(default=-1, init=False, repr=False)

    def _tick_keys(self) -> list[Tick]:
        """Ascending lattice keys, refreshed when the key set may have changed."""
        keys = self._sorted_keys
        if keys is None or self._sorted_keys_len != len(self.ticks):
            keys = sorted(self.ticks)
            self._sorted_keys = keys
            self._sorted_keys_len = len(self.ticks)
        return keys

    def _boundary_x96(self, tick: Tick) -> int:
        """Exact Q64.96 boundary for ``tick``, memoised."""
        boundary = self._tick_sqrt_x96.get(tick)
        if boundary is None:
            boundary = tick_to_sqrt_price_x96(int(tick))
            self._tick_sqrt_x96[tick] = boundary
        return boundary

    # -- Lattice state ---------------------------------------------------
    def replace_ticks(self, ticks: Mapping[Tick, TickState]) -> None:
        """Replace the whole lattice and invalidate the crossing index (ADR-033).

        ``LpEnvironment`` restores its construction-time base lattice on every
        reset and owns the ``TickState`` copies it passes in.
        """
        self.ticks.clear()
        self.ticks.update(ticks)
        self._sorted_keys = None

    # -- Snapshot seeding -------------------------------------------------
    @classmethod
    def from_snapshot(
        cls,
        state: PoolState,
        snapshot_rows: Iterable[Mapping[str, object]],
    ) -> PoolEngine:
        """Build an engine from a ``FEE_GROWTH_SCHEMA`` snapshot (ADR-018).

        Mirrors ``FeeGrowthReplay.from_snapshot``: the snapshot carries exactly
        one global sentinel row (``tick == GLOBAL_TICK_SENTINEL``) with the
        block's globals, ``current_tick``, ``current_liquidity`` and
        ``fee_protocol``, plus one row per initialized tick.  ``state`` is used
        as the template for the fee tier / tick spacing (and any other config
        not carried by the snapshot).  Q128 integers are divided by ``Q128`` so
        the engine's float convention matches the rest of the sim; the raw sqrt
        price is recovered from ``current_tick``.

        The snapshot leaves the pre-snapshot sqrt price unknown (it carries only
        the tick), so the first crossing swap uses the tick-boundary price —
        the same bounded approximation the exact tracker makes from a snapshot.
        """
        rows = list(snapshot_rows)
        globals_rows = [
            r for r in rows if _as_int(r.get("tick")) == GLOBAL_TICK_SENTINEL
        ]
        if len(globals_rows) != 1:
            # Mirrors ``FeeGrowthReplay.from_snapshot``, which raises
            # ``ValueError`` for the same malformed snapshot.
            raise ValueError(
                "PoolEngine.from_snapshot: expected exactly one global sentinel "
                f"row (tick == {GLOBAL_TICK_SENTINEL}), got {len(globals_rows)}"
            )
        g = globals_rows[0]
        ticks: dict[Tick, TickState] = {}
        for r in rows:
            tick = _as_int(r.get("tick"))
            if tick == GLOBAL_TICK_SENTINEL:
                continue
            ticks[Tick(tick)] = TickState(
                fee_growth_outside_0=_as_int(r.get("fee_growth_outside_0_x128")) / Q128,
                fee_growth_outside_1=_as_int(r.get("fee_growth_outside_1_x128")) / Q128,
                liquidity_gross=_as_int(r.get("liquidity_gross")),
                liquidity_net=_as_int(r.get("liquidity_net")),
                initialized=bool(r.get("initialized", True)),
            )
        current_tick = _as_int(g.get("current_tick"))
        seeded = PoolState(
            sqrt_price=float(calc_sqrt_price_a(Tick(current_tick))),
            tick=Tick(current_tick),
            liquidity=float(_as_int(g.get("current_liquidity"))),
            fee_growth_global_0=_as_int(g.get("fee_growth_global_0_x128")) / Q128,
            fee_growth_global_1=_as_int(g.get("fee_growth_global_1_x128")) / Q128,
            fee_tier_bps=int(state.fee_tier_bps),
            tick_spacing=int(state.tick_spacing),
            fee_protocol=_as_int(g.get("fee_protocol")),
        )
        return cls(state=seeded, ticks=ticks)

    # -- Internal helpers -------------------------------------------------

    def _cross_one(self, tick: Tick, upward: bool) -> None:
        """Cross a single initialized ``tick`` in the exact ``Tick.cross`` order.

        Flips ``fee_growth_outside = fee_growth_global - fee_growth_outside``
        for both tokens and applies the tick's ``liquidity_net`` to the active
        liquidity (add upward, subtract downward).  Called after the crossed
        segment's fee has been added to the global accumulator.
        """
        tick_state = self.ticks[tick]
        tick_state.fee_growth_outside_0 = (
            self.state.fee_growth_global_0 - tick_state.fee_growth_outside_0
        )
        tick_state.fee_growth_outside_1 = (
            self.state.fee_growth_global_1 - tick_state.fee_growth_outside_1
        )
        if upward:
            self.state.liquidity += tick_state.liquidity_net
        else:
            self.state.liquidity -= tick_state.liquidity_net

    # -- External liquidity events ----------------------------------------
    def apply_liquidity_event(self, row: Mapping[str, object]) -> None:
        """Apply one ``mint``/``burn`` row to the external lattice (ADR-019).

        Mirrors ``FeeGrowthTracker.apply_liquidity_event``: both boundary ticks
        get ``liquidity_gross``/``liquidity_net`` updates; a newly initialized
        tick is seeded with ``fee_growth_outside = fee_growth_global`` when it
        is at or below the current tick (else 0); a tick whose gross returns to
        zero is removed; and ``state.liquidity`` moves by the signed delta when
        the range spans the current tick (half-open ``[lower, upper)``).  A
        zero-amount event is a fee-collection no-op (the tracker's ``burn(0)``).
        """
        event_type = str(row.get("event_type", "")).lower()
        if event_type not in ("mint", "burn"):
            raise PositionError(
                f"apply_liquidity_event expects a mint or burn row, got {event_type!r}"
            )
        # The key set may gain/lose a boundary tick below; drop the crossing index.
        self._sorted_keys = None
        tick_lower = _as_int(row.get("tick_lower"))
        tick_upper = _as_int(row.get("tick_upper"))
        liquidity_amount = _as_int(row.get("liquidity_amount"))
        delta = liquidity_amount if event_type == "mint" else -liquidity_amount
        if delta == 0:
            return
        # ``liquidity_gross`` moves by the signed amount on *both* ticks (both
        # reference the same position liquidity); only ``liquidity_net`` differs
        # in sign, exactly as in ``Tick.update``.  This mirrors the tracker.
        for tick, net_delta in ((tick_lower, delta), (tick_upper, -delta)):
            before = self.ticks.get(Tick(tick))
            # ADR-032: snapshot rows are seeded as Python ints and ticks first
            # touched by a tape event start from int ``0``, so the production
            # path does exact integer accounting — a float64 rounding can never
            # hide a sub-ULP remainder (or reject an exact cancellation).  Only
            # explicitly float-seeded states take the ADR-020 float64 path with
            # its delta-relative residue tolerance.
            if before is None:
                gross_before: float | int = 0
                net_before: float | int = 0
            else:
                gross_before = before.liquidity_gross
                net_before = before.liquidity_net
            if isinstance(gross_before, int):
                gross_after: float | int = gross_before + delta
            else:
                gross_after = gross_before + float(delta)
            if isinstance(gross_after, int) and gross_after < 0:
                raise PositionError(
                    f"{event_type} of liquidity {liquidity_amount} at tick {tick} "
                    "would drive liquidity_gross negative — the event stream is "
                    "inconsistent"
                )
            if isinstance(gross_after, float) and gross_after < 0.0:
                if gross_after >= -_LIQUIDITY_ROUNDING_TOLERANCE * abs(float(delta)):
                    # Exact cancellation above 2**53 leaves a float64 residue;
                    # clamp it so the tick is removed rather than rejecting a
                    # consistent stream (ADR-020).
                    gross_after = 0.0
                else:
                    raise PositionError(
                        f"{event_type} of liquidity {liquidity_amount} at tick {tick} "
                        "would drive liquidity_gross negative — the event stream is "
                        "inconsistent"
                    )
            if gross_before == 0.0:
                if self.state.tick >= tick:
                    outside_0 = self.state.fee_growth_global_0
                    outside_1 = self.state.fee_growth_global_1
                else:
                    outside_0 = 0.0
                    outside_1 = 0.0
            else:
                assert before is not None  # gross_before > 0 implies it exists
                outside_0 = before.fee_growth_outside_0
                outside_1 = before.fee_growth_outside_1
            if gross_after == 0.0:
                del self.ticks[Tick(tick)]
            else:
                if isinstance(net_before, int):
                    net_after: float | int = net_before + net_delta
                else:
                    net_after = net_before + float(net_delta)
                self.ticks[Tick(tick)] = TickState(
                    outside_0, outside_1, gross_after, net_after, True
                )
        if tick_lower <= self.state.tick < tick_upper:
            self.state.liquidity += float(delta)

    # -- Position management ----------------------------------------------
    def open_position(
        self,
        tick_lower: Tick,
        tick_upper: Tick,
        liquidity: float,
    ) -> int:
        """Register a new **marginal** agent position; returns a fresh id.

        Bounds are snapped to the nearest tick-spacing multiple before
        validation (``tick_lower < tick_upper``, ``liquidity > 0``).  Each
        boundary tick is seeded only when absent, with the protocol seeding
        convention and the sentinel ``liquidity_gross=1, liquidity_net=0`` of
        ``FeeGrowthReplay.register_position``; the agent's ``L`` is **not** added
        to ``state.liquidity``.  The ``Position`` is recorded and its ADR-035
        per-segment fee ledger is initialised at zero.
        """
        spacing = int(self.state.tick_spacing)
        if spacing <= 0:
            raise PositionError("tick_spacing must be > 0")
        if liquidity <= 0:
            raise PositionError("liquidity must be > 0")
        # Sentinels may add fresh keys; drop the crossing index.
        self._sorted_keys = None
        lower = _snap_tick(tick_lower, spacing)
        upper = _snap_tick(tick_upper, spacing)
        if lower >= upper:
            raise PositionError("tick_lower must be < tick_upper after snapping")

        amount = float(liquidity)
        for tick in (lower, upper):
            if tick in self.ticks:
                continue
            if self.state.tick >= tick:
                outside_0 = self.state.fee_growth_global_0
                outside_1 = self.state.fee_growth_global_1
            else:
                outside_0 = 0.0
                outside_1 = 0.0
            self.ticks[tick] = TickState(outside_0, outside_1, 1, 0, True)

        position = Position(lower, upper, amount, TickSpacing(spacing))
        position_id = self._next_position_id
        self._next_position_id += 1
        self.positions[position_id] = position
        self.position_accrued[position_id] = (0.0, 0.0)

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
        """Forget a **marginal** position and return its accrued raw fees.

        The position never contributed external liquidity, so neither
        ``state.liquidity`` nor the boundary-tick lattice is touched; the
        sentinel ticks are deliberately left in place (the backtester's tracker
        leaves ``register_position`` sentinels too, so a later real mint at the
        same tick does not re-seed ``fee_growth_outside``).
        """
        if position_id not in self.positions:
            raise PositionError(f"unknown position_id {position_id}")
        fees = self.position_fees(position_id)
        del self.positions[position_id]
        del self.position_fee_snapshots[position_id]
        del self.position_accrued[position_id]
        return fees

    def position_fees(self, position_id: int) -> tuple[float, float]:
        """Current uncollected **raw** fees for ``position_id``.

        The ledger is written by :meth:`step` during the per-segment swap walk:
        each marginal position earns ``lp_fee * L_pos / (L_pos + L_external)`` of
        the segment's LP fee (ADR-035), so a position can never be credited more
        than the fee actually taken.  The result is **raw** token units and is
        deliberately *not* humanised here; the reporting accessor divides by
        ``10**dec0``/``10**dec1`` (ADR-009).
        """
        accrued = self.position_accrued.get(position_id)
        if position_id not in self.positions or accrued is None:
            raise PositionError(f"unknown position_id {position_id}")
        return accrued

    def _accrue_marginal_fees(
        self, seg_tick: Tick, liquidity: float, lp_fee: float, input_token: int
    ) -> None:
        """Credit each marginal position its exact share of a segment's LP fee.

        The denominator is the **augmented** liquidity
        ``L_external + sum(L_marginal_active)``, so a position earns
        ``lp_fee * L_pos / (L_pos + L_external)`` — the documented marginal
        share (PLAN §7).  The previous global-growth path implied
        ``lp_fee * L_pos / L_external``, which can exceed the fee actually taken
        when a thin tick segment's external liquidity is smaller than the
        position's own ``L`` (ADR-035).
        """
        active = [
            (position_id, position.liquidity)
            for position_id, position in self.positions.items()
            if position.tick_lower <= seg_tick < position.tick_upper
        ]
        if not active:
            return
        denominator = liquidity + math.fsum(amount for _pid, amount in active)
        if denominator <= 0.0:
            return
        for position_id, amount in active:
            share = lp_fee * amount / denominator
            accrued_0, accrued_1 = self.position_accrued[position_id]
            if input_token == 0:
                self.position_accrued[position_id] = (accrued_0 + share, accrued_1)
            else:
                self.position_accrued[position_id] = (accrued_0, accrued_1 + share)

    # -- Core step --------------------------------------------------------
    def step(
        self,
        new_sqrt_price: SqrtPrice,
        new_tick: Tick,
        swap_volume_0: float = 0.0,
        swap_volume_1: float = 0.0,
        *,
        new_sqrt_price_x96: int | None = None,
    ) -> dict[str, Any]:
        """Advance the pool by one step, mirroring ``apply_swap`` in float64.

        The input token is the leg whose ``swap_volume_*`` is ``> 0`` and
        ``gross_in`` is that leg.  Ticks crossed between the current tick and
        ``new_tick`` bound the swap into segments, each walked with the
        liquidity active in it; the gross input is apportioned across segments
        proportionally to their net inputs (the last segment absorbs the float
        remainder); each segment's fee is ``gross_seg * fee_tier_bps / 1e6``
        added to the input token's global accumulator over that segment's ``L``;
        and the segment's boundary tick is crossed **after** the accrual
        (update-then-cross), so ``fee_growth_outside`` sees the new global.

        Ticks are still crossed on the caller's zero-volume alignment step when
        the segment liquidity is non-zero, since the price move itself changes
        the active liquidity.  Returns the documented dict: ``fees_accrued``
        (per position, raw), ``ticks_crossed`` (ascending set),
        ``active_liquidity``, and ``pool_fee_earned_0/1`` (gross raw fee per
        token, reported even when no liquidity could accrue it).
        """
        pre_price = float(self.state.sqrt_price)
        post_price = float(new_sqrt_price)
        post_tick = Tick(new_tick)

        volume_0 = float(swap_volume_0)
        volume_1 = float(swap_volume_1)
        if volume_0 > 0.0:
            input_token = 0
            gross_in = volume_0
        elif volume_1 > 0.0:
            input_token = 1
            gross_in = volume_1
        else:
            input_token = -1
            gross_in = 0.0

        fee_pips = self.state.fee_tier_bps
        pool_fee_0 = gross_in * fee_pips / _FEE_DENOMINATOR if input_token == 0 else 0.0
        pool_fee_1 = gross_in * fee_pips / _FEE_DENOMINATOR if input_token == 1 else 0.0

        # The set of initialized ticks to cross, selected by **pool tick** rather
        # than by price (ADR-021).  A downward swap that ends exactly on a tick
        # boundary makes the contract set ``slot0.tick`` to ``T-1`` while the
        # price stays at ``boundary(T)``; a price-based rule would then let the
        # next upward move start on a boundary and miss crossing ``T``, silently
        # corrupting ``state.liquidity``.  The protocol's own interval is
        # ``(min_tick, max_tick]`` — exactly what ``_select_crossed_ticks``
        # implements and what ``tick_cross`` already uses.  The physical crossing
        # order is ascending upward and descending downward, so the segment walk
        # reverses the always-ascending list.
        pre_tick = self.state.tick
        upward = post_tick > pre_tick
        ticks_crossed = _select_crossed_ticks(
            self.ticks, pre_tick, post_tick, sorted_keys=self._tick_keys()
        )
        walk = ticks_crossed if upward else list(reversed(ticks_crossed))

        # Build the segment list: [pre_price, b1] .. [bn, post_price], each with
        # the liquidity active in it.  We walk the same boundary prices the
        # lattice exposes (``calc_sqrt_price_a``) and keep ``lo <= hi`` so the
        # per-segment amount formulas are direction-free.
        segments: list[tuple[float, float, float, Tick]] = []
        price = pre_price
        liquidity = float(self.state.liquidity)
        seg_tick = pre_tick
        for tick in walk:
            boundary = self._boundary_x96(tick) / float(Q96)
            lo, hi = (price, boundary) if price <= boundary else (boundary, price)
            segments.append((lo, hi, liquidity, seg_tick))
            net = float(self.ticks[tick].liquidity_net)
            liquidity = liquidity + net if upward else liquidity - net
            price = boundary
            # Downward, the segment after crossing ``tick`` spans ticks below
            # it; the half-open activity predicate then needs ``tick - 1``.
            seg_tick = tick if upward else Tick(int(tick) - 1)
        if post_price != price:
            lo, hi = (
                (price, post_price) if price <= post_price else (post_price, price)
            )
            segments.append((lo, hi, liquidity, seg_tick))

        # Per-segment net inputs and the proportional gross allocation.  If no
        # segment has liquidity there is nothing to accrue (and, mirroring the
        # tracker, no tick to cross: the pool is empty).
        nets: list[float] = []
        for lo, hi, liquidity, _seg_tick in segments:
            if input_token == 0:
                net = liquidity * (1.0 / lo - 1.0 / hi)
            elif input_token == 1:
                net = liquidity * (hi - lo)
            else:
                net = 0.0
            nets.append(net)
        net_total = math.fsum(nets)
        # Crossings happen whenever a segment has liquidity, independent of the
        # volume: the caller's zero-volume alignment step still moves the price
        # and must rebalance the active liquidity.  (For a real swap this is the
        # tracker's own ``net_total > 0`` gate, since a positive segment net is
        # equivalent to positive segment liquidity.)
        has_liquidity = any(liq > 0.0 for _lo, _hi, liq, _seg_tick in segments)
        do_fees = input_token in (0, 1) and net_total > 0.0

        if has_liquidity:
            fee_protocol = int(self.state.fee_protocol)
            fp_per_token = (
                fee_protocol & 0x0F
                if input_token == 0
                else (fee_protocol >> 4) & 0x0F
            )
            allocated = 0.0
            for i, ((_lo, _hi, liquidity, seg_tick), net) in enumerate(
                zip(segments, nets, strict=True)
            ):
                if do_fees:
                    if i == len(segments) - 1:
                        # The last segment absorbs the float remainder so the
                        # apportioned gross sums to ``gross_in`` exactly-ish.
                        gross_seg = gross_in - allocated
                    else:
                        gross_seg = gross_in * net / net_total
                        allocated += gross_seg
                    fee_seg = gross_seg * fee_pips / _FEE_DENOMINATOR
                    # Mirror ``FeeGrowthTracker.apply_swap`` (feegrowth.py:339):
                    # only accrue when the segment has liquidity and the fee is
                    # positive.  A large swap can exhaust the pool past its last
                    # initialized tick, leaving a zero-liquidity final segment
                    # whose float remainder is a tiny (possibly negative)
                    # non-zero; guarding also floors that remainder to 0.
                    if liquidity > 0.0 and fee_seg > 0.0:
                        lp_fee = (
                            fee_seg - fee_seg // fp_per_token
                            if fp_per_token > 0
                            else fee_seg
                        )
                        if lp_fee > 0.0:
                            if input_token == 0:
                                self.state.fee_growth_global_0 += lp_fee / liquidity
                            else:
                                self.state.fee_growth_global_1 += lp_fee / liquidity
                            self._accrue_marginal_fees(
                                seg_tick, liquidity, lp_fee, input_token
                            )
                if i < len(walk):
                    self._cross_one(walk[i], upward)

        # slot0 stores the actual sqrt price on every step, including moves that
        # stay within a tick, so downstream valuation never sees a stale price.
        self.state.tick = post_tick
        self.state.sqrt_price = post_price
        self.state.sqrt_price_x96 = new_sqrt_price_x96

        fees_accrued = {
            position_id: self.position_fees(position_id)
            for position_id in self.positions
        }
        return {
            "fees_accrued": fees_accrued,
            "ticks_crossed": ticks_crossed,
            "active_liquidity": self.state.liquidity,
            "pool_fee_earned_0": pool_fee_0,
            "pool_fee_earned_1": pool_fee_1,
        }
