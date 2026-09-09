"""Exact integer per-position fee accrual for Uniswap V3 concentrated liquidity (T10).

This module is the scientific heart of ``undertow.data``: every PnL number in the
thesis flows through it. It mirrors the protocol's own accounting — the
``feeGrowth*X128`` accumulators of the roadmap §10.2.4 (equations (3)–(6), the
water-meter analogy) and the tick-indexed state of the V3 whitepaper §6.3–6.5 —
in Python, on integers, with no ``float`` anywhere on the accrual path.

Reference implementations mirrored here:

- ``Pool._modifyPosition`` (Uniswap v3-core) — the ``feeGrowthBelow/Above/Inside``
  branch logic of equations (4)/(5)/(3).
- ``Position.update`` — ``tokensOwed = mulDiv(L, inside - last, 2**128)``,
  equation (6).
- ``Tick.cross`` / ``Tick.update`` — the outside-flip on crossing and the
  ``feeGrowthOutside`` seeding convention (whitepaper eq. 6.20 / 6.21).
- ``Pool.swap`` — the per-step ``feeGrowthGlobalX128 += mulDiv(fee, Q128, L)``
  update and the tick-crossing loop.

Modelling decisions (see the T10 PR body; the reviewer is instructed to treat a
silently-wrong numeric result as Critical, so every approximation below is named
and flagged):

1. **Fee formula.** ``fee_amount = amount_in * fee_pips // 1_000_000`` per the
   task brief (CONTRACTS.md §4.1: "the fee is taken on the input (positive)
   side"). The protocol computes the per-step fee as
   ``mulDivRoundingUp(net_in, fee, 1e6 - fee)``; the brief's formula is the
   contract here and differs from the protocol by a relative ``fee/1e6``
   (≈ 0.3% of the fee, ≈ 1e-5 of the input) — a documented, bounded residual.
2. **Single-segment swaps are exact.** A swap that crosses no initialized tick
   accrues ``(fee_amount << 128) // L_active`` with the single active liquidity;
   the pre-swap price does not enter, so this is exact even at replay start.
3. **Multi-tick swaps are re-simulated through the tick lattice.** The swap is
   split into segments bounded by the initialized ticks it crosses; each
   segment's net input is ``fixedpoint.get_amount{0,1}_delta(..., round_up=True)``
   (the protocol's own per-step formula, same boundaries, same rounding). The
   event's gross input is then apportioned across segments proportionally to
   those net inputs (the last segment absorbs the floor remainder, so the
   apportioned amounts sum to the event's input exactly), and each segment
   accrues ``(fee_segment << 128) // L_segment`` before its boundary tick is
   crossed — the protocol's update-then-cross order, so ``feeGrowthOutside``
   sees the global including the segment's fee. With the pre-swap price known
   (tracked from the previous swap event's ``sqrt_price_x96``), this
   apportionment is exact per the brief's fee formula.
4. **Unknown pre-swap price.** At replay start, or after restoring a checkpoint
   that lacks ``current_sqrt_price_x96``, the pre-swap price is approximated as
   the current tick's boundary ratio. This only affects the *apportionment* of a
   crossing swap (single-segment swaps stay exact); the affected swap raises a
   ``FeeGrowthApproximation`` warning and permanently sets the tracker's
   ``exact`` provenance flag to False, which propagates to every subsequent
   ``FeeAccrual``. No path returns ``exact=True`` on an approximated computation.
5. **Flash fees are out of scope** (CONTRACTS.md §4.3.1): a flash swap's
   ``paid - amount`` moves ``feeGrowthGlobal`` without a liquidity change, and
   the frozen ``FeeGrowthTracker`` interface has no flash method. On the pinned
   pools flash volume is negligible; T13's tolerance accounts for the small
   negative bias. The ``FeeGrowthTracker`` interface is frozen in CONTRACTS.md
   §6.1; the only extensions are two trailing defaulted fields on
   ``FeeGrowthState`` (``current_sqrt_price_x96``, ``exact``) so T14 checkpoints
   can resume a replay exactly and preserve the provenance flag.

No ``float`` anywhere on the accrual path: all accumulators are arbitrary-
precision ``int``, all deltas go through ``fixedpoint.wrapping_sub_256``
(Solidity ``unchecked`` semantics — accumulators legitimately wrap), and the
Q128 division in ``uncollected_fees`` is a right shift, matching
``FullMath.mulDiv``. The only floats in this module are the relative-delta
statistics in ``ReconciliationReport``, which are diagnostics, not accrual.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field

import pyarrow as pa  # type: ignore[import-untyped]  # no stubs shipped

from undertow.data.config import GLOBAL_TICK_SENTINEL, PoolConfig
from undertow.data.fixedpoint import (
    get_amount0_delta,
    get_amount1_delta,
    tick_to_sqrt_price_x96,
    wrapping_add_256,
    wrapping_sub_256,
)
from undertow.data.types import Address, BlockNumber

LOGGER = logging.getLogger("undertow.data.transforms.feegrowth")


def _as_int(value: object) -> int:
    """Coerce a schema value (decimal string or int) to an int."""
    if isinstance(value, int):
        return value
    return int(str(value))

# ---------------------------------------------------------------------------
# The pure functions — roadmap §10.2.4 equations (3)–(6).
# ---------------------------------------------------------------------------


def fee_growth_below(current_tick: int, tick_lower: int, g_global: int, g_out_lower: int) -> int:
    """Fee growth accumulated below ``tick_lower`` — roadmap §10.2.4 eq (4).

    Mirrors the ``feeGrowthBelow`` branch of ``Pool._modifyPosition``: when the
    current tick is at or above the lower bound the "outside" value already
    covers everything below; otherwise the growth below is the global minus the
    outside value. Integer, wrapping (Solidity ``unchecked``).
    """
    if current_tick >= tick_lower:
        return g_out_lower
    return wrapping_sub_256(g_global, g_out_lower)


def fee_growth_above(current_tick: int, tick_upper: int, g_global: int, g_out_upper: int) -> int:
    """Fee growth accumulated above ``tick_upper`` — roadmap §10.2.4 eq (5).

    Mirrors the ``feeGrowthAbove`` branch of ``Pool._modifyPosition``: when the
    current tick is at or above the upper bound the growth above is the global
    minus the outside value; otherwise the outside value already covers it.
    Integer, wrapping.
    """
    if current_tick >= tick_upper:
        return wrapping_sub_256(g_global, g_out_upper)
    return g_out_upper


def fee_growth_inside(
    current_tick: int,
    tick_lower: int,
    tick_upper: int,
    g_global: int,
    g_out_lower: int,
    g_out_upper: int,
) -> int:
    """Interior fee growth of the range ``[tick_lower, tick_upper)`` — eq (3).

    ``feeGrowthInside = feeGrowthGlobal - feeGrowthBelow - feeGrowthAbove``,
    the ``feeGrowthInside`` computation of ``Pool._modifyPosition``. The range is
    half-open: ``current_tick == tick_lower`` is in range, ``current_tick ==
    tick_upper`` is out of range (the protocol's ``[lower, upper)`` convention).
    Integer, wrapping — an out-of-range evaluation legitimately wraps.
    """
    below = fee_growth_below(current_tick, tick_lower, g_global, g_out_lower)
    above = fee_growth_above(current_tick, tick_upper, g_global, g_out_upper)
    return wrapping_sub_256(wrapping_sub_256(g_global, below), above)


def uncollected_fees(liquidity: int, g_inside_now: int, g_inside_last: int) -> int:
    """Uncollected fees for one token since the last snapshot — eq (6).

    ``fees = L * wrapping_sub_256(now, last) >> 128`` — a right shift, i.e. floor
    division by ``2**128``, matching ``Position.update``'s
    ``FullMath.mulDiv(L, inside - last, Q128)``. Returns raw integer token units;
    human-unit conversion is a presentation concern (T11/T16), never done here.
    """
    return (liquidity * wrapping_sub_256(g_inside_now, g_inside_last)) >> 128


# ---------------------------------------------------------------------------
# Result / state dataclasses — CONTRACTS.md §6.1, frozen.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PositionKey:
    """Identifies one position: (owner, tick_lower, tick_upper)."""

    owner: Address
    tick_lower: int
    tick_upper: int


@dataclass(frozen=True, slots=True)
class FeeAccrual:
    """Fees accrued by one position since its last snapshot, plus the new snapshots.

    ``exact`` is the tracker's replay provenance: False if any swap in the replay
    was approximated (unknown pre-swap price) or the tracker was restored from a
    checkpoint that recorded an approximation. The fixed ``accrue`` signature
    cannot carry an interpolation flag for the *input* ``g_inside_*_last`` values;
    per CONTRACTS.md §6.1 the caller (T11) is responsible for that part of the
    provenance — this field is the tracker's honest contribution.
    """

    fees0: int  # raw token0 units
    fees1: int  # raw token1 units
    g_inside_0_last: int  # new snapshot, Q128
    g_inside_1_last: int  # new snapshot, Q128
    block_number: BlockNumber
    exact: bool  # False if any input fee-growth value was interpolated / approximated


@dataclass(frozen=True, slots=True)
class TickState:
    """The per-tick struct of the whitepaper Table 2 (fee-growth subset)."""

    fee_growth_outside_0_x128: int
    fee_growth_outside_1_x128: int
    liquidity_gross: int
    liquidity_net: int  # signed
    initialized: bool


@dataclass(frozen=True, slots=True)
class FeeGrowthState:
    """Serializable tracker state — T14 checkpoints this to resume a long replay.

    The six leading fields are the frozen CONTRACTS.md §6.1 shape. Three trailing
    defaulted extensions were added so a resumed replay stays exact:
    ``current_sqrt_price_x96`` (the last swap's post-swap price, which makes the
    next crossing swap's apportionment exact instead of approximated),
    ``exact`` (the replay provenance flag, so an approximation that happened
    before a checkpoint is not silently forgotten after it), and
    ``fee_protocol`` (the packed uint8 from slot0: bits 0-3 = fp0, bits 4-7 =
    fp1; 0 when protocol fees are off, so T10's original behaviour is the
    default).
    """

    block_number: BlockNumber
    fee_growth_global_0_x128: int
    fee_growth_global_1_x128: int
    current_tick: int
    current_liquidity: int
    ticks: Mapping[int, TickState]
    # --- T10 extensions (defaulted, keyword-only; the frozen shape still constructs) ---
    current_sqrt_price_x96: int | None = field(default=None, kw_only=True)
    exact: bool = field(default=True, kw_only=True)
    fee_protocol: int = field(default=0, kw_only=True)


@dataclass(frozen=True, slots=True)
class Mismatch:
    """One field-level disagreement between the replayed state and an observed row."""

    block_number: BlockNumber
    tick: int | None  # None for a global-accumulator mismatch
    field_name: str
    replayed: int
    observed: int
    abs_delta: int
    rel_delta: float  # abs_delta / max(abs(observed), 1)


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    """Returned by ``FeeGrowthTracker.reconcile``; rendered by T13. Reports, never raises."""

    n_compared: int
    n_exact: int
    max_abs_delta_g0: int
    max_abs_delta_g1: int
    max_rel_delta: float
    mismatches: tuple[Mismatch, ...]


class FeeGrowthApproximation(RuntimeWarning):
    """A swap's fee apportionment was approximated (unknown pre-swap price).

    Raised via ``warnings.warn`` (the replay continues — the approximation is
    bounded and flagged, not fatal) and permanently sets the tracker's ``exact``
    provenance flag to False. See the module docstring, modelling decision 4.
    """


# ---------------------------------------------------------------------------
# The state machine.
# ---------------------------------------------------------------------------


class FeeGrowthTracker:
    """Replays swaps + tick crossings to maintain feeGrowthGlobal and per-tick
    feeGrowthOutside, so fee accrual can be evaluated at any block without an
    ``eth_call`` per block.

    State: ``fee_growth_global_{0,1}_x128``, ``current_tick``,
    ``current_liquidity``, and a ``dict[int, TickState]`` of initialized ticks.
    The pre-swap price of the next swap is tracked from the previous swap
    event's ``sqrt_price_x96`` (mints/burns do not move the price), which makes
    multi-tick apportionment exact; at replay start it is unknown and the first
    crossing swap is flagged (see module docstring, decision 4).
    """

    def __init__(self, pool: PoolConfig, initial: FeeGrowthState) -> None:
        self.pool = pool
        self.block_number = initial.block_number
        self.fee_growth_global_0_x128 = initial.fee_growth_global_0_x128
        self.fee_growth_global_1_x128 = initial.fee_growth_global_1_x128
        self.current_tick = initial.current_tick
        self.current_liquidity = initial.current_liquidity
        self.ticks: dict[int, TickState] = dict(initial.ticks)
        self._current_sqrt_price_x96: int | None = initial.current_sqrt_price_x96
        self._exact = initial.exact
        self._fee_protocol: int = initial.fee_protocol

    # -- event application --------------------------------------------------

    def apply_swap(self, row: Mapping[str, object]) -> None:
        """Replay one ``swap`` row: accrue fees per in-range segment, cross ticks.

        The input token is the one with a positive amount (CONTRACTS.md §4.1:
        "Never assume the input token"). A swap that crosses no initialized tick
        is a single segment and is exact; a crossing swap is re-simulated through
        the tick lattice (module docstring, decision 3). Raises ``ValueError`` on
        a row whose amount signs are inconsistent with a real swap, or whose
        direction disagrees with the price movement.
        """
        amount0 = _as_int(row["amount0"])
        amount1 = _as_int(row["amount1"])
        if (amount0 > 0) == (amount1 > 0):
            raise ValueError(
                "swap row must have opposite-signed amounts (input positive, output "
                f"negative); got amount0={amount0}, amount1={amount1}"
            )
        input_token = 0 if amount0 > 0 else 1
        gross_in = amount0 if amount0 > 0 else amount1
        # Per-token protocol fee denominator from the packed uint8 (contract's
        # SwapCache: zeroForOne → fp0 = bits 0-3, oneForZero → fp1 = bits 4-7).
        fp_per_token = (
            self._fee_protocol & 0x0F if input_token == 0
            else (self._fee_protocol >> 4) & 0x0F
        )
        post_tick = _as_int(row["tick"])
        post_price = _as_int(row["sqrt_price_x96"])

        pre_tick = self.current_tick
        pre_price = self._current_sqrt_price_x96
        approximated = False
        if pre_price is None:
            pre_price = tick_to_sqrt_price_x96(pre_tick)
            approximated = True
        if (amount0 > 0) != (post_tick < pre_tick) and post_tick != pre_tick:
            # The pool's raw price is token1-per-token0, so a token0-in swap (amount0
            # > 0) drives the tick DOWN and a token1-in swap drives it UP. A swap
            # whose price moves but stays within the same tick (post_tick == pre_tick)
            # is legitimate — it still accrues fees — and carries no direction
            # signal, so it is allowed for either input token.
            raise ValueError(
                f"swap row direction disagrees with its amounts: amount0={amount0} "
                f"implies token0 {'in' if amount0 > 0 else 'out'} but the price moved "
                f"from tick {pre_tick} to {post_tick}"
            )

        upward = post_price > pre_price
        if upward:
            crossed = sorted(
                t for t in self.ticks if pre_price < tick_to_sqrt_price_x96(t) <= post_price
            )
        else:
            crossed = sorted(
                (t for t in self.ticks if post_price <= tick_to_sqrt_price_x96(t) < pre_price),
                reverse=True,
            )

        # Walk the segments: [pre_price, boundary_1] .. [boundary_n, post_price],
        # each with the liquidity active in it (after the crossings so far).
        segments: list[tuple[int, int, int]] = []
        price = pre_price
        liq = self.current_liquidity
        for t in crossed:
            boundary = tick_to_sqrt_price_x96(t)
            segments.append((price, boundary, liq))
            net = self.ticks[t].liquidity_net
            liq = liq + net if upward else liq - net
            price = boundary
        if post_price != price:  # skip a degenerate zero-length final segment
            segments.append((price, post_price, liq))

        # Per-segment net inputs (the protocol's own per-step formula).
        nets: list[int] = []
        for a, b, L in segments:
            if input_token == 0:
                nets.append(get_amount0_delta(a, b, L, True))
            else:
                nets.append(get_amount1_delta(a, b, L, True))
        net_total = sum(nets)

        if net_total > 0:
            fee_pips = self.pool.fee_tier.value
            allocated = 0
            for i, ((_a, _b, L), net) in enumerate(zip(segments, nets, strict=True)):
                if i == len(segments) - 1:
                    gross_seg = gross_in - allocated  # last segment absorbs the floor remainder
                else:
                    gross_seg = gross_in * net // net_total
                    allocated += gross_seg
                fee_seg = gross_seg * fee_pips // 1_000_000
                if L > 0 and fee_seg > 0:
                    lp_fee = fee_seg - fee_seg // fp_per_token if fp_per_token > 0 else fee_seg
                    if lp_fee > 0:
                        inc = (lp_fee << 128) // L
                        if input_token == 0:
                            self.fee_growth_global_0_x128 = wrapping_add_256(
                                self.fee_growth_global_0_x128, inc
                            )
                        else:
                            self.fee_growth_global_1_x128 = wrapping_add_256(
                                self.fee_growth_global_1_x128, inc
                            )
                if i < len(crossed):
                    # update-then-cross order: the tick's outside sees the global
                    # including this segment's fee, exactly like Pool.swap.
                    self.cross_tick(crossed[i], upward)

        self.current_tick = post_tick
        self._current_sqrt_price_x96 = post_price
        self.block_number = BlockNumber(_as_int(row["block_number"]))

        if approximated and crossed:
            warnings.warn(
                f"swap at block {self.block_number} crossed {len(crossed)} tick(s) with an "
                "unknown pre-swap price (replay start, or a restored checkpoint without "
                "current_sqrt_price_x96); fee apportionment is approximated and the "
                "tracker's exact flag is now False. See the modelling-decisions section "
                "of undertow.data.transforms.feegrowth.",
                FeeGrowthApproximation,
                stacklevel=2,
            )
            self._exact = False

    def apply_liquidity_event(self, row: Mapping[str, object]) -> None:
        """Replay one ``mint``/``burn`` row: update both boundary ticks and, if the
        position spans the current tick, ``current_liquidity``.

        Mirrors ``Pool._modifyPosition`` + ``Tick.update``: ``liquidity_gross``
        and ``liquidity_net`` are updated on both boundary ticks; a newly
        initialized tick (gross was 0) is seeded with ``fee_growth_outside =
        fee_growth_global`` when the tick is at or below ``current_tick``, else 0
        (whitepaper eq. 6.21); a tick whose gross returns to 0 is uninitialized
        (removed), exactly like ``Tick.update``'s struct clear.
        """
        etype = str(row["event_type"])
        if etype not in ("mint", "burn"):
            raise ValueError(
                f"apply_liquidity_event expects a mint or burn row, got {etype!r}"
            )
        tick_lower = _as_int(row["tick_lower"])
        tick_upper = _as_int(row["tick_upper"])
        liquidity_amount = _as_int(row["liquidity_amount"])
        delta = liquidity_amount if etype == "mint" else -liquidity_amount

        if delta == 0:
            # A zero-amount burn is the pool's fee-collection no-op (burn(0) then
            # collect): it moves no liquidity and so touches no tick state. Real-chain
            # finding: such events target an arbitrarily wide tick range whose
            # boundaries are often uninitialized, and treating them as real updates
            # crashes (deleting a tick that was never added). No-op: just advance the
            # block number.
            self.block_number = BlockNumber(_as_int(row["block_number"]))
            return

        for t, d in ((tick_lower, delta), (tick_upper, -delta)):
            before = self.ticks.get(t)
            gross_before = before.liquidity_gross if before is not None else 0
            gross_after = gross_before + delta  # signed: +L on mint, -L on burn
            if gross_after < 0:
                raise ValueError(
                    f"{etype} of liquidity {liquidity_amount} at tick {t} would drive "
                    "liquidity_gross negative — the event stream is inconsistent"
                )
            if gross_before == 0:
                # newly initialized: seed per Tick.update (whitepaper eq. 6.21)
                if t <= self.current_tick:
                    out0, out1 = self.fee_growth_global_0_x128, self.fee_growth_global_1_x128
                else:
                    out0, out1 = 0, 0
            else:
                assert before is not None  # gross_before > 0 implies the tick exists
                out0, out1 = before.fee_growth_outside_0_x128, before.fee_growth_outside_1_x128
            if gross_after == 0:
                del self.ticks[t]  # uninitialized: Tick.update clears the struct
            else:
                net_after = (before.liquidity_net if before is not None else 0) + d
                self.ticks[t] = TickState(out0, out1, gross_after, net_after, True)

        if tick_lower <= self.current_tick < tick_upper:  # half-open [lower, upper)
            self.current_liquidity += delta
        self.block_number = BlockNumber(_as_int(row["block_number"]))

    def cross_tick(self, tick: int, upward: bool) -> None:
        """Mirror ``Tick.cross``: flip ``fee_growth_outside`` to the global minus
        itself for both tokens, and apply the tick's ``liquidity_net`` to
        ``current_liquidity`` (add when crossing upward, subtract downward).

        The outside flip uses ``wrapping_sub_256`` (Solidity ``unchecked``
        semantics; CONTRACTS.md §4: "all accumulator subtraction goes through
        wrapping_sub_256"). Raises ``KeyError`` for an uninitialized tick.
        """
        state = self.ticks.get(tick)
        if state is None:
            raise KeyError(f"cross_tick: tick {tick} is not initialized")
        outside_0 = wrapping_sub_256(self.fee_growth_global_0_x128, state.fee_growth_outside_0_x128)
        outside_1 = wrapping_sub_256(self.fee_growth_global_1_x128, state.fee_growth_outside_1_x128)
        net = state.liquidity_net
        liq = self.current_liquidity + net if upward else self.current_liquidity - net
        self.ticks[tick] = TickState(outside_0, outside_1, state.liquidity_gross, net, True)
        self.current_liquidity = liq

    # -- accrual ------------------------------------------------------------

    def accrue(
        self,
        key: PositionKey,
        liquidity: int,
        g_inside_0_last: int,
        g_inside_1_last: int,
    ) -> FeeAccrual:
        """Accrue fees for one position: eq (3) then eq (6) for both tokens.

        Returns the fees plus the new ``g_inside_*`` snapshots (the caller stores
        them as the position's ``feeGrowthInside*Last``). Boundary ticks missing
        from the tracker read as ``fee_growth_outside = 0``, the protocol's
        default for an uninitialized tick.
        """
        lower = self.ticks.get(key.tick_lower)
        upper = self.ticks.get(key.tick_upper)
        out0_lower = lower.fee_growth_outside_0_x128 if lower is not None else 0
        out1_lower = lower.fee_growth_outside_1_x128 if lower is not None else 0
        out0_upper = upper.fee_growth_outside_0_x128 if upper is not None else 0
        out1_upper = upper.fee_growth_outside_1_x128 if upper is not None else 0

        g0 = fee_growth_inside(
            self.current_tick,
            key.tick_lower,
            key.tick_upper,
            self.fee_growth_global_0_x128,
            out0_lower,
            out0_upper,
        )
        g1 = fee_growth_inside(
            self.current_tick,
            key.tick_lower,
            key.tick_upper,
            self.fee_growth_global_1_x128,
            out1_lower,
            out1_upper,
        )
        return FeeAccrual(
            fees0=uncollected_fees(liquidity, g0, g_inside_0_last),
            fees1=uncollected_fees(liquidity, g1, g_inside_1_last),
            g_inside_0_last=g0,
            g_inside_1_last=g1,
            block_number=self.block_number,
            exact=self._exact,
        )

    # -- checkpointing -----------------------------------------------------

    def snapshot(self) -> FeeGrowthState:
        """Serializable checkpoint of the tracker state (T14 resumes from it)."""
        return FeeGrowthState(
            block_number=self.block_number,
            fee_growth_global_0_x128=self.fee_growth_global_0_x128,
            fee_growth_global_1_x128=self.fee_growth_global_1_x128,
            current_tick=self.current_tick,
            current_liquidity=self.current_liquidity,
            ticks=dict(self.ticks),
            current_sqrt_price_x96=self._current_sqrt_price_x96,
            exact=self._exact,
            fee_protocol=self._fee_protocol,
        )

    def restore(self, state: FeeGrowthState) -> None:
        """Reset the tracker from a checkpoint (the resume half of snapshot/restore)."""
        self.block_number = state.block_number
        self.fee_growth_global_0_x128 = state.fee_growth_global_0_x128
        self.fee_growth_global_1_x128 = state.fee_growth_global_1_x128
        self.current_tick = state.current_tick
        self.current_liquidity = state.current_liquidity
        self.ticks = dict(state.ticks)
        self._current_sqrt_price_x96 = state.current_sqrt_price_x96
        self._exact = state.exact
        self._fee_protocol = state.fee_protocol

    # -- reconciliation -----------------------------------------------------

    def reconcile(self, observed: pa.Table) -> ReconciliationReport:
        """Compare the replayed state against observed ``FEE_GROWTH_SCHEMA`` rows.

        The caller positions the tracker at the observed block (by replaying the
        event stream up to it) and passes that block's rows: a global row
        (``tick == GLOBAL_TICK_SENTINEL``) is compared on the four global fields,
        a tick row on the five ``TickState`` fields. Reports per-field exact
        counts and, for mismatches, absolute and relative deltas
        (``rel_delta = abs_delta / max(abs(observed), 1)``). It reports, never
        raises — T13 decides severity.
        """
        blocks = observed.column("block_number").to_pylist()
        for block in blocks:
            if block != self.block_number:
                raise ValueError(
                    f"reconcile: observed row at block {block} but the tracker is at "
                    f"block {self.block_number}; position the tracker at the observed "
                    "block first (replay the event stream up to it)"
                )
        ticks_col = observed.column("tick").to_pylist()
        g0_col = observed.column("fee_growth_global_0_x128").to_pylist()
        g1_col = observed.column("fee_growth_global_1_x128").to_pylist()
        cur_tick_col = observed.column("current_tick").to_pylist()
        cur_liq_col = observed.column("current_liquidity").to_pylist()
        out0_col = observed.column("fee_growth_outside_0_x128").to_pylist()
        out1_col = observed.column("fee_growth_outside_1_x128").to_pylist()
        gross_col = observed.column("liquidity_gross").to_pylist()
        net_col = observed.column("liquidity_net").to_pylist()
        init_col = observed.column("initialized").to_pylist()
        fp_col = observed.column("fee_protocol").to_pylist()

        mismatches: list[Mismatch] = []
        n_compared = 0
        n_exact = 0
        max_abs_g0 = 0
        max_abs_g1 = 0
        max_rel: float = 0

        def compare(
            block: int, tick: int | None, field: str, replayed: int, observed_val: int
        ) -> None:
            nonlocal n_compared, n_exact, max_abs_g0, max_abs_g1, max_rel
            n_compared += 1
            if replayed == observed_val:
                n_exact += 1
                return
            abs_delta = abs(replayed - observed_val)
            rel_delta = abs_delta / max(abs(observed_val), 1)
            mismatches.append(
                Mismatch(
                    block_number=BlockNumber(block),
                    tick=tick,
                    field_name=field,
                    replayed=replayed,
                    observed=observed_val,
                    abs_delta=abs_delta,
                    rel_delta=rel_delta,
                )
            )
            if "_0_" in field:
                max_abs_g0 = max(max_abs_g0, abs_delta)
            elif "_1_" in field:
                max_abs_g1 = max(max_abs_g1, abs_delta)
            max_rel = max(max_rel, rel_delta)

        for block, tick, g0, g1, cur_tick, cur_liq, out0, out1, gross, net, init, fp in zip(
            blocks,
            ticks_col,
            g0_col,
            g1_col,
            cur_tick_col,
            cur_liq_col,
            out0_col,
            out1_col,
            gross_col,
            net_col,
            init_col,
            fp_col,
            strict=True,
        ):
            if tick == GLOBAL_TICK_SENTINEL:
                compare(
                    block, None, "fee_protocol", self._fee_protocol, int(fp)
                )
                compare(
                    block, None, "fee_growth_global_0_x128", self.fee_growth_global_0_x128, int(g0)
                )
                compare(
                    block, None, "fee_growth_global_1_x128", self.fee_growth_global_1_x128, int(g1)
                )
                compare(block, None, "current_tick", self.current_tick, int(cur_tick))
                compare(block, None, "current_liquidity", self.current_liquidity, int(cur_liq))
            else:
                state = self.ticks.get(tick, TickState(0, 0, 0, 0, False))
                compare(
                    block,
                    tick,
                    "fee_growth_outside_0_x128",
                    state.fee_growth_outside_0_x128,
                    int(out0),
                )
                compare(
                    block,
                    tick,
                    "fee_growth_outside_1_x128",
                    state.fee_growth_outside_1_x128,
                    int(out1),
                )
                compare(block, tick, "liquidity_gross", state.liquidity_gross, int(gross))
                compare(block, tick, "liquidity_net", state.liquidity_net, int(net))
                compare(block, tick, "initialized", int(state.initialized), int(bool(init)))

        return ReconciliationReport(
            n_compared=n_compared,
            n_exact=n_exact,
            max_abs_delta_g0=max_abs_g0,
            max_abs_delta_g1=max_abs_g1,
            max_rel_delta=max_rel,
            mismatches=tuple(mismatches),
        )
