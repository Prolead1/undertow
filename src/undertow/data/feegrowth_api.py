"""Public fee-growth replay facade for the sim backtester (ADR-013).

This module is the **only** ``undertow.data`` surface that exposes the exact
integer fee-growth engine to ``undertow.sim``.  It wraps the internal
:class:`~undertow.data.transforms.feegrowth.FeeGrowthTracker` in a small,
immutable-interface facade so the sim never imports an ``undertow.data``
submodule (``docs/plans/sim/CONTRACTS.md`` §0, ADR-008) and never touches the
mutable tracker type directly::

    from undertow.data import FeeGrowthReplay

    replay = FeeGrowthReplay(pool, current_tick=entry_tick)
    for row in tape:
        replay.apply_event(row)
    fees0, fees1, g0_new, g1_new = replay.accrue(
        tick_lower=lower, tick_upper=upper, liquidity=L,
        g_inside_last_0=g0_last, g_inside_last_1=g1_last,
    )

Everything on the accrual path stays in raw Q128 integers: ``accrue`` returns
integer token units and Q128 inside-snapshots, never floats.  The caller is
responsible for dividing to human units (and is responsible for persisting the
returned snapshots).

The underlying tracker is the reconciled state machine of
``docs/plans/data-pipeline/CONTRACTS.md`` §6.1; this facade adds no arithmetic
of its own, so the exactness (and the documented ``FeeGrowthApproximation``
residual) is exactly the tracker's.
"""

from __future__ import annotations

from collections.abc import Mapping

from undertow.data.config import PoolConfig
from undertow.data.transforms.feegrowth import (
    FeeGrowthState,
    FeeGrowthTracker,
    PositionKey,
    TickState,
)
from undertow.data.types import Address, BlockNumber

__all__ = ["FeeGrowthReplay"]

#: Owner placeholder for the internal ``PositionKey``.  The tracker's ``accrue``
#: keys only on the tick range (the owner is part of the frozen key shape, not
#: of the arithmetic), so a single canonical placeholder is sufficient here.
_KEY_OWNER = Address("0x" + "00" * 20)


class FeeGrowthReplay:
    """Exact fee-growth replay over an event tape (ADR-013).

    Public facade over the internal :class:`FeeGrowthTracker`; the sim imports
    only this.  Construct it once per backtest run with the pool and the pool
    state at the start of the tape, feed each tape row to :meth:`apply_event`,
    and call :meth:`accrue` for the position's fees after each row.

    The facade deliberately does not expose the tracker's tick lattice or the
    mutable accumulator fields; callers persist :meth:`snapshot` /
    :meth:`restore` if they need a checkpoint.
    """

    def __init__(
        self,
        pool: PoolConfig,
        *,
        fee_pips: int | None = None,
        block_number: int = 0,
        current_tick: int = 0,
        current_liquidity: int = 0,
    ) -> None:
        """Start an empty replay.

        ``pool`` supplies the pool's fee tier and identity.  ``fee_pips`` is an
        optional override of the pool's fee tier (the sim's ``fee_tier_bps`` may
        be an ablation value such as ``0`` that has no matching ``PoolConfig``);
        ``None`` uses ``pool.fee_tier.value``.  ``block_number``,
        ``current_tick`` and ``current_liquidity`` are the pre-tape pool state —
        the sim seeds ``current_tick`` from the first tape observation so the
        tracker can resolve swap direction and cross ticks on the real grid.
        """
        self._pool = pool
        self._tracker = FeeGrowthTracker(
            pool,
            FeeGrowthState(
                block_number=BlockNumber(block_number),
                fee_growth_global_0_x128=0,
                fee_growth_global_1_x128=0,
                current_tick=int(current_tick),
                current_liquidity=int(current_liquidity),
                ticks={},
                fee_pips=fee_pips,
            ),
        )

    @classmethod
    def from_state(cls, pool: PoolConfig, state: FeeGrowthState) -> FeeGrowthReplay:
        """Resume a replay from a checkpoint returned by :meth:`snapshot`.

        The checkpoint carries the global accumulators, the tick lattice, the
        last tracked sqrt price and the exactness/fee-pips provenance, so a
        resumed replay is byte-identical to an uninterrupted one.
        """
        obj = cls.__new__(cls)
        obj._pool = pool
        obj._tracker = FeeGrowthTracker(pool, state)
        return obj

    def apply_event(self, row: Mapping[str, object]) -> None:
        """Replay one tape row.

        ``swap`` rows go through the tracker's exact swap replay
        (``apply_swap``); ``mint``/``burn`` rows update the tick lattice and
        active liquidity (``apply_liquidity_event``).  ``collect`` and
        ``flash`` rows move no pool fee-growth state in the T10 engine (the
        frozen tracker interface has no flash method; flash fees are a
        documented T10 out-of-scope residual), so they only advance the block
        number.  Any other ``event_type`` raises ``ValueError``.
        """
        event_type = str(row.get("event_type", "")).lower()
        if event_type == "swap":
            self._tracker.apply_swap(row)
        elif event_type in ("mint", "burn"):
            self._tracker.apply_liquidity_event(row)
        elif event_type in ("collect", "flash"):
            block = row.get("block_number")
            if block is not None:
                self._tracker.block_number = BlockNumber(int(str(block)))
        else:
            raise ValueError(
                f"FeeGrowthReplay.apply_event: unknown event_type {event_type!r}; "
                "expected one of swap/mint/burn/collect/flash"
            )

    def register_position(self, *, tick_lower: int, tick_upper: int) -> None:
        """Seed a synthetic position's boundary ticks in the tracker lattice.

        The backtester's position is *marginal*: it does not contribute pool
        liquidity, so it is not minted on-chain and its boundary ticks may not
        appear in the tape.  If a later real mint at one of those ticks arrives
        after fee growth has accrued, the tracker would seed the (previously
        absent) tick's ``feeGrowthOutside`` from the then-current global and the
        synthetic position's ``feeGrowthInside`` would jump — producing a
        wrapped-negative `uncollected_fees` credit.

        Registering the boundaries up front prevents that: the tick exists with
        the protocol's seeding convention for a position opened now, so a later
        real mint sees ``liquidity_gross > 0`` and does not re-seed it.  A
        sentinel ``liquidity_gross=1`` with ``liquidity_net=0`` keeps the tick
        initialised without contributing to ``current_liquidity`` (the
        marginal-agent model).  An already-initialised tick is left untouched.
        """
        tracker = self._tracker
        current_tick = tracker.current_tick
        global_0 = tracker.fee_growth_global_0_x128
        global_1 = tracker.fee_growth_global_1_x128
        for tick in (int(tick_lower), int(tick_upper)):
            if tick in tracker.ticks:
                continue
            if tick <= current_tick:
                outside_0, outside_1 = global_0, global_1
            else:
                outside_0, outside_1 = 0, 0
            tracker.ticks[tick] = TickState(outside_0, outside_1, 1, 0, True)

    def accrue(
        self,
        *,
        tick_lower: int,
        tick_upper: int,
        liquidity: int,
        g_inside_last_0: int,
        g_inside_last_1: int,
    ) -> tuple[int, int, int, int]:
        """Raw ``(fees0, fees1, new_inside_last_0, new_inside_last_1)``.

        ``fees0``/``fees1`` are integer token units; ``new_inside_last_*`` are
        Q128 inside snapshots the caller must persist as the position's
        ``feeGrowthInside*Last``.  Never returns a float.
        """
        accrual = self._tracker.accrue(
            PositionKey(owner=_KEY_OWNER, tick_lower=int(tick_lower), tick_upper=int(tick_upper)),
            int(liquidity),
            int(g_inside_last_0),
            int(g_inside_last_1),
        )
        return (
            accrual.fees0,
            accrual.fees1,
            accrual.g_inside_0_last,
            accrual.g_inside_1_last,
        )

    @property
    def exact(self) -> bool:
        """``False`` once any replayed swap's apportionment was approximated.

        This is the tracker's own provenance flag; it does not by itself claim
        the replay matches chain (see ADR-013's named residuals).
        """
        return self._tracker.exact

    def snapshot(self) -> FeeGrowthState:
        """Serialisable checkpoint of the replay state (for :meth:`from_state`)."""
        return self._tracker.snapshot()

    def restore(self, state: FeeGrowthState) -> None:
        """Reset the replay from a checkpoint (the resume half of ``snapshot``)."""
        self._tracker.restore(state)
