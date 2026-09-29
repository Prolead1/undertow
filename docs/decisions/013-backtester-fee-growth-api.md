# ADR 013 — Backtester fee growth: exact integer replay through the public `FeeGrowthReplay` facade

**Status:** accepted (implementation and export)
**Affects:** `src/undertow/sim/backtest/engine.py` · `src/undertow/sim/marketview.py` · `src/undertow/sim/backtest/ledger.py` · `src/undertow/data/__init__.py` · `src/undertow/data/feegrowth_api.py` · `src/undertow/data/transforms/feegrowth.py` · sim `CONTRACTS.md` §3/§13 · tasks S02, S11, S14
**Raised by:** S11 (`feature-sim-backtester`); implemented on `feature-data-feegrowth-api`

## Context

`CONTRACTS.md` §3 states that `FeeGrowthTracker` is **not** exported by S02 and forbids the sim
from importing `undertow.data.<submodule>`; the sim may use only the top-level `undertow.data`
surface (PLAN.md §0.7/§4).  The S11 brief nonetheless asks the backtester to use "the data module's
`FeeGrowthTracker` for exact integer fee math" and, failing that export, to either consume the
event tape's pre-computed `fee_growth_global_0_x128` / `_1_x128` columns and apportion per position,
or raise an ADR proposing a deliberate public-API extension.

The event tape (`undertow.data.schemas.EVENT_TAPE_SCHEMA`) carries the two **global** Q128
accumulators per event, plus the per-event pool `tick`, `sqrt_price_x96`, `liquidity` and the
mint/burn `tick_lower` / `tick_upper` / `liquidity_amount`.  It does **not** carry the per-tick
`fee_growth_outside_0/1_x128` values that `FeeGrowthTracker` maintains.

## Problem

A position's fee entitlement is `L · ΔfeeGrowthInside`, where
`feeGrowthInside = feeGrowthGlobal − feeGrowthBelow − feeGrowthAbove` (data
`transforms/feegrowth.py`, roadmap §10.2.4 eq (3)).  The global columns alone give
`ΔfeeGrowthInside = ΔfeeGrowthGlobal` **only while the current tick is inside the position and no
boundary is crossed during the event**:

* in range, boundaries uncrossed: `Δinside = Δglobal` — the tape-columns path is exact;
* price crosses a boundary mid-event: the swap is split into segments with different active
  liquidity, so `Δinside ≠ Δglobal` and the event-level apportionment is an approximation;
* a position whose boundary ticks already had non-zero `feeGrowthOutside` before the tape starts:
  the opening snapshot cannot be reconstructed from the global columns at all.

For a policy that rebalances (every baseline except HODL, and the RL agent in evaluation), both
situations are routine.  The tape-columns path therefore **cannot in general reproduce the data
module's Collect reconciliation** — only its in-range case.

## Decision

1. **Ship the tape-columns path now.**  `run_backtest` replays the tape's Q128 global accumulators
   as Python integers, gates accrual on `tick_lower <= pool_tick < tick_upper`, and computes
   `(L_int · ΔG) >> 128`, dividing by `10**dec0/1` only when marking to USDC (PLAN.md §4, "Two
   numeric worlds").  It imports only top-level `undertow.data`.  This is exact for an in-range
   position whose boundaries are not crossed — including all HODL/passive-deployment sweeps where
   the position stays in range — and it is flagged as approximate otherwise.

   Two implementation caveats keep the exactness claim honest:

   * **Nullable leading snapshots.**  The tape's global columns are nullable before the first
     recorded snapshot.  The engine treats a `None` cell as "no observation": it credits nothing and
     leaves the baseline untouched, and the first non-null cell *seeds* the baseline without
     crediting the unobservable pre-snapshot gap.  Without this, a leading null coerced to `0` would
     credit all fee growth since pool genesis in a single step — a silent windfall.  Pinned by
     `test_leading_null_fee_growth_does_not_windfall` / `test_all_null_fee_growth_accrues_nothing`.
   * **Float-derived liquidity.**  `Position.liquidity` is a `float64` produced by S04's
     `initial_deposit`; the integer accrual rounds it to `int` (`int(round(L))`) before
     `(L · ΔG) >> 128`.  The Q128 accumulator itself never touches `float` (the requirement of
     PLAN.md §4), but `L` is only 53-bit precise, so the accrual is exact **relative to the
     position's float liquidity**.  The fully exact path would need the protocol-integer `L` that
     the proposed data export could supply.

2. **Ship the deliberate additive export — accepted and implemented.**  The follow-up data task
   (`feature-data-feegrowth-api`) adds a small public facade, `FeeGrowthReplay`, to the top-level
   `undertow.data` surface; the sim imports only that.  The facade wraps the internal
   `FeeGrowthTracker` (which stays unexported) and drives it event-by-event:

   ```python
   # undertow.data public API (additive; ADR-008 superset continues)
   from undertow.data import FeeGrowthReplay

   class FeeGrowthReplay:
       def __init__(self, pool: PoolConfig, *, fee_pips: int | None = None,
                    block_number: int = 0, current_tick: int = 0,
                    current_liquidity: int = 0) -> None: ...
       @classmethod
       def from_state(cls, pool: PoolConfig, state: FeeGrowthState) -> FeeGrowthReplay: ...
       def apply_event(self, row: Mapping[str, object]) -> None: ...
       def register_position(self, *, tick_lower: int, tick_upper: int) -> None: ...
       def accrue(self, *, tick_lower: int, tick_upper: int, liquidity: int,
                  g_inside_last_0: int, g_inside_last_1: int) -> tuple[int, int, int, int]:
           """Raw (fees0, fees1, new_inside_last_0, new_inside_last_1) — Q128 ints,
           never float."""
       @property
       def exact(self) -> bool: ...
       def snapshot(self) -> FeeGrowthState: ...
       def restore(self, state: FeeGrowthState) -> None: ...
   ```

   `apply_event` dispatches swap rows to `FeeGrowthTracker.apply_swap` and mint/burn rows to
   `apply_liquidity_event`; collect/flash rows move no T10 pool state and only advance the block
   number.  `register_position` seeds a *marginal* position's boundary ticks in the lattice (with
   the protocol's seeding convention and a zero-net sentinel ``liquidity_gross=1``) without adding
   pool liquidity, so a later real mint at one of those ticks does not re-seed `feeGrowthOutside`.  The facade keeps the exact-integer engine, the per-tick lattice and the wrapping
   arithmetic inside `undertow.data` where they are already reconciled, while giving S11/S14 a
   stable, top-level surface.  `fee_pips` is an additive override so the sim's ablation fee tier
   (e.g. `0`) is honoured exactly even though it has no matching `PoolConfig`; it is carried on
   `FeeGrowthState` so checkpoints preserve it.  The tape's global columns remain useful for
   cross-checking.

3. **Wire the backtester to the exact path and record the residual.**  `run_backtest` constructs
   one `FeeGrowthReplay` per run from the `MarketView`'s pool (an additive
   `MarketView.pool: PoolConfig | None` field, populated by `build_market_view` from
   `DataConfig.pool`; a config-derived pool is the fallback for hand-built views), replays every
   tape event in order, and accrues the position's inside growth from the returned Q128 snapshots.
   The whole path stays integer until it is divided to human units for the USDC mark.  The ledger
   carries the standard `config_hash`/`git_commit` plus an additive `BacktestLedger.fee_growth_exact`
   field recording the tracker's provenance.  S14 (parity) quantifies the residual.  No silent
   approximation ships.

### As-built exactness (supersedes the tape-columns claim in Decision §1)

With the exact replay wired in, **boundary-crossing swaps and pre-existing `feeGrowthOutside` are
now handled exactly**, not approximated:

* a swap that crosses a position boundary is split by the tracker into segments with the correct
  active liquidity, and the position's `ΔfeeGrowthInside` is taken from the crossed tick's flipped
  `feeGrowthOutside` — the data module's own update-then-cross order;
* a boundary tick that already had non-zero `feeGrowthOutside` (initialised before the position) is
  represented in the tracker's lattice and enters equation (3) exactly;
* the backtester's marginal position registers its own boundaries at deployment and at each
  rebalance, so a later real mint at the same tick cannot re-seed `feeGrowthOutside` and make the
  synthetic position's inside delta wrap negative.

The tape-columns apportionment of Decision §1 is **retained in the tests only**, as the ADR-013
cross-check: it agrees with the exact path for the in-range, no-boundary-crossing case and differs
(correctly) when a boundary is crossed.

Residuals that remain, all inherited from the tracker and surfaced on the ledger:

* **Unknown pre-swap price at replay start.**  The opening tick is seeded from the tape's first
  observed tick, but the first swap's pre-price is legitimately unknown.  A swap that crosses a
  tick as the first swap sets the tracker's `exact` flag to `False` (with a
  `FeeGrowthApproximation` warning); `BacktestLedger.fee_growth_exact` records it.  Single-segment
  swaps stay exact even at replay start.
* **Window-relative tick lattice / active liquidity.**  The tracker builds its lattice from the
  events present in the tape window; a position whose boundary ticks were crossed before the
  window (and whose mints are not in the tape) is not represented.  A precomputed
  `load_fee_growth_state` checkpoint would remove this; it is not needed for S11's tests.
* **Flash fees are out of scope** (T10 module decision 5): the frozen tracker interface has no
  flash method.

## Consequences

- `src/undertow/sim/backtest/engine.py` implements the exact `FeeGrowthReplay` path (Decision §1's
  tape-columns apportionment lives on in the tests as the cross-check); `Position.uncollected_fees`
  is **not** used (its float snapshots would round-trip the Q128 accumulator through `float64`).
- **The settled surface grows by exactly one name**, `FeeGrowthReplay` (ADR-008's superset
  continues; 54 → 55 names), pinned by `tests/data/test_public_api.py` and
  `tests/sim/test_data_api.py`.
- **S14 (parity)** measures residuals other than boundary crossing: the unknown pre-swap price at
  replay start (now flagged) and the window-relative lattice.  Boundary-crossing drift is expected
  to be zero.
- **S15 (evaluation)** inherits the flagged residual when it reports backtest PnL for rebalancing
  policies: a run whose tape opens on a crossing swap records `fee_growth_exact=False`.
- `CONTRACTS.md` §13's claim that S11 "wires FeeGrowthTracker through S02's exports" is now true in
  substance: S11 wires the exact tracker through the `FeeGrowthReplay` facade, which is the
  deliberate export this ADR adds.  No contract field is renamed or removed.
- Rejected — **importing `undertow.data.transforms.feegrowth` directly.**  It is a Critical review
  finding under PLAN.md §4/§0.7 and would make S11 depend on an unstable internal.
- Rejected — **re-implementing the tick lattice inside the backtester.**  That duplicates the exact
  arithmetic the data module already reconciles, guaranteeing two divergent "exact" engines — the
  precise failure the two-artifact split exists to prevent.
- Vault edits are **not** made here; only repo files change.
