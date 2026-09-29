# ADR 013 — Backtester fee growth: consume the tape's Q128 global columns; propose a public inside-accrual extension

**Status:** accepted (implementation); the proposed data export is **proposed**
**Affects:** `src/undertow/sim/backtest/engine.py` · `src/undertow/data/__init__.py` · `src/undertow/data/loader.py` · sim `CONTRACTS.md` §3/§13 · tasks S02, S11, S14
**Raised by:** S11 (`feature-sim-backtester`)

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

2. **Propose, for a future data task, a deliberate additive export** that lets the backtester reach
   the exact path without violating the boundary rule:

   ```python
   # undertow.data public API (additive; ADR-008 superset continues)
   from undertow.data.loader import (
       load_fee_growth_state,   # DataConfig, block_number -> serialisable FeeGrowthState
   )
   ```

   combined with a small public accrual adapter:

   ```python
   def accrue_position_fees(
       state: FeeGrowthState,
       *,
       tick_lower: int,
       tick_upper: int,
       liquidity: int,
       g_inside_last_0: int,
       g_inside_last_1: int,
   ) -> tuple[int, int, int, int]:
       """Raw (fees0, fees1, new_inside_last_0, new_inside_last_1) — the data
       module's FeeGrowthTracker.accrue() without exposing the tracker itself."""
   ```

   That keeps the exact-integer engine, the per-tick lattice and the wrapping
   arithmetic inside `undertow.data` where they are already reconciled, while giving S11/S14 a
   stable, top-level surface.  The tape's global columns remain useful for the in-range fast path
   and for cross-checking.

3. **Document the residual in S11's ledger provenance.**  The ledger carries the standard
   `config_hash`/`git_commit`; the approximation is recorded in this ADR and in the module
   docstring, and S14 (parity) is the task that quantifies it.  No silent approximation ships.

## Consequences

- `src/undertow/sim/backtest/engine.py` implements the tape-columns path; `Position.uncollected_fees`
  is **not** used (its float snapshots would round-trip the Q128 accumulator through `float64`).
- **S02's settled surface is unchanged** by this task; the extension in §2 is a proposal for a
  follow-up data task, not something S11 needs to merge to be correct for its tests.
- **S14 (parity)** must treat boundary-crossing swaps and pre-existing `feeGrowthOutside` as the
  named sources of the fee drift it measures; the expected drift there is not zero.
- **S15 (evaluation)** inherits the same caveat when it reports backtest PnL for rebalancing
  policies.
- `CONTRACTS.md` §13's claim that S11 "wires FeeGrowthTracker through S02's exports" is superseded
  in part: S11 wires the **tape's** exact Q128 columns; the tracker export is the follow-up
  proposed here.  No contract field is renamed or removed.
- Rejected — **importing `undertow.data.transforms.feegrowth` directly.**  It is a Critical review
  finding under PLAN.md §4/§0.7 and would make S11 depend on an unstable internal.
- Rejected — **re-implementing the tick lattice inside the backtester.**  That duplicates the exact
  arithmetic the data module already reconciles, guaranteeing two divergent "exact" engines — the
  precise failure the two-artifact split exists to prevent.
- Vault edits are **not** made here; only repo files change.
