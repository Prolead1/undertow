# ADR 021 — Liquidity-replay tick crossings follow the pool tick, not the price

**Status:** accepted (implementation)
**Affects:** `src/undertow/sim/core/pool.py` · `src/undertow/data/transforms/feegrowth.py` · `src/undertow/data/validation/checks.py` · `tests/sim/test_pool.py` · `tests/data/test_feegrowth.py` · `tests/data/test_checks.py` · sim `CONTRACTS.md` §6 · sim `STATUS.md` · data `STATUS.md` · ADR-013, ADR-019, ADR-020
**Raised by:** real-data active-liquidity divergence found during the RL kick-off (ADR-020 follow-up)

## Context

ADR-013/ADR-019 selected the initialized ticks to cross during a swap by comparing the swap's
``sqrt_price_x96`` against each tick's boundary.  The stated reason was that a swap's recorded
``slot0.tick`` can disagree with its ``sqrt_price_x96`` by one at an exact boundary, so a tick-derived
selection was believed to cross the set with the wrong sign.  Both the exact ``FeeGrowthTracker`` and
the float64 ``PoolEngine`` used the price rule (upward ``pre < price(t) <= post``; downward
``post <= price(t) < pre``), which made sim ↔ tracker parity hold while both diverged from mainnet.

## Problem

The price is not the pool's crossing state at a boundary.  A Uniswap V3 swap that moves *down* and ends
exactly on ``boundary(T)`` executes the contract's loop branch
``state.tick = tickNext - 1``, so it records ``slot0.tick = T-1`` while ``slot0.sqrtPriceX96`` stays
exactly ``boundary(T)`` and ``T`` **has been crossed**.  The next swap that moves *up* therefore starts
with ``pre_tick = T-1`` and the price sitting on ``boundary(T)``.  The price rule sees
``pre == boundary(T)`` and, being strict at ``pre``, skips crossing ``T`` — so the external position
becomes active in the sim/tracker state but not on-chain (or vice versa).

On the real 2022–24 USDC/WETH tape this first diverged at block **15,369,829**: after the downward
landing on ``boundary(201600)`` (recorded tick 201599), the upward move to tick 201604 failed to cross
201600, leaving active liquidity 0.49% high.  The error is persistent and accumulated; by block
**15,997,326** the engine reached ``-3.65e19`` against an on-chain ``+1.19e19``, and late training
windows ran on corrupted fee denominators and valuations.

## Decision

Select crossings with the protocol's own half-open **tick** interval
``(min(pre_tick, post_tick), max(pre_tick, post_tick)]``, where ``pre_tick`` is the pool's recorded
``slot0.tick`` from the previous event and ``post_tick`` the swap's recorded tick.  The sim reuses
``_select_crossed_ticks`` — the same helper ``PoolState.tick_cross`` and position management already
use — and the tracker applies the identical rule inline.  The recorded tick is authoritative because it
encodes the contract's loop state (including the ``T-1`` down-landing convention); the price is used
only for the per-segment fee apportionment, never to decide which ticks were crossed.

The price-based ``PoolEngine._crossed_ticks_exact`` / ``_crossed_ticks_by_price`` helpers are deleted.

## Evidence

Against mainnet ``liquidity()`` on 224 sampled swap blocks spanning 2022–2024:

- **0** structural divergences (> 1e-6 relative); worst relative difference **1.35e-12** (float64 ULP).
- The engine at the eval-lane snapshot block ``18,908,911`` matches the on-chain ``current_liquidity``
  to 1.3e-13 and the tick exactly.
- Active liquidity never goes negative across the full lane (minimum ``6.2e17``).
- Real-data sim ↔ backtester parity on a late (previously corrupted) window: ``fee_drift_mean =
  5.1e-8``, ``il_drift_mean = 0``, ``pnl_drift_mean = 5.1e-8``, ``passed = True``;
  ``BacktestLedger.fee_growth_exact = True``.

## Consequences

- Real-data active liquidity now tracks the chain, which is a precondition for trustworthy RL training
  and RQ1–RQ3 results.
- The tests that encoded the price rule are rewritten to pin the protocol convention, including the
  exact block-15,369,829 down-landing → reverse-up regression, in both the engine and the tracker.
- `data.validation.check_liquidity_conservation` had the same wrong downward interval
  (`[post_tick, current_tick)`); it is corrected to `(post_tick, current_tick]` and its tests now use the
  `T-1` convention.
- The ``T-1`` down-landing convention is now load-bearing; any future crossing code must key off the
  recorded tick, not the price.
- The zero-volume alignment step used by fixture views still passes its caller-computed tick; at an
  exact downward boundary that tick is the physical ``T`` rather than ``T-1``, so a fixture-only
  alignment could differ from a real swap.  Tape-driven (snapshot-seeded) replays never take that path.
