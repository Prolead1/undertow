# ADR 020 — Real-data liquidity replay in the float64 sim: rounding tolerance and checkpointed warmup

**Status:** accepted (implementation)
**Affects:** `src/undertow/sim/core/pool.py` · `src/undertow/sim/env/lp_env.py` · `tests/sim/test_pool.py` · `tests/sim/test_env.py` · sim `STATUS.md` · S12, S14
**Raised by:** RL kick-off on the real 2022–23 train split (post ADR-018/019)

## Context

ADR-019 made `PoolEngine` the float64 twin of the backtester's exact external-pool fee engine, and
ADR-018 gave it a boundary `pool_state` snapshot plus a pre-window warmup so an episode can start
mid-lifecycle. Both were validated against fixtures and against the S14 parity replay, but the
combination — *training* on the real 2024 dataset — had never been exercised. The first real run
crashed and was then found to be impractically slow.

## Problem

**1. A false "inconsistent stream" crash above `2**53`.** The seed at block `13,916,167` plus the
lane's mints put `liquidity_gross` at tick `214800` at exactly `11,806,162,276,124,034` before the
burn at block `14,718,509` removes exactly that amount. Integer accounting is consistent, but
`TickState.liquidity_gross` is float64 and `2**53 ≈ 9.007e15`, so the accumulated float lands 2 units
below the integer burn. `gross_after = 1.1806162276124032e16 - 1.1806162276124034e16 = -2.0`, and
`PoolEngine.apply_liquidity_event` rejected it as inconsistent — aborting training. The backtester
never saw this because its `FeeGrowthTracker` uses Python ints.

**2. Warmup replay was O(distance from the lane start) on every reset.** `_reset_external_pool`
reseeding from the lane snapshot and replaying *every* prior row cost ~16s for a late-2022 episode
(the median episode whose start is ~265k events in). At ~1,157 resets for the default 5 seeds × 1M
steps, that is ~5 CPU-hours of pure warmup, on top of stepping.

## Decision

**1. Absorb float64 rounding, never real gaps.** A negative `gross_after` within
`_LIQUIDITY_ROUNDING_TOLERANCE = 1e-12` of the applied `|delta|` is clamped to `0.0` (the tick is
removed as fully unwound). Anything larger still raises `PositionError`. The tolerance is relative to
the delta, so it scales with the magnitudes that can carry float error.

**2. Memoise warmup checkpoints per environment.** On the first reset, `LpEnvironment` replays the
backed lane once from the snapshot (`warmup` + active rows) and records a deep copy of the lattice
every `WARMUP_CHECKPOINT_STRIDE = 10,000` events. Later resets restore the nearest checkpoint with
`bisect_left` — strictly *before* the episode start, because a checkpoint holds the state *after* its
row — and replay only the remaining gap, bounding warmup at one stride. The checkpoints are local to
the env instance (each `DummyVecEnv` slot builds its own once) to avoid cross-instance coupling.

## Consequences

- Training on the real train split completes: `undertow-sim train` on a smoke config ran end-to-end
  (previously it aborted on the first late window).
- Late-window resets drop from ~16s to ~0.6–0.7s (~25×); the one-time ~20s lane replay amortises over
  hundreds of resets.
- The checkpoint path is *bit-identical* to a full replay: a checkpoint is the exact float state
  produced by the same replay sequence, and the remaining rows are applied in the same order. Tests
  pin this for starts on and between checkpoints.
- The simulator remains a float64 approximation of the exact tracker (ADR-019); this ADR bounds the
  approximation's observable failure mode instead of changing the modelling.
- The tolerance is one-sided (negative residues only) and never masks a genuine over-burn, which the
  suite asserts separately.
