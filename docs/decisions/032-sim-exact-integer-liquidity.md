# ADR 032 — Exact integer liquidity bookkeeping on the snapshot/tape path

**Status:** accepted (implementation)
**Affects:** `src/undertow/sim/core/pool.py` (`TickState`, `from_snapshot`, `apply_liquidity_event`, `open_position`) · `tests/sim/test_pool.py` · `docs/decisions/020-sim-realdata-liquidity-replay.md` (amended status) · sim `STATUS.md` · S07, S12, S14
**Raised by:** the first training run on the real `eth_usdc_500` dataset (2022–23 train split), after `undertow-data pool-state` captured its boundary snapshot

## Context

ADR-020 absorbs float64 rounding in `PoolEngine.apply_liquidity_event`: a
negative `liquidity_gross` within `_LIQUIDITY_ROUNDING_TOLERANCE = 1e-12` of the
applied `|delta|` is clamped to zero, and anything larger raises
`PositionError`. That tolerance is relative to the *current* delta, on the
reasoning that the delta scales with the magnitudes that can carry float error.
ADR-019/020 keep the whole simulator in float64 on purpose.

Real 0.05 % pool data breaks that reasoning twice in the 2022 train lane:

1. **Carried residue.** The boundary snapshot seeds tick `192530` at
   `21,024,082,281,029,979,573`. A burn of `21,024,078,024,294,254,872` leaves
   an exact remainder of `4,256,735,724,701`, but float64 stores
   `4,256,735,723,520` (1,181 low). A second burn of the exact remainder then
   computes `gross_after = -1,181`; the delta-relative allowance for that burn
   is only `1e-12 × 4.26e12 ≈ 4.25`, so the consistent stream is rejected.
2. **Sub-ULP remainder.** At tick `204360` the running gross reaches
   `1,362,030,533,161,524,474` and a burn of `1,362,030,533,161,524,413` leaves
   exactly `61`. Both large values round to the same representable float64
   (`…480`), so the subtraction lands on `0.0`; the tick is deleted with the
   `61` lost, and the following, consistent burn of `61` then raises. No
   tolerance constant can recover this — the information is gone.

Both failures abort `LpEnvironment` warmup and every downstream training run.

## Decision

**Make the production accounting exact instead of tolerating its drift.**
`liquidity_gross` and `liquidity_net` are integer counts; the simulator keeps
them as Python `int`s on the snapshot/tape path:

- `PoolEngine.from_snapshot` seeds snapshot rows with `_as_int(...)` (not
  `float(...)`).
- `apply_liquidity_event` starts an absent tick from int `0`, and when
  `gross_before`/`net_before` are ints performs `gross_before + delta` in
  integer arithmetic. A negative result is a genuine stream inconsistency and
  always raises; an exact zero removes the tick, exactly as before.
- `open_position` seeds its marginal sentinel ticks with int `1`/`0`.

The ADR-020 delta-relative tolerance remains **only for explicitly
float-seeded `TickState`s** (hand-built engines and fixtures that pass floats).
That path is unchanged: a small residue within the delta allowance is clamped,
anything larger raises.

Fee-growth arithmetic (`fee_growth_outside`, the Q128 globals, `state.liquidity`
and the swap/fee path) stays float64; only the mint/burn bookkeeping becomes
exact. `state.liquidity` is still a float and converts the int net on crossing,
which is the modelling precision ADR-019 pins.

## Consequences

- The real `eth_usdc_500` warmup completes: both sequences above replay exactly
  (verified on the full 2022–23 lane and by the env smoke run).
- The whole class of "carried residue"/"sub-ULP remainder" crashes disappears
  from production data; no heuristic can mask a genuine gap on that path.
- Float-seeded fixtures keep their existing behaviour and tests; the two code
  paths are selected by the type of the stored state (int = exact, float =
  ADR-020 fallback).
- Python ints add arbitrary-precision adds only on mint/burn rows (swaps never
  touch gross); the real train-lane warmup and 1,500-step smoke completed with no
  measurable throughput regression against the float path.
  A crossing still converts the int `liquidity_net` to float64 when it updates
  `PoolState.liquidity`, which is deliberate: the active-liquidity field stays
  float and only the mint/burn bookkeeping is exact.
- The exact path agrees with the backtester's integer `FeeGrowthTracker` by
  construction on the liquidity dimension (S14 parity).
