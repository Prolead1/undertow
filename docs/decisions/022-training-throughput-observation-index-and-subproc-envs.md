# ADR 022 — Training throughput: O(log n) as-of lookups and process-parallel environments

**Status:** accepted (implementation)
**Affects:** `src/undertow/sim/env/observations.py` · `src/undertow/sim/train/ppo_runner.py` · `src/undertow/sim/train/envs.py` · `src/undertow/sim/cli.py` · `src/undertow/sim/config.py` · `src/undertow/sim/marketview.py` · `src/undertow/sim/evaluate/protocol.py` · `configs/sim_default.toml` · sim `CONTRACTS.md` §2 · sim `STATUS.md` · ADR-010, ADR-019
**Raised by:** RL kick-off throughput

## Context

The real-data RL run was measured at **67–95 steps/s** in a single process. With
`DummyVecEnv` stepping its four "parallel" slots sequentially, the default
5 seeds × 1M steps budget was ~17 hours. The machine has 8 usable CPUs, so the
obvious fix looked like process-parallel environments.

## Problem

Profiling one decision step (11 ms wall) showed the cost was **not** the replay
engine:

| part | time | share |
|---|---|---|
| `ObservationBuilder._reference_closes_through` — `reference.filter(close_time <= t).sort(...)` over the full 1.5M-row feed, every step | 8.3 ms | 75% |
| `MarketView.regime_at` — full as-of scan of the 1.5M-row regime feed | 1.2 ms | 11% |
| `_gas_price_recent` → `MarketView.gas_at_block` — full scan of the 7.6M-row gas feed | 0.9 ms | 8% |
| the exact `PoolEngine` replay itself | 0.15 ms | 1% |

`DummyVecEnv` was already saturating all 8 cores because polars parallelises
those scans. `SubprocVecEnv` alone therefore could not help — it would only
oversubscribe further. The right fix was algorithmic.

## Decision

1. **Precompute sorted as-of indexes in `ObservationBuilder`.** On construction,
   the reference closes, regime labels and gas fees are sorted once into
   `int64` epoch/block arrays; each step uses `np.searchsorted` (returning a
   *view* into the sorted closes) instead of filtering and sorting the full
   feeds. Complexity goes from O(rows) per step to O(log rows), with identical
   semantics (closed-on-the-right, the look-ahead wall is unchanged).
2. **Process-parallel environments as a configured backend.** `TrainingConfig`
   gains `vec_env: Literal["dummy", "subproc"] = "dummy"`; `sim_default.toml`
   sets `"subproc"`. `SubprocVecEnv` (SB3's default `forkserver`/`spawn` — plain
   `fork` after torch import can deadlock) runs one process per `parallel_envs`
   slot. The CLI builds a picklable `EnvFactory` that stores only the data-config
   path/config and rebuilds the `MarketView` in each worker; `__getstate__`
   drops the in-process cache so the multi-gigabyte frame is never shipped.
3. **Pin BLAS/OpenMP to one thread per process** (`OMP_NUM_THREADS` & friends
   before workers start, `torch.set_num_threads(1)` in the parent), so four
   workers do not spawn four thread pools each.
4. **Binary-search the `MarketView` as-of lookups too.** Re-profiling after (1)
   showed `_rebalance` still calling `MarketView.gas_at_block` twice per
   rebalance (once for the gas row, once inside `ReplayGasModel`), each a scan
   of the 7.6M-row feed.  `regime_at`/`gas_at_block` now use
   `Series.search_sorted` on the by-construction-sorted feeds instead of
   filter+sort.

## Evidence

- Single-env step: **67–95 steps/s → 6,850 steps/s** hold (0.15 ms) and
  **1,269 steps/s** with random/rebalance-heavy actions (0.79 ms).
- Four envs, random actions (PPO exploration): `dummy` 1,112 → **`subproc`
  2,631 steps/s** (2.4×).
- Real CLI run: 100,000 steps, one seed, `subproc` = **2m31s** (was 2m57s after
  step 1 and ~7m before).  That is ~53 s of per-seed startup plus ~98 s of
  stepping/updates, so a 1M-step seed is roughly 17 min and the default
  5 × 1M budget **~1.5 h** (a naive linear extrapolation of the 100k wall time,
  which includes startup, gives ~2 h).  Was ~17 h before this ADR.

## Consequences

- The replay engine is no longer the bottleneck; a step is Python-bound, so the
  process backend now gives real multicore scaling.  PPO updates are now a
  meaningful share of the remaining time; using more torch threads trades update
  time for slower action sampling and does not improve wall-clock (measured), so
  the parent stays at one thread.
- `Series.search_sorted` itself costs ~0.2 ms per call (polars builds a lazy
  selection internally).  It is still much cheaper than the old full scans; a
  numpy block-index cache on `MarketView` could remove that residual, but the
  view is frozen/slots and the feeds are memory-bound, so it is not done here.
- Workers each hold a `MarketView` (~4 GB resident), so the worker count is
  memory-bound: the pinned `parallel_envs = 4` fits in the 30 GB host; raising it
  needs memory headroom (and changes the PPO rollout, since it is `n_steps ×
  parallel_envs`).
- `dummy` remains the default for synthetic/test views; `subproc` needs a factory
  that can rebuild its view in the worker, which is why the CLI no longer passes
  the closure over the parent frame to `train_ppo`.
- `build_replay_env` gains an optional `price_process` so calibrated training
  reuses the exact same engine/frictions construction as replay.  That also
  unifies the **initial deployment width**: calibrated training now deploys the
  parity band, matching replay training, the backtester and the evaluate runner,
  where it previously used `lp_env`'s standalone default of 10.  The width is
  derived from the config spacing as
  ``DEFAULT_INITIAL_HALF_WIDTH_TICKS // tick_spacing`` (2 at the 0.30% tier's
  spacing 60; 12 at the primary 0.05% pool's spacing 10), so the two instruments
  start from the same token mix on every pool.  The derivation is pinned by
  `tests/sim/test_train_envs.py` and
  `tests/sim/test_validation.py::TestRunParityCheck`.
