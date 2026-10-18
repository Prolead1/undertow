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

## Evidence

- Single-env step: **67–95 steps/s → ~700 steps/s** (1.4 ms, hold).
- Four envs, random actions (PPO exploration): `dummy` 646 → **`subproc`
  1833 steps/s** (2.8×).
- Real CLI run: 100,000 steps, one seed, `subproc` = **2m57s**; the default
  5 × 1M budget is therefore **~1.5–2 hours** (was ~17h).

## Consequences

- The replay engine is no longer the bottleneck; a step is Python-bound, so the
  process backend now gives real multicore scaling.
- Workers each hold a `MarketView` (~4 GB resident), so the worker count is
  memory-bound: the pinned `parallel_envs = 4` fits in the 30 GB host; raising it
  needs memory headroom (and changes the PPO rollout, since it is `n_steps ×
  parallel_envs`).
- `dummy` remains the default for synthetic/test views; `subproc` needs a factory
  that can rebuild its view in the worker, which is why the CLI no longer passes
  the closure over the parent frame to `train_ppo`.
- `build_replay_env` gains an optional `price_process` so calibrated training
  reuses the exact same engine/frictions construction as replay.
