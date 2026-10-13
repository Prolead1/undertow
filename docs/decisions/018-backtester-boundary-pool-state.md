# ADR 018 — Seed the backtester's exact fee engine from an on-chain boundary snapshot

**Status:** accepted (implementation and export)
**Affects:** `src/undertow/data/fetchers/abi.py` · `src/undertow/data/fetchers/rpc.py` · `src/undertow/data/feegrowth_api.py` · `src/undertow/data/loader.py` · `src/undertow/data/pipeline.py` · `src/undertow/data/cli.py` · `src/undertow/sim/marketview.py` · `src/undertow/sim/backtest/engine.py` · `configs/eth_usdc_*.toml` · data `CONTRACTS.md`
**Raised by:** training-readiness follow-up (backtester/RQ3)

## Context

The backtester is the thesis's **measurement instrument** (`PLAN` §1: "correct, not
fast"); RQ3 is `Δ = sim_PnL − onchain_PnL`. Its fee path is the exact integer
engine (ADR-013), which is a state machine: given the pool's state at the window
start, it replays the tape's swaps and mints/burns exactly.

The study window starts **2022-01-01**, but the pinned pool deployed at block
12,370,624 (2021-05). A split window therefore starts mid-lifecycle: positions
minted in 2021 and burned inside the window appear as burns with no preceding
mint. Measured: **1,715** such burns in the 2022–23 train split and **2,555** in
the 2024 eval split. The exact tracker starts from an empty tick lattice and
raises `burn of liquidity … would drive liquidity_gross negative`. The data
module already flags the window as unseedable (`check_liquidity_conservation` →
`seed_block=none`), i.e. the boundary state simply isn't in the dataset — the
`fee_growth` stream was reserved by the data plan and never implemented.

## Problem

Without the boundary state the exact engine cannot run on **any** real split, so
the backtester, the S14 parity gate, and RQ3 are all blocked. The obvious "fix" —
tolerate pre-window liquidity events and flag the residual — keeps the backtester
*running* but no longer *correct*, which defeats the instrument's whole purpose
and would inject an unquantified bias into the headline number.

## Decision

Capture the pool's exact state at each walk-forward boundary and seed the engine
from it.

1. **`RpcFetcher.pool_state_at(pool, block)`** returns a complete
   `FEE_GROWTH_SCHEMA` table: one global sentinel row (`slot0`, `liquidity()`,
   `feeGrowthGlobal0/1X128()`) plus one row per **initialized tick**, enumerated
   from the pool's `tickBitmap(int16)` (all words covering `[MIN_TICK, MAX_TICK]`
   — 116 words for the 0.3% pool) and read via `ticks(int24)`. Real cost: **570
   initialized ticks** at 2022-01-01, ~1,300 rows across both boundaries, ~1,100
   `eth_call`s once.
2. **`pipeline.capture_pool_state(config)`** writes those snapshots as the
   optional `fee_growth` stream, taken at **`first_tape_block - 1`** — the state
   *before* the lane's first event. (`pool_state_at(B)` is post-block-`B` state,
   and the backtester replays block `B`'s rows, so seeding at `B` would
   double-apply them; verified by replaying block 13,916,168 from the seed at
   13,916,167 and matching the independent snapshot exactly.) Exposed as
   `undertow-data pool-state`; requires the `WindowConfig` split hook
   (`train_end_utc`/`eval_start_utc`, now set in both pinned configs). The
   snapshot leaves `current_sqrt_price_x96` unset, so a first crossing swap is
   flagged approximate (`fee_growth_exact=False`); the facade does not report the
   underlying `eth_call` count, so the stream's `n_requests` is `0`.
3. **`loader.load_fee_growth(config)`** (public API) returns the stream or a typed
   empty table.
4. **`MarketView.pool_state`** carries the snapshot for the active split's first
   block (selected by `build_market_view`); `slice` propagates it.
5. **`FeeGrowthReplay.from_snapshot(pool, table)`** builds the tracker state
   (globals, `current_tick`, `current_liquidity`, `fee_protocol`, the tick
   lattice) from the table and rejects a snapshot for a different pool.
   `run_backtest` seeds from `market_view.pool_state` when present and
   **requires the snapshot block to equal `first_block - 1`** (the state before
   the window's first event; any other block is a hard error, never a silent
   approximation). Without a snapshot it keeps the old empty-lattice fallback
   (correct for fixtures, which contain every mint).
6. **Walk-forward windows reuse the lane seed by warming up.** `MarketView.slice`
   keeps the lane-start `pool_state` plus a `warmup` frame of the parent lane's
   rows before the window (composed across nested slices); `run_backtest` replays
   those warmup rows through the exact engine before accounting. So every window
   of `walk_forward_windows` (RQ3's `run_gap_analysis`, S15 `evaluate_episodes`,
   and `run_parity_check`'s per-episode backtests) starts from the lane seed —
   the seed block check compares against the first *replayed* event. Each window
   is exact subject to the lane-seed approximation, if any (a boundary swap that
   crosses a tick with unknown pre-price flags `fee_growth_exact=False` for the
   whole lane, conservatively).

Snapshot fetches are throttled (small JSON-RPC batches, a minimum interval) and
`_jsonrpc_batch` now retries **per-item** transient errors (e.g. a provider's
CU/s cap on one call in a batch), which the HTTP-level retry layer did not cover.

## Consequences

- `run_backtest` now runs **exactly** on real splits: verified 2022–23 train,
  406,245 events, `BacktestLedger.fee_growth_exact is True`. With the `slice`
  warmup replay, the walk-forward windows that drive the S14 parity gate, the RQ3
  on-chain gap (`run_gap_analysis`), and the `undertow-sim parity` report are
  exact too. (Warmup re-applies up to a lane's rows per window, so a walk-forward
  sweep over a long lane is slower than a single full-lane run; the tracker
  application itself is not the bottleneck.)
- The dataset gains an optional `fee_growth` stream. Its globals also populate the
  tape's `fee_growth_global_*` columns (as-of backward, `fee_growth_source`
  marks them); with only two boundaries most in-window rows are `stale_prior`,
  which is honest and does not affect the exact replay.
- The snapshot must be refreshed if the split boundary moves; `pool-state` is
  idempotent and cached, so re-running it is cheap.
- Rejected alternatives:
  - *Tolerate pre-window liquidity events.* Keeps the instrument running but not
    correct; rejected.
  - *Pull the whole tape from deployment.* Drags in pre-London blocks that have no
    EIP-1559 base fee, fighting the pinned post-1559 window, and adds a warmup
    concept through the whole backtester.
  - *Per-block `feeGrowthGlobal` snapshots.* ~450k `eth_call`s; the boundary
    lattice already makes the tape replay exact.
