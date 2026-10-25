# ADR 033 — Training throughput: cached lattice keys, episode index, window extraction

**Status:** accepted (implementation)
**Affects:** `src/undertow/sim/core/pool.py` (`PoolEngine`, `_select_crossed_ticks`) · `src/undertow/sim/marketview.py` (`MarketView.build_episode`, `window`, `active_split_data`) · `src/undertow/sim/env/lp_env.py` (`_prepare_replay_events`, `_warmup_rows_between`, `_warmup_frames`) · `tests/sim/test_pool.py` · `tests/sim/test_marketview.py` · sim `STATUS.md` · S12, S14, S18
**Raised by:** the first RL pilot on `eth_usdc_500` (ADR-032 worktree), which ran at ~150–175 training steps/s

## Context

The 0.05 % pool tape is 6.86 M swaps / 4.86 M train rows. The pilot measured
~310 steps/s in a bare env loop and ~150–175 steps/s end-to-end, so a 500 k-step
seed took ~55 min. Three per-reset costs were O(tape) and one per-swap cost was
O(lattice):

- `MarketView.build_episode` converted `block_timestamp` and `seq` for the whole
  active split to Python lists on every `reset()` (4.9 M rows): **1.88 s per
  reset**.
- `MarketView.slice(...).active_split_data()`, called from
  `LpEnvironment._prepare_replay_events`, materialised the lane's entire
  pre-window history into `warmup` on every reset even though the caller only
  needs the window's rows.
- `_warmup_rows_between` filtered the full 4.9 M-row frame with a boolean mask on
  every reset.
- `_select_crossed_ticks` ran `sorted(ticks)` for **every swap** (~30–50 per
  decision), re-sorting a ~1–2 k-key lattice each time.

## Decision

1. **Cache the lattice key index.** `PoolEngine` keeps an ascending
   `_sorted_keys` list, rebuilt lazily; `apply_liquidity_event` and
   `open_position` clear it (they can add/remove keys), and a key-count check
   covers direct fixture mutation. `_select_crossed_ticks` gains an optional
   `sorted_keys` argument and selects with two `bisect` calls instead of sorting.
   The default (no argument) keeps the helper self-contained for `tick_cross`
   and tests.
2. **Cache the episode index.** `MarketView` lazily stores
   `(block_timestamp, seq)` as numpy arrays (`datetime64[us]`, `int64`) and
   `build_episode` uses `np.searchsorted` (`"right"` for candidates, `"left"` for
   the window end), reproducing the previous `bisect` sampling exactly. The
   frozen dataclass is updated through `object.__setattr__`.
3. **Add `MarketView.window`.** It returns `[start_seq, end_seq)` active rows via
   two `search_sorted` calls plus one `slice`, without building `warmup`;
   `_prepare_replay_events` uses it. `slice` remains for the backtester's
   warmup-aware windows.
4. **Binary-search the warmup gap.** `_warmup_rows_between` uses
   `search_sorted` + `slice` instead of a full-column mask. `active_split_data`
   also skips its defensive `sort` when `seq` is already sorted (it always is
   for frames built by the data pipeline / `slice`).

Behaviour is unchanged: the samples, the crossing sets, the warmup rows and the
fee accounting are identical (pinned by the existing tests plus the new
equivalence tests).

## Consequences

Measured on the real 4.86 M-row train split (same machine, same window):

| metric | before | after |
|---|---|---|
| `build_episode` (per reset) | **1.88 s** | **1.7 ms** |
| raw env stepping | **310 steps/s** | **782 steps/s** |
| one-time warmup replay | **279 s** | **55 s** |
| warm reset (memoised checkpoints) | seconds | **1.08 s** |

- `build_episode`'s numpy sampling matches the original `bisect` algorithm on
  the real view for 20/20 seeds (and is pinned by a fixture equivalence test).
- Crossing selection drops the per-swap sort; the warmup lane replay and every
  stepping interval get proportionally faster.  The cache is invalidated by the
  engine's own mint/burn/open mutators, and the key-count check covers direct
  fixture setup.
- The cached arrays add 8 bytes/row for timestamps and 8 bytes/row for seq
  (~78 MB for the 4.86 M-row train split), replacing a much larger transient
  list build per reset; the `is_sorted` fast path also avoids a defensive sort
  allocation on every `active_split_data()` call.
- Behaviour is unchanged: the sampled windows, crossing sets, warmup rows and
  fee accounting are identical (pinned by the existing tests plus the new
  equivalence tests).
- End-to-end training throughput follows the env gain; the pilot is re-run at
  the same 3 x 500 k budget and the per-seed wall time is recorded in
  `STATUS.md`.
