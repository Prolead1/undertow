# S03 — MarketView + walk-forward split

**Wave 2 · size M · depends on: S01, S02 · blocks: S06, S08, S11, S12, S15**  
**Also supplies data to: S06 (calibration), S08 (gas stream)**  
**Branch:** `feature-sim-marketview`

## Why this task exists

Every consumer of historical data in the sim must go through a single, look-ahead-safe door.
`MarketView` is that door. It owns the train/eval split, enforces the wall between them, and
provides a query API that makes it impossible for a consumer to accidentally read evaluation-window
data while building a training view. This is the single most important correctness mechanism in the
entire sim package — if it fails, every result is tainted.

The walk-forward split boundaries are pinned in PLAN.md §7 and recorded in `SplitConfig`. This task
makes those boundaries real.

## Files you own

```
src/undertow/sim/marketview.py            (MarketView + build_market_view)
tests/sim/test_marketview.py
tests/sim/conftest.py                     (appends tiny_market_view fixture)
```

## What to build

### 1. `marketview.py` — CONTRACTS.md §4

Implement `MarketView` as a frozen dataclass with:

- Fields: `train` (pl.DataFrame | None), `eval` (pl.DataFrame | None), `reference` (pl.DataFrame),
  `regimes` (pl.DataFrame), `gas` (pl.DataFrame), `train_start_utc`, `train_end_utc`,
  `eval_start_utc`, `eval_end_utc`, `is_train` (bool)

- `__post_init__`: enforce the invariants from CONTRACTS.md §4.1:
  1. `train` and `eval` are never both non-None
  2. The active split (whichever is non-None) has at least one row
  3. The reference feed spans at least from the active split's start to its end

- Query methods (all with look-ahead enforcement):
  - `slice(start_seq, end_seq)` → `MarketView` sub-view. Raises `LookAheadError` if `end_seq` exceeds
    the active split's last seq
  - `tape_at(seq)` → `dict[str, object]` for a single tape row. Out-of-split → `LookAheadError`
  - `reference_close(at_time)` → `float | None`. Last reference close with `close_time <= at_time`.
    Returns None before the feed starts (don't raise — the first few steps may predate the reference
    feed)
  - `reference_bars(start_time, end_time)` → `pl.DataFrame`. Clamped to active split's time bounds
  - `regime_at(at_time)` → `str`. As-of join on the regime labels table. Returns `"unknown"` if no
    regime label covers that time
  - `gas_at_block(block_number)` → `dict[str, object] | None`
  - `active_split_data()` → `pl.DataFrame` (the non-None of `train`/`eval`)
  - `step_count()` → `int` (length of the active split's tape)
  - `build_episode(seed, max_steps)` → `tuple[int, int]`. Sample a contiguous episode window. Uses a
    `np.random.Generator(seed)` internally. Never crosses the split boundary. The window length is
    config-driven (you take `EpisodeConfig` as a parameter). Returns `(start_seq, end_seq)` where
    `end_seq` is exclusive

- Standalone function `build_market_view(data_config, split_config, *, split)` → `MarketView`.
  Uses the S02 `load_*` functions. If a loader raises `NotImplementedError` (pending T16), this
  function should raise `NotImplementedError` with the same message — don't silently create an
  empty view. Raises `MarketViewError` if the split window contains no data

### 2. `tests/sim/conftest.py` — shared `tiny_market_view` fixture

Append (do not replace S00's `rng` fixture) a `tiny_market_view` fixture that:

- Synthesizes a `MarketView` with the train split active
- ~1000 tape rows with columns matching `EVENT_TAPE_SCHEMA` (trim a few real rows if available;
  synthesize if the data loader isn't available yet)
- ~100 reference bars (1-minute closes) spanning the tape's time range
- ~100 gas rows
- ~100 regime label rows (alternating `"bull"`, `"bear"`, `"sideways"`, `"high_vol"` every ~25 rows)
- Train split: `2022-01-01 → 2022-02-01`; eval: None
- Use numpy seeded from `rng` fixture so it's deterministic

This fixture is the shared test data for S04–S15. Build it carefully — cross-task bugs are caught
here before they become integration failures.

## Tests you must write

1. `build_market_view` raises `MarketViewError` when given a split window with no data
2. `MarketView` `__post_init__` rejects both `train` and `eval` non-None
3. `MarketView` `__post_init__` rejects both `train` and `eval` None
4. `slice()` raises `LookAheadError` when `end_seq` exceeds the active split
5. `tape_at()` raises `LookAheadError` for out-of-split seq
6. `reference_close()` returns the correct bar for a given time (test edge: exactly at `close_time`)
7. `reference_close()` returns None before the feed starts
8. `reference_bars()` is clamped to the active split's time bounds
9. `regime_at()` returns the correct regime label for a given time
10. `regime_at()` returns `"unknown"` before the first regime label
11. `gas_at_block()` returns the correct gas row
12. `active_split_data()` returns the correct DataFrame
13. `build_episode()` returns a valid `(start, end)` within the split
14. `build_episode()` with the same seed returns the same window (determinism)
15. `build_episode()` never produces a window that ends past the split boundary
16. `tiny_market_view` fixture is importable and has `step_count() > 0`
17. All `slice()`/`tape_at()` calls on the `eval`-mode view with train data present raise
    `LookAheadError` if they access the train data

## Definition of done

- All files exist, typed, no TODO stubs
- `uv run pytest tests/sim/test_marketview.py -q` green
- `tiny_market_view` fixture is importable from `tests/sim/conftest.py`
- `code-reviewer` APPROVE
- PR open against `main` from `feature-sim-marketview`
- `STATUS.md` updated