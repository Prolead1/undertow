# T11 — Stream alignment → the event tape

**Wave 4 · size L · depends on: T02, T03, T09, T10 (and consumes T05–T08, T12 output) · blocks: T13, T14, T16**
**Branch:** `feature-data-event-tape`

## Why this task exists

Roadmap §10.1.6 requires the five (here: seven) streams **"aligned in block order"**, and §10.2.3 states
the property the whole thesis rests on: *"Events are replayed in block order — no looking ahead, ever…
Any accidental peek of future data is look-ahead bias, the single most common way a backtest lies."*
§10.7 repeats it as a named risk with the mitigation *"the strictly block-ordered loop of §10.2.3 and
code review of every array index."*

This module produces the artifact `undertow.sim`'s backtester will consume. If look-ahead enters the
pipeline, it enters here — in an as-of join with the wrong direction, or a forward-fill that reaches
backwards in time. Treat every join as guilty until tested.

## Files you own

```
src/undertow/data/transforms/align.py
tests/data/test_align.py
tests/data/conftest.py                    (the shared `tiny_dataset()` factory — T13/T14/T16 use it)
```

## What to build

Implement `CONTRACTS.md` §6.2: `Dataset`, `build_event_tape`, `assert_no_lookahead`.

### 1. The merge
- Union the four log streams (`swap`, `mint`, `burn`, `collect`) into one table with
  `EVENT_TAPE_SCHEMA`'s superset of columns, each row tagged by `event_type`. Columns not applicable to a
  row are null (e.g. `sqrt_price_x96` on a `collect`) — **not** zero. Zero is a value; null is an
  absence, and the difference matters when T13 checks conventions.
- Sort by `(block_number, log_index)`. Assert the key is **unique** — duplicates raise
  `ValidationError` listing the offending keys. Never `drop_duplicates`: a duplicate means a fetcher
  paginated wrong, and silently swallowing it hides a real bug (this is why T05's boundary de-dup is
  T05's job, not yours).
- Assign `seq = 0..N-1` densely after sorting. `seq` is the canonical step index the RL environment will
  use.

### 2. The joins — all backward-only
| Column | Source | Join |
|---|---|---|
| `price_reference` | T08 reference | as-of **backward** on `close_time <= block_timestamp` |
| `regime` | T12 regime | as-of **backward** on `timestamp <= block_timestamp` |
| `base_fee_per_gas`, `priority_fee_p50_wei` | T07 gas | **exact** equi-join on `block_number` |
| `fee_growth_global_{0,1}_x128` | T06/T10 snapshots | as-of **backward** on `block_number` |
| `price_pool` | derived | `fixedpoint.sqrt_price_x96_to_price` on this row |
| `gas.eth_usd_price` | T08 reference | as-of **backward** on `close_time <= block_timestamp`, written back into the **gas** table (T07 leaves it null) so a gas cost can be expressed in USD at the block's prevailing price |

- Gas is an **exact** join, not as-of: §10.2.3 requires *that block's* gas price. A missing block in the
  gas stream must therefore raise, not fill — that is why T07 guarantees no gaps. Assert the join
  produced no nulls and raise `ValidationError` naming missing blocks.
- Reference and regime are as-of backward, and the tape's `block_timestamp` is the join key on the
  *left*. Use `polars.join_asof(strategy="backward")` (or `merge_asof(direction="backward")`). A
  `"nearest"` or `"forward"` strategy is look-ahead — the reviewer is told this is Critical.
- Fee-growth globals are forward-filled from the nearest **prior** snapshot, and
  `fee_growth_source` records `"exact"` when a snapshot exists at this exact block, `"stale_prior"`
  when carried forward. There is deliberately no `"interpolated_future"` value in the contract: if you
  find yourself wanting one, you are about to introduce look-ahead.
- Rows before the first available reference bar / regime label get nulls and `regime = "unknown"`,
  never a back-filled value.

### 3. `price_pool` orientation
Use T02's `sqrt_price_x96_to_price(sqrt_price_x96, dec0, dec1)` and quote T02's orientation sentence in
your docstring. `price_pool` and `price_reference` must be in the **same units** (USDC per WETH,
i.e. ~2000–4000 over the window) — otherwise every IL number downstream is nonsense. Add a test
asserting the two agree within a few percent on real-shaped fixture data; a systematic 1e12 discrepancy
is the decimals bug this test exists to catch.

### 4. `assert_no_lookahead(table, time_col)`
A reusable guard: for each as-of-joined column, verify the joined value could have been known at
`time_col`. Practical implementation: carry the **source timestamp** of each as-of join through as a
hidden `_src_ts_<col>` column during the join, assert `_src_ts <= block_timestamp` for every row, then
drop the helpers. Raise `ValidationError` naming the column and the first offending row. This function is
the mechanical expression of §10.2.3's rule and is worth more than any amount of careful reading.

### 5. `Dataset` construction — optional fields and route samples

`CONTRACTS.md` §6.2 makes every table field on `Dataset` `pa.Table | None`, plus a `route_samples`
mapping and a `require(name)` accessor. Two reasons, both of which you must honor:

- T16's `load_dataset(streams=[...])` fast path returns a **partial** dataset. `None` means "not
  requested / not present"; an empty table means "requested, present, zero rows". Collapsing the two
  would make a missing stream indistinguishable from an empty one.
- T13's `crosscheck_routes` needs *both* routes' tables for the same range, which the merged tape cannot
  provide. `route_samples[(route, stream)]` carries them; `build_event_tape` does not populate it (T14's
  route policy does), so just plumb it through and default it to empty.

`require(name)` raises `ValidationError` naming the stream **and** the `undertow-data pull` command that
would produce it — use it internally instead of scattering `assert x is not None`.

### 6. `tiny_dataset()` — the shared fixture factory (in `tests/data/conftest.py`)
A hand-built, ~30-event, few-hundred-block synthetic dataset spanning all seven streams, deliberately
containing: a tick crossing, an out-of-range position, a mint and its matching burn+collect, a gas spike,
a forward-filled reference gap, an incomplete regime window, and a swap in each direction. T13, T14 and
T16 all build on this, so make it a well-documented `pytest` fixture with a docstring listing exactly
which awkward cases it covers. **This factory is a deliverable, not a test detail** — design it as an
API.

## Tests you must write

1. Merge: four input streams → one tape, correct row count, `seq` dense and starting at 0,
   `(block_number, log_index)` strictly increasing.
2. Duplicate key in the input → `ValidationError` naming the key; assert **no** silent de-dup.
3. Column applicability: a `collect` row has null `sqrt_price_x96`, not `0`.
4. **Backward-only reference join:** construct a reference bar *after* a tape row and assert the row
   does **not** pick it up; the row uses the last prior bar.
5. **Backward-only regime join:** same shape, on regime labels.
6. Gas exact join: a tape row whose block is absent from the gas stream → `ValidationError` naming that
   block. A complete gas stream → no nulls.
7. `fee_growth_source` is `"exact"` at a snapshot block and `"stale_prior"` between snapshots; assert no
   row ever takes a value from a *later* snapshot.
8. `assert_no_lookahead` catches a deliberately-forward-joined table (build one by using a `forward`
   strategy in the test) and passes on the correct one. **Both directions of this test are required** —
   a guard that never fires is not a guard.
9. Price orientation: `price_pool` and `price_reference` agree within 5% on the fixture; and a
   deliberately decimals-swapped config produces a wildly different value (proving the test is sensitive
   to the bug it is guarding).
10. `EVENT_TAPE_SCHEMA` conformance under `validate_table(strict=False)`.
11. Pre-history rows: tape rows before the first reference bar have null `price_reference` and
    `regime == "unknown"`, and the row is **retained**.
12. Determinism: building the tape twice from the same inputs yields identical `content_hash` (T09's).
13. `tiny_dataset()` itself: a test asserting it contains each of the awkward cases its docstring claims
    (a tick crossing exists, a gas spike exists, etc.). Fixtures rot; this test stops it.
14. `Dataset.require("gas")` returns the table when present and raises `ValidationError` mentioning
    `undertow-data pull` when the field is `None`; a `None` field and an `empty_table` field are
    distinguishable.
15. `gas.eth_usd_price` is populated by the join and is null only where no prior reference bar exists.

## Acceptance criteria

- `uv run pytest tests/data/test_align.py` green; `mypy` clean.
- Every join in the module is annotated with a comment naming its direction and why.
- `assert_no_lookahead` is called inside `build_event_tape`, not merely available for callers to forget.
- No fetching, no disk I/O beyond what T09 provides.

## Handoff notes for your PR body

- The `Dataset` dataclass's final field list (T13/T14/T16 consume it).
- `tiny_dataset()`'s coverage list.
- Any stream whose absence you had to tolerate (e.g. if T05 could not fetch `collect`), and how the tape
  represents that.

## Process (mandatory)

Branch (stacked per `PLAN.md` §3.2 if dependencies are unmerged) → implement → tests green →
`code-reviewer` → fix → commit → push → PR → update `STATUS.md`.
