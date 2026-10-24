# T18 — De-duplicate subgraph reorg artifacts instead of aborting the pull

**Wave 6 · size S · depends on: T05 · blocks: the 0.05 % snapshot (S23)**
**Branch:** `bug-data-thegraph-reorg-duplicates`

## Why this task exists

The ETH/USDC 0.05 % pull (config `eth_usdc_500.toml`) aborted after ~1,856 of
1,858 swap pages with:

```
Validation error: ... blocks=16672773..16676868: duplicate (block_number, log_index)
key (16675001, 9) within a single GraphQL page; refusing to emit a table with
duplicate keys
```

The duplicate is a **real subgraph reorg artifact**, not a fetcher bug. Verified
directly against the pinned subgraph deployment:

```
{ swaps(first: 50, where: {pool: "0x88e6…5640", transaction_: {blockNumber: 16675001}}) }
→ 2 rows, both (blockNumber=16675001, logIndex=9), different ids
  (…<txhash>#4398002: amount0=-63072.49246, amount1=37.177979305
   …<txhash>#4398001: amount0=-63080.10805, amount1=37.177979305)
```

The two rows differ by 0.012 % on `amount0`; one is the orphaned reorg branch.
The tape's canonical key is `(block_number, log_index)` (`EVENT_TAPE_SCHEMA`), so
exactly one must be kept. Aborting the whole pull on one reorg row makes the
0.05 % dataset unbuildable.

The cached 1,856 pages contain **zero** within-page duplicates, so this is rare
(a handful of reorg rows across ~1.9 M swaps) — but fatal under the current
contract.

## Files you own

```
src/undertow/data/fetchers/thegraph.py     (_fetch_chunk within-page dedup + _dedupe across chunks)
tests/data/test_thegraph_fetcher.py        (or wherever the fetcher tests live — read first)
```

## What to build

1. **Within-page duplicates** (`_fetch_chunk`, the `if key in page_keys` branch):
   instead of raising `ValidationError`, keep one row and drop the other(s) with a
   **deterministic** rule — keep the row with the lexicographically largest `id`
   (the reorg re-mine has the higher suffix in the observed case; more
   importantly, `id` is the only stable ordering key the subgraph gives us), so
   the same page always resolves to the same row. Log a `WARNING` naming the
   key, the kept id, the dropped id(s), and the block range.
2. **Across-chunk duplicates** (`_dedupe`, the `if key in seen` branch): the same
   policy — keep the largest id, drop the rest, warn. Do not raise.
3. **Visible, not silent.** The dropped-duplicate count must appear in the pull
   report (a new additive `PullReport`/`StreamPullResult` field or an existing
   `notes`-style field — read the dataclasses first; additive/defaulted per
   PLAN §0.2) so a dataset can be audited for reorg rows. If adding a field is
   disproportionate, at minimum emit one summary `WARNING` per stream with the
   count.
4. **Do not change** the overlap/`seen_boundary` pagination logic, the short-page
   termination, or the indexer-lag check — those are correct and load-bearing.

## Tests

1. A page with two rows sharing `(block_number, log_index)` and different ids →
   exactly one survives, the larger id wins, a warning is logged, no exception.
2. The same key appearing in two different pages (across-chunk) → one survives,
   larger id wins.
3. A clean page → unchanged (no warning, same rows, same order).
4. The dropped-duplicate count is surfaced in the pull report / summary warning.
5. Existing pagination/overlap tests pass untouched.

Use the existing fetcher test fixtures (read the test file first to match the
fake-transport pattern).

## Acceptance criteria

- `uv run pytest tests/data/ -q -n 4 --cov-fail-under=0` green; full suite
  `uv run pytest -q -n 4` green with the 90 % coverage gate.
- After the fix, `uv run undertow-data pull --config configs/eth_usdc_500.toml`
  **resumes from the 1,856 cached swap pages** and completes (the 0.05 % cache is
  in `data/cache/eth_usdc_500`). Report the duplicate count and the final row
  counts per stream.

## Addendum (cross-stream collisions in `build_event_tape`)

The 0.05 % snapshot then fails at the tape union:

```
Validation error: build_event_tape: union of the four log streams: duplicate merge keys
... 16 distinct offending key(s), the first is (block_number=16675039, log_index=2)
```

The subgraph kept **both branches of an orphaned fork** for a ~2 100-block region
(block 16 675 039 ≈ 2023-02-21, verified: 7/12 sampled blocks in that region do
not match the canonical chain from `eth_getLogs`). A `swap` and a `mint`/`burn`
can therefore claim the same `(block_number, log_index)`. The RPC route cannot
replace them (the pinned provider's free tier caps `eth_getLogs` at 10 blocks).

The corrupted region is in the **train** split; the eval view seeds from the
2024-01-01 boundary snapshot after it, so eval results are unaffected. For the
train split the impact is ~0.017 % of events.

### What to build

In `_build_base_tape` (`src/undertow/data/transforms/align.py`), replace the
`_assert_unique_keys` abort for cross-stream collisions with a **deterministic
collapse**: keep one row per `(block_number, log_index)` using the documented
priority `mint > burn > swap > collect` (liquidity events over flow events; the
sampled canonical collisions were mints), drop the rest, and emit one `WARNING`
with the count and the first few keys. `_assert_unique_keys` keeps its abort for
any residual duplicate the collapse cannot resolve. Stamp `seq` after the
collapse so the tape is dense.

### Tests

1. A union with a cross-stream collision (swap + mint at one key) → one row
   survives, the mint wins, a warning is emitted, no exception.
2. A union with no collisions → unchanged (same rows, same order, no warning).
3. The `seq` column is dense (`0..N-1`) after a collapse.
4. Existing align/tape tests pass untouched.

## Out of scope

- Switching the 0.05 % pull to the RPC route (the free tier cannot; a paid plan
  or a targeted ~210-call repair of the corrupted region is a separate task).
- Any change to `EVENT_TAPE_SCHEMA` or the tape's key.
