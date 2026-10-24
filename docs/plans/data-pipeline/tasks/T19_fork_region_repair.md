# T19 — Targeted RPC repair of the 0.05 % orphaned-fork region

**Wave 6 · size S · depends on: T18 · unblocks: trustworthy train-split data (S23)**
**Branch:** `bug-data-tape-cross-stream-collapse` (folded into PR #96 with the T18 addendum)

## Why this task exists

The T18 addendum stopped `build_event_tape` from aborting on the 16 cross-stream
collision keys left by an orphaned subgraph fork, but the **underlying rows were
still non-canonical**. A canonical comparison over the enclosing 2,305-block
envelope (`eth_getLogs`, 10-block chunks) found **1,422 blocks** whose
swap/mint/burn `(log_index, tx_hash)` multiset does not match the chain
(16,675,000..16,677,195 ≈ 2023-02-21). The collapse therefore kept 4 rows with no
canonical log at all, and the region's other rows could still be orphaned. The
region is in the **train** split; eval (2024) seeds from the 2024-01-01 boundary
snapshot after it and is unaffected.

## Executed procedure (2026-10-24, dataset `d5dbfefc…`)

1. **Probe** — stride-64 single-block `eth_getLogs` over 16,670,000..16,680,000:
   24/157 sampled blocks mismatched, span 16,675,184..16,677,104.
2. **Fetch** — `RpcFetcher` (route RPC, 10-block chunks, `min_request_interval_s=0.75`,
   `max_retries=10`) for `swap`/`mint`/`burn` over 16,674,992..16,677,296
   (2,305 blocks → 231 chunks/stream; 84 uncached chunks after the throttled first
   attempt; the rest reused the RPC cache under `data/cache/eth_usdc_500/rpc`).
3. **Locate** — canonical-vs-subgraph comparison on `(log_index, event_type,
   tx_hash)` inside the envelope: exact fork span **16,675,000..16,677,195**
   (1,422 blocks).
4. **Replace** — drop subgraph rows in 16,674,992..16,677,203 (8-block margin),
   append the canonical RPC rows, rewrite each stream with `write_stream`
   (full-bucket rewrite: 36 s swap, ~1 s/mint/burn).
5. **Re-prove + publish** — post-repair comparison found **0 residual mismatching
   blocks**; rebuild the tape, run all checks, rewrite `manifest.json` (+ per-stream
   `T19 fork repair` notes) and `validation_report.md`.

## Results

| Stream | Before | After | Content hash |
|---|---|---|---|
| swap | 6,862,355 | 6,862,602 (+247 canonical rows) | `fdf0da5b5a82…` |
| mint | 119,781 | 119,781 | `0c866ccd9a2f…` |
| burn | 130,516 | 130,516 | `441d482d6b50…` |

- Event tape 7,112,636 → **7,112,899 rows**; the T18 cross-stream collapse
  **no longer fires** (zero collisions remain).
- All critical checks pass; the only warning failures are the known systemic
  ones (128 real zero-leg swaps; 1-minute reference-bar edge starvation).
- `check_liquidity_conservation` now sees partial liquidity (the RPC route fills
  it for the repaired span only) and still returns an info skip.

## Caveats / follow-ups

- The repaired span carries `liquidity` (RPC) while the rest of the window is null
  (subgraph). Filling the whole window would need ~200 k further `eth_getLogs`
  calls — not worth the CU; the check skips on partial coverage.
- The **TheGraph cache still contains the fork pages**. `pull` skips streams whose
  manifest `row_count > 0`, so a normal `snapshot` is safe; a `pull --force` would
  reintroduce the fork rows. The canonical RPC chunks are cached, so re-running the
  repair is cheap.
- `collect` remains absent (the pinned subgraph has zero `collects` globally). The
  sim never reads collect rows, so only `reconcile_fees_against_collect` is lost.
  T05's own contingency says `TheGraphFetcher.supported_streams()` should be
  narrowed if the entity stays unindexed — a separate task if the empty stream is
  to be represented honestly.

## Acceptance

- [x] 0 residual non-canonical blocks in the repaired span
- [x] tape rebuild with **no** T18 collapse warning
- [x] all critical validation checks pass; manifest + report carry the repair provenance

The one-off runner (probe → fetch → merge → re-verify) was executed from
`/tmp/repair_fork_region.py` and is deliberately not committed: it is un-tested
one-shot maintenance code, and this brief records every parameter needed to
reproduce it.
