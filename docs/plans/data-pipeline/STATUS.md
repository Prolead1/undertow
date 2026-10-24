# STATUS — data-pipeline-v1 handoff board

Update this file when you **start** and when you **finish**. Keep it terse. This is how the next agent
knows whether to branch off `main` or off your branch (see `PLAN.md` §3.2).

Legend: `todo` · `in-progress` · `in-review` (code-reviewer running / findings open) · `pr-open`
(approved + PR raised, awaiting human merge) · `merged` · `blocked`

| Task | State | Branch | PR | pytest | Reviewer | Notes |
|---|---|---|---|---|---|---|
| T00 scaffold | merged | chore-data-scaffold | #2 | 4 passed, 1 skipped | APPROVE (merged by human) | |
| T01 config/types | merged | feature-data-config | #3 | 53 passed, 1 skipped | APPROVE (x2) | rescued from crashed agent; ADR-003 raised |
| T02 fixedpoint | merged | feature-data-fixedpoint | #4 | 27 passed, 1 skipped | APPROVE (x2) | exact integer TickMath/price/liquidity ports |
| T03 schemas | merged | feature-data-schemas | #8 | 237 passed, 1 skipped | APPROVE | |
| T04 fetcher base | merged | feature-data-fetcher-base | #9 | 237 passed, 1 skipped | APPROVE | retry/chunk/cache; schema-free per §5.1 |
| T05 thegraph | merged | feature-data-thegraph | #12 | 20 tests | APPROVE | cursor pagination, 4 log streams |
| T06 rpc | merged | feature-data-rpc | #13 | 25 tests | APPROVE | eth_getLogs + eth_call + ABI decode |
| T07 gas | merged | feature-data-gas | #14 | 16 tests | APPROVE | per-block fees, EIP-1559 |
| T08 reference | merged | feature-data-reference | #15 | 18 tests | APPROVE | Binance 1m klines, gap-fill |
| T09 storage | merged | feature-data-storage | #16 | 26 tests (parquet) + 18 (manifest) | APPROVE | hive-partitioned parquet + deterministic manifest |
| T10 feegrowth | merged | feature-data-feegrowth | #10 | 41 tests | APPROVE | exact multi-tick apportionment |
| T11 align | merged | feature-data-event-tape | #22 | 21 tests | APPROVE | tiny_dataset() fixture; backward-only as-of joins |
| T12 regimes | merged | feature-data-regimes | #18 | 15 tests | APPROVE | rolling σ_rv + μ labeller |
| T13 validation | merged | feature-data-validation | #23 | 15 test_checks + 17 test_crosscheck | APPROVE | 11 checks + route-agreement + fee reconciliation |
| T14 cli | merged | feature-data-cli | #25 | 25 test_cli + 15 test_pipeline | APPROVE (merged by human) | pull/verify/snapshot/info subcommands; JSON + human output |
| T15 dune | merged | feature-data-dune-queries | #17 | 17 tests | APPROVE | 6 SQL queries + magnitude expectations |
| T16 public api | merged | feature-data-cli | #25 | included in T14 | APPROVE (merged by human) | `load_dataset()` entry point; surface widened for sim by ADR-008 |

## ADRs raised

- `adr/001-regime-drift-definition.md` — **accepted**: `μ` is total window log return, not per-step mean.
- `adr/003-window-blocks-optional.md` — **accepted**: `WindowConfig.start_block/end_block` widened to `BlockNumber | None` with XOR validation.
- `adr/005-gas-flat-tip-surcharge.md` — **accepted**: per-block priority-fee resolution replaced with `eth_feeHistory(no pctls)` + flat `tip_surcharge_pct` in `gas_cost_wei`. 2000× CU reduction; priority fees on Alchemy archive are undeliverable.
- `adr/006-gas-timestamps-feehistory-only.md` — **accepted**: `GAS_SCHEMA` is base-fee only; per-block `block_timestamp`/`eth_usd_price` removed (~122M CU saved; the event tape already carries timestamps free from the event logs).
- `adr/014-reference-symbol-selection.md` — **accepted**: `DataConfig.reference_symbol` (default `ETHUSDT`, optional `[reference] symbol`) and a keyword-only `build_event_tape(reference_symbol=...)` restrict the multi-symbol reference and regime streams to one series before the backward joins (the stream stores the primary plus its cross-check); the tape is derived at load time, so no re-snapshot is needed.
- `adr/016-validation-swap-sign-and-symbol.md` — **accepted**: `check_swap_sign_convention` downgraded critical→warning (26 real zero-amount swaps in 447 714, the T13 brief's anticipated case), and `check_reference_coverage`/`check_regime_labels_complete` filter to `DataConfig.reference_symbol`.
- `adr/017-reference-cross-fill.md` — **accepted**: `forward_fill_klines` cross-fills a secondary reference symbol from the densest one's effective close (median-ratio scaled, fresher-series-aware) instead of freezing it across a multi-month outage; the densest symbol is never cross-filled; cross-filled bars stay `is_gap_filled=True` and are counted in `GapStats.n_cross_filled`.
- `adr/018-backtester-boundary-pool-state.md` — **accepted**: `RpcFetcher.pool_state_at` enumerates the pool's `tickBitmap` at each walk-forward boundary (`undertow-data pool-state`), stores it as the optional `fee_growth` stream, and `FeeGrowthReplay.from_snapshot` seeds the backtester's exact fee engine so it can replay a window that starts mid-lifecycle (the state the tape alone cannot reconstruct).
- `adr/021-tick-based-crossing-selection.md` — **accepted**: `FeeGrowthTracker.apply_swap` now selects the initialized ticks to cross from the recorded `slot0` tick interval `(min_tick, max_tick]` rather than from the swap's price — a downward swap landing exactly on `boundary(T)` records `tick = T-1` while the price stays on the boundary, so a price rule missed the next upward crossing of `T` and corrupted active liquidity on the real tape (first at block 15,369,829; see ADR-021 for mainnet evidence).

## Post-v1 data fixes

| Task | State | Branch | PR | pytest | Reviewer | Notes |
|---|---|---|---|---|---|---|
| T17 liquidity-check null | merged | bug-data-liquidity-check-null | #93 | 1826 passed, 7 skipped, 94.20% cov | APPROVE | check skips on absent/all/partially-null swap liquidity; snapshot unblocked |
| T18 thegraph reorg duplicates (fetch) | merged | bug-data-thegraph-reorg-duplicates | #95 | 1831 passed, 7 skipped, 94.23% cov | APPROVE | within-page + across-chunk dedup by largest id; pull no longer aborts |
| T18 addendum (tape collapse) | pr-open | bug-data-tape-cross-stream-collapse | #96 | 1942 passed, 7 skipped, 94.42% cov | APPROVE (x4) | cross-stream `(block, log)` collisions collapse `mint > burn > swap > collect`; 0.05% tape builds |
| T19 fork-region RPC repair | pr-open | bug-data-tape-cross-stream-collapse | #96 | n/a (one-off data repair) | n/a | 1,422-block fork span replaced with canonical RPC rows; 0 residual mismatches; tape 7,112,636 → 7,112,899 |

## All tasks complete — data module v1 merged 🎉

T14 + T16 merged via PR #25. The sim-facing surface was later widened from T16's documented
list by **ADR-008** (`docs/decisions/008-data-sim-public-api-reconciliation.md`), enforced by
`tests/data/test_public_api.py`. T16's remaining brief items — the generated data dictionary and its
tests — shipped in PR #37 (`docs/data_dictionary.md`, `tests/data/test_data_dictionary.py`,
`src/undertow/data/dictionary.py`). T16 is now fully delivered.