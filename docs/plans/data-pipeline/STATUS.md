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
| T14 cli | pr-open | feature-data-cli | #25 | 25 test_cli + 15 test_pipeline | awaiting merge | pull/verify/snapshot/info subcommands; JSON + human output |
| T15 dune | merged | feature-data-dune-queries | #17 | 17 tests | APPROVE | 6 SQL queries + magnitude expectations |
| T16 public api | pr-open | feature-data-cli | #25 | included in T14 | awaiting merge | load_dataset() entry point; 16-name stable __all__ |

## ADRs raised

- `adr/001-regime-drift-definition.md` — **accepted**: `μ` is total window log return, not per-step mean.
- `adr/003-window-blocks-optional.md` — **accepted**: `WindowConfig.start_block/end_block` widened to `BlockNumber | None` with XOR validation.

## All tasks complete — data module v1 done 🎉

Last remaining action: merge PR #25 (T14 + T16) into `main`.