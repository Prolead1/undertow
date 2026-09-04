# T06 — RPC fetcher (Route C) + ABI decoders

**Wave 3 · size L · depends on: T02, T03, T04 · blocks: T10 (data), T13, T14**
**Branch:** `feature-data-rpc`

## Why this task exists

Roadmap §10.1.3: direct RPC is *"the only route for the exact `feeGrowthOutside` values you need for
position-level fee accounting"*, and §10.1.5 assigns it the verification role — *"use direct RPC to
verify the exact feeGrowth numbers your backtester's fee accounting depends on."* Two jobs, then:
(1) an independent second opinion on the four log streams so T13 can prove the subgraph is not lying,
and (2) the sole source of the fee-growth snapshots T10 reconciles against.

The roadmap's analogy is the one to keep in mind: The Graph is the bookkeeper's summary; this is reading
the ledger by hand. When a number has to survive a skeptical examiner, it comes from here.

## Files you own

```
src/undertow/data/fetchers/abi.py                    (topic0 constants + pure decoders)
src/undertow/data/fetchers/rpc.py                    (transport + orchestration)
tests/data/test_abi.py
tests/data/test_rpc.py
tests/data/fixtures/rpc_logs_swap.json
tests/data/fixtures/rpc_logs_mint.json
tests/data/fixtures/rpc_logs_burn.json
tests/data/fixtures/rpc_logs_collect.json
tests/data/fixtures/rpc_ticks_call.json
tests/data/fixtures/collect_reconciliation/*.json     (raw captures; T13 adds cases.json)
```

**The `collect_reconciliation/` captures are a deliverable, not an afterthought.** T13's acceptance
criterion — reconstructed fees matching real on-chain `Collect` amounts, the thesis's §10.2.4 claim —
needs real data offline, and you are the only task that can capture it. Provide **three** real position
lifecycles from the pinned pool: for each, the `Mint`, the swaps in between, the `Burn`/`Collect`, and
`ticks()`/`slot0()`/`feeGrowthGlobal` snapshots at the mint and collect blocks. At least one lifecycle
must have the price **exit the range**. Record each lifecycle's block range and tx hashes in a header
comment. Keep them small — a lifecycle spanning a few hundred blocks with a handful of swaps is ideal.

Note the split: `abi.py` is pure functions over hex strings (no I/O), `rpc.py` does transport. Do not
merge them — the reviewer rejects god modules, and `abi.py`'s decoders must be unit-testable without any
HTTP.

## What to build

### 1. `abi.py` — topic0 constants and decoders
Event signatures are given verbatim in roadmap §10.1.3. Compute `topic0 = keccak256(signature)` and
**hardcode the resulting constants with the signature string in a comment**, then add a test that
recomputes them (via `eth_utils.keccak` or a vendored keccak) and asserts equality — so the constants are
both stable and verified. The `Swap` topic0 is given in the roadmap as
`0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67`; use it as a known-answer test.

Decoders — one pure function per event, taking a raw log dict (`topics: list[str]`, `data: str`,
`blockNumber`, `logIndex`, `transactionHash`) and returning a row dict matching the schema:

- **`Swap`**: `sender` and `recipient` are `indexed` → they are in `topics[1]`, `topics[2]`;
  `amount0`, `amount1`, `sqrtPriceX96`, `liquidity`, `tick` are in `data`, in that order.
  `amount0`/`amount1` are `int256` → **two's-complement sign extension** on 32-byte words.
  `tick` is `int24` → sign-extend from 24 bits, and note it is right-aligned in a 32-byte word.
- **`Mint`**: `owner`, `tickLower`, `tickUpper` are indexed (`topics[1..3]`); `sender`, `amount`,
  `amount0`, `amount1` are in `data`. Careful: `sender` is **non-indexed** on `Mint` despite appearing
  first in the signature — the indexed/non-indexed split does not follow declaration order in `data`.
- **`Burn`**: `owner`, `tickLower`, `tickUpper` indexed; `amount`, `amount0`, `amount1` in `data`.
- **`Collect`**: `owner`, `tickLower`, `tickUpper` indexed; `recipient`, `amount0`, `amount1` in `data`.
- **`Flash`** (optional but in scope per `EventType.FLASH`): contributes fee growth without liquidity
  change — decode it so T10's replay does not miss that fee source. If you skip it, say so in the PR and
  note the effect on T10's reconciliation.

The indexed `int24` tick values are 32-byte topics carrying a sign-extended `int24` — decode via
two's complement over the full word, then range-check against `MIN_TICK`/`MAX_TICK` and raise
`SchemaViolationError` if outside. This range check is your best defense against a silent decode bug.

### 2. `rpc.py` — `RpcFetcher(BaseHttpFetcher)`
Per `CONTRACTS.md` §5.2. Streams: the four logs plus `flash` plus `fee_growth`.

**You own table-building.** Per `CONTRACTS.md` §5.1's layering rule, `BaseHttpFetcher` returns raw rows
and never touches schemas. So *your* module implements `_rows_to_table(rows, stream)`, calls
`schemas.validate_table` on the result, and wraps it in `FetchResult`. Use `empty_table(schema)` for an
empty range.

- **Logs**: `eth_getLogs` with `address` + `topics: [topic0]`, chunked by the base class. Providers cap
  ranges and response sizes differently — feed your provider's error phrasing into T04's adaptive-shrink
  pattern constant.
- **JSON-RPC batching**: batch requests where the provider supports it, but keep a per-request fallback,
  because batch support and batch size limits vary. A batch response can return results **out of order**
  and can contain per-item `error` objects — handle both (match on `id`, never on position).
- **`fee_growth_at(pool, block, ticks)`**: `eth_call` at a historical block (`blockNumber` as the third
  param — **this requires an archive node**; a non-archive node returns "missing trie node" or
  "state not available", which you must classify as `PermanentFetchError` with a message that says
  *archive node required*, because that error will otherwise waste hours). Calls needed per block:
  - `slot0()` → `sqrtPriceX96`, `tick` (`current_tick`)
  - `liquidity()` → `current_liquidity`
  - `feeGrowthGlobal0X128()`, `feeGrowthGlobal1X128()`
  - `ticks(int24)` for each requested tick → `liquidityGross`, `liquidityNet`,
    `feeGrowthOutside0X128`, `feeGrowthOutside1X128`, …, `initialized`
  Hand-encode these calldata selectors and decode the returns in `abi.py` (add function selectors
  alongside the event topics). Emit one `FEE_GROWTH_SCHEMA` row per tick plus a global row keyed by
  `GLOBAL_TICK_SENTINEL`, all with `source="rpc_call"`.
- **Reorg / finality guard**: refuse to fetch beyond `latest - N` confirmations (N from config, default
  64) and raise `ReorgDetectedError` if a cached block hash no longer matches on re-fetch. Store the
  block hash alongside cached log responses so this check is possible at all.
- **Block timestamps**: `eth_getLogs` returns no timestamp. Resolve `block_number → timestamp` via a
  batched `eth_getBlockByNumber` and cache it. **Keep this resolver private to your module** (a leading
  underscore) and do not coordinate with T07 about it: T07 owns the *public* timestamp↔block index
  (`block_for_timestamp` / `timestamp_for_block`), you own a private one for annotating logs. You are
  both running in wave 3 and cannot negotiate mid-flight, so the plan accepts this small duplication
  deliberately — it is cheaper than two agents blocking on each other. T14 may later consolidate.

## Tests you must write (no network)

1. Known-answer: recomputed `keccak256` of each event signature equals the hardcoded topic0, including
   the roadmap's `Swap` value.
2. Decode a real `Swap` log fixture field-by-field against hand-computed expected values, including a
   **negative** `amount0` (sign extension) and a **negative** `tick`.
3. Decode `Mint` — assert the non-indexed `sender` is read from `data`, not from a topic. This is the
   trap; test it explicitly.
4. Decode `Burn` and `Collect`, each with a negative tick range (e.g. `[-201000, -199800]`).
5. A tick decoding to outside `MIN_TICK..MAX_TICK` raises `SchemaViolationError`.
6. A log whose `topics[0]` does not match the expected event raises rather than silently decoding.
7. A log with a truncated `data` field raises `SchemaViolationError` (not `IndexError`).
8. `fee_growth_at` against `rpc_ticks_call.json`: one global row (sentinel tick) + one row per requested
   tick, `source="rpc_call"`, all Q128 values decoded as exact integers, `liquidity_net` correctly
   **signed**.
9. Archive-node absence: a mocked `"missing trie node"` error → `PermanentFetchError` whose message
   contains "archive".
10. Batch handling: a batch response returned in shuffled `id` order is reassembled correctly; a batch
    containing one error item raises for that item only, with the failing params named.
11. Finality guard: requesting `end_block > latest - 64` raises; a cached entry whose block hash changed
    raises `ReorgDetectedError`.
12. Schema conformance for all streams, empty-range behaviour, and cache-hit-makes-zero-calls (same as
    T05's).
13. No RPC URL or key in `caplog`.

## Acceptance criteria

- `uv run pytest tests/data/test_abi.py tests/data/test_rpc.py` green; `mypy` clean.
- `abi.py` imports nothing that does I/O.
- Every decoder's docstring quotes the event signature it decodes.
- Tables returned for `swap`/`mint`/`burn`/`collect` are **field-identical** to T05's for the same
  range — you cannot test this here without T05's fixtures, so leave it to T13, but design for it:
  same units, same signs, same lowercase-hex conventions.

## Handoff notes for your PR body

- Whether `Flash` is decoded (T10 needs to know).
- The archive-provider you assumed and the confirmations default.
- Your private block-timestamp resolver's name (so T14 can consolidate later if it wants).
- Any provider-specific error phrasing you added to T04's transient/shrink constants.

## Process (mandatory)

Branch → implement → tests green → `code-reviewer` → fix → commit → push → PR → update `STATUS.md`.
