# T05 — The Graph fetcher (Route A)

**Wave 3 · size L · depends on: T02, T03, T04 · blocks: T11, T13, T14**
**Branch:** `feature-data-thegraph`

## Why this task exists

Roadmap §10.1.5's pragmatic plan: *"use The Graph to pull your curated training/eval datasets
quickly"*. This is the lowest-effort correct route and therefore the pipeline's default bulk source for
the four log streams. T06 (RPC) exists to *verify* it; T13 asserts they agree exactly.

## Files you own

```
src/undertow/data/fetchers/thegraph.py
src/undertow/data/fetchers/queries/*.graphql          (queries as files, not inline strings)
tests/data/test_thegraph.py
tests/data/fixtures/graph_swaps_page.json
tests/data/fixtures/graph_mints_page.json
tests/data/fixtures/graph_burns_page.json
tests/data/fixtures/graph_collects_page.json
```

## What to build

`TheGraphFetcher(BaseHttpFetcher)` per `CONTRACTS.md` §5.2, supporting streams
`{"swap", "mint", "burn", "collect"}`, returning tables conforming to `SCHEMA_REGISTRY[stream]`.

### Pitfalls that will bite you — handle each explicitly

1. **`skip` pagination is a trap.** The hosted service caps `skip` at 5000 and *silently* returns
   nothing beyond it. Use **cursor pagination on the ordering key**: order by `transaction.blockNumber`
   then `logIndex`, and page with `where: { ... , transaction_: {blockNumber_gte: $cursorBlock} }`
   plus a de-dup on `(block_number, log_index)` at the boundary. Never use `skip` for anything but
   intra-page tie-breaking. Add a test that a >1000-row range paginates correctly and returns every row
   exactly once.
2. **The endpoint URL has moved.** The `api.thegraph.com/subgraphs/name/uniswap/uniswap-v3` host quoted
   in the roadmap is the deprecated hosted service. Use the decentralized gateway
   (`https://gateway.thegraph.com/api/<key>/subgraphs/id/<subgraph-id>`) with the key from
   `${GRAPH_API_KEY}`, configured in `config.py` — not hardcoded. Record the exact subgraph ID you used
   in the module docstring and in your PR, because reproducibility depends on it. If the gateway is
   unavailable, note the fallback you used and flag it; do not silently switch sources.
3. **Decimal-adjusted vs raw amounts.** The Uniswap subgraph exposes `amount0`/`amount1` as
   **decimal-adjusted `BigDecimal`** (e.g. `-1234.567891`), *not* raw integer token units. Our schema
   requires **raw integer units** (`CONTRACTS.md` §4.1). You must therefore either (a) query the raw
   fields the subgraph exposes (`amount0`/`amount1` are decimals; `sqrtPriceX96`, `liquidity`, `tick`
   are `BigInt` and safe) and reconstruct raw amounts as `Decimal(amount) * 10**decimals`, validating
   the result is integral, or (b) prefer the `messari/uniswap-v3-ethereum` subgraph which carries raw
   amounts. **Whichever you choose, a `BigDecimal` → raw-int conversion that silently rounds is a
   correctness bug.** Validate integrality and raise `SchemaViolationError` on a non-integral result;
   test with a real fixture value that has full 18-decimal precision.
4. **Sign convention.** The subgraph's `Swap.amount0`/`amount1` follow the contract convention
   (positive = into the pool). Assert it on a real fixture row: for an ETH-sell swap, `amount1 > 0` and
   `amount0 < 0`. Do not flip signs. `Mint`/`Burn` amounts are unsigned.
5. **`Collect` entity availability.** Not every subgraph version indexes `Collect`. If the chosen
   subgraph lacks it, declare `collect` unsupported by this fetcher (`supported_streams()` reflects
   reality) and note in your PR that T06 (RPC) is the sole source for it — T13 and T11 must know. Do not
   fabricate an empty `collect` table and call it fetched.
6. **`transaction.gasPrice`/`gasUsed`** exposed by the subgraph are *transaction*-level, not the
   per-block base fee T07 needs. Do not use them for the gas stream.

### Implementation shape
- Queries live in `queries/*.graphql` and are loaded by name — reviewable, diffable, and quotable in
  the thesis appendix. Parameterize with GraphQL variables; never string-format a query.
- Implement `_fetch_chunk` and `_rows_to_table` from T04's template; let the base class own retry,
  chunking, caching and concurrency.
- A GraphQL `errors` array in a 200 response must raise — classify as `PermanentFetchError` unless the
  message is a known transient (rate limit / indexer lag), which is `TransientNetworkError`. Silent
  partial data is the failure mode to prevent.
- Detect **indexer lag**: the response's `_meta { block { number } }` tells you how far the subgraph has
  indexed. If `_meta.block.number < request.end_block`, raise `FetchError` naming both numbers — do not
  return a short table that looks complete. Add `_meta` to every query for exactly this reason.

## Fixtures you create

Small (≤ 50 rows each), hand-trimmed real responses, committed. Each must include at least: a
full-precision 18-decimal amount, a row where `amount0` and `amount1` have the signs described above,
and — in one file — an **empty** `data` array (the empty-page case). Record in a header comment which
real block range each came from.

On tick values: for USDC/WETH the pool tick is **positive** (~+190k to +210k) — see `CONTRACTS.md` §3.0.
Include a *position* tick range with negative bounds only if you actually find one in the real data; do
not manufacture one. What matters is that the values are real.

**Pagination fixtures may be synthetic envelopes** (`CONTRACTS.md` §10's "real values, synthetic
envelopes"): test 5 needs >1000 rows, which no ≤50-row fixture provides, so build that payload by
replicating real rows across synthetic pages with incrementing `(blockNumber, logIndex)`. Field *values*
stay real; only the paging wrapper is fabricated. Note it in the fixture header.

## Tests you must write (no network; `responses`-mocked)

1. Fixture → table conforms to `SCHEMA_REGISTRY[stream]` under `validate_table(strict=True)`, for all
   supported streams.
2. Field-by-field decode of one real swap row: `amount0`, `amount1`, `sqrt_price_x96`, `liquidity`,
   `tick`, `tx_hash` (lowercase), `block_number`, `log_index`, `block_timestamp` (UTC-aware).
3. Raw-amount reconstruction: a `BigDecimal` amount with 18 significant decimals converts to the exact
   expected integer; a value that would be non-integral raises `SchemaViolationError`.
4. Sign convention assertion on a real row (as described above).
5. Cursor pagination: mock three pages spanning >1000 rows → all rows present, none duplicated,
   `(block_number, log_index)` strictly increasing.
6. Boundary de-dup: two pages whose cursor overlaps by one row → that row appears exactly once.
7. GraphQL `errors` in a 200 → `PermanentFetchError`; a rate-limit-phrased error → `TransientNetworkError`
   and the base class retries.
8. Indexer lag: `_meta.block.number` below `end_block` → `FetchError` naming both numbers.
9. Empty range → `empty_table(SWAP_SCHEMA)`, zero rows, schema-valid, no exception.
10. Ordering: output is sorted by `(block_number, log_index)` even when the mocked pages arrive
    out of order (the base class parallelizes chunks).
11. Cache: second identical fetch makes zero HTTP calls (assert on the `responses` call count).
12. No API key appears in any log record (`caplog`) or in the cache filename.

## Acceptance criteria

- `uv run pytest tests/data/test_thegraph.py` green; `mypy` clean.
- The subgraph ID and endpoint form are documented in the module docstring.
- `supported_streams()` tells the truth about `collect`.
- Queries are files; no inline query strings.

## Handoff notes for your PR body

- The subgraph ID used, and whether `collect` is supported by it.
- The raw-amount conversion approach taken (option (a) or (b) above) and how integrality is enforced.
- Confirmed sign convention, quoting T02's orientation sentence.
- Rows-per-page limit and observed pagination behaviour.

## Process (mandatory)

Branch → implement → tests green → `code-reviewer` → fix → commit → push → PR → update `STATUS.md`.
