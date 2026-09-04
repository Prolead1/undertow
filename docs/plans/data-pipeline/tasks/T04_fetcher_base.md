# T04 — Fetcher base layer (retry / chunking / raw cache)

**Wave 2 · size M · depends on: T00, T01 · blocks: T05, T06, T07, T08**
**Branch:** `feature-data-fetcher-base`

## Why this task exists

Roadmap §10.7 names "data-engineering burden eats the schedule" as the most common way this thesis dies,
and prescribes the mitigation: *cache to parquet so you never re-query*. Four fetchers are about to be
written in parallel; without a shared base they will each invent their own retry loop, their own
chunking, and their own cache — and three of the four will get backoff wrong. Build the plumbing once.

## Files you own

```
src/undertow/data/fetchers/__init__.py
src/undertow/data/fetchers/base.py
tests/data/test_fetcher_base.py
```

## What to build

Implement `CONTRACTS.md` §5.1: `FetchRequest`, `FetchResult`, the `Fetcher` protocol, `BaseHttpFetcher`.

### 1. Retry with backoff — a precise contract
- Retry **only** `RateLimitError` and `TransientNetworkError`. Never retry `PermanentFetchError`.
- Exponential backoff with **full jitter**: `sleep = random.uniform(0, min(cap, base * 2**attempt))`,
  `base = 0.5s`, `cap = 30s`, attempts from `EndpointConfig.max_retries`.
- Honor a `Retry-After` header when present — it overrides the computed backoff.
- Classify responses: `429` → `RateLimitError`; `5xx`, connection reset, read timeout →
  `TransientNetworkError`; other `4xx` → `PermanentFetchError`. A JSON-RPC body carrying an `error`
  object is a `PermanentFetchError` **unless** its message matches a known transient set (rate limit
  phrasing, "please try again") — keep that set as a module-level constant, not inline regexes.
- The sleep function must be injectable so tests do not actually sleep. A test suite that takes 30s
  because it really backed off is a finding.
- You may use `tenacity` or hand-roll it; if you hand-roll, keep it in one small well-named class.

### 2. Block-range chunking with adaptive shrink
- `chunks(start, end, size) -> Iterator[tuple[int, int]]` — **inclusive** bounds, no overlap, no gap,
  last chunk possibly short. Test the arithmetic hard: off-by-one here silently drops or duplicates
  events at every chunk boundary, and a duplicated `(block_number, log_index)` will blow up in T11 with
  a confusing error.
- Adaptive shrink: when a chunk fails with a "too many results" / "response size exceeded" style error,
  halve the chunk size and retry the *sub-ranges*, down to a floor of 1 block; if a single block still
  fails, raise `PermanentFetchError` with the block number. Providers phrase this error differently
  (`-32602`, "query returned more than 10000 results", "Log response size exceeded") — collect the
  patterns in a module constant and let T05/T06 extend it.

### 3. On-disk raw-response cache
- `cache_key(route, stream, params)` — a deterministic hash (sha256, hex, truncated to 32 chars) over a
  **canonically serialized** `params` mapping: sorted keys, JSON with `separators=(",", ":")`, ints as
  strings. Same logical request ⇒ same key across runs and machines. Nested dicts and lists must
  canonicalize too (the RPC fetcher passes topic lists).
- Secrets must never enter the key material or the filename. Strip URL credentials/API-key path
  segments before hashing; hash the *host*, not the full URL. Test this.
- Layout: `cache_dir/<route>/<stream>/<key>.json.gz`, gzip-compressed, written atomically
  (`.tmp` + `os.replace`) so an interrupted run never leaves a truncated cache entry that later reads
  as valid.
- Cache entries carry a small envelope: `{"params": ..., "fetched_at_utc": ..., "body": ...}` so a
  stale entry is diagnosable. Reads validate the envelope; a corrupt entry is treated as a miss and
  logged at WARNING, never crashed on.
- `FetchResult.from_cache` / `n_requests` must be accurate — T14's acceptance criterion is that a warm
  re-run prints `n_requests=0`, and that number comes from here.

### 4. `BaseHttpFetcher` shape

**Layering rule (this is why you can be built in parallel with T03):** this module must **not** import
`schemas.py` and must **not** validate tables. Your template method's core is
`fetch_rows(request) -> (rows, n_requests, from_cache, warnings)` returning raw row dicts; each concrete
fetcher (T05–T08) owns `_rows_to_table` and calls `validate_table` itself. See `CONTRACTS.md` §5.1's
layering comment. If you find yourself needing `empty_table` or `SCHEMA_REGISTRY`, stop — that is the
subclass's job, and reaching for it creates a same-wave dependency the plan deliberately removed.
- `requests.Session` with a connection pool sized from `max_concurrency`, a `User-Agent` identifying the
  project, and `request_timeout_s` applied to every call.
- Template method: `fetch_rows()` handles chunking + cache + retry and returns
  `(rows, n_requests, from_cache, warnings)`. It does **no** schema validation and imports no schemas.
  Subclasses implement `_fetch_chunk(request, start, end) -> list[dict]` (raw rows) and own
  `_rows_to_table(rows, stream) -> pa.Table` plus the `validate_table` call and the `FetchResult`
  wrapping. Keep the concurrency mechanism (thread pool over chunks) here so subclasses stay
  single-threaded and simple; bound it by `max_concurrency` and preserve deterministic output order
  regardless of completion order.
- Never log a URL containing a key. Log `route`, `stream`, block range, host, attempt number.

## Tests you must write

Use `responses` (already a dev dep) to mock HTTP. No network.

1. `chunks` exactness: `(0, 9, 3)` → `[(0,2),(3,5),(6,8),(9,9)]`; `(5,5,10)` → `[(5,5)]`;
   `size <= 0` raises; total coverage equals the input range with no duplicates (property-style loop
   over several ranges/sizes).
2. Retry: a 429 then 200 → one retry, correct result, injected sleep called once with a value in the
   jittered bound. `Retry-After: 2` → sleep called with ~2.
3. `max_retries` exhausted → `RateLimitError` propagates, and the message contains the block range.
4. A 404 → `PermanentFetchError` and **zero** retries (assert the mock was called exactly once).
5. Adaptive shrink: a mocked endpoint failing on ranges wider than 4 blocks but succeeding below →
   assert the full range is still covered exactly once, and that the chunk size shrank.
6. Cache: cold call hits the network and writes a file; second identical call makes **zero** requests,
   returns an equal table, and `from_cache is True`, `n_requests == 0`.
7. Cache key determinism: same params in a different dict insertion order ⇒ same key; a changed param
   ⇒ different key; two different `stream`s ⇒ different keys.
8. Secret hygiene: a `rpc_url` containing `https://x.example/v2/SECRETKEY` produces a cache key and
   filename containing neither `SECRETKEY` nor the full URL; and `caplog` contains no `SECRETKEY`.
9. Atomic write: simulate a crash mid-write (patch `gzip.open`/write to raise) and assert no `.json.gz`
   entry was left behind and the next read is a clean miss.
10. Corrupt cache entry (truncated gzip, or valid gzip with garbage JSON) ⇒ treated as a miss, WARNING
    logged, network path taken.
11. Empty range (`start > end` after clamping) → `fetch_rows` returns `([], 0, False, [...])` and makes
    no requests. (No `empty_table` here — that is the subclass's concern, per the layering rule.)

## Acceptance criteria

- `uv run pytest` green and **fast** (< 5s for this module — proof that sleeps are injected).
- No parsing of Uniswap-specific payloads in this file. This module knows about HTTP, ranges, and
  bytes; it does not know what a swap is.
- `mypy` clean.

## Handoff notes for your PR body

- The `_fetch_chunk` signature T05–T08 must implement, and `fetch_rows`'s return tuple, verbatim.
- Confirmation that this module imports nothing from `schemas.py` (the reviewer will check).
- The transient-error pattern constant's name and how to extend it.
- The cache layout and how to clear it (`--force` in T14 depends on this).

## Process (mandatory)

Branch → implement → tests green → `code-reviewer` → fix → commit → push → PR → update `STATUS.md`.
