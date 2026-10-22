# S19 — Funding-rate stream (Binance USDT-perp)

**Wave 6 · size M · depends on: T08 (reference fetcher), T14 (schema/dictionary) · blocks: S20**
**Branch:** `feature-data-funding-stream`

## Why this task exists

The thesis's hedge leg (ADR-030 C3) needs a real cost of carry. Measured on
Binance ETHUSDT perp 2022–24: mean funding **+0.0067 %/8 h (+7.3 %/yr)**,
positive in **84 %** of periods (+13.0 %/yr in 2024). A short hedge *receives*
this, which roughly pays for the perp trading cost — the hedge is carry, not a
free lunch, and the sim must price it from the tape rather than assume zero.

`undertow.sim` never fetches. This task adds the stream to `undertow.data` and
the public API; S20 consumes it from the snapshot.

## Files you own

```
src/undertow/data/fetchers/funding.py     (new: FundingFetcher)
src/undertow/data/fetchers/__init__.py    (export)
src/undertow/data/schemas.py              (FUNDING_SCHEMA + units + SORT_KEYS)
src/undertow/data/dictionary.py           (STREAM_SOURCES + units)
src/undertow/data/pipeline.py             (_DEFAULT_STREAMS / _VALID_STREAMS / fetcher wiring)
src/undertow/data/config.py               ([endpoints] futures_base_url)
src/undertow/data/loader.py               (load_funding)
src/undertow/data/__init__.py             (public export)
configs/eth_usdc_3000.toml                (futures_base_url)
tests/data/test_funding.py                (new)
```

Follow the existing `ReferenceFetcher` / `GasFetcher` structure for the
`BaseHttpFetcher` pattern, retry/backoff, caching, and `FetchResult`.

## What to build

### 1. Schema — `FUNDING_SCHEMA`

| field | type | units |
|---|---|---|
| `funding_time` | `timestamp[us, UTC]` | Binance funding settlement time (every 8 h) |
| `symbol` | `large_string` | perp symbol, e.g. `ETHUSDT` |
| `funding_rate` | `float64` | fraction per 8 h (e.g. `0.00027858`), signed |
| `mark_price` | `float64` | perp mark price at settlement, USDT |
| `is_gap_filled` | `bool` | true when a missing 8 h settlement was forward-filled |

`SORT_KEYS["funding"] = ("symbol", "funding_time")`. Add units to the dictionary
and a `STREAM_SOURCES` entry ("Binance USDT-M futures fundingRate REST").

### 2. Fetcher — `FundingFetcher(BaseHttpFetcher)`

- Endpoint: `GET {futures_base_url}/fapi/v1/fundingRate?symbol={symbol}&startTime={ms}&endTime={ms}&limit=1000`.
- The response already carries `markPrice`, so **one call per page** supplies both
  `funding_rate` and `mark_price`. Paginate by advancing `startTime` to
  `last_fundingTime + 1`; stop when a page is short or `endTime` is reached.
- De-duplicate by `(symbol, funding_time)` (page boundaries overlap).
- Gap handling: if a scheduled 8 h settlement is missing between the window's
  first and last record, forward-fill `funding_rate`/`mark_price` and set
  `is_gap_filled = True` (mirror the reference feed's ADR-017 semantics — read it
  first).
- Symbol: reuse `DataConfig.reference_symbol` (the perp symbol string is
  identical to the spot symbol). Do not add a second symbol field.
- Window: clamp to the configured `[window]` bounds and request only that range.

### 3. Config

`[endpoints] futures_base_url = "https://fapi.binance.com"` in `DataConfig`
(default in code so existing configs keep loading), plus the value in
`configs/eth_usdc_3000.toml`. It is not a secret, so no `${ENV_VAR}`.

### 4. Pipeline + loader

- Add `"funding"` to `_DEFAULT_STREAMS` and `_VALID_STREAMS`, wire the fetcher in
  `_build_fetchers`, and let the generic `_pull_one_stream` path handle it.
- `load_funding(config) -> pl.DataFrame` in `loader.py`, sorted by
  `SORT_KEYS["funding"]`, exported from `undertow.data.__init__` (the sim's S20
  consumer imports it from the top-level public API — never an internal module).

### 5. Tests (`tests/data/test_funding.py`)

All fixture-based; no network (mark any live test `@pytest.mark.network`).

1. Schema conformance: fetched rows validate against `FUNDING_SCHEMA`; dtypes and
   the `is_gap_filled` default.
2. Pagination: >1000 records span two pages, no duplicates, strictly increasing
   `funding_time`.
3. Gap fill: a missing 8 h settlement is forward-filled and flagged; a complete
   series has `is_gap_filled` all false.
4. Window clamp: records outside `[window]` are dropped; an empty window yields
   an empty frame with the right schema (not an exception).
5. Loader: `load_funding` returns the documented sort order and is deterministic
   across two calls on the same cache.
6. Error paths: a non-2xx page raises the fetcher's typed error; a malformed
   `fundingRate` string is reported with the offending record.

## Acceptance criteria

- `uv run pytest tests/data/ -q -n 4` green; full suite green.
- `uv run undertow-data pull --config configs/eth_usdc_3000.toml --stream funding`
  succeeds against the live endpoint (run once, report the row count); the
  snapshot rebuild itself is left to the human if it is long.
- No `undertow.sim` import anywhere in this change.

## Out of scope

- Consuming the stream in the sim (S20), the hedge model (S20).
- Fetching perp klines separately — `markPrice` from the funding response is the
  pinned mark source; basis modelling is S20's problem.
