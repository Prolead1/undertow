# T08 — Reference price feed fetcher (Binance klines)

**Wave 3 · size M · depends on: T01, T03, T04 · blocks: T11, T12, T14**
**Branch:** `feature-data-reference`

## Why this task exists

Roadmap §10.1.7 is unambiguous about why this stream is not optional: the pool's own tick is *"stale and
noisy — it only moves when someone trades against it, and it is distorted by that very trade's
slippage."* Two downstream uses depend on an **external** price:

1. **IL / LVR** must be measured against the price arbitrageurs actually track, not the pool's echo of
   it. Use the pool tick and you understate the very cost the thesis is about.
2. **Regime labels** (T12, gap **G4**) must come from this feed so "regime" means a macro market
   condition rather than a local pool artifact.

§10.2.2 adds a third: this is the replay source for the simulator's price process.

## Files you own

```
src/undertow/data/fetchers/reference.py
tests/data/test_reference.py
tests/data/fixtures/binance_klines_api.json      (REST-shaped payload)
tests/data/fixtures/binance_klines_bulk.zip     (one small bulk-archive CSV zip)
```

## What to build

`ReferenceFetcher(BaseHttpFetcher)` per `CONTRACTS.md` §5.3, producing `REFERENCE_SCHEMA`.

**You own table-building.** Per `CONTRACTS.md` §5.1's layering rule, `BaseHttpFetcher` returns raw rows
and never touches schemas. So *your* module implements `_rows_to_table(rows, stream)`, calls
`schemas.validate_table` on the result, and wraps it in `FetchResult`. Use `empty_table(schema)` for an
empty range.

### 1. Source and symbols
- Primary: **Binance `ETHUSDT` 1m klines**. Secondary/cross-check: `ETHUSDC` 1m.
  (Pinned in `PLAN.md` §7: ETHUSDT has the deepest and longest 1m history; the USDC pair is thinner.)
- Two access paths — implement the bulk one, keep the API one for tails:
  - **Bulk archive** (`data.binance.vision` monthly/daily zipped CSVs) — the right tool for three years
    of minute bars; one HTTP GET per month instead of ~1600 paginated API calls.
  - **REST API** (`/api/v3/klines`, 1000-bar pages) — for the current partial month, and as a fallback.
  Both must produce identical `REFERENCE_SCHEMA` output; test that they agree on an overlapping day.
- Kline field order in both sources is: `open_time, open, high, low, close, volume, close_time,
  quote_volume, trades, taker_buy_base, taker_buy_quote, ignore`. Times are **milliseconds** since
  epoch — convert to UTC-aware microsecond timestamps; a naive or second-resolution timestamp is a bug.
- **Interval convention:** Binance klines are left-closed/right-open with `close_time = open_time +
  59_999ms`. Keep both columns and document it, because T11's as-of join uses `close_time <=
  block_timestamp` and an off-by-one-minute here shifts every price in the dataset by a bar.

### 2. Gaps: forward-fill and flag, never interpolate, never drop
Exchange outages and maintenance windows leave genuine holes in the minute series.
- Reindex to a **complete** 1-minute grid over the requested range.
- Missing bars: forward-fill `open/high/low/close` from the last observed close, set volumes and
  `trades` to `0`, and set `is_gap_filled = True`.
- If the leading edge is missing (no prior bar to fill from), leave nulls and warn — do not back-fill,
  which is look-ahead.
- Report gap statistics in `FetchResult.warnings`: count of filled bars and the longest run. T13's
  `check_reference_coverage` consumes this; a multi-hour hole is something the human must see, not
  something the pipeline papers over.

### 3. Cross-check helper
```python
def compare_symbols(a: pa.Table, b: pa.Table, *, tolerance_bps: float = 25.0) -> CheckResult
```
Aligns two symbols' closes on `open_time` and reports the fraction of bars whose relative difference
exceeds `tolerance_bps`. ETHUSDT vs ETHUSDC should track within a few basis points; a sustained
divergence means one series is wrong or stale — exactly the sanity check §10.1.7 implies.

Returns a **`CheckResult`**, imported from `undertow.data.types` (`CONTRACTS.md` §1). It lives in
`types.py` — owned by T01 in wave 1 — specifically so you can return one without waiting for T13's
`validation/` package in wave 5. Do not define a local variant and do not create anything under
`validation/`.

### 4. USD vs USDT
`REFERENCE_SCHEMA`'s prices are "ETH in USD-stable terms". USDT is not exactly USD. For this thesis the
depeg risk over the window is second-order, but it must be **stated, not assumed**: put the assumption in
the module docstring and in `docs/data_dictionary.md`'s entry (flag it to T16), and make the symbol
choice config-driven so it can be revisited.

## Tests you must write (no network)

1. Both fixtures → `REFERENCE_SCHEMA` conformance. Note per `CONTRACTS.md` §4.0/§4.6 that this
   stream has **no** block columns and is **not** pool-partitioned — do not add `block_number` or
   `log_index` sentinels.
2. Millisecond → UTC microsecond conversion is exact for a known bar; `close_time - open_time` is
   exactly 59,999,000 µs.
3. A price value round-trips exactly as `float64` for a realistic ETH price string.
4. **Gap filling:** a fixture with a 3-minute hole yields 3 rows with `is_gap_filled=True`, prices equal
   to the last prior close, volumes `0`, and the grid is complete with no duplicates.
5. A leading-edge gap produces nulls + a warning, **not** a back-fill. Assert no value was copied
   backwards in time (this is the look-ahead guard).
6. Gap statistics appear in `FetchResult.warnings` with the correct count and longest run.
7. Grid completeness: over a 24h fixture range the output has exactly 1440 rows.
8. Bulk-CSV path and REST path produce identical tables for an overlapping day (mock both).
9. `compare_symbols`: identical inputs → zero exceedances; a synthetic 100bp divergence on 10% of bars
   → the reported fraction is 0.10.
10. Range clamping: a request outside the fixture's coverage returns only covered bars **and warns**,
    rather than silently returning a short table with no signal.
11. Cache-hit makes zero HTTP calls.

## Acceptance criteria

- `uv run pytest tests/data/test_reference.py` green; `mypy` clean.
- No interpolation anywhere. Forward-fill-and-flag or fail.
- The USDT-vs-USD assumption is documented in the module docstring.

## Handoff notes for your PR body

- The exact archive URL pattern and symbol used, for reproducibility.
- The interval convention sentence (T11 quotes it when writing the as-of join).
- Gap statistics observed on a real month, if you ran a live pull manually.
- Confirmation that `compare_symbols` returns `types.CheckResult` unmodified, so T13 can wire it in.

## Process (mandatory)

Branch → implement → tests green → `code-reviewer` → fix → commit → push → PR → update `STATUS.md`.
