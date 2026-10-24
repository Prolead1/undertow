# T17 — Liquidity-conservation check must skip when swap liquidity is unavailable

**Wave 6 · size S · depends on: T13 · blocks: the funding-stream snapshot rebuild (S19)**
**Branch:** `bug-data-liquidity-check-null`

## Why this task exists

`uv run undertow-data snapshot --config configs/eth_usdc_3000.toml` now crashes in
`check_liquidity_conservation`:

```
File "src/undertow/data/validation/checks.py", line 503, in check_liquidity_conservation
    _feed("swap", swaps)
  File ".../checks.py", line 494, in _feed
    _as_int(table.column("liquidity")[i]) if kind == "swap" else None
  File ".../checks.py", line 114, in _as_int
    return int(str(value))
ValueError: invalid literal for int() with base 10: 'None'
```

Root cause: the default route fetches log streams from the Uniswap-v3 **subgraph**,
whose `Swap` entity has **no `liquidity` field** (the comment in
`fetchers/queries/swaps.graphql` says exactly this; only the RPC `Swap` event log
carries post-swap active liquidity). Every swap row's `liquidity` is therefore null
(measured: 447,714 / 447,714 rows). The check was previously skipped because the
dataset had no `fee_growth` (`validation_report.md`: `seed_block=none`); now that the
dataset carries `fee_growth`, the check runs and crashes instead of skipping.

This blocks the S19 funding-stream snapshot rebuild and therefore S20.

## Files you own

```
src/undertow/data/validation/checks.py    (check_liquidity_conservation + a defensive guard)
tests/data/test_checks.py                 (new tests)
```

## What to build

At the top of `check_liquidity_conservation`, before `_feed("swap", swaps)`:

1. **Absent/all-null `liquidity`** — if the swaps table has no `liquidity` column, or
   every value is null, return an **info skip** via `_result(...)`:
   `"swap liquidity is not recorded on this route (the subgraph Swap entity has no
   liquidity field; only the RPC route supplies it) — liquidity conservation cannot be
   compared; skipped"`, metrics `{"swap_liquidity": "unavailable", "rows": <n>}`,
   `severity="info"`, `passed=True`.
2. **Partially-null** — if some swap rows have null `liquidity`, skip the comparison
   with an info result reporting `{"swap_liquidity": "partial", "null_rows": <n>,
   "rows": <n>}` and the same rationale. Do **not** silently compare only the non-null
   rows: a partial comparison can hide drift, and the caller should see the coverage.
3. **Defensive nulls elsewhere** — `_as_int` raises on null, and the mint/burn
   `liquidity_amount` path has the same shape. If any required value is null, return an
   info skip naming the column rather than crashing. `_as_int` itself keeps raising
   (its contract is "non-null"); handle nulls at the call site.
4. **Unchanged when data is present** — a swaps table with a fully-populated
   `liquidity` column must run exactly the existing algorithm and return the same
   result (the existing tests in `tests/data/test_checks.py` cover this; they must pass
   untouched).

## Tests (`tests/data/test_checks.py`)

1. All-null swap `liquidity` → info skip, `passed=True`, metric marks it unavailable,
   and **no exception**.
2. Missing `liquidity` column entirely → the same info skip.
3. Partial nulls → info skip reporting `null_rows`.
4. Populated `liquidity` → unchanged behaviour (assert the existing expected metric,
   e.g. the max-deviation path still runs).
5. Null mint/burn `liquidity_amount` → info skip naming the column, no crash.

Assert the returned `CheckResult` fields (`passed`, `severity`, `message`/metrics),
not just "does not raise".

## Acceptance criteria

- `uv run pytest tests/data/ -q -n 4 --cov-fail-under=0` green; full suite
  `uv run pytest -q -n 4` green with the 90 % coverage gate.
- After the fix: `uv run undertow-data snapshot --config configs/eth_usdc_3000.toml`
  **completes** and writes `data/datasets/eth_usdc_3000/manifest.json` (report the
  validation summary line). The dataset currently has no manifest — the failed rebuild
  left it half-written.
- No change to `check_liquidity_conservation`'s result when the data is complete.

## Out of scope

- Fetching swap liquidity via RPC (expensive; a future route decision).
- Any other check.
