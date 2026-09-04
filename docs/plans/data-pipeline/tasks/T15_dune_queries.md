# T15 — Dune exploration queries + dashboard

**Wave 3 · size S · depends on: T01 · blocks: T13 (imports your `dune_expectations` module)**
**Branch:** `feature-data-dune-queries`

## Why this task exists

The Week-1 deliverable explicitly includes *"a Dune dashboard sanity-checking volume/positions"* (§10.6),
and §10.1.5's pragmatic plan assigns Dune the exploration role: *"use Dune to explore and sanity-check
(what's the volume, how many positions, what do regimes look like)"*. The point is not to build a
production data path — §10.1.2 is clear that Dune *"is not where the backtester's inner loop runs — it is
where you design and sanity-check it."*

Its concrete value to this pipeline: **independent expected magnitudes**. When T13 asks "is our swap
count for March 2023 plausible?", the answer comes from here. Without it, the pipeline can be
self-consistently wrong.

## Files you own

```
sql/dune/01_pool_overview.sql
sql/dune/02_daily_volume_fees.sql
sql/dune/03_swap_counts_by_month.sql
sql/dune/04_active_positions.sql
sql/dune/05_tick_range_distribution.sql
sql/dune/06_gas_price_context.sql
sql/dune/README.md
src/undertow/data/validation/dune_expectations.py    (library code, so T13 can import it)
tests/data/test_dune_expectations.py
tests/data/fixtures/dune_expected_magnitudes.json
```

Note you own `validation/dune_expectations.py` but **not** the rest of `validation/` (T13 owns
`checks.py`, `crosscheck.py` and `__init__.py`). If `validation/__init__.py` does not exist yet because
T13 has not run, create it empty and say so in your PR so T13 knows not to be surprised.

This task writes **SQL and one small fixture**, not pipeline code. That is intentional and in scope —
it is a thesis deliverable.

## What to build

### 1. The six queries
Against `uniswap_v3_ethereum.Pair_evt_Swap` / `_Mint` / `_Burn` / `_Collect` and Dune's `ethereum.blocks`,
for the two pinned pools (`PLAN.md` §7). Each file starts with a comment block: what it answers, which
`CONTRACTS.md` stream it sanity-checks, and the expected order of magnitude.

1. **`01_pool_overview.sql`** — pool address, fee tier, token0/token1 + decimals, creation block and
   date, lifetime swap/mint/burn/collect counts. This is the query that confirms our config's token
   ordering and decimals are right.
2. **`02_daily_volume_fees.sql`** — daily volume **and** fee income. Roadmap §10.1.2 shows a version of
   this query and immediately flags its own bug: summing `abs(amount1)` only captures fees paid in WETH,
   *"swaps paid in USDC are missed unless you also sum `abs(amount0)`"*, and the fee is taken on the
   **input** amount. **Write the corrected query**: split by swap direction (sign of `amount0`), apply
   the fee to the input side, and report fees in both tokens plus a USD total. Put a comment in the file
   explaining what the naive version gets wrong — this correction is itself a thesis-worthy detail.
   Also carry §10.1.2's second caveat in the comment: this is *volume/activity* truth, **not**
   position-level fee truth, which only the `feeGrowth` accumulators give (T10).
3. **`03_swap_counts_by_month.sql`** — monthly event counts per stream. This is T13's row-count
   cross-reference.
4. **`04_active_positions.sql`** — count of distinct `(owner, tickLower, tickUpper)` with net positive
   liquidity, monthly. Note in the comment that `owner` is overwhelmingly the NFT position manager
   (`0xC364…FE88`), so this counts *positions*, not *users* — the same caveat `PLAN.md` §7 records as
   out-of-scope.
5. **`05_tick_range_distribution.sql`** — histogram of mint range widths in ticks, and the share of
   mints whose ticks are off the fee tier's spacing grid (should be **zero**; if not, our grid assumption
   is wrong). This directly validates T13's `check_ticks_on_spacing_grid`.
6. **`06_gas_price_context.sql`** — daily median/p90 base fee over the window, so T07's numbers have an
   independent reference and the gas spikes §10.2.3 cares about are visible.

Style: CTEs over nested subqueries, explicit column aliases, parameterized pool address at the top via a
single CTE (`WITH params AS (SELECT 0x… AS pool)`) so the queries are re-runnable on the second pool by
editing one line. Dune's engine is Trino/DuckDB-flavored — use `date_trunc`, `approx_percentile`, and
mind that `contract_address` is a `varbinary`, so compare with `0x…` literals, not quoted strings.

### 2. The magnitudes fixture — the actual handoff artifact
Run the queries on Dune, then record the results in
`tests/data/fixtures/dune_expected_magnitudes.json`:

```json
{
  "source": "Dune Analytics",
  "queried_at_utc": "2025-...",
  "dashboard_url": "https://dune.com/...",
  "pool": "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",
  "monthly": {
    "2023-03": {"swaps": 123456, "mints": 4567, "burns": 4321, "collects": 3210},
    "...": {}
  },
  "tolerance_pct": 2.0,
  "notes": "…"
}
```

`tolerance_pct` exists because Dune's decoded tables and our fetchers may differ marginally at month
boundaries (block vs timestamp bucketing) — pick a tolerance you can defend and write down *why* in
`notes`. If you cannot access Dune (no account), say so explicitly in the PR, populate the fixture with
`null`s and a `"status": "unavailable"` flag, and make the test skip rather than pass vacuously. **Do not
invent numbers** — a fabricated expected magnitude is worse than none, because T13 will trust it.

### 3. `sql/dune/README.md`
The dashboard URL, one line per query, how to re-run them on a different pool, and the §10.1.2 caveat
(Dune for aggregates, accumulators for position-level fees) stated plainly so nobody later mistakes this
directory for a data source.

## Tests you must write

`tests/data/test_dune_expectations.py` is thin but real:

1. The fixture parses, has the required keys, and every monthly entry has all four counts (or is
   explicitly `null` under `"status": "unavailable"`).
2. Month keys are `YYYY-MM`, sorted, contiguous, and fall inside the pinned window.
3. Counts are positive integers where present; `tolerance_pct` is in `(0, 10]`.
4. `expected_counts(month, stream) -> tuple[int, float] | None` — **in
   `src/undertow/data/validation/dune_expectations.py`**, returning `(expected_count, tolerance_pct)` or
   `None` when unknown/unavailable. It lives in `src/`, not in a test module, because T13 imports it as
   library code. Tested for a present month, an absent month, and the `unavailable` case.
5. Every `.sql` file in `sql/dune/` is non-empty, starts with a `--` comment block, and mentions its pool
   parameter CTE (a cheap guard against a half-finished query landing).

## Acceptance criteria

- All six queries committed and *actually run* on Dune (or the unavailability is explicit).
- Dashboard URL recorded in `sql/dune/README.md` and the fixture.
- `02_daily_volume_fees.sql` correctly handles **both** swap directions and applies the fee to the input
  side — the reviewer should be able to read the comment and see the roadmap's naive version being fixed.
- `uv run pytest tests/data/test_dune_expectations.py` green.

## Handoff notes for your PR body

- The dashboard URL.
- Observed magnitudes: lifetime swap count, a typical daily volume, and the count of off-grid mints
  (expected: zero — if not, flag it loudly, because it breaks a core assumption).
- Whether Dune access was available.
- `expected_counts`'s exact import path and signature, for T13.
- Whether you had to create `validation/__init__.py`.

## Process (mandatory)

Branch → write queries + fixture + tests → `uv run pytest` → `code-reviewer` (it will review SQL and the
test; that is fine and expected) → fix → commit → push → PR → update `STATUS.md`.
