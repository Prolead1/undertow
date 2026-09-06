# Dune exploration queries (T15)

Exploration + independent-sanity-check layer for the `undertow.data` pipeline. The roadmap
(§10.1.2, §10.1.5) assigns Dune exactly one job: **explore and sanity-check magnitudes** — how much
volume, how many positions, what do the regimes/gas context look like. This directory is that job.
It is **NOT a data source**: the backtester and the RL inner loop never read from here. Only humans
and T13's `expected_counts` cross-reference do.

## Status

- **Dune access: UNAVAILABLE** in the sandbox that authored this task (no Dune account, no
  network). All six queries were authored and reviewed, but **none were executed on Dune Analytics**.
- Consequently `tests/data/fixtures/dune_expected_magnitudes.json` records **no magnitudes**
  (`"status": "unavailable"`, all fields `null`). This is deliberate: a fabricated expected
  magnitude is worse than none, because T13 would trust it. `expected_counts(...)` returns `None`
  (skip, never vacuous-pass) until real numbers exist.
- **Dashboard URL: none yet.** When a dashboard is published, pasted the URL here AND into the
  fixture's `dashboard_url` field. It must visualise 01 (pool overview), 02 (daily volume+fees),
  04 (active positions) and 06 (gas context) so the §10.6 "dashboard sanity-checking
  volume/positions" deliverable is satisfied.

## The queries (one line each)

| Query | Answers | Sanity-checks stream (CONTRACTS §4) |
|---|---|---|
| `01_pool_overview.sql` | On-chain token0/token1 + decimals, fee tier, spacing, creation block, lifetime event counts | config values (token ordering!), plus swap/mint/burn/collect counts |
| `02_daily_volume_fees.sql` | Daily volume + fee income, per token and USD, fee on the **input** side, both swap directions | `swap` |
| `03_swap_counts_by_month.sql` | Monthly event counts for swap/mint/burn/collect/flash | all five log streams — T13's row-count cross-reference |
| `04_active_positions.sql` | Monthly count of `(owner, tickLower, tickUpper)` with net positive liquidity | `mint` + `burn` |
| `05_tick_range_distribution.sql` | Histogram of mint range widths; off-grid mint share (**must be 0**) | `mint` (grid assumption) |
| `06_gas_price_context.sql` | Daily p50/p90/p99/mean base fee (gwei), gas used/limit | `gas` |

## The standing caveat (say it plainly, from §10.1.2)

**Dune is for aggregates; the fee-growth accumulators are for position-level fees.**
`02_daily_volume_fees.sql` derives a per-swap fee from the flow (input amount × fee tier). That is an
excellent *volume/activity* sanity check, but it is **not** what a position actually accrued — real
per-position fee accrual only comes from the `feeGrowth*X128` accumulators replayed by T10 and is the
only thing T13's fee↔Collect reconciliation should trust. Do not build any downstream consumer on this
directory.

## Re-running on the other pinned pool

Every query (except `06_gas_price_context.sql`, which is chain-wide — its `params` CTE is a no-op
kept for uniformity) is re-runnable on the second pinned pool by editing **one line**: the `0x…`
literal in the `WITH params AS (SELECT 0x… AS pool)` CTE.

- Primary pool (current default): `0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8` — USDC/WETH, 0.30%
  tier, tick spacing 60, decimals 6/18.
- Secondary pool: `0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640` — USDC/WETH, 0.05% tier, tick
  spacing 10, decimals 6/18.

Fee tier, tick spacing and token decimals are **derived from the chain** inside each query
(`PoolFactory_evt_PoolCreated` + `tokens.erc20`), so switching the address literal is sufficient —
no other constant needs to move. Do not invent per-config fee numbers in SQL; the chain is the source.

## Workflow when Dune access exists

1. Run `01` → `06` in order (each file parameterized as above). Dune's engine is Trino/DuckDB
   flavored: the queries use `date_trunc`, `approx_percentile`, and compare `contract_address`
   (a `varbinary`) with `0x…` literals — never quoted strings.
2. Check the expectations in each file's header comment hold (lifetime swaps ~1e6; off-grid mints
   exactly 0; both swap directions > 0 in `02`).
3. Transcribe the monthly counts from `03` into the fixture's `monthly` map, set `queried_at_utc`,
   `dashboard_url`, pick `tolerance_pct` (suggested **2.0**: Dune buckets by event timestamp while
   our fetchers bucket by `(block_number, log_index)`, so a few events legitimately float across
   month boundaries — 1–2% is defensible), flip `status` to `"available"`, and update the notes.
4. If anything deviates (e.g. off-grid mints ≠ 0, or swap counts off by 10×), **stop and investigate
   the pipeline** — that deviation is exactly what this directory exists to surface.