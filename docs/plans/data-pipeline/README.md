# Plan: `undertow.data` — Uniswap V3 data pipeline

> **Source of truth.** This directory — `docs/plans/data-pipeline/` — is the canonical plan. ADRs
> mentioned below as `adr/*.md` live in [`docs/decisions/`](../decisions/).

Multi-agent execution plan for the thesis's Week-1 critical-path deliverable.

## Read in this order

| File | What it is | Who reads it |
|---|---|---|
| **`PLAN.md`** | scope, architecture, task graph, dependencies, conventions, DoD, launch order | everyone, first |
| **`CONTRACTS.md`** | **frozen** cross-task interfaces: schemas, signatures, units, sign conventions | everyone, second |
| `STATUS.md` | live handoff board — task states, branches, PRs, blockers | everyone, on start & finish |
| `tasks/T*.md` | 17 self-contained task briefs | the agent assigned that task |
| `adr/*.md` | decisions that deviate from the source requirement, with justification | anyone touching the affected area |

## The requirement this implements

`~/Documents/fyp/lesson_plan/10_data_evaluation_roadmap.md` — §10.1 (four data routes), §10.1.6 (the
per-position stream inventory), §10.1.7 (external reference feed), §10.2.4 (fee-growth accumulators),
§10.3 (regime labels), §10.6 Week 1–2, §10.7 (risks). Gap context: `09_research_gap_problem_statement.md`
§9.2 (G1–G4).

## Task index

| Wave | Tasks | Theme |
|---|---|---|
| 0 | T00 | scaffold |
| 1 | T01, T02 | config/types · fixed-point & tick math |
| 2 | T03, T04, T10 | schemas · fetcher base · fee-growth engine |
| 3 | T05, T06, T07, T08, T09, T12, T15 | the four routes · gas · reference feed · storage · regimes · Dune |
| 4 | T11 | alignment → event tape |
| 5 | T13 | validation & cross-check |
| 6 | T14 | CLI + snapshot |
| 7 | T16 | public API + data dictionary |

Full dependency table: `PLAN.md` §3.1. Launch commands: `PLAN.md` §9.

## Non-negotiables (from `undertow/AGENTS.md`)

Branch first · tests with real assertions · `code-reviewer` before every commit · PR against `main`,
never self-merged · `undertow.data` and `undertow.sim` stay separate · `.pi/` never committed.

## The three things most likely to go wrong

1. **Float precision on Q128 accumulators.** `2**128 ≈ 3.4e38`; float64 has 53 mantissa bits. Big
   integers stay `int` in Python and strings in Parquet. See `PLAN.md` §4 and `tasks/T02`, `tasks/T10`.
2. **Look-ahead bias.** Every as-of join is backward-only; every rolling window is backward-looking.
   `tasks/T11`'s `assert_no_lookahead` and `tasks/T12`'s truncated-series test are the guards.
3. **Price orientation.** USDC is token0 (6 dp), WETH is token1 (18 dp); prices are USDC per WETH
   (~2000–4000). A decimals slip inverts or 1e12-scales every IL number in the thesis. `tasks/T02`
   pins the sentence; `tasks/T11` tests it end to end.
