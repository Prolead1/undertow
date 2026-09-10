# Plan: `undertow.sim` — simulator, backtester & RL environment

> **Source of truth.** This directory — `docs/plans/sim/` — is the canonical plan. ADRs live in
> [`docs/decisions/`](../decisions/) (numbering shared with the data plan).

Multi-agent execution plan for the thesis's Week-2-to-6 deliverables: the correct backtester, the fast
training simulator, the friction-realistic RL environment, the baseline ladder, and the evaluation
harness that produces the RQ1–RQ3 artifacts.

## Read in this order

| File | What it is | Who reads it |
|---|---|---|
| **`PLAN.md`** | scope, architecture, task graph, dependencies, conventions, DoD, pinned decisions, launch order | everyone, first |
| **`CONTRACTS.md`** | **frozen** cross-task interfaces: dataclasses, protocols, signatures, units, sign conventions | everyone, second |
| `STATUS.md` | live handoff board — task states, branches, PRs, blockers | everyone, on start & finish |
| `tasks/S*.md` | 17 self-contained task briefs | the agent assigned that task |

## The requirement this implements

`~/Documents/fyp/lesson_plan/10_data_evaluation_roadmap.md` — §10.2 (the two artifacts + four
components), §10.2.3 (backtest loop), §10.3 (evaluation methodology), §10.4 (metrics), §10.5
(baseline ladder), §10.6 Weeks 2–6. Gap/RQ context: `09_research_gap_problem_statement.md` §9.2–§9.3.
Mechanics: `05_concentrated_liquidity_lp_problem.md` §5.3–§5.8. Env design:
`07_reinforcement_learning.md` §7.8–§7.9. Anchor setup + baselines: `08_rl_market_making_sota.md`
§8.4–§8.6, §8.10.

## Task index

| Wave | Tasks | Theme |
|---|---|---|
| 0 | S00 | scaffold + RL-library ADR |
| 1 | S01, S02 | types/config · data public-API extension |
| 2 | S03, S04, S05, S06 | market view/split · position math · metrics · price processes |
| 3 | S07, S08, S09, S10 | pool engine · frictions · reward · baselines |
| 4 | S11, S12 | backtester · Gymnasium env |
| 5 | S13, S14 | training harness · parity/look-ahead validation |
| 6 | S15 | evaluation runner (RQ1 ablation · RQ2 regime matrix · RQ3 gap) |
| 7 | S16 | CLI + public API + docs |

Full dependency table: `PLAN.md` §3.1. Launch commands: `PLAN.md` §9.

## Non-negotiables (from `undertow/AGENTS.md`)

Branch first · tests with real assertions · `code-reviewer` before every commit · PR against `main`,
never self-merged · `undertow.data` and `undertow.sim` stay separate · `.pi/` never committed.

## The three things most likely to go wrong

1. **Simulator/backtester conflation.** They share the position math and frictions but never a loop:
   the simulator is fast float64 with injectable models; the backtester is exact and replay-only. A
   simulator bug must not be able to "improve" a backtest — S14's parity check is the tripwire.
2. **Look-ahead bias.** `MarketView` (S03) is the only door to historical data and enforces the
   train/eval wall; every rolling feature is backward-looking; S14 attacks the composed system.
3. **Silent friction zeroing.** The whole thesis is eq (1) of §9.2 with *no term silently zero*.
   Ablations flip flags explicitly through config; a default that disables a cost term is a Critical
   review finding everywhere except the pinned ablation configs.
