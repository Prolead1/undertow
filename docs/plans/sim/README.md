# Plan: `undertow.sim` — AMM simulator, RL environment, and training harness

> **Source of truth.** This directory — `docs/plans/sim/` — is the canonical plan. ADRs mentioned
> below live in [`docs/decisions/`](../decisions/).

Implementation plan for the thesis's core research deliverable — a PPO agent that learns
concentrated-liquidity provision in Uniswap V3.

## Read in this order

| File | What it is | Who reads it |
|---|---|---|
| **`PLAN.md`** | scope, architecture, task graph, dependencies, conventions, DoD | everyone, first |
| **`CONTRACTS.md`** | frozen cross-task interfaces — module signatures, defaults, units | everyone, second |
| `STATUS.md` | live handoff board — task states, branches, PRs | everyone, on start & finish |
| `tasks/S*.md` | self-contained task briefs | the agent assigned that task |

## Task index

| Wave | Tasks | Theme |
|---|---|---|
| 0 | S00 | scaffold — math, config, pool, price, env, metrics |
| 1 | S01 | PPO training harness (stable-baselines3) |
| 2 | S02 | backtester + episode runner |
| 3 | S03 | baseline LP strategies |
| 4 | S04 | analysis + visualisation |

Full dependency table: `PLAN.md` §3.1.

## Non-negotiables (from `undertow/AGENTS.md`)

Branch first · tests with real assertions · `code-reviewer` before every commit · PR against `main`,
never self-merged · `undertow.data` and `undertow.sim` stay separate · `.pi/` never committed.

## The three things most likely to go wrong

1. **Tick ordering.** Higher USDC-per-WETH price → *lower* tick. Every place that turns a price
   range into tick bounds must swap `price_low` and `price_high`. S00's tests gate this.
2. **Fee-growth wrapping.** Uniswap V3 uses unchecked subtraction; fee-growth accumulators wrap.
   All delta subtractions go through `wrapping_sub_256`. The reviewer checks every one.
3. **Silent position loss.** When a rebalance's tick alignment collapses the range (`tick_lower >=
   tick_upper`), the agent can end up with no position. S00 has a fallback to minimum-width range;
   later tasks must preserve this invariant.