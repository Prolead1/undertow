# Plan: `undertow.ui` — Live Policy Cockpit

> **Source of truth.** This directory — `docs/plans/ui/` — is the canonical plan for the dashboard.
> ADRs live in [`docs/decisions/`](../decisions/) (numbering shared with the data and sim plans).

A live, dark-themed web dashboard for the Undertow RL pipeline: watch a policy train, replay what a
checkpoint actually did, and rank the best policies discovered so far. Delivered as a third top-level
package, `undertow.ui`, plus an isolated React frontend under `web/`.

## Origin

The requirement came from a design discussion (not a roadmap section): *"a cool visualization which
shows me the policy being trained, the progress on it, and what are the best policies discovered so
far — a nice clean modern visualization, dark theme."* This plan is the engineering translation of
that discussion. There is no external spec to satisfy; **this directory is the spec.**

## Read in this order

| File | What it is | Who reads it |
|---|---|---|
| **`PLAN.md`** | scope, architecture, task graph, conventions, DoD, pinned decisions, subagent dispatch | everyone, first |
| **`CONTRACTS.md`** | **frozen** schemas, endpoints, events, figure specs, design tokens | everyone, second |
| `STATUS.md` | live handoff board — task states, branches, PRs, blockers | everyone, on start & finish |
| `tasks/U*.md` | 12 self-contained task briefs | the agent assigned that task |

## The three views

| View | The question it answers | Hero element |
|---|---|---|
| **Live Cockpit** (`/live`) | What is the policy doing *right now*? | animated action-grid heatmap + learning curve + PPO diagnostics |
| **Policy Replay** (`/replay`) | What did *this* checkpoint actually do? | price path with the agent's position bands, rebalances and in/out-of-range shading |
| **Leaderboard / Compare** (`/leaderboard`, `/compare`) | Which policies are best so far, and how do they trade off? | ranked run table with provenance + overlay comparison |

## Task index

| Wave | Tasks | Theme |
|---|---|---|
| 0 | U00, U01 | sim telemetry instrumentation · `undertow.ui` scaffold + stack ADR |
| 1 | U02, U03, U06 | run registry · live streaming · rollout recording |
| 2 | U04 | figure builders + Plotly dark theme |
| 3 | U05 | FastAPI app: REST + WebSocket + static + `undertow-ui serve` |
| 4 | U07 | React/Vite scaffold, design tokens, API client |
| 5 | U08, U09, U10 | Live Cockpit · Policy Replay · Leaderboard/Compare |
| 6 | U11 | docs, CI job, integration smoke, polish |

Full dependency table: `PLAN.md` §3.1. Launch order & subagent protocol: `PLAN.md` §9–§10.

## Non-negotiables (from `undertow/AGENTS.md`)

Branch first · tests with real assertions · `code-reviewer` before every commit · PR against `main`,
never self-merged · `.pi/` never committed · `undertow.ui` imports `sim`/`data` public APIs only, never
the reverse · the dashboard is **read-only** (it never trains or mutates a run).

## The three things most likely to go wrong

1. **The dashboard becomes a second source of truth.** It must only *read* artifacts. All compute
   lives in `undertow.sim`; U00/U06 record sidecars, everyone else reads. If a task finds itself
   re-deriving PnL or re-running the env for a chart, it is doing it wrong.
2. **Torn/partial telemetry lines.** Training writes while the reader tails. U00 writes one complete
   line per `write()`; U03 skips an unparseable trailing line and only advances past complete lines.
   Getting this wrong shows up as random JSON decode errors mid-training — a Critical bug.
3. **A JS toolchain quietly breaking the Python quality gate.** `web/` is quarantined: `node_modules`
   and `dist` are gitignored, its CI job is separate, and `uv run pytest` must pass with `web/`
   absent. If a Python task starts requiring Node, that is a Critical review finding.
