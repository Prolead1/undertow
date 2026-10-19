# STATUS — ui-v1 handoff board

Update this file when you **start** and when you **finish**. Keep it terse. This is how the next agent
knows whether to branch off `main` or off your branch (see `PLAN.md` §3.2).

Legend: `todo` · `in-progress` · `in-review` (code-reviewer running / findings open) · `pr-open`
(approved + PR raised, awaiting human merge) · `merged` · `blocked`

| Task | State | Branch | PR | pytest / typecheck | Reviewer | Notes |
|---|---|---|---|---|---|---|
| U00 sim telemetry instrumentation | pr-open | `feature-sim-telemetry` | [#70](https://github.com/Prolead1/undertow/pull/70) | 24 passed; full 1497 passed, 6 skipped (93.64%) | approved | telemetry.jsonl + status.json + config flags; stacked on #68 |
| U01 ui scaffold + stack ADR | pr-open | `feature-ui-scaffold` | [#69](https://github.com/Prolead1/undertow/pull/69) | 24 passed; full 1497 passed, 6 skipped (93.64%) | approved | `undertow.ui` pkg, `ui` extra, `undertow-ui` script, ADR-023; stacked on #68 |
| U02 registry + schemas + results | todo | `feature-ui-registry` | — | — | — | scan runs/, parse results/ |
| U03 telemetry reader + streaming hub | todo | `feature-ui-streaming` | — | — | — | tail jsonl → WS |
| U04 figure builders + dark theme | todo | `feature-ui-figures` | — | — | — | pure Plotly specs |
| U05 FastAPI app + CLI serve | todo | `feature-ui-api` | — | — | — | REST + WS + static |
| U06 rollout recorder + reader | todo | `feature-ui-rollouts` | — | — | — | parquet sidecar (sim) + reader |
| U07 React scaffold + design system | todo | `feature-ui-web` | — | — | — | Vite/TS/Tailwind + generated client |
| U08 Live Cockpit view | todo | `feature-ui-live` | — | — | — | learning + diagnostics + actions |
| U09 Policy Replay view | todo | `feature-ui-replay` | — | — | — | ribbon + filmstrip + inspector |
| U10 Leaderboard + Compare view | todo | `feature-ui-leaderboard` | — | — | — | ranking + overlay |
| U11 docs + CI + integration smoke | todo | `feature-ui-polish` | — | — | — | `running_the_dashboard.md`, `ui.yml` |

## ADRs raised

- **ADR-023** — UI stack & live server: FastAPI + WebSocket backend, React/Vite + Plotly.js frontend, third top-level package `undertow.ui`, backend-owned figure specs, Node quarantined in `web/`, in `docs/decisions/023-ui-stack-and-live-server.md`

## Blockers / notes

- **PR #68** (`docs-ui-plan`) is the plan itself and is the base for the Wave-0 PRs (#69, #70).
  Merge it first; the stacked PRs then clean up to their own diff.
- `feature-sim-report` carries an **uncommitted** `plotly` addition to `pyproject.toml`/`uv.lock`.
  U01 supersedes it with the `ui` extra; do not build on the uncommitted change.
- Wave 1 (U02, U03, U06) may start once #68 is merged, or by stacking on #70/#69 as noted in each
  task file.
