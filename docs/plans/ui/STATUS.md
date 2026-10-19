# STATUS — ui-v1 handoff board

Update this file when you **start** and when you **finish**. Keep it terse. This is how the next agent
knows whether to branch off `main` or off your branch (see `PLAN.md` §3.2).

Legend: `todo` · `in-progress` · `in-review` (code-reviewer running / findings open) · `pr-open`
(approved + PR raised, awaiting human merge) · `merged` · `blocked`

| Task | State | Branch | PR | pytest / typecheck | Reviewer | Notes |
|---|---|---|---|---|---|---|
| U00 sim telemetry instrumentation | todo | `feature-sim-telemetry` | — | — | — | telemetry.jsonl + status.json + config flags |
| U01 ui scaffold + stack ADR | todo | `feature-ui-scaffold` | — | — | — | `undertow.ui` pkg, `ui` extra, `undertow-ui` script, ADR-023 |
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

_(none yet — U01 owns ADR-023 for the UI stack & live server)_

## Blockers / notes

- `feature-sim-report` carries an **uncommitted** `plotly` addition to `pyproject.toml`/`uv.lock`.
  U01 supersedes it with the `ui` extra; do not build on the uncommitted change.
