# STATUS — ui-v1 handoff board

Update this file when you **start** and when you **finish**. Keep it terse. This is how the next agent
knows whether to branch off `main` or off your branch (see `PLAN.md` §3.2).

Legend: `todo` · `in-progress` · `in-review` (code-reviewer running / findings open) · `pr-open`
(approved + PR raised, awaiting human merge) · `merged` · `blocked`

| Task | State | Branch | PR | pytest / typecheck | Reviewer | Notes |
|---|---|---|---|---|---|---|
| U00 sim telemetry instrumentation | merged | `feature-sim-telemetry` | [#70](https://github.com/Prolead1/undertow/pull/70) | 24 passed; full 1497 passed, 6 skipped (93.64%) | approved | telemetry.jsonl + status.json + config flags |
| U01 ui scaffold + stack ADR | merged | `feature-ui-scaffold` | [#69](https://github.com/Prolead1/undertow/pull/69) | 24 passed; full 1497 passed, 6 skipped (93.64%) | approved | `undertow.ui` pkg, `ui` extra, `undertow-ui` script, ADR-023 |
| U02 registry + schemas + results | merged | `feature-ui-registry` | [#72](https://github.com/Prolead1/undertow/pull/72) | 79 passed; full 1576 passed, 6 skipped (93.92%) | approved | scan runs/, parse results/; rank() = eval-first then Sharpe→worst-regime→reward |
| U03 telemetry reader + streaming hub | merged | `feature-ui-streaming` | [#73](https://github.com/Prolead1/undertow/pull/73) | 92 passed; full 1589 passed, 6 skipped (93.84%) | approved | tail jsonl → WS |
| U04 figure builders + dark theme | pr-open | `feature-ui-figures` | [#75](https://github.com/Prolead1/undertow/pull/75) | 50 passed; full 1659 passed, 6 skipped (94.12%) | approved | pure Plotly specs; 6 builders + dark theme |
| U05 FastAPI app + CLI serve | pr-open | `feature-ui-api` | [#76](https://github.com/Prolead1/undertow/pull/76) | 205 passed; full 1713 passed, 6 skipped (94.10%) | approved | REST + WS + static; ADR-025; split ws/static/assembly |
| U06 rollout recorder + reader | merged | `feature-ui-rollouts` | [#74](https://github.com/Prolead1/undertow/pull/74) | 20 passed; full 1596 passed, 6 skipped (93.91%) | approved | parquet sidecar (sim) + reader; costs in raw USDC; ADR-024 |
| U07 React scaffold + design system | todo | `feature-ui-web` | — | — | — | Vite/TS/Tailwind + generated client |
| U08 Live Cockpit view | todo | `feature-ui-live` | — | — | — | learning + diagnostics + actions |
| U09 Policy Replay view | todo | `feature-ui-replay` | — | — | — | ribbon + filmstrip + inspector |
| U10 Leaderboard + Compare view | todo | `feature-ui-leaderboard` | — | — | — | ranking + overlay |
| U11 docs + CI + integration smoke | todo | `feature-ui-polish` | — | — | — | `running_the_dashboard.md`, `ui.yml` |

## ADRs raised

- **ADR-023** — UI stack & live server: FastAPI + WebSocket backend, React/Vite + Plotly.js frontend, third top-level package `undertow.ui`, backend-owned figure specs, Node quarantined in `web/`, in `docs/decisions/023-ui-stack-and-live-server.md`
- **ADR-024** — additive `info` fields on `LpEnvironment` for rollout recording (`timestamp`, `price`, `tick`, `position_value`, `in_range`, `regime`, `gas_price_wei`, `action_index`); public return types and existing keys unchanged, in `docs/decisions/024-env-info-rollout-fields.md`

## Blockers / notes

- **Smart ordering:** U02 first (it owns the frozen `schemas.py`); U03 and U06 are stacked on it. Wave-0 and the boundary fix are merged/in PR.
- **UI-wide follow-up (U11):** readers must reject a higher `schema_version` major with a typed error (CONTRACTS §0). Not yet implemented in `telemetry.py`/`rollouts.py`/`registry.py`; assigned to U11.
- `feature-sim-report` carries an **uncommitted** `plotly` addition to `pyproject.toml`/`uv.lock` (now stashed). U01 superseded it with the `ui` extra; do not build on the uncommitted change.
