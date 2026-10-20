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
| U07 React scaffold + design system | merged | `feature-ui-web` | [#77](https://github.com/Prolead1/undertow/pull/77) | 17 tests; typecheck/lint/build green; Python 1713 passed, 6 skipped (94.10%) | approved | Vite/TS/Tailwind + generated client; ADR-026 |
| U08 Live Cockpit view | pr-open | `feature-ui-live` | [#79](https://github.com/Prolead1/undertow/pull/79) | 32 tests; typecheck/build/lint green | approved | learning + diagnostics + actions; view-local state |
| U09 Policy Replay view | pr-open | `feature-ui-replay` | [#80](https://github.com/Prolead1/undertow/pull/80) | 37 tests; typecheck/build/lint green | approved | ribbon + filmstrip + inspector; replay theme/rows modules |
| U10 Leaderboard + Compare view | pr-open | `feature-ui-leaderboard` | [#78](https://github.com/Prolead1/undertow/pull/78) | 38 tests; typecheck/build/lint green | approved | ranking + overlay; URL-only pins; ADR-027 |
| U11 docs + CI + integration smoke | pr-open | `feature-ui-polish` | [#82](https://github.com/Prolead1/undertow/pull/82) | Python 1736 passed, 6 skipped (94.11%); web 74 tests; smoke PASS | approved | `running_the_dashboard.md`, `ui.yml`, schema-version close-out, lazy Plotly split |

## ADRs raised

- **ADR-023** — UI stack & live server: FastAPI + WebSocket backend, React/Vite + Plotly.js frontend, third top-level package `undertow.ui`, backend-owned figure specs, Node quarantined in `web/`, in `docs/decisions/023-ui-stack-and-live-server.md`
- **ADR-024** — additive `info` fields on `LpEnvironment` for rollout recording (`timestamp`, `price`, `tick`, `position_value`, `in_range`, `regime`, `gas_price_wei`, `action_index`); public return types and existing keys unchanged, in `docs/decisions/024-env-info-rollout-fields.md`
- **ADR-025** — UI API surface: single-scan registry with the `(RunRecord, RunDir)` invariant + `<experiment>/seed_<n>` alias, per-directory leased WS hubs, `FigureSpecModel` response typing, additive `refresh`/`metric`/heartbeat, in `docs/decisions/025-ui-api-surface.md`
- **ADR-026** — frontend state management for U07: React state + typed client now, Zustand/TanStack Query deferred to U08; `gen:api` reads the committed `web/openapi.json` (offline/reproducible) with `gen:api:live` for a running server, in `docs/decisions/026-frontend-state-management.md`
- **ADR-027** — leaderboard contract gaps: `n_rebalances` absent from the frozen `EvalMetrics` read defensively with an `annualized_return` fallback; URL-only pin selection (App sidebar tray intentionally unwired, cross-view store deferred), in `docs/decisions/027-leaderboard-contract-gaps.md`
- **ADR-028** — minimal frontend dependencies: drop `react-router-dom` (built-in `src/router/`), `react-plotly.js` (direct `Plotly.react`), and the ESLint/Prettier stack (strict `tsc` + tests + review); runtime deps are `react`, `react-dom`, `plotly.js-dist-min`; TypeScript held at 5.x, in `docs/decisions/028-minimal-frontend-dependencies.md`

## Blockers / notes

- **Smart ordering:** U02 first (it owns the frozen `schemas.py`); U03 and U06 are stacked on it. Wave-0 through wave-5 are merged.
- **UI-wide follow-up (U11): CLOSED.** `SchemaVersionError` + `check_schema_version()` live in `ui/schemas.py`, used by `telemetry.py` (`read_status`, `read_points`), `rollouts.py` (reader) and `registry.py` (degrades to `unknown`); REST surfaces `422` via a FastAPI exception handler. Tests cover `schema_version: 2` rejection and version-1/missing tolerance for telemetry, status, rollouts and registry.
- **U11 final state:** `docs/running_the_dashboard.md`, `.github/workflows/ui.yml` (non-blocking, separate from `quality.yml`), `README.md` Dashboard section + Status line, `scripts/smoke_ui.sh` + committed fixture, route-level `React.lazy` Plotly split (entry chunk ~226 kB, Plotly deferred to a 4.85 MB on-demand chunk), `.gitignore` symlink-safe `node_modules` entries. Python `quality` remains the merge gate.
- `feature-sim-report` carries an **uncommitted** `plotly` addition to `pyproject.toml`/`uv.lock` (now stashed). U01 superseded it with the `ui` extra; do not build on the uncommitted change.
