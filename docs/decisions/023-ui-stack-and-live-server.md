# ADR 023 — UI stack: FastAPI + WebSocket backend, React/Vite + Plotly.js frontend, third top-level package

**Status:** accepted
**Affects:** `pyproject.toml` · `.gitignore` · `src/undertow/ui/` · `tests/ui/` · `web/` (U07+) · `docs/plans/ui/PLAN.md` §7 · ADR-004
**Raised by:** U01 (ui-v1)

## Context

`docs/plans/ui/PLAN.md` specifies a live, dark-themed cockpit for the RL training
pipeline — three linked views (Live Cockpit, Policy Replay, Leaderboard/Compare)
that read artifacts the simulator already writes and stream training telemetry
while a run is in progress.  The plan pins the stack in §7 and states that it must
not be re-litigated by an individual task:

> **Do not re-litigate the stack in a task.**  The choice (FastAPI + React/Vite +
> Plotly.js, third package, isolated Node toolchain) was made deliberately and is
> pinned in §7.

U01 therefore does not *choose* the stack; it **records** it, together with the
layout and packaging consequences, so later agents and the thesis have one
citation and so the rejected alternatives are documented rather than
re-discovered.  The ADR is required by `PLAN.md` §0.2 ("U01 owns the stack ADR")
and by the repo's rule that every non-obvious deviation is written down.

The relevant shape of the existing repo: `undertow.data` and `undertow.sim` are
sibling top-level packages in one distribution (`src/undertow/`), with a frozen
public-API boundary between them (`AGENTS.md`; ADR-008).  `SimConfig` is a frozen
dataclass loaded from TOML; the CLI follows a `0/1/2` exit-code convention.

## Problem

There is no single artefact that answers "what should `undertow.ui` be built
with, where does the frontend live, and who owns figure rendering?"  Without one:

- a later agent may introduce a second dashboard technology or a second
  distribution, splitting the wire contract (`CONTRACTS.md`) across two sources;
- the Node toolchain could leak into the Python gate (`uv run pytest`) and make
  CI flaky or slow (`PLAN.md` §8 risk register);
- the browser could be asked to render millions of telemetry rows for a
  sparkline, or two implementations of the Plotly figure spec (one Python, one
  JS) could drift.

The alternatives are real and each is tempting for a small team:

| Alternative | Why it was rejected |
|---|---|
| **Dash** (Plotly's Python framework) | Server-rendered callbacks are fine for static analysis, but the plan needs a WebSocket push and stateful, cross-linked client views without losing zoom/selection.  Dash fights per-component streaming and custom dark-theme layout, and its callback graph scales poorly with linked charts. |
| **Streamlit** | Reruns the whole script on every interaction; no stable component model for a live multi-view cockpit, weak WebSocket story, and no OpenAPI surface to generate TS types from.  Excellent for notebooks, wrong for a long-lived research dashboard. |
| **Static-only report** (pre-rendered HTML/PDF) | Cannot show a run *while it trains*, which is the plan's central requirement.  Also pushes figure generation into a one-shot export rather than a reusable read API. |
| **Separate distribution / separate repo** for the UI | Duplicates the wire schema (`CONTRACTS.md` §4) and the `RunManifest`/telemetry types; forces a second version pin and a second CI job for no isolation benefit — the third package already keeps UI imports one-directional. |
| **Client-side figure generation** (ship raw series, build specs in JS) | Slower to first value and duplicates the Plotly spec that the Python figure builders (`undertow/ui/figures/`, U04) own; the plan explicitly chooses server-side figure specs for v1. |

The plan also needs a stable HTTP surface because U07 generates TypeScript types
from OpenAPI (`PLAN.md` §7, "the biggest iteration win"), which rules out the
callback frameworks above.

## Decision

1. **Third top-level package `undertow/ui/`** inside the existing distribution
   (`src/undertow/ui/`).  It mirrors `data`/`sim`, keeps one `pyproject.toml` and
   one version, and is importable as `undertow.ui`.  The React frontend lives in a
   top-level `web/` directory, not under `src/`, so it is visibly not a Python
   package and is never added to the wheel (`PLAN.md` §7).
2. **Backend: FastAPI + uvicorn.**  FastAPI gives a typed HTTP/WS surface and an
   OpenAPI document that U07 consumes to generate `web/src/api/types.ts`.  The
   `ui` extra pins `fastapi>=0.115`, `uvicorn[standard]>=0.30`; `plotly>=6.0` is
   included because figure builders run server-side and are importable from the
   same distribution.
3. **Live transport: WebSocket** at `/api/stream`, with a REST
   `GET .../telemetry?since=` tail as the polling fallback (`CONTRACTS.md` §5).
4. **Frontend: React 18 + Vite + TypeScript (strict)**, styled with Tailwind v4
   and the CSS-variable dark tokens in `CONTRACTS.md` §9.  Charts are Plotly.js
   via `react-plotly.js`; the server ships Plotly figure specs and typed raw
   series.
5. **Backend owns figures.**  U04 builds figure payloads from plain dicts, never
   from readers, so figure construction is testable without I/O.  This keeps one
   plotting implementation and lets the server pre-aggregate downsampled
   payloads (`max_points`, default 2000).
6. **Node is quarantined in `web/`.**  `web/node_modules/`, `web/dist/` and
   `*.tsbuildinfo` are gitignored; the frontend has its own CI job (U11) and
   `uv run pytest` must pass with `web/node_modules` absent.  The Python gate is
   unchanged.
7. **Read-only consumer.**  `undertow.ui` imports only the *public* surfaces of
   `undertow.sim`/`undertow.data`; neither may ever import `undertow.ui`.  The
   dashboard never trains, backtests, or writes into a run directory
   (`PLAN.md` §0.6–§0.7).  U01 enforces the direction with two tests in
   `tests/ui/test_cli.py`: a fresh-interpreter import test asserting neither
   `undertow.sim` nor `undertow.data` lands in `sys.modules`, and an `ast`-based
   scan of every `src/undertow/ui/**/*.py` source that rejects any actual
   `import`/`from` statement naming those packages (prose mentions in comments
   and docstrings are allowed).

## Consequences

- **New files.** `src/undertow/ui/{__init__,__main__,config,cli}.py`, the ADR,
  `tests/ui/` scaffolding, the `ui` extra, dev deps, and the `undertow-ui`
  console script.  U01 deliberately does not create `app.py`, `registry.py`,
  `telemetry.py`, `schemas.py`, `figures/`, or `rollouts.py`; later tasks own
  them (`PLAN.md` §2).
- **Packaging.** The wheel still lists `packages = ["src/undertow"]`, so `web/`
  is excluded by construction.  `uv.lock` gains fastapi/uvicorn/plotly (and
  their transitive deps: pydantic, starlette, httpx, etc.).
- **Later tasks must know.** U02–U11 code against `CONTRACTS.md`; U05 replaces
  the `serve` stub but keeps `cli.py`'s parser and `0/1/2` exit codes; U07
  generates TS types from the OpenAPI doc rather than hand-writing them; U11
  adds `.github/workflows/ui.yml` and must not make the Python `quality` job
  depend on Node.
- **`CONTRACTS.md` needs no edit.** The stack and layout were already frozen
  there and in `PLAN.md` §7; this ADR records them without changing a byte.
  No vault/lesson-plan edit is implied.
- **Rejected, and the human will ask:** Dash and Streamlit (no live WebSocket /
  weak OpenAPI story), a static report (cannot show a live run), a separate
  distribution (splits the schema and doubles the release surface), and
  client-side figures (duplicates the U04 spec and costs first-load speed).
- **Trade-off accepted.** Tailwind + Plotly adds a frontend toolchain to the
  project.  It is isolated to `web/` and to a dedicated CI job; contributors who
  only touch Python never install it.
