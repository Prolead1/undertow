# ADR 028 — Minimal frontend dependencies (no react-router, no react-plotly.js, no ESLint)

**Status:** accepted
**Affects:** `web/package.json`, `web/src/router.tsx`, `web/src/router/primitives.tsx`,
`web/src/components/PlotlyFigure.tsx`, `.github/workflows/ui.yml`, `web/README.md`,
`docs/plans/ui/PLAN.md`, `docs/plans/ui/CONTRACTS.md`
**Raised by:** U07/U11 follow-up (security + minimalism pass)

## Context

`PLAN.md` §7 pinned the frontend as React + Vite + Tailwind + `react-router-dom` + `react-plotly.js`,
with `eslint`/`typescript-eslint`/`prettier` as the lint/format toolchain. The dashboard is a
read-only four-route viewer: it lists runs, streams telemetry, renders six server-generated Plotly
figures, and keeps selection in the query string (`?run_id=&seed=&episode=`).

## Problem

A dependency audit (`npm audit`) reported 14 advisories (2 critical, 9 high) and the lint stack was
**ESLint 8, which is EOL** and dragged in deprecated `glob@7`/`inflight`. The two most replaceable
runtime deps were also the source of churn:

- `react-router-dom` was upgraded 6 → 7 to clear a production advisory, but the app uses none of its
  data APIs (loaders/actions/nested routes) — only `useLocation`/`useNavigate`/`useSearchParams`/
  `Link` for four static paths.
- `react-plotly.js` is a thin wrapper; the app already imports `plotly.js-dist-min` directly.

"Keep a 25+ package tree current and vulnerability-free" is ongoing maintenance for a dashboard that
does not need most of it.

## Decision

Keep only what the dashboard actually uses.

- **Runtime (3):** `react`, `react-dom`, `plotly.js-dist-min`.
- **Build/test/types:** `vite`, `@vitejs/plugin-react`, `typescript`, `tailwindcss`,
  `@tailwindcss/vite`, `vitest`, `jsdom`, `@testing-library/react` (+ `dom` peer),
  `openapi-typescript`, and `@types/{react,react-dom,plotly.js}`.
- **Routing:** a dependency-free router in `web/src/router/primitives.tsx` (~180 lines) providing
  `BrowserRouter`, `MemoryRouter` (tests), `useLocation`, `useNavigate`, `useSearchParams`, `Link`,
  and `Navigate`; `web/src/router.tsx` owns the route table and lazy-loads the chart-heavy views.
- **Charts:** `PlotlyFigure` calls `Plotly.react`/`Plotly.purge` directly, memoising `data`/`layout`/
  `config` by JSON signature (view state is preserved via the layout's `uirevision`).
- **Quality gate:** strict `tsc --noEmit` + `vitest` + the `code-reviewer` subagent. The ESLint and
  Prettier toolchains are removed, and `ui.yml` no longer runs a lint step.
- **TypeScript stays on 5.x** (5.9.3): TypeScript 7.0.2 passes `tsc --noEmit` but crashes
  `typescript-eslint`; revisit when the tooling supports it.

Result: **17 direct dependencies** (3 runtime + 14 dev), all current, `npm audit` and
`npm audit --omit=dev` both report **0 vulnerabilities**.

## Consequences

- `web/package.json`/`package-lock.json` shrink; `react-router-dom`, `react-plotly.js`, the ESLint
  stack, Prettier, and `@types/react-plotly.js` are gone. `@types/node` is retained (dev-only): the
  design-token test imports `node:fs`/`node:path`/`node:url` to read `theme/tokens.css` from disk.
- `PLAN.md` §7 (charts) and `CONTRACTS.md` §8 (frontend perf note) are updated: figures are rendered
  by direct `Plotly.react`, not `react-plotly.js`'s `revision` prop.
- `.github/workflows/ui.yml` drops its lint step; `web/README.md` documents the policy and the new
  `src/router/` module.
- Tests: the router is exercised through the local `MemoryRouter`; Plotly is mocked at the
  `plotly.js-dist-min` boundary (`Plotly.react`/`purge`).
- Rejected: keeping `react-router-dom`/`react-plotly.js` (unnecessary surface for this app); keeping
  ESLint (the EOL/vulnerable-churn source, and `tsc` + tests already gate correctness); TypeScript 7
  (tooling incompatibility); hand-rolling charts (the six figure builders are Plotly specs and are
  the product).
- No `~/Documents/fyp/lesson_plan/` text depends on the frontend dependency choice.
