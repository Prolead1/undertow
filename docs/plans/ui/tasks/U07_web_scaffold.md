# U07 — React/Vite scaffold, design system, generated API client

**Wave 4 · size M · depends on: U05 (OpenAPI) · blocks: U08, U09, U10**
**Branch:** `feature-ui-web`

## Why this task exists

Stand up the isolated frontend toolchain and the shared shell (routing, layout, dark tokens, API
client, live-stream hook) so the three views are pure composition, not plumbing.

## Files you own

```
web/package.json  web/vite.config.ts  web/tsconfig.json  web/tailwind.config.ts
web/index.html  web/postcss.config.js  web/.eslintrc.cjs  web/.prettierrc
web/src/main.tsx  web/src/App.tsx  web/src/router.tsx
web/src/theme/tokens.css
web/src/api/client.ts  web/src/api/types.ts  web/src/api/useLiveStream.ts
web/src/components/            (Panel, Stat, Sparkline, Toolbar, EmptyState, RunBadge, Toast)
web/src/views/                 (placeholder stubs only — real views are U08–U10)
web/src/__tests__/smoke.test.tsx
web/README.md
```

Do **not** implement the three views here (stubs that U08–U10 replace). Do not touch `src/undertow/ui/**`.

## What to build

Read `CONTRACTS.md` §8 (frontend contract) and §9 (tokens) first.

1. **Toolchain.** Vite + React 18 + TypeScript (strict). Tailwind v4 with the §9 tokens exposed as CSS
   variables in `theme/tokens.css` and mapped in `tailwind.config.ts`. `recharts` is **not** used —
   Plotly via `react-plotly.js` + `plotly.js-dist-min`. Add `vitest` + `@testing-library/react`.
   Scripts: `dev`, `build`, `preview`, `typecheck`, `test`, `lint`, `gen:api`.

2. **API client.** `gen:api` runs `openapi-typescript http://127.0.0.1:8050/openapi.json -o
   src/api/types.ts` (document that the server must be running; commit the generated file). `client.ts`
   is a thin typed `fetch` wrapper reading `VITE_API_BASE` (default `""`, proxied by Vite to
   `127.0.0.1:8050`). Include `getRuns`, `getRun`, `getCurve`, `getTelemetry`, `getFigure`,
   `listRollouts`, `getRollout`, `compare`.

3. **`useLiveStream`.** A hook owning one WebSocket to `/api/stream`, exposing
   `{ status, runs, telemetryByRun, lastStatus }`. Reconnect with exponential backoff (cap 30 s);
   on `close`/error fall back to polling `getTelemetry` every `telemetry_poll_seconds` and mark the
   connection `degraded` in the UI. Live-merge rule from §8 (dedupe by `(run_id, timestep)`).

4. **Shell.** `App.tsx` = left rail (run selector + nav) + top bar (global state, connection badge,
   pin tray) + routed content. `router.tsx` = the §8 routes with query-param selection preserved.
   `EmptyState`, `Panel`, `Stat`, `RunBadge`, `Toolbar`, `Sparkline`, `Toast` styled from tokens.
   Keyboard reachable, visible focus rings, `prefers-reduced-motion` respected.

5. **Plotly wrapper.** A single `components/PlotlyFigure.tsx` that renders a `FigureSpec` with
   `react-plotly.js`, applies `useResizeHandler`, and bumps `revision` only on data change. Views use
   this, never `react-plotly.js` directly.

## Tests (`web/src/__tests__`)

1. `client.ts` builds the right URL/params (mock `fetch`) and throws a typed error on non-2xx.
2. `useLiveStream` merges deltas, dedupes by `(run_id, timestep)`, and enters `degraded` mode on
   socket close (use a mock WebSocket).
3. Router preserves `run_id`/`seed`/`episode` query params on navigation.
4. `PlotlyFigure` renders without throwing for an empty spec.
5. Tokens: assert every §9 token exists in the generated CSS (parse `tokens.css`).

Keep tests DOM-light; no snapshots of chart internals.

## Definition of done

- `cd web && npm ci && npm run typecheck && npm run test && npm run build` all green.
- `uv run pytest` still green **with `web/node_modules` absent** (the Python suite must not need Node).
- Dark theme visibly applied on the placeholder shell; no light-mode fallback.
- `code-reviewer` APPROVE; PR from `feature-ui-web`; `STATUS.md` updated.
