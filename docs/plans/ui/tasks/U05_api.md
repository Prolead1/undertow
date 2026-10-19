# U05 — FastAPI app: REST + WebSocket + static + `undertow-ui serve`

**Wave 3 · size L · depends on: U02, U03, U04, U06 · blocks: U07, U08**
**Branch:** `feature-ui-api`

## Why this task exists

Compose the readers and figure builders behind a typed HTTP/WS surface, generate the OpenAPI schema
that the frontend consumes, and make the whole thing start with one command.

## Files you own

```
src/undertow/ui/app.py        (new — create_app, routes, WS, static mount)
src/undertow/ui/cli.py        (edit — implement `serve`)
src/undertow/ui/__init__.py   (edit — append create_app, serve to __all__)
tests/ui/test_api.py          (new)
docs/decisions/024-ui-api-surface.md   (new — optional, only if you make a non-obvious choice)
```

## What to build

Read `CONTRACTS.md` §5 (streaming) and §7 (REST) first — the surface is frozen.

1. **`create_app(config: UiConfig) -> FastAPI`.**
   - Build a `RunRegistry` + `TelemetryHub` on `app.state`.
   - Routes exactly as §7. Every response is a `schemas` model (or list of them). `run_id` lookups
     return `404` with a helpful detail. `figures/{name}` looks up `FIGURE_REGISTRY`; unknown name →
     `404`. The `figures` endpoint assembles the builder's plain-data input from the registry/readers,
     then calls the pure builder — **no chart logic in this file**.
   - `GET /api/runs/{run_id}/curve` → `LearningCurve`; `telemetry` → §5.1 payload.
   - `POST /api/compare` accepts `{"run_ids":[...],"metric":"sharpe"}` and returns the compare figure
     + the records; validate non-empty, max 6 ids.
   - `WS /api/stream`: on connect send `hello`, then subscribe to the hub and forward frames; support
     `run_id` omitted (status-only). Handle client disconnect without leaking the subscription.
   - Static: when `config.web_dist` is set, mount it at `/` **after** API routes with an SPA fallback
     to `index.html`; otherwise `GET /` returns a small JSON hint.
   - CORS: only when `config.cors_origins` is non-empty (dev). Default same-origin.
   - Enable `/openapi.json` (FastAPI default) — U07 generates types from it.

2. **`cli.py` `serve`.** Resolve `UiConfig` (CLI flags override TOML), run
   `uvicorn.run(app, host=..., port=..., reload=...)`. `--reload` only for dev. Print the bound URL
   and the runs dir. Keep exit codes as `undertow-sim`.

3. **`__init__.py`.** Append `create_app` and `serve` to `__all__` and imports.

4. **Performance.** The registry is scanned on startup and refreshed lazily (e.g. on a `?refresh=true`
   or a short TTL) — do **not** rescan on every request. No blocking calls in async routes: run
   parquet/json reads through `asyncio.to_thread` or make the handlers sync (`def`) so FastAPI offloads
   them. Pick one and be consistent.

## Tests (`tests/ui/test_api.py`, FastAPI `TestClient`)

1. `/api/health` → `{"status":"ok","schema_version":1}`.
2. `/api/runs` on an empty dir → `[]`; on the fixture run tree → the records, in deterministic order.
3. `/api/runs/{id}` present → 200 with provenance; absent → 404.
4. `/api/runs/{id}/telemetry?since=0&limit=2` returns exactly 2 points and the right `next_offset`;
   `since` beyond the end returns `[]` with `done` reflecting status.
5. `/api/runs/{id}/figures/learning` → a themed `FigureSpec`; unknown name → 404.
6. `/api/runs/{id}/rollouts` and `/api/rollouts/{id}/{seed}/{ep}` on the fixture; missing → 404.
7. `POST /api/compare` with 6 ids works; with 0 or 7 → 422.
8. `WS /api/stream`: connect, receive `hello`, append a telemetry line, receive a `telemetry` frame;
   closing the socket does not error the server. (Use `TestClient.websocket_connect`.)
9. Static mount: with a temp `web_dist` containing `index.html`, `GET /some/route` returns the SPA
   fallback; with `web_dist=None`, `GET /` returns the hint JSON.
10. OpenAPI includes every §7 path (assert the path set).

## Definition of done

- `uv run pytest tests/ui/test_api.py -q` green; full suite green at ≥90%; `ruff` + `mypy` clean.
- `uv run undertow-ui serve --runs <fixtures>` binds and serves `/openapi.json`.
- Surface matches `CONTRACTS.md` §5/§7 exactly.
- `code-reviewer` APPROVE; PR from `feature-ui-api`; `STATUS.md` updated.
