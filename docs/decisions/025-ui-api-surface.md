# ADR 025 — UI API surface: compare equity source, per-run streaming hubs, reload factory

**Status:** accepted
**Affects:** `src/undertow/ui/app.py`, `src/undertow/ui/assembly.py`, `src/undertow/ui/ws.py`,
`src/undertow/ui/static.py`, `src/undertow/ui/cli.py`, `tests/ui/test_api.py`
**Raised by:** U05 (ui-v1)

## Context

`CONTRACTS.md` §7 freezes the REST surface, including:

```text
POST /api/compare  {"run_ids":[str,...],"metric?":str}  ->  {"figures": {...}, "records": RunRecord[]}
WS   /api/stream   run_id?                              ->  CONTRACTS §5.2 envelopes
```

Two things the contract deliberately leaves to the implementation are (a) **which series the compare
overlay plots** — it only names the response key `figures` — and (b) **how the WebSocket composes a
per-run tail**, because U03 ships a `TelemetryHub` that watches exactly one run directory:

> `CONTRACTS.md` §5.2: "Only the requested `run_id` is streamed (`run_id` omitted = server status
> only)."  · `PLAN.md` §2 (U05): "Build a `RunRegistry` + `TelemetryHub` on `app.state`."

The UI is a pure artifact consumer (`PLAN.md` §0.6): it has readers for telemetry, status and rollout
parquet, and a `results.py` parser for eval metrics — but **no equity reader**. The only per-step
equity series the package can read is the `equity` column of a rollout parquet (`CONTRACTS.md` §2.1),
and those exist only when a run was trained with `training.record_rollouts = true` (on by default).

The U05 brief also asks for `undertow-ui serve --reload`, but uvicorn only reloads when the app is an
import string; a resolved `UiConfig` cannot ride along inside an app instance into a fresh worker.

## Problem

1. Without a pinned equity source, `/api/compare` cannot be implemented consistently: `EvalMetrics`
   has point estimates (`sharpe`, `total_pnl`, …), not curves; training telemetry has reward, not
   equity; and `equity_curve.csv` lives next to the backtest outputs that U02's `results.py` does not
   parse. Two implementers could return two different overlays from the same ids.
2. `TelemetryHub` is one-run-only, so a naive reading of "a TelemetryHub on app.state" cannot serve
   `/api/stream?run_id=<any>`; it would either stream the wrong run or require a second streaming
   implementation.
3. `--reload` silently degrades to a normal run if the app is passed as an instance.

## Decision

1. **Compare overlay = first recorded rollout episode per run.**  For each requested id,
   `assembly.compare_series` picks the deterministic first parquet via `list_rollouts` (sorted by
   `(seed, episode)`) and plots its `equity` column with `compare_equity_figure`.  The response is
   `{"figures": {"compare": <FigureSpec>}, "records": <RunRecord[]>}`.  Runs that exist but recorded
   no rollouts still appear in `records`; they are simply omitted from the overlay (and the figure is
   the themed "No data" figure when nothing is plottable).  `metric` ranks `records` with the pinned
   leaderboard rule; unknown metrics are a `422`.
2. **Streaming is a lazy, leased per-run hub cache.**  `app.state.hubs` is a `TelemetryHubs` manager
   (:mod:`undertow.ui.ws`) that creates and caches one `TelemetryHub` per `run_id` on first
   subscribe, closes/evicts it when its last subscriber disconnects, and closes them all on lifespan
   shutdown.  Each hub starts at the telemetry line count at subscribe time, so a client receives
   **deltas only** (CONTRACTS §5.2/§8 "Live merge rule"); history is backfilled through
   `GET .../telemetry`.  `run_id` omitted sends `hello` then heartbeat pings plus `run_added` for
   runs discovered by the status-only loop's registry refresh.
3. **Additive, contract-compatible extensions:** `?refresh=true` on `/api/runs` forces a registry
   re-scan (otherwise a 2 s TTL bounds re-scans — never per request); the figure route accepts an
   optional `metric` query (defaults `policy_loss` for diagnostics, `sharpe` for the leaderboard); the
   WebSocket heartbeat is a `{"type":"ping", ...}` JSON envelope, which clients ignore; the
   status-only loop also emits the contract's `run_added` event for newly discovered runs; and
   `--reload` serialises the resolved `UiConfig` into `UNDERTOW_UI_CONFIG_JSON` and points uvicorn at
   `undertow.ui.app:create_app_from_env` (falling back to `UiConfig()` when the variable is absent).
4. **Single-scan run resolution, keyed by directory.**  `RunRegistry.scan()` returns records and
   `RunDir`s together, and `resolve()` accepts the CONTRACTS §7 `<experiment>/seed_<n>` path as a
   disambiguation escape hatch for a duplicate manifest `run_id`.  Every route (REST, figures,
   rollouts, WebSocket) resolves the same `(record, RunDir)` pair, so an id can never read two
   different directories.  Because a `run_id` can collide, every **cache is keyed by the resolved
   directory** (`str(RunDir.path)`), never by `run_id`: the learning-curve and telemetry-timestep
   maps in `RunRegistry` (`_Snapshot.curves`/`_Snapshot.timesteps`) and the lease/hub maps in
   `TelemetryHubs` all use the directory path, so a duplicate cannot overwrite another directory's
   entry.  `status`/`telemetry`/`rollouts`/`WS /api/stream` read the alias's **own** directory, and
   so do `curve`/`figures/learning`: the curve aggregates by the **resolved** directory's
   experiment rather than re-canonicalising by `run_id`, so an alias for a duplicate-`run_id`
   directory returns that directory's own per-seed numbers (the raw id still resolves to the
   canonical directory).  The aggregate spans every directory of the resolved experiment —
   including duplicate-`run_id` directories whose records lost the canonical dedup — because the
   curve is a per-seed aggregate; `_Snapshot.directories` exists to make that grouping possible.
   `LearningCurve.run_id` is the **requested** id string — a canonical request keeps the canonical
   id, an alias request echoes the alias (e.g. `exp_b/seed_0`) — so a frontend cache key can never
   collide with the canonical entry.  `GET .../curve` resolves the curve exactly once (its
   existence re-check runs only on the error path), so a concurrent refresh cannot pair a directory
   from one scan with a curve from another.  `scan()` builds the whole index in locals and publishes
   it with a **single immutable `_Snapshot` assignment**, so a concurrent lock-free `resolve()` can
   never observe a torn view (records from one scan, curves from another).
5. **Leaderboard alias provenance.**  `/figures/leaderboard` ranks one representative per `run_id`.
   The requested run's resolved directory supersedes its canonical namesake for that request, so an
   alias to a duplicate-`run_id` directory always appears in its own leaderboard instead of being
   hidden behind the canonical record (and the surviving set is then what `leaderboard_figure`
   plots).  `POST /api/compare` keeps the other colliding aliases alive instead of collapsing them:
   it pairs ranked records back to their directories by object identity, and `compare_series`
   disambiguates colliding legend labels with the `<experiment>/seed_<n>` alias.  The compare
   request list is de-duped by the **resolved `RunDir.path`**, so a raw id and one of its aliases
   that name the same directory collapse to a single `records` entry and a single overlay trace
   (de-duping by request string would leave two records behind one overwritten trace).  The full
   per-directory provenance stays available through `GET /api/runs/{id}` and the alias itself; the
   figure never grows duplicate bars for one `run_id`.
6. **app.py is route wiring only.**  The WebSocket pump/heartbeat/hub lifecycle lives in
   `ws.py`, the SPA mount in `static.py`, and figure/compare plain-data assembly in `assembly.py`;
   the U05 wire models (`FigureSpecModel`, `TelemetryTail`, compare request/response, health/hint)
   stay in `app.py` because `schemas.py` is U02-owned.

## Consequences

- `app.py` owns the wire models the frozen `schemas.py` (U02) does not: `HealthResponse`,
  `TelemetryTail`, `RootHint` and the compare request/response pair.  `schemas.py` is not edited
  (U02 owns it), and every one of these is additive to `CONTRACTS.md` §7.
- `FigureSpecModel` is the OpenAPI response model for every figure route and for
  `CompareResponse.figures`, so the generated TypeScript sees a typed `{data, layout, config}`.
- **U09/U10 must know** compare draws from rollout equity, not eval point estimates: a run without
  `record_rollouts` will be in `records` but absent from the overlay legend.  The view already needs
  rollouts for replay, so the same "no rollouts" empty state applies.
- The `ping`/`run_added` frames and the `metric`/`refresh` query params are additive; no consumer is
  required to handle them, and `CONTRACTS.md` needs no edit.
- `tests/ui/test_cli.py` (owned by U01) is updated because two of its assertions encoded the U01
  stub: `__all__` now includes `create_app`/`serve`, and `serve` starts uvicorn instead of returning
  exit `2`.  No other U01 behaviour changes.
- **Cache-key policy (non-obvious, pinned above):** directory paths, not `run_id`s, key the curve,
  timestep and hub/lease caches; the leaderboard uses the resolved directory as the representative
  for a colliding `run_id`; and compare de-dupes by directory path.  A duplicate `run_id` therefore
  never mixes two directories' numbers on `/curve`, `/figures/learning`, `/api/compare` or
  `WS /api/stream`.
- Rejected: plotting telemetry reward or `EvalMetrics` scalars as a pseudo-equity (mixes units and
  misleads); adding `equity_curve.csv` parsing to U02's `results.py` (out of U05's file ownership and
  not needed for the frozen surface); a single global `TelemetryHub` (cannot address arbitrary
  `run_id`s); and running uvicorn with `reload=True` on an app instance (uvicorn ignores reload for
  non-import-string apps, so `--reload` would be a no-op).
