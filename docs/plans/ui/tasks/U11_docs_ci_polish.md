# U11 — Docs, CI workflow, integration smoke, polish

**Wave 6 · size M · depends on: U08, U09, U10 · blocks: —**
**Branch:** `feature-ui-polish`

## Why this task exists

Close the loop: document how to run the dashboard, put the frontend under CI without disturbing the
Python gate, and prove the whole thing works end-to-end on a real fixture run.

## Files you own

```
docs/running_the_dashboard.md          (new)
.github/workflows/ui.yml               (new)
README.md                              (edit — Status checklist + Dashboard quickstart; keep the diff scoped)
web/playwright.config.ts               (new — optional)
web/e2e/smoke.spec.ts                  (new — optional)
```

Do not change the Python package or the three views' internals; file issues / small fixes back to the
owning task's branch instead.

## What to build

1. **`docs/running_the_dashboard.md`.** A fresh-clone walkthrough:
   - `uv sync` → build a snapshot (link `docs/running_experiments.md` §"Build a data snapshot") →
     `uv run undertow-sim train --output runs/experiment_1/`;
   - `uv run undertow-ui serve --runs runs/` → open `http://127.0.0.1:8050`;
   - developer mode: two terminals (`uv run undertow-ui serve --reload` + `cd web && npm run dev`,
     Vite proxy to 8050), and `npm run gen:api` after API changes;
   - how to enable replay data (`training.record_rollouts = true`), where artifacts land, and the
     telemetry/status/rollout file layouts (link `docs/plans/ui/CONTRACTS.md`);
   - troubleshooting: no runs found, WS blocked (polling fallback), port in use, stale `status.json`,
     `web_dist` not built.
   Keep it copy-pasteable and free of secrets.

2. **`.github/workflows/ui.yml`.** A **separate** job from `quality.yml` (which must stay untouched):
   checkout → setup-node (22, cache `npm`) → `cd web && npm ci` → `npm run typecheck` → `npm run test`
   → `npm run build`. Trigger on PRs touching `web/**`, `src/undertow/ui/**`, or the workflow itself.
   Start non-blocking (`continue-on-error: true`) with a comment saying it becomes required once the
   frontend stabilises; the Python `quality` job remains the merge gate.

3. **`README.md`.** Add a short "Dashboard" section under the sim status table: one command to start
   it, a one-line description of the three views, and a link to `docs/running_the_dashboard.md`.
   Extend the Status checklist with the UI line. Keep the diff to those additions.

4. **Integration smoke (recommended).** A Playwright test that: starts `undertow-ui serve` against a
   committed fixture run dir, loads `/live`, asserts the run header + a rendered chart, and navigates
   to `/leaderboard`. Skip gracefully (not fail) when a browser cannot be installed in CI. If you
   choose not to add Playwright, say why in the PR and instead add a `scripts/smoke_ui.sh` that curls
   the API endpoints and greps for `FigureSpec` keys.

5. **Polish pass.** Verify: keyboard navigation, focus rings, reduced-motion, empty/error states on
   every panel, no console errors, no `NaN` in any figure JSON, and the dark theme applied on every
   route. Fix small issues directly; file anything structural back to the owning task.

6. **Close the schema-version follow-up.** CONTRACTS §0 requires readers to reject a higher
   `schema_version` major with a **typed** error. Add a shared `SchemaVersionError` (in
   `undertow/ui/schemas.py`) plus a `check_schema_version()` helper, and use it in `telemetry.py`,
   `rollouts.py` and `registry.py` (replacing the ad-hoc `ValueError`). Add tests for a `schema_version: 2`
   artifact being rejected. This is a UI-wide contract gap flagged by review; it must not ship open.

## Definition of done

- `docs/running_the_dashboard.md` works verbatim from a fresh clone (state the command you ran).
- `ui.yml` passes on the PR; `quality.yml` unchanged and passing.
- `cd web && npm run typecheck && npm run test && npm run build` green; `uv run pytest` green.
- Smoke test (or `scripts/smoke_ui.sh`) passes against a fixture run.
- `code-reviewer` APPROVE; PR from `feature-ui-polish`; `STATUS.md` marks the plan complete.
