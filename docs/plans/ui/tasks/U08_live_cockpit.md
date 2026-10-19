# U08 — Live Cockpit view

**Wave 5 · size L · depends on: U07 · blocks: U11**
**Branch:** `feature-ui-live`

## Why this task exists

The cockpit is the answer to "watch the policy being trained". It streams telemetry for the selected
run and turns it into a legible picture of training health and behaviour, updating live.

## Files you own

```
web/src/views/LiveCockpit.tsx
web/src/components/live/LearningCurve.tsx
web/src/components/live/DiagnosticsGrid.tsx
web/src/components/live/ActionHeatmap.tsx
web/src/components/live/RunHeader.tsx          (status badge, provenance, throughput)
web/src/__tests__/live.test.tsx
```

Do not edit `web/src/api/**`, `web/src/components/{Panel,Stat,...}`, or other views.

## What to build

Read `CONTRACTS.md` §5, §7, §8 first.

Layout (responsive; panels collapse gracefully on narrow screens):

1. **RunHeader** — run id, experiment/seed, `state` badge (`running` pulses, `done`/`failed` static),
   `config_hash` + short `git_commit`, latest timestep / total, wall-clock elapsed. Uses
   `useLiveStream` + the status endpoint.
2. **LearningCurve** — `GET /figures/learning` (or the curve endpoint rendered locally), mean ± std,
   per-seed toggle, checkpoint markers, log-y switch. Appends live points from the stream without
   losing the x-range the user zoomed to.
3. **DiagnosticsGrid** — small multiples for `policy_loss`, `value_loss`, `entropy_loss`, `approx_kl`,
   `clip_fraction`, `explained_variance`; each fed by `GET /figures/diagnostics?metric=...` and
   appended live. A metric that is null/absent shows "not reported" rather than a broken axis.
4. **ActionHeatmap** — `GET /figures/actions` (counts derived server-side from the latest telemetry /
   rollouts). If a `step` scrubber is present, it animates the heatmap over training checkpoints;
   otherwise it shows the current distribution. Legend explains the `offset × width` grid and the
   `hold` action.
5. **Live behaviour.** On `status` `done`/`failed`, stop appending and show the final state. If the
   connection is `degraded`, show a non-blocking banner and keep the last data. A run with no
   telemetry shows `EmptyState` with the command to start training.

## Tests (`web/src/__tests__/live.test.tsx`)

Mock the API client + WebSocket:
1. Header reflects `state`, and flips to `done` when a `status` frame arrives.
2. Learning curve appends a new point from a `telemetry` frame and dedupes a repeat.
3. A null diagnostic metric renders "not reported", not `NaN`.
4. Empty run (no telemetry) renders `EmptyState`.
5. Degraded connection shows the banner and retains prior points.

## Definition of done

- `cd web && npm run typecheck && npm run test && npm run build` green.
- Visually verified against a fixture run dir served by `undertow-ui serve` (include a screenshot in
  the PR body).
- `code-reviewer` APPROVE; PR from `feature-ui-live`; `STATUS.md` updated.
