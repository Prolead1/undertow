# U09 — Policy Replay / filmstrip view

**Wave 5 · size L · depends on: U07 · blocks: U11**
**Branch:** `feature-ui-replay`

## Why this task exists

The cockpit shows training progress; this view shows what a trained checkpoint *actually did*. It is
the most domain-specific screen: the agent's concentrated-liquidity range over the price path, the
rebalances it chose, and the costs it paid. This is the "very interesting" part of the visualization.

## Files you own

```
web/src/views/PolicyReplay.tsx
web/src/components/replay/RibbonChart.tsx
web/src/components/replay/EquityVsHodl.tsx
web/src/components/replay/CostWaterfall.tsx
web/src/components/replay/EpisodeFilmstrip.tsx
web/src/components/replay/StateInspector.tsx
web/src/__tests__/replay.test.tsx
```

## What to build

Read `CONTRACTS.md` §2 (rollout schema), §7 (rollout endpoints) first.

1. **Selector.** Run → seed → episode (from `GET /api/runs/{id}/rollouts`); selection lives in the
   query string so a view is shareable. Missing rollouts → `EmptyState` explaining
   `training.record_rollouts = true`.

2. **RibbonChart** — the hero. Price on a log axis with the position `[tick_lower, tick_upper]` drawn
   as a band, recolored by `in_range`; rebalance steps marked (size ∝ `gas`); HODL counterfactual;
   regime bands in the background. A shared time cursor (from the shell store) highlights the same
   step across every panel.

3. **EquityVsHodl** — equity vs a HODL baseline computed client-side from the price series and the
   initial capital; drawdown fill below, sharing the x-axis/cursor.

4. **CostWaterfall** — end-of-episode decomposition (fees → gas → slippage → IL → net) plus a stacked
   area of per-step components over time.

5. **EpisodeFilmstrip** — small-multiple mini equity curves for every episode of the selected
   seed/run, colored by dominant regime; click to load into the main panels. If there are many
   episodes, virtualize or cap with a "show all" toggle.

6. **StateInspector** — at the cursor step, show the action taken (`hold` vs `rebalance(offset,width)`),
   in/out-of-range, position value, gas price, and regime. If the observation vector is available in
   the rollout, render the 18 dims as labelled sparklines.

## Tests (`web/src/__tests__/replay.test.tsx`)

Mock the API:
1. Selecting seed/episode updates the URL and fetches the right rollout.
2. Ribbon receives the right `tick_lower/tick_upper` series (assert via the figure payload passed to
   the mocked Plotly wrapper) and marks exactly the rebalance rows.
3. HODL baseline math: a known price series yields the hand-computed baseline.
4. Cost waterfall components sum to the displayed net.
5. Filmstrip renders one tile per episode and clicking changes the selection.
6. Missing rollouts → `EmptyState`, no crash.

## Definition of done

- `cd web && npm run typecheck && npm run test && npm run build` green.
- Visually verified against a fixture run produced with `training.record_rollouts = true` (screenshot
  in the PR body).
- `code-reviewer` APPROVE; PR from `feature-ui-replay`; `STATUS.md` updated.
