# U10 — Leaderboard + Compare view

**Wave 5 · size M · depends on: U07 · blocks: U11**
**Branch:** `feature-ui-leaderboard`

## Why this task exists

"Which policies are best so far?" needs a ranked, provenance-visible answer and a way to put two or
more policies side by side. This is also the view that makes the thesis comparison defensible: the
ranking rule is explicit and every row is traceable to a config hash and commit.

## Files you own

```
web/src/views/Leaderboard.tsx
web/src/views/Compare.tsx
web/src/components/leaderboard/LeaderboardTable.tsx
web/src/components/leaderboard/PolicyFingerprint.tsx
web/src/components/leaderboard/ProvenanceCard.tsx
web/src/__tests__/leaderboard.test.tsx
```

## What to build

Read `CONTRACTS.md` §4 (`RunRecord`, `EvalMetrics`), §7 (`/api/runs`, `/api/compare`) first.

1. **LeaderboardTable.** Sortable columns: rank, run id, experiment/seed, state, `best_reward`,
   eval Sharpe, total PnL, MaxDD, worst-regime PnL, config hash (short), git commit (short), updated
   at. Default sort follows the pinned rule (eval Sharpe → worst-regime PnL → `best_reward`); the
   column header tooltips explain it. Rows without eval artifacts are visibly marked "train-only" and
   sorted after evaluated runs. A sparkline of the run's learning curve sits in an expandable row.
   Filter by state and by experiment; search by run id.

2. **ProvenanceCard.** On row expand: full config hash, git commit, library versions, checkpoint path,
   and a copy button. This is the answer to "can I trust this number?".

3. **PolicyFingerprint.** A radar (or small-multiple bars) of normalized metrics (Sharpe, PnL, MaxDD,
   worst-regime PnL, n_rebalances) per selected run; if a `radar` figure is added to U04's registry
   later, consume it — otherwise build it client-side from `EvalMetrics`. Normalize per-metric across
   the selected set and label axes clearly.

4. **Compare.** Pin up to 6 runs (from the global pin tray or row checkboxes); render
   `POST /api/compare` equity overlay + a metric table + the fingerprints side by side. Guard the
   ≤6 limit in the UI (disable further pinning with a tooltip) and rely on the API's 422 as the
   backstop. When fewer than 2 are pinned, explain how to add more.

5. **Export.** A "Download CSV" button serialising the visible ranking (client-side) so the thesis can
   cite the exact leaderboard. No server write.

## Tests (`web/src/__tests__/leaderboard.test.tsx`)

Mock the API:
1. Default sort follows the pinned rule on a fixture with mixed eval coverage.
2. Train-only runs sort after evaluated runs and show the marker.
3. Search/filter narrows rows correctly.
4. Pinning 7 runs is prevented client-side; 6 is allowed.
5. Compare calls the API with the pinned ids and renders one legend entry per run.
6. CSV export contains a header plus one row per visible run.
7. ProvenanceCard shows the full hash/commit (not truncated).

## Definition of done

- `cd web && npm run typecheck && npm run test && npm run build` green.
- Ranking rule matches `PLAN.md` §7 and is visible in the UI.
- `code-reviewer` APPROVE; PR from `feature-ui-leaderboard`; `STATUS.md` updated.
