# ADR 027 — Leaderboard/Compare contract gaps: defensive `n_rebalances`, URL-only pinning

**Status:** accepted
**Affects:** `web/src/views/Compare.tsx` · `web/src/views/Leaderboard.tsx` ·
`web/src/components/leaderboard/LeaderboardTable.tsx` · `web/src/components/leaderboard/pinned.ts` ·
`web/src/components/leaderboard/ranking.ts` · `web/src/components/leaderboard/csv.ts` ·
`web/src/components/leaderboard/fingerprint.ts` · `web/src/App.tsx` · `docs/plans/ui/CONTRACTS.md` §4/§8 · U10–U11
**Raised by:** U10 (leaderboard)

## Context

`CONTRACTS.md` §4 freezes the eval payload and the run record:

```python
class EvalMetrics(BaseModel):
    """Parsed from a results/ directory when present."""
    sharpe: float | None = None
    total_pnl: float | None = None
    annualized_return: float | None = None
    max_drawdown: float | None = None
    worst_regime_pnl: float | None = None
    per_regime: dict[str, float] = {}      # regime -> mean total_pnl
    source: str | None = None              # relative path the numbers came from
```

`n_rebalances` is **not** a member of `EvalMetrics` or `RunRecord`, and the committed
`web/src/api/types.ts` (generated from `web/openapi.json`) therefore has no field for it.

The U10 brief asks for both of these anyway:

> **3. PolicyFingerprint.** A radar (or small-multiple bars) of normalized metrics (Sharpe, PnL,
> MaxDD, worst-regime PnL, **n_rebalances**) per selected run … build it client-side from
> `EvalMetrics`.

> **4. Compare.** Pin up to 6 runs (**from the global pin tray** or row checkboxes) …

The "global pin tray" is `App.tsx`'s sidebar, which keeps `pinned` in local `useState` and is not
backed by the URL. `CONTRACTS.md` §8 also names a Zustand store as the owner of the pinned
comparison set, but ADR-026 explicitly deferred installing Zustand/TanStack Query to U08, and U08 is
not on this branch.

## Problem

1. **A missing wire field must not be treated as present.** If the fingerprint reads
   `run.eval.n_rebalances` directly, `tsc` rejects it against the generated types; a cast that
   *assumes* the field exists would render a permanently empty "Rebalances" axis against today's
   API, and a future server adding the field could still return it at the top level rather than
   nested in `eval`. The view needs a shape that is correct today and lights up cleanly if the field
   is ever added.
2. **Two unconnected pin sets are worse than one.** The Compare view resolves its selection solely
   from the URL (`?run_ids=a,b,c`). `App.tsx`'s tray toggles a local array that Compare never reads,
   and rows pinned in Compare never appear in the tray. Wiring them without a shared store would
   mean either duplicating URL state into `App` or lifting `App`'s tray into a context that no other
   view yet needs.

## Decision

1. **Treat `n_rebalances` as an optional, additive extension and degrade gracefully.**
   `web/src/views/Compare.tsx` reads it through one defensive helper (`rebalanceCount`) that accepts
   the field either at the top level of `RunRecord` or nested in `eval`, and only returns a finite
   number. The fingerprint field list includes the "Rebalances" axis **only when at least one
   selected record carries a finite value**; otherwise it falls back to `annualized_return`
   ("Annualised return"), which *is* in the frozen `EvalMetrics`. No `CONTRACTS.md` or `schemas.py`
   edit is made — the axis activates automatically if U02/U05 ever add the field.
2. **The URL is the single source of truth for the pinned set in U10.** The leaderboard row
   checkboxes and Compare's picker both read/write `?run_ids=` via the pure helpers in
   `web/src/components/leaderboard/pinned.ts`. `App.tsx`'s sidebar tray stays local UI state and is
   **not** wired to Compare on this branch. A cross-view pinned store is deferred to **U11**, which
   already owns the integration/docs pass and can adopt ADR-026's deferred Zustand store once more
   than one view consumes it.

## Consequences

- **Code.** `Compare.tsx` gains the `rebalanceCount`/`fingerprintFields` fallback; the pinned-set and
  ranking/CSV/fingerprint helpers move into `pinned.ts`, `ranking.ts`, `csv.ts` and `fingerprint.ts`
  (non-component modules, so the `react-refresh/only-export-components` ESLint gate stays at zero
  warnings). The CSV `rank` column is taken from `rankRuns` positions so it always reflects the
  pinned rule, not the visible row index.
- **U11 must know** that the sidebar pin tray and Compare are intentionally not synchronised, and
  that it owns the cross-view store decision. Until then a comparison link is the hand-off between
  the two views.
- **`CONTRACTS.md` needs no immediate edit.** §4's `EvalMetrics` stays frozen and §8's Zustand
  sentence stays the U08/U11 target. This ADR only timestamps two U10 deviations.
- **Vault edits are flagged, not made.** The U10 brief's `n_rebalances` and "global pin tray" wording
  describes the intended end state; no thesis text under `~/Documents/fyp/lesson_plan/` was touched.
- **Rejected:** adding `n_rebalances` to the frozen `EvalMetrics` (U02 owns `schemas.py`, it would
  churn every generated client and is not needed to ship the leaderboard); installing Zustand in U10
  (per ADR-026, deferred until the first genuinely shared state; a single-view store here would be
  speculative and untested); and making Compare scrape `App`'s tray through props/context (couples
  the views without removing the underlying dual-source problem).
