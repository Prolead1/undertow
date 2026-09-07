# ADR 001 — Regime drift `μ` is the total window log return, not the per-step mean

**Status:** accepted (pre-committed by the planner before any data was seen)
**Affects:** `src/undertow/data/transforms/regimes.py` (T12) — `window_drift` and
`label_series`; and the thesis text of the lesson plan (§10.3 of
`~/Documents/fyp/lesson_plan/10_data_evaluation_roadmap.md`, a vault file).
**Raised by:** T12 (planner decision; T12 owns this record per `PLAN.md` §0.3/§2)

The plan references this decision under the shorthand path `adr/001-regime-drift-definition.md`;
the canonical copy is **this file**, `docs/decisions/001-regime-drift-definition.md`,
and `regimes.py` cites this path from `window_drift`'s docstring.

## Context

`undertow.data` produces per-regime evaluation labels — `bull`, `bear`, `sideways`,
`high_vol` — from the reference price feed (Binance klines) over a 30-day
**backward-looking** rolling window of minute closes, so that downstream results can be
reported *per regime* (thesis roadmap gap **G4**, regime robustness: "beats passive in a
quiet sideways market" is a different claim from "beats passive in a crash").

The thesis roadmap (§10.3 of the lesson plan, the source text this plan implements)
defines two statistics over a window of closes `S_1 … S_T`:

- `σ_rv = std(log(S_t / S_{t-1})) · sqrt(365·1440)` — annualized realized volatility;
- `μ = (1/T) Σ log(S_t / S_{t-1})` — described as "the 30-day log drift";

and thresholds them: high-vol if `σ_rv > 0.80`; else bull if `μ > +0.05`, bear if
`μ < −0.05`, sideways otherwise.

## Problem — the two are dimensionally inconsistent

`(1/T) Σ log(S_t/S_{t−1})` is the **mean per-minute log return**. A 30-day window has
`T = 43,200` minute steps, so a mean per-minute log return of `0.05` corresponds to a
**total** log return of `0.05 × 43,200 = 2,160` — a factor of `e^2160`. The ±5% cutoff
is plainly intended for a *30-day* move, not a per-minute one. Implemented literally,
every real window has `|μ|` of order `1e-5` per minute, so **every window on earth is
labelled `sideways`** and the regime split — the load-bearing evaluation for G4 —
becomes vacuous. This is a defect in the source text, not a tuning choice.

## Decision

Define the drift as the **total** log return across the window:

```
μ₃₀ = Σ_t log(S_t / S_{t−1}) = log(S_end / S_start)
```

This is the quantity a ±5% cutoff is meaningful for: `RegimeConfig.drift_threshold =
0.05` now reads "the reference price moved by more than ~5% (in log terms) over the
trailing 30 days". `σ_rv` is unchanged — it was already annualized and internally
consistent.

## Consequences

- `undertow.data`'s `window_drift` returns `log(closes[-1] / closes[0])` and cites this
  file; `label_series` thresholds on that quantity with the identical decision order.
- The 5% cutoff is a *30-day move* cutoff. All thresholds stay pre-committed in
  `config.RegimeConfig` and are swept by `sensitivity_sweep`, so the headline's
  robustness to ±10% on the cutoff is reportable either way.
- The thesis chapter must be corrected — **a vault edit, flagged to the human, never
  made from a code task**: replace `μ = (1/T) Σ log(S_t/S_{t−1})` with
  `μ = Σ log(S_t/S_{t−1}) = log(S_T/S_0)` and call it the *cumulative* 30-day log
  return.
- Rejected alternative: keep the per-step mean but annualize it
  (`μ_ann = mean · 365 · 1440`) and threshold an annualized number. Either choice is
  defensible as a coherent pair; what is not defensible is a per-minute mean
  thresholded at 5%. The total-log-return reading was chosen because it needs no new
  number and matches the intuitive "the market moved X% over the last 30 days".
