# ADR 001 — Regime drift `μ` is the total window log return

**Status:** accepted (planner decision, pre-committed before any data is seen)
**Source of truth:** this file is the canonical ADR-001. T12 writes the standalone final version here and cites *that* path from `regimes.py`'s docstring (see `PLAN.md` §0.3).
**Affects:** T12 (`transforms/regimes.py`), the thesis text of `lesson_plan/10_data_evaluation_roadmap.md` §10.3

## Context

§10.3 of the roadmap defines the regime statistics from the reference feed's minute closes over a
30-day rolling window as:

- `σ_rv = std(log(S_t / S_{t-1})) * sqrt(365 * 1440)` — annualized realized volatility
- `μ = (1/T) Σ log(S_t / S_{t-1})` — described as "the 30-day log drift"

and then thresholds them: high-vol if `σ_rv > 80%`; else bull if `μ > +5%`, bear if `μ < −5%`, sideways
if `|μ| ≤ 5%`.

## Problem

The two are dimensionally inconsistent. `(1/T) Σ log(S_t/S_{t−1})` is the **mean per-minute** log
return. Over a 30-day window `T = 43,200`, so a mean per-minute log return of `0.05` corresponds to a
total log return of `2160` — a factor of `e^2160`. The ±5% threshold is obviously intended for a
*30-day* move, not a per-minute one. Implemented literally, every window on earth is labelled
`sideways` and the regime split — the load-bearing evidence for gap **G4** — becomes vacuous.

## Decision

Define

```
μ₃₀ = Σ_t log(S_t / S_{t−1}) = log(S_end / S_start)
```

i.e. the **total log return across the window**, which is the quantity ±5% is meaningful for. `σ_rv`
keeps the roadmap's definition unchanged (it is already annualized and internally consistent).

`RegimeConfig.drift_threshold = 0.05` therefore means "the reference price moved more than ~5% (in log
terms) over the trailing 30 days".

## Consequences

- `transforms/regimes.py::window_drift` returns `log(closes[-1] / closes[0])`, and its docstring cites
  this ADR.
- Thresholds stay pre-committed in `config.py` and are swept by `sensitivity_sweep` per §10.3, so the
  headline's robustness to the cutoff is reportable either way.
- The thesis chapter should be corrected: replace `μ = (1/T) Σ log(S_t/S_{t−1})` with
  `μ = Σ log(S_t/S_{t−1}) = log(S_T/S_0)` and call it the *cumulative* 30-day log return. **This is a
  vault edit, not a repo edit** — flag it to the human; do not edit the lesson plan from a code task.
- If the human prefers to keep the per-step mean, the fix is to annualize it instead
  (`μ_ann = mean * 365 * 1440`) and threshold at an annualized number. Either is defensible; what is not
  defensible is a per-minute mean thresholded at 5%.
