# T12 — Regime labeller

**Wave 3 · size M · depends on: T01, T03 · blocks: T11, T14**
**Branch:** `feature-data-regimes`

## Why this task exists

Gap **G4** (regime robustness) is one of the four the thesis exists to close, and §10.3's third
non-negotiable evaluation practice is the regime split: *"report metrics per regime… 'beats passive' in a
quiet sideways market is a different claim from 'beats passive in a crash'."* §10.7 names
"non-reproducible regime labels" as a named risk, mitigated by *pre-committed thresholds plus a
sensitivity sweep*. This module is that mitigation.

The labels must be **producible by anyone** from public data — that is the standard §10.3 sets.

## Files you own

```
src/undertow/data/transforms/regimes.py
tests/data/test_regimes.py
tests/data/fixtures/reference_series.json
docs/decisions/001-regime-drift-definition.md
```

`docs/decisions/001-regime-drift-definition.md` is the ADR you own here (`PLAN.md` §0.3). Write a short
standalone version — context, the dimensional-inconsistency problem, the decision, the consequence — that
makes sense to someone who has never seen this plan, and cite `docs/decisions/001-…` from
`window_drift`'s docstring.

You consume `REFERENCE_SCHEMA` tables. You do not fetch anything, and you do **not** import
`fetchers/reference.py` — you code against the **schema** (T03), which is why you can run in wave 3
alongside T08 (see `PLAN.md` §3.1 on code vs data dependencies).

**You therefore own your own fixture.** `reference_series.json` needs three `REFERENCE_SCHEMA` series,
built by hand to the schema: (a) flat-for-30-days-then-jump, for the look-ahead test; (b) a crash series
(a ~30% single-day drop with elevated vol); (c) a series containing gap-filled bars. Do not wait for or
import T08's fixture.

## Required reading

- `~/Documents/fyp/lesson_plan/10_data_evaluation_roadmap.md` §10.3 ("Defining regimes concretely").
- `~/Documents/fyp/lesson_plan/09_research_gap_problem_statement.md` §9.2 G4 — the four canonical bins
  and the reason they must come from the reference feed rather than the pool tick.
- **`docs/decisions/001-regime-drift-definition.md`** — read this before you write
  `window_drift`. It changes the roadmap's formula for `μ`, deliberately, and explains why.

## What to build

Implement `CONTRACTS.md` §6.3.

### 1. The two statistics
- `realized_volatility(closes, periods_per_year)` = `std(log returns, ddof=1) * sqrt(periods_per_year)`.
  With minute bars, `periods_per_year = 365 * 1440 = 525_600`. Sample std (`ddof=1`), stated in the
  docstring — the population/sample choice changes the third decimal and someone will ask.
- `window_drift(closes)` = `log(closes[-1] / closes[0])`, the **total** log return over the window.
  **Per ADR 001**, not the per-step mean: the roadmap's `μ = (1/T) Σ log(S_t/S_{t−1})` thresholded at
  ±5% is dimensionally inconsistent (a 5% mean *per-minute* log return is `e^2160` over 30 days), so
  every window would label `sideways` and the regime split would be vacuous. Cite the ADR in the
  docstring.

### 2. Labelling
`label_regime(sigma_rv, mu, cfg)` implements §10.3's decision order **exactly**, and the order matters:
1. `sigma_rv > cfg.vol_threshold` → `HIGH_VOL` (vol dominates — a crash is a crash regardless of drift)
2. else `mu > cfg.drift_threshold` → `BULL`
3. else `mu < -cfg.drift_threshold` → `BEAR`
4. else → `SIDEWAYS`

Comparisons are strict (`>`, `<`) so a value exactly on a threshold falls to the *less extreme* label.
Test all three boundaries at exactly the threshold value.

### 3. `label_series(reference, cfg)` — the rolling labeller
- Window: `cfg.lookback_days` of minute bars, **backward-looking, closed on the right**. The label at
  time `t` uses only bars with `close_time <= t`. This is the look-ahead guard, and it is the single most
  important property of this module.
- Rows whose window is not yet full (the first 30 days) get `regime = Regime.UNKNOWN` and
  `window_complete = False`. **They are not dropped.** Dropping them silently shortens the dataset and
  makes downstream row counts mysteriously disagree.
- Gap-filled reference bars (`is_gap_filled = True`) participate in the window but must be counted: emit
  a warning if more than a configurable fraction (default 1%) of a window's bars were filled, because a
  vol estimate over forward-filled prices is biased **downward** (zero returns) and could mislabel a
  crash as sideways. This is a real trap; handle it explicitly.
- Output conforms to `REGIME_SCHEMA`. One row per input bar (so the as-of join in T11 is dense), or per
  a configurable stride — if you use a stride, default it to 1 and document that T11's as-of join is
  backward-looking so a coarser stride is safe but lossy.
- Efficiency: three years of minute bars is ~1.6M rows and the window is 43,200 bars. A naive O(n·w)
  loop is ~7e10 operations. Use a rolling/streaming computation (Welford or a cumulative sum of `x` and
  `x²` over log returns) — and be careful: the cumulative-sum approach loses precision over 1.6M terms.
  Prefer polars' native rolling aggregations, or Welford with a compensated summation. **Add a test
  that the fast path matches a naive brute-force implementation on a small window.**

### 4. `sensitivity_sweep(reference, cfg, vol_grid, drift_grid)`
Returns a table of `(vol_threshold, drift_threshold, regime, share_of_time, n_windows)` over the grid —
the §10.3 pre-commitment sweep that answers *"does the headline survive ±10% on the cutoff?"*. This is a
deliverable in its own right; the thesis reports it.

## Tests you must write

1. `realized_volatility` on a synthetic series with known constant per-step log return `σ` matches
   `σ * sqrt(periods_per_year)` analytically; `ddof=1` verified against a hand-computed 5-point series.
2. `realized_volatility` on a constant price series is exactly `0.0`.
3. `window_drift` on a series doubling over the window equals `log(2)`; on a flat series, `0.0`;
   on a halving series, `-log(2)`.
4. `label_regime` truth table: all four labels reachable, with hand-picked `(sigma_rv, mu)` pairs.
5. Precedence: `sigma_rv = 1.2, mu = +0.5` → `HIGH_VOL`, **not** `BULL`. This asserts the decision order.
6. Boundaries: exactly `sigma_rv == vol_threshold` → not `HIGH_VOL`; exactly `mu == drift_threshold` →
   not `BULL`; exactly `mu == -drift_threshold` → not `BEAR`.
7. **Look-ahead test (mandatory):** build a reference series that is flat for 30 days then jumps 50%.
   Assert every label at times *before* the jump is unaffected by it — concretely, label the truncated
   series (ending just before the jump) and the full series, and assert the shared prefix of labels is
   **identical**. This is the test that proves the window is backward-looking.
8. Incomplete window: the first `lookback_days` of output are `UNKNOWN` with `window_complete=False`,
   and the row count equals the input row count (nothing dropped).
9. Fast path vs brute force: on a 500-bar series with a 60-bar window, the rolling implementation
   matches a naive per-window recomputation to within 1e-10.
10. Gap-fill warning: a window where 5% of bars are `is_gap_filled` triggers the warning; 0.1% does not.
11. `label_series` output conforms to `REGIME_SCHEMA`; timestamps UTC-aware.
12. `sensitivity_sweep`: shares sum to 1.0 (± float tolerance) within each threshold combination; a
    higher `vol_threshold` monotonically decreases the `HIGH_VOL` share.
13. A crash fixture (a real-shaped 30% single-day drop with elevated vol) labels `HIGH_VOL` — the
    end-to-end sanity check that the thresholds mean what §10.3 intends.

## Acceptance criteria

- `uv run pytest tests/data/test_regimes.py` green; `mypy` clean.
- Thresholds come from `RegimeConfig` only. A magic number in this module is a rejection.
- The ADR-001 deviation from the roadmap's `μ` formula is cited in the docstring, so a reader of the code
  is never confused about which formula is implemented.

## Handoff notes for your PR body

- The regime shares observed over the pinned window if you ran it on real data (bull/bear/sideways/
  high_vol percentages). The human will want to see that all four bins are non-empty — if one is, the
  thresholds need revisiting **before** any RL training, and that is a finding to surface now.
- The rolling implementation chosen and its measured runtime on 1.6M rows.
- Whether a stride other than 1 was used.

## Process (mandatory)

Branch → implement → tests green → `code-reviewer` → fix → commit → push → PR → update `STATUS.md`.
