# ADR 015 — Anchor the calibrated MRSJD to the data module's regime labels

**Status:** proposed (implementation shipped on `bug-sim-mrsjd-calibration`; human accepts in PR review)
**Affects:** `src/undertow/sim/prices/regime_jump.py` · sim `CONTRACTS.md` §7
**Raised by:** training-readiness audit (calibrated price mode)

## Context

`PLAN.md` §7 pins the calibrated price mode as "Markov-regime-switching
jump-diffusion … 4 latent states matching the regime labels, Student-t jump sizes,
fit on the **train** split only", and pins regime labels as "consumed from the
data module's regime table only; **never recomputed in sim**".

The shipped `CalibratedPriceProcess.fit` instead fitted a **Gaussian-emission
HMM** with Baum-Welch EM directly on 10-minute log returns and then *post-hoc*
assigned the four data-label names by drift/volatility order. So the sim did
recompute its own regimes rather than anchor to the data module's.

## Problem

On the real `eth_usdc_3000` train split, the unsupervised fit is degenerate. On
clean single-symbol ETHUSDT data (after ADR-014) it still produces an
economically absurd parameter set:

```
per-state drift (log/yr):  ( 84.7, -86.9,  11.5, -10.7)   ~350x the global mean
transition matrix:          near-uniform (~0.25 rows)       no regime persistence
jump_intensity (/yr):      (2465, 2530, 0, 0)
diffusion (vol/yr):        (0.80, 0.82, 0.47, 0.47)
```

The global annualised log-mean is a sane `-0.24`; the per-state means are ~350×
that and cancel almost exactly through near-uniform switching. This is the
classic Gaussian-HMM-EM degeneracy on near-zero-mean, fat-tailed financial
returns: the likelihood is improved by alternating large-opposite-mean states at
high frequency. A persistence prior does **not** fix it — re-running the EM with
a sticky transition initialisation (`0.97` diagonal) and a Dirichlet diagonal
pseudo-count (`50`) still yields per-state drifts of `-193` / `+195` / `-7.7` /
`+8.5` per year.

By contrast, the data module's own regime labels are persistent and their
volatility structure is sensible. Conditioning each 10-minute return on the
regime label at the interval's **start** (`MarketView.regime_at(t_i)` for the
return over `(t_i, t_{i+1}]`) over the same train split gives:

```
regime      n        ann drift   ann vol
bull       29 874       +0.50      0.56
bear       14 105       -1.09      0.62
sideways   24 585       +0.55      0.43
high_vol   32 091       -0.62      0.92
```

with an empirical transition matrix whose diagonals are `[0.9937, 0.9852,
0.9842, 0.9996]` (mean dwell ≈ `[1.1, 0.5, 0.4, 17.8]` days) — the persistent
regime structure the process is meant to model.

**Why the start label, not the end.** `regime_at` uses a trailing 30-day
window, so the label at the *end* of `(t_i, t_{i+1}]` already contains that
return: conditioning on it leaks the target into its own label and inflates the
apparent signal. Using the **end** label gives drifts of `(+2.05, −2.98, −0.31,
−0.58)` — roughly 3–4× the leakage-free bull/bear magnitudes. The start label is
the only one computable before the return. Consequently the labels predict
**volatility** strongly (`high_vol` 0.92 vs `sideways` 0.43) but
next-10-minute **direction** only weakly and not monotonically in the label
names; that is the data's honest signal, and the model reflects it rather than
the leakage artefact.

## Decision

`CalibratedPriceProcess.fit` anchors its states to the data module's regime
labels:

1. Build 10-minute log returns as today (`stride = step_minutes`, `dt` from the
   feed's own cadence).
2. As-of join `MarketView.regime_at(t_i)` for the return over `(t_i, t_{i+1}]` —
   the label at the interval **start**. It is backward-looking (not look-ahead)
   and, unlike the end label, is not contaminated by the return being fitted.
3. If **every** one of `("bull", "bear", "sideways", "high_vol")` has at least
   `_MIN_LABELLED_RETURNS_PER_STATE = 30` returns, fit the MRSJD from the labels:
   - `drift_s = mean(r_s) / dt`, `diffusion_s = std(r_s) / sqrt(dt)` (floored);
   - jump tail/`Student-t` parameters per state exactly as before, from
     residuals beyond `3σ_s`, with `intensity = tail_count / (n_s · dt)`;
   - the transition matrix from consecutive known-label transitions, with a
     `0.5` pseudo-count smoothing (rows always sum to 1).
4. Otherwise fall back to the existing unsupervised HMM + post-hoc ordering, so
   hand-built views without regime labels (and the dense synthetic fit test)
   still work.

The state order is the fixed `("bull", "bear", "sideways", "high_vol")` label
order; no post-hoc reordering is needed on the label path.

## Consequences

- Calibrated-mode training now samples a regime model whose per-state drifts and
  volatilities are pinned to the same labels the environment observes, rather
  than to an EM artefact. This is the "single source of truth" the plan asks
  for.
- The unsupervised HMM remains only as a fallback for label-less views; it is no
  longer on the production fit path. Its tests stay.
- `CalibratedPriceProcess.step` now draws each step's return from the state
  *entering* the step and then transitions, matching the start-label
  conditioning. This changes the RNG draw order (and therefore a seeded path)
  but not the process distribution. `MRSJDParams` and the
  `fit(market_view) -> MRSJDParams` signature are unchanged, so no consumer
  changes.
- When the labels are absent or too sparse for the anchored fit, `fit` logs a
  warning before running the HMM fallback (which is known to be degenerate on
  real data).
- Rejected alternatives:
  - *Keep the HMM, add regularisation.* Tested; the sticky-prior EM is still
    degenerate (above), so it does not fix the numbers.
  - *Fit latent states and never name them.* Still recomputes regimes in sim and
    leaves an unidentifiable ordering; the plan explicitly ties states to the
    labels.
  - *Use the synthetic daily-return HMM of the literature.* Changes the model's
    data basis and cadence, and discards the data module's regime work.
  - *Condition on the interval-end label to match `step`'s old
    transition-then-draw order.* Rejected as target leakage; instead `step` was
    aligned to the leakage-free start-label convention.
