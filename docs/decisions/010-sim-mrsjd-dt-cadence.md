# ADR 010 — MRSJD step size: `dt` is the 10-minute cadence, not the literal `1/(6·365.25)`

**Status:** accepted
**Affects:** `src/undertow/sim/prices/regime_jump.py` · sim `CONTRACTS.md` §7 · tasks S06, S12
**Raised by:** S06 (`feature-sim-price-processes`)

## Context

`docs/plans/sim/CONTRACTS.md` §7 freezes the calibrated price-process constructor as:

```python
def __init__(self, params: MRSJDParams, dt: float = 1 / (6 * 365.25)) -> None: ...
```

and `docs/plans/sim/tasks/S06_price_processes.md` repeats the same expression while describing it:
*"`dt` defaults to 10 minutes in annualized units."*

Everything else in the plan pins a **10-minute** decision cadence, not the 4-hour step the literal
expression evaluates to:

- `docs/plans/sim/PLAN.md` (Pinned decisions): *"Decision cadence | every **10 minutes** (10 reference
  bars in sim; nearest tape block boundary in backtest)"* and *"Episode | … 10-min steps"*.
- `src/undertow/sim/config.py` `EpisodeConfig.step_minutes = 10  # decision cadence`.
- `src/undertow/sim/metrics/performance.py`: `DEFAULT_PERIODS_PER_YEAR = 365 * 24 * 6` — six
  ten-minute periods per hour, i.e. 52 596 periods/year.

## Problem

The literal formula is arithmetically wrong for a 10-minute cadence:

```
1 / (6 * 365.25)      = 1 / 2191.5   = 4.5631e-04 years  = 4 hours
1 / (6 * 24 * 365.25) = 1 / 52596    = 1.9013e-05 years  = 10 minutes
```

The factor is exactly 24. Implementing the literal value would annualise the fitted drift,
diffusion and jump intensity against a 4-hour step while the simulator advances 10 minutes:
at simulation time every fitted parameter would be wrong by 24× (drift and diffusion applied over
a step 24× too long relative to their calibration). This is not a cosmetic mismatch — it corrupts
the generative model that S12 trains on. The error is invisible from the type signature alone, so
S12 coding against the frozen text would silently inherit it.

## Decision

`CalibratedPriceProcess`'s default step is the **10-minute** cadence:

```python
DEFAULT_DT: float = 1.0 / (6.0 * 24.0 * 365.25)   # 10 minutes in years
```

`MRSJDParams` carries annualised `drift`/`diffusion`/`jump_intensity`; the constructor's `dt` is the
step over which they are applied, and defaults to `DEFAULT_DT`. `fit()` derives the annualisation
`dt` directly from the reference bars' own timestamps (`_decision_dt_years(close_times, stride)`),
so a 1-minute feed yields exactly `DEFAULT_DT`; no cadence is hardcoded. The module docstring records
the slip, and tests pin both the default (`test_default_dt_is_ten_minutes`) and the
timestamp-derived value (`test_decision_dt_years_tracks_bar_spacing`).

`CONTRACTS.md` §7 should be edited to `dt: float = 1 / (6 * 24 * 365.25)` (or simply to reference
`regime_jump.DEFAULT_DT`). The vault lesson-plan text is **not** touched here.

## Consequences

- `src/undertow/sim/prices/regime_jump.py`: `DEFAULT_DT = 1/(6·24·365.25)`; `fit` annualises with
  the timestamp-derived `dt`; docstring documents the correction.
- **S12 (env)** must construct `CalibratedPriceProcess` with the default (or its own 10-minute `dt`)
  and expect `DEFAULT_DT == 1.9013e-5` years, matching `EpisodeConfig.step_minutes = 10` and
  `metrics.performance.DEFAULT_PERIODS_PER_YEAR`.
- **S14 (parity)** already matches decision boundaries every 10 minutes, so this keeps the sim
  cadence identical to the backtester's.
- `CONTRACTS.md` §7 needs the one-token edit; this is flagged in the S06 PR, not edited by the code
  task.
- **Rejected — implement the literal `1/(6*365.25)`:** it would make the fitted MRSJD 24×
  inconsistent with the simulator step and contradict the plan's own pinned 10-minute cadence,
  `EpisodeConfig`, and the metrics annualisation. There is no reading under which a 4-hour step is
  intended.
- **Rejected — keep the literal value in the signature but override it in `fit`/the env:** two
  different `dt`s would coexist, and a caller omitting `dt` would still get the wrong 4-hour step.

## Related deviation (σ = 0) recorded here for S12

The S06 brief's §3 validation rule *"σ > 0 for all states"* contradicts its own required tests #5
(`σ = 0`, no jumps ⇒ constant price) and #6 (known drift with `σ = 0`). `MRSJDParams.__post_init__`
therefore requires `σ >= 0` (rejecting only negative diffusion) as the pure-drift / pure-jump limit;
all **fitted** values are strictly positive (variance floor). `CONTRACTS.md` §7 imposes no positivity
constraint, so this does not amend the contract. Pinned by
`test_allows_zero_diffusion_for_deterministic_limit`.
