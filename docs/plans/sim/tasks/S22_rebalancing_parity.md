# S22 — Decision-time execution + trustworthy rebalancing parity (revision 2)

> **Supersedes the first S22 attempt**, which the code-reviewer rejected: the
> parity gate passed only on a hand-picked fixture and gave false passes on a
> one-step execution shift. This revision fixes the root cause (ADR-031) and
> reworks the gate to be trustworthy. The first attempt's plumbing is kept where
> the review did not object.

**Wave 6 · size L · depends on: S14, S18 · blocks: S20, S21, S23**
**Branch:** `feature-sim-rebalance-parity` (off `bug-sim-rebalance-swap-cost`; rebase onto `main` once #90 merges)

## Why this task exists

`LpEnvironment.step()` advances the market **before** executing the action, so a
decision made at boundary `D_{i-1}` executes at `D_i` — one 10-minute bar late.
The backtester executes at the decision event. The S22 review reproduced a false
parity pass on a mid-episode jump and showed the test fixture was hand-picked to
hide the divergence. ADR-031 fixes the simulator to execute at the decision
price; this task implements that and makes the parity gate prove it.

## Files you own

```
src/undertow/sim/env/lp_env.py              (reset/step ordering only; _rebalance body is S18's)
src/undertow/sim/validation/parity.py
src/undertow/sim/validation/__init__.py
tests/sim/test_env.py
tests/sim/test_validation.py
tests/sim/test_backtester.py                (only where the reorder changes expectations)
tests/sim/test_evaluation.py                (only where the reorder changes expectations)
docs/plans/sim/CONTRACTS.md                 (§12 step-ordering note)
```

The worktree already contains the rejected first attempt (uncommitted). Keep its
`_Decision` / `_assert_decisions_match` / gas-slippage plumbing where useful, but
fix the findings below.

## What to build

### 1. Decision-time execution (ADR-031) — `lp_env.py`

Reorder `step()` to:

1. **Capture** the decision state: `pool_engine.state.sqrt_price`/`tick`, the
   mark `price` the observation was valued at, the decision tape row and time.
2. **Execute** the rebalance at that state (gas/slippage from the decision tape
   row). On success set `_last_fees_usdc = 0` and
   `_last_il_total = _current_il_total(decision_sqrt, decision_price)`. A
   rejected rebalance leaves the position untouched. Remove the
   `price_exhausted` skip — the action no longer needs a next bar.
3. **Advance** one interval (`_advance_replay` / `_advance_calibrated`), accruing
   the interval's events to the post-action position.
4. **Measure** on the post-action position: `fees_earned`, `il_change`, equity,
   reward, observation.

`reset()` must leave a fully-populated decision state — set the mark price, tape
row and time from the entry bar (today `_last_tape_row` is `{}` until the first
advance, which would price first-step gas from an empty row). The
`self._last_mark_price`/`_last_tape_row`/`_last_at_time` fields are the decision
state; document that in the step docstring.

Do not change `_rebalance`'s body, the reward function, or the observation
builder.

### 2. Parity harness — `parity.py`

Keep `run_parity_check(..., *, policy=None)` with HOLD default and
`policy=None` byte-identical to the old behaviour.

- **Executed-action guard.** Replace the observation-time comparison with a check
  that the action **executed in each numeric bucket** matches: for each decision
  index, the sim's executed action (type + params + executed/rejected) must equal
  the backtester's decision-log entry for the same bucket. Raise `ParityError`
  naming the first divergent index. This is the guard the first attempt lacked.
- **Per-step pass criterion.** `passed` must require the per-step gas and
  slippage series to be equal within tolerance, not just their means: check
  `max|drift|` (and keep the mean/std fields for reporting). A one-step shift
  must fail.
- **Provenance.** When `policy is not None`, **always re-run** the backtester over
  the identical window with that policy — do not reuse the supplied ledger
  (policy `name` alone is not a fingerprint, and same-named policies with
  different parameters would produce false parity). The supplied ledger is only
  reused on the `policy=None` HOLD path, exactly as today.
- **Tolerant bucketing.** The sim's decision times are reference-bar closes; the
  backtester's are first-event times. Bucket with the existing `<=` rule and
  assert a one-to-one bucket mapping; do not require exact datetime equality
  (which would raise spuriously on real lanes).
- Keep `ParityReport`'s original fields first; the appended gas/slippage fields
  stay defaulted.
- **RNG note.** The sim seeds the policy RNG with `default_rng(seed)`, the
  backtester with `config.rng()`. This task uses a deterministic policy
  (`TauResetPolicy`); add a docstring note that a stochastic policy needs the
  seeds threaded, and leave that to S20.

### 3. Tests

**Sim timing (`tests/sim/test_env.py`):**
1. A rebalance issued on the first `step()` executes at the **entry** price (not
   the next bar's): assert the resulting `tick_lower`/`tick_upper` are centred on
   the entry tick and the gas cost equals the decision row's block price.
2. A mid-episode rebalance executes at the decision price: build a path with a
   step change, rebalance, and assert the position's entry price equals the
   decision mark (within float tolerance).
3. `_last_tape_row` is non-empty immediately after `reset()` (first-step gas is
   priced from the entry row, not block 0).
4. Update every existing env test whose expectations change because of the
   reorder; explain each change in the test or the PR.

**Parity (`tests/sim/test_validation.py`):**
5. **Real mid-episode rebalance on a moving path.** Replace `_REBALANCE_PRICES`
   with a path whose first bar moves and whose step change triggers a genuine
   mid-episode `TauResetPolicy` rebalance with a non-zero swapped imbalance.
   Assert `passed=True`, gas/slippage max drift `< 1e-6`, and that the decision
   log contains a mid-episode (not step-0) rebalance with non-zero slippage.
6. **A one-step shift fails.** Construct/mock a case where the sim and
   backtester execute the same action one bucket apart and assert `passed=False`
   (the first attempt returned `True`).
7. **Divergent executed actions raise** `ParityError` naming the index.
8. **HOLD unchanged**: `policy=None` gives the same numeric fields as before.
9. Guard-branch coverage: a decision-log step outside the window, and duplicate
   decisions at one timestamp.

## Acceptance criteria

- `uv run pytest tests/sim/test_env.py tests/sim/test_validation.py tests/sim/test_backtester.py -q -n 4` green (use `--no-cov` or `--cov-fail-under=0` for the subset; the repo's global 90% gate only passes on the full suite).
- Full suite `uv run pytest -q -n 4` green with the 90% gate.
- `policy=None` behaviour unchanged.
- Report the measured rebalancing fee/IL/gas/slippage drifts as numbers in the PR.
- CONTRACTS §12 step-ordering note updated.

## Out of scope

- The backtester (it is already decision-time and stays unchanged).
- The hedge leg (S20) — but the harness must accept any deterministic `Policy`.
