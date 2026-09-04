# T13 — Validation & cross-check suite

**Wave 5 · size L · depends on: T02, T05, T06, T10, T11, T15 (its `dune_expectations` module) · blocks: T14, T16**
**Branch:** `feature-data-validation`

## Why this task exists

Roadmap §10.7: *"Treat the data loader's correctness as its own deliverable with tests, not an
afterthought."* §10.6's Week-2 milestone requires *"verify fee accrual matches on-chain `Collect`
events"*. §10.1.5 assigns RPC the job of *verifying* the numbers the backtester depends on. This module
is where all of that becomes a pass/fail artifact.

The thesis's credibility on gap **G3** reduces to one question: *can you show that your reconstructed
fees equal the fees the chain actually paid out?* That is `reconcile_fees_against_collect`, and it is the
most important function in this task — arguably in the whole pipeline.

## Files you own

```
src/undertow/data/validation/__init__.py
src/undertow/data/validation/checks.py          (single-stream invariants)
src/undertow/data/validation/crosscheck.py      (route agreement + fee reconciliation)
tests/data/test_checks.py
tests/data/test_crosscheck.py
tests/data/fixtures/collect_reconciliation/cases.json   (ONLY this file; T06 owns the raw captures)
```

## What to build

Implement `CONTRACTS.md` §8. `run_all_checks` **returns** results and never raises on a failed check —
only the CLI (T14) decides to exit non-zero. A validator that crashes cannot report the other nine
problems.

### 1. `checks.py` — invariants (each independently testable)
- `check_block_ordering` — `(block_number, log_index)` strictly increasing.
- `check_key_uniqueness` — no duplicate keys; report the first few offenders.
- `check_no_block_gaps` — the gas stream covers every block in the window. Critical.
- `check_swap_sign_convention` — for every swap, `amount0` and `amount1` have **opposite** signs and
  neither is zero. Report violations with `tx_hash`. (Rare edge: a zero-amount swap is possible in
  degenerate cases — if you find any in real data, downgrade to warning and document the count rather
  than failing the dataset; say which you chose.)
- `check_tick_price_consistency` — `sqrt_price_x96_to_tick(row.sqrt_price_x96)` vs `row.tick` for every
  swap. A **very** strong end-to-end check: it validates T02's TickMath port, T05/T06's decoding and the
  schema's fidelity at once. **But allow ±1 and document why:** after a *downward* tick crossing the pool
  writes `slot0.tick = tickNext - 1`, which is not always equal to
  `getTickAtSqrtRatio(sqrtPriceX96)` at the boundary. So: a deviation of 0 or ±1 passes; |deviation| > 1
  is **Critical**; and report the distribution of deviations (how many 0, how many ±1) in the metrics so a
  systematic drift toward ±1 is visible rather than absorbed. Without the tolerance this check fails on
  every real dataset; without the reported distribution the tolerance hides real bugs.
- `check_ticks_on_spacing_grid` — every `tick_lower`/`tick_upper` on mints/burns is a multiple of the
  pool's `tick_spacing`. Critical (a violation means our fee-tier→spacing mapping is wrong).
- `check_liquidity_conservation` — **deltas only, seeded from a snapshot.** The naive version of this
  check is unsatisfiable on a windowed dataset: positions minted before `start_block` are invisible in
  the window, so summing in-window `liquidity_net` can never reproduce absolute active liquidity, and the
  check would report drift from row 1 on every run. Instead:
  1. **Seed** absolute active liquidity from the `fee_growth` snapshot's `current_liquidity` at (or
     immediately before) `start_block`.
  2. From there, verify **changes**: between consecutive swaps, the change in swap-reported `liquidity`
     must equal the net effect of the intervening mints/burns spanning the tick, plus the net
     `liquidity_net` of any ticks crossed.
  3. Report max absolute and relative deviation and **the block where drift first appears** — a drift
     starting at a specific block is a decoding bug with a findable cause, and that diagnostic is the
     point of this check.
  Severity: **warning**, not critical, and say so in the report, because tick-crossing bookkeeping over a
  sampled fee-growth seed has legitimate slack. If it comes out exact, note that in your PR — that is a
  strong signal T06's decoding and T10's replay are both right.
- `check_reference_coverage` — reference bars span the window; report gap-filled count and longest run;
  warn above a threshold.
- `check_regime_labels_complete` — no null labels; `unknown` only in the initial lookback period; all
  four bins non-empty over the full window (warning if one is empty — see T12's handoff note, this is a
  finding the human needs).

Also add (not in the contract, worth having — extend, don't rename):
- `check_monotonic_timestamps` — `block_timestamp` non-decreasing with `block_number`.
- `check_no_future_data` — nothing beyond the configured `end_block`/`end_utc`.

### 2. `crosscheck.py` — the two headline checks

**`crosscheck_routes(graph, rpc, stream)`** — field-level equality between The Graph's and the RPC's
tables for the same block range. §10.1.5's whole point. Compare on the key, then every shared column;
report per-column mismatch counts and the first few offending keys. **Any mismatch is Critical** — the
two routes decode the same chain events, so a disagreement means one of them is wrong and we do not yet
know which. Handle these expected asymmetries explicitly rather than by loosening the comparison:
rows present in one route only (report as a mismatch, with the count), and streams one route does not
support (skip with an `info` result naming the reason — e.g. if T05 reported `collect` unsupported).

**`reconcile_fees_against_collect(tape, fee_growth, pool, rel_tolerance=1e-6)`** — the acceptance test
for §10.2.4. Method:

1. Find real position lifecycles in the tape: a `Mint` at `(owner, tickLower, tickUpper)`, later
   `Burn`/`Collect` at the same key. Pick at least **3** lifecycles (the plan's DoD) that are short
   enough to replay cheaply and that include at least one where the price **exited the range**.
2. Replay T10's `FeeGrowthTracker` from the mint block to the collect block to obtain
   `uncollected_fees` for both tokens.
3. **Separate principal from fees.** A `Burn` returns principal (`amount0`, `amount1` on the Burn event)
   and a following `Collect` in the same transaction withdraws principal + accrued fees together. So the
   fee component is `Collect.amountX - Burn.amountX` when they are paired in one tx, and
   `Collect.amountX` when the collect stands alone (a fee-only collect with no burn). Implement both
   cases and identify them by `tx_hash` equality. **This is the subtlety that makes naive
   reconciliations fail** — `CONTRACTS.md` §4.3 flags it and it is why the tolerance is defined on the
   fee component, not the total.
4. Compare reconstructed vs actual per token; report absolute and relative deltas.
5. Tolerance: the target is **exact** integer agreement (the protocol's arithmetic is deterministic).
   `rel_tolerance` exists only to absorb the multi-tick-swap apportionment residual if T10 took the
   approximation route (see `docs/decisions/002-swap-segment-apportionment.md`, if it exists). **State the tolerance actually
   achieved in the report and in your PR.** If agreement is exact, say so — that is the strongest
   sentence available for the thesis's §10.2.4 claim, and it should end up in the write-up.

Wire in T15's `expected_counts(month, stream)` — imported from
`undertow.data.validation.dune_expectations` (**T15 owns that module**; it is library code, not a test
helper, precisely so you can import it) — as an additional `warning`-severity check comparing our monthly
row counts against Dune's within T15's `tolerance_pct`. If Dune was unavailable, the fixture carries
`status: "unavailable"` and this check returns an `info` result saying it was skipped and why.

### 3. `render_report(results, path)`
A markdown report: a summary table (check, severity, pass/fail, key metric), then a section per failure
with detail. This file is a thesis appendix artifact — make it readable by a human who has not read the
code, and put the dataset id and manifest hash at the top so a report can be tied to the data it
describes.

## Tests you must write

Build on T11's `tiny_dataset()`. **Every check needs both a passing and a failing case** — a check that
has never been seen to fire is not known to work.

1. Per check: one test on clean data (passes) and one on deliberately corrupted data (fails, correct
   severity, message names the offending row/column). That is ~22 tests; they are the task.
2. `check_tick_price_consistency` fails when one swap's `tick` is perturbed by 1.
3. `check_ticks_on_spacing_grid` fails on a mint at `tick_lower = 61` for a spacing-60 pool.
4. `crosscheck_routes`: identical tables → pass; a single differing `amount0` → Critical naming the
   column and key; a row present only in one route → Critical with the count; an unsupported stream →
   `info`, not a failure.
5. `reconcile_fees_against_collect`:
   - a synthetic lifecycle with hand-computed expected fees reconciles **exactly** (delta 0);
   - the **paired** `Burn`+`Collect` case correctly subtracts principal — construct it so that a naive
     implementation comparing against the raw `Collect` amount would fail, and assert yours passes;
   - the standalone fee-only `Collect` case;
   - a lifecycle where the price exits the range accrues no fees while out of range;
   - a deliberately wrong tracker state → the check fails with a sensible relative delta.
6. `run_all_checks` on a dataset with three distinct problems returns three failures and does **not**
   raise.
7. `render_report` output contains every failed check's name, the dataset id, and no secrets.
8. Severity mapping is stable: a fixture asserting each check's severity, so a later edit that quietly
   downgrades a Critical to a warning breaks a test.

## Acceptance criteria

- `uv run pytest tests/data/test_checks.py tests/data/test_crosscheck.py` green; `mypy` clean.
- `run_all_checks` never raises on check failure (test it with a maximally broken dataset).
- The fee reconciliation runs against **real** `Collect` amounts in the committed fixtures, not only
  synthetic ones. Getting real fixture data here is the point of the task — three real lifecycles from
  the pinned pool, trimmed small, with their block ranges recorded.

## Handoff notes for your PR body

- **The reconciliation result: exact, or within what tolerance, over which real lifecycles.** This is the
  headline the human is waiting for.
- Whether the two routes agreed exactly on the sampled range.
- Any check you downgraded from Critical, and why.
- The liquidity-conservation drift, if any, and where it starts.

## Process (mandatory)

Branch (stacked per `PLAN.md` §3.2) → implement → tests green → `code-reviewer` → fix → commit → push →
PR → update `STATUS.md`.
