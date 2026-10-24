# ADR 031 — Decision-time execution: the sim acts at the observation price

**Status:** accepted
**Affects:** `src/undertow/sim/env/lp_env.py` (`reset`/`step`) · `src/undertow/sim/validation/parity.py` · `CONTRACTS.md` §12 · task S22 (revision)
**Raised by:** S22 rebalancing-parity review (code-reviewer REQUESTS CHANGES)

## Context

S22 was the ADR-030 gate that validates the rebalancing path against exact
accounting. It found a structural execution-timing divergence:

- `LpEnvironment.step()` calls `_advance_replay()` **first** (advancing to the
  next reference bar and applying that interval's events) and only then
  `_rebalance()` at the **new** bar's price. An action chosen from the
  observation at boundary `D_{i-1}` therefore executes at `D_i` — one full
  10-minute bar later.
- `BacktestEngine` accrues the decision event, builds the observation, calls
  `policy.act` and executes `_apply_rebalance` at the **same** event/price.

The reviewer reproduced three consequences on the S22 worktree:

1. A moving first bar gives `il_drift = +4.15e-3`, `gas_drift = −7.28e-3` →
   parity fails.
2. A mid-episode jump returns `passed=True` while the gas/slippage series are
   shifted by one step (`[0, −197.3, 0]` vs `[0, 0, −197.3]`) — a **false
   pass**, because `passed` compares means only and a shift cancels.
3. The committed test fixture was hand-picked (first bar closes at the entry
   price) so the two worlds agree, and its only "rebalance" was the step-0
   deploy — so the slippage assertion was vacuous.

This is not a measurement artifact. The one-bar lag is baked into every
simulator rollout and every sim-to-on-chain gap (RQ3), and it would flow
directly into the hedge PnL, which is price-timing sensitive.

## Problem

The two worlds disagree on when a decision executes. On-chain, a transaction
submitted at `T` mines within ~12 s, not 10 min later, so the backtester's
same-time semantics are the faithful ones and the simulator is wrong.

## Decision

**The simulator executes an action against the decision state it was chosen
from, before advancing the market.**

`LpEnvironment.step(action)` order becomes:

1. Capture the decision state: pool `sqrt_price`/`tick`, the mark `price` the
   observation was valued at, the tape row and the decision time.
2. If the action is a rebalance, execute it at that state (gas/slippage from the
   decision tape row). A rejected rebalance leaves the position untouched, as
   today. On success the new position starts with zero uncollected fees and its
   own IL level (`_last_fees_usdc = 0`, `_last_il_total = il_total(decision)`).
3. Advance the market one interval (`_advance_replay` / `_advance_calibrated`),
   accruing the interval's events to the post-action position.
4. Measure the step on the post-action position over that interval: fees earned,
   `il_change`, equity, reward, observation.

`reset()` must leave a fully-populated decision state (mark price, pool sqrt,
tape row, time). The current base already sets `_last_tape_row = entry_row`
(`lp_env.py`) and the decision mark price; the S22 tests pin this so a first-step
rebalance prices gas from the entry row rather than an empty row.

The `price_exhausted` skip is removed: the action no longer needs a next bar, so
a rebalance at the final decision executes at the decision price and the episode
then terminates. The backtester already behaves this way.

## Consequences

- The simulator now matches the backtester and on-chain latency; the S22 gate
  can validate a genuine mid-episode rebalance.
- **ADR-029's "the IL step change is measured on the pre-action position" note is
  superseded**: the action now executes at the interval's *start*, so the
  interval's IL is measured on the post-action position. The fixed-HODL
  benchmark itself is unchanged.
- `CONTRACTS.md` §12's step ordering note is updated.
- Every prior checkpoint, run and rollout is invalid (they were already invalid
  under S17/S18). Re-training is required — expected, the thesis is restarting.
- The S22 guard must compare the action **executed in each numeric bucket**, not
  the observation-time action (nearly tautological), and `passed` must include a
  per-step gas/slippage max/std check, not mean-only.
- The backtester is **unchanged**.

## Rejected alternatives

- **Add the one-bar lag to the backtester.** Makes the measurement instrument
  less faithful to on-chain latency for no gain; the sim is the one that is
  wrong.
- **Keep the lag and special-case it in the parity harness.** Leaves a 10-minute
  execution bias in every rollout and every reported gap; the gate would be
  measuring the wrong thing.
