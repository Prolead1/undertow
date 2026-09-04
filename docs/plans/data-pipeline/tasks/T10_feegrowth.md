# T10 — Fee-growth engine (eqs 3–6 + tick-crossing state machine)

**Wave 2 · size L · depends on: T00, T02 · blocks: T11, T13**
**Branch:** `feature-data-feegrowth`

## Why this task exists

This is the scientific heart of `undertow.data`. Roadmap §10.1.6 calls the tick/fee-growth stream
"non-negotiable" and §10.2.4 spells out why: *"a position's fees are not a function of the swap amounts
alone — they are a function of how much liquidity was active at each tick while those swaps flowed
through it… skipping them is how backtests quietly overstate or understate fees by an order of
magnitude."* Every PnL number in the thesis flows through this module. Gap **G3** (on-chain validation)
is only credible if this is exact.

You are implementing the pool's own accounting, in Python, on integers.

## Files you own

```
src/undertow/data/transforms/__init__.py
src/undertow/data/transforms/feegrowth.py
tests/data/test_feegrowth.py
tests/data/fixtures/feegrowth_replay.json
```

**On fixtures:** you are in wave 2 and T06 (which fetches real snapshots) is in wave 3, so you cannot use
its fixture. Write your own `feegrowth_replay.json`: the ~20-event replay sequence of test 11 plus a few
synthetic `FEE_GROWTH_SCHEMA` rows (§4.4) for `reconcile`. Build them to the schema, hand-computing the
expected accumulator values — that hand-computation *is* the test's value. T13 later re-runs your
`reconcile` against T06's real captures.

You do **not** fetch anything. Your inputs are tables/rows handed to you; T06 fetches the observed
snapshots and T11 wires you into the tape.

## Required reading before you code

- `~/Documents/fyp/lesson_plan/10_data_evaluation_roadmap.md` §10.2.4 — equations (3)–(6), the
  water-meter analogy, and the worked example. Read it fully; it is the spec.
- `~/Documents/fyp/lit_review/papers/uniswap_v3_whitepaper.pdf` §6.3–6.5 (tick-indexed state, fee
  growth). The roadmap explicitly says to re-read this right before implementing.
- Uniswap v3-core `Pool.sol` (`_modifyPosition`, `swap`'s tick-crossing branch), `Tick.sol`
  (`cross`, `update`), `Position.sol` (`update`) — the reference implementation you are mirroring.

## What to build

Implement `CONTRACTS.md` §6.1 exactly.

### 1. The pure functions — eqs (3)–(6)
`fee_growth_below`, `fee_growth_above`, `fee_growth_inside`, `uncollected_fees`. All integer, all using
`fixedpoint.wrapping_sub_256`. Each docstring cites its equation number from §10.2.4 and states the
branch condition.

`uncollected_fees(liquidity, g_inside_now, g_inside_last)` = `liquidity * wrapping_sub_256(now, last)
>> 128` — a right shift, i.e. floor division by `2**128`, matching the protocol's `FullMath.mulDiv`.
Return raw integer token units. **Do not** convert to human units here; that is a presentation concern
and belongs in T11/T16.

### 2. `FeeGrowthTracker` — the state machine
The reason this class exists: you cannot afford an archive `eth_call` per block for three years of
history (that is millions of calls). Instead, **replay** the event stream and maintain the accumulators
yourself, then *verify* against sampled real snapshots (T06 fetches those, T13 asserts agreement).

State: `fee_growth_global_{0,1}_x128`, `current_tick`, `current_liquidity`, and a
`dict[int, TickState]` of initialized ticks (`fee_growth_outside_{0,1}_x128`, `liquidity_gross`,
`liquidity_net`, `initialized`).

- `apply_swap(row)` — a swap moves the price from the previous tick to `row["tick"]`. Two effects:
  (a) fee growth accrues on the **input** token: `fee_amount = amount_in * fee_pips // 1_000_000` and
  `fee_growth_global += (fee_amount << 128) // liquidity_active` — computed **per in-range segment**,
  because a swap that crosses ticks pays different segments with different active liquidity;
  (b) each tick crossed calls `cross_tick`.
  You do not have per-segment amounts in the `Swap` event (it reports only net amounts and the final
  tick). **This is the central modelling decision of this task**: for swaps that cross no initialized
  tick, the computation is exact; for swaps that cross ticks, apportioning fees across segments requires
  re-simulating the swap through the tick lattice. Implement the exact single-segment case, implement
  tick-lattice re-simulation for the crossing case using `fixedpoint.get_amount{0,1}_delta`, and
  **mark the result's provenance** so T13 can quantify any residual. If you conclude full
  re-simulation is out of scope for this task, implement the single-segment exact path, raise a clearly
  named `FeeGrowthApproximation` warning on crossing swaps, record the decision in
  `.pi/plans/data-pipeline/adr/002-swap-segment-apportionment.md` **and** its repo-visible copy
  `docs/decisions/002-swap-segment-apportionment.md` (per `PLAN.md` §0.3 — cite the repo path from the
  code), and make sure the `exact` flag on `FeeAccrual` reflects it.
  Do not silently approximate.
- `cross_tick(tick, upward)` — mirrors `Tick.cross`: `fee_growth_outside = wrapping_sub_256(
  fee_growth_global, fee_growth_outside)` for both tokens, and `current_liquidity +=
  liquidity_net` (upward) or `-= liquidity_net` (downward).
- `apply_liquidity_event(row)` — a `Mint`/`Burn` updates `liquidity_gross`/`liquidity_net` on both
  boundary ticks and, if the position spans the current tick, `current_liquidity`. Newly initialized
  ticks get `fee_growth_outside` seeded per `Tick.update`: `fee_growth_global` if the tick is at or
  below `current_tick`, else `0`. Getting this seeding wrong is a classic silent-error source.
- `accrue(key, liquidity, g_inside_0_last, g_inside_1_last) -> FeeAccrual` — eq (3) then eq (6) for both
  tokens, returning the new snapshots too.
- `snapshot()` / restore — so T14 can checkpoint and resume a long replay without redoing it.
- `reconcile(observed: pa.Table) -> ReconciliationReport` — compare replayed state against real sampled
  snapshots at the same blocks; report per-field exact-match counts and, for mismatches, the absolute
  and relative delta. This is the method T13 turns into a blocking check.

### 3. `ReconciliationReport`
A frozen dataclass: `n_compared`, `n_exact`, `max_abs_delta_g0`, `max_abs_delta_g1`,
`max_rel_delta`, `mismatches: tuple[Mismatch, ...]`. It **reports**, it does not raise — T13 decides
severity.

## Invariants that must be tested (these are the task)

1. **Roadmap §10.2.4's own worked example** reproduces the printed values (`g_global=1e36`,
   `g_out_low=0.4e36`, `g_out_up=0.2e36`, `L=1e20`, `g_last=0.1e36`, `ic=50`, range `[-200, 200]`).
   Use integers, and assert the exact integer result.
2. **Roadmap exercise Q3 (explicitly assigned in the chapter):** a position **entirely below** the
   current tick (`tick_upper < current_tick`) receives **zero new** interior fee growth even while
   `fee_growth_global` grows. Test it by growing the global accumulator and asserting
   `fee_growth_inside` is unchanged across the growth.
3. Symmetric case: a position entirely **above** the current tick likewise accrues nothing new.
4. Full-range: `fee_growth_inside(MIN_TICK, MAX_TICK)` with `g_out = 0` on both ⇒ equals
   `fee_growth_global`.
5. **Branch flip:** with fixed `g_out_*`, evaluate `fee_growth_inside` at `current_tick` just below
   `tick_lower`, inside, and at/above `tick_upper` — assert each takes the intended branch of eqs (4)/(5)
   (assert the *value*, computed by hand in the test, not just that the branches differ).
6. **Half-open range convention:** `current_tick == tick_lower` is **in range**;
   `current_tick == tick_upper` is **out of range**. Assert both.
7. **Wrapping:** construct a genuinely wrapped case — `g_inside_last` near `2**256 - k` and
   `g_inside_now` a small value that resulted from the accumulator wrapping past `2**256` — and assert
   `uncollected_fees` returns exactly the true growth `(2**256 - g_inside_last) + g_inside_now` scaled by
   `L >> 128`, computed by hand in the test. (Note the naive reading "last > now implies wrap" is *not*
   a valid test on its own: any `last > now` pair yields a large value through wrapping arithmetic, so
   the test must be built from a real wrap, not merely from an inverted pair.)
8. `cross_tick` twice in opposite directions returns `fee_growth_outside` to its original value
   (involution) when no growth happened in between — and does **not** when growth did happen.
9. Tick seeding on `Mint`: a tick initialized below `current_tick` gets `g_out = g_global`; above gets
   `0`. Then accruals from that position are correct from the next swap onward.
10. Liquidity bookkeeping: after a `Mint` spanning the current tick, `current_liquidity` increased by the
    minted amount; after the matching `Burn`, it returns exactly to the prior value.
11. A replay over a hand-built ~20-event sequence (mint, swaps in both directions crossing a tick,
    burn, collect) ends with state matching a hand-computed expected state. Build this fixture
    deliberately — it is the test that will catch real regressions.
12. No `float` anywhere on the accrual path (same `ast` guard style as T02).

## Acceptance criteria

- `uv run pytest tests/data/test_feegrowth.py` green.
- Every equation-implementing function cites §10.2.4's equation number and the corresponding
  Solidity function.
- `exact` provenance is propagated honestly through `FeeAccrual` — no path returns `exact=True` on an
  approximated computation. The reviewer is instructed to treat a silently-wrong numeric result as
  Critical; this is the module where that applies most.
- If you took the approximation route on multi-tick swaps, both copies of ADR-002 exist (vault + 
  `docs/decisions/`) and quantify the expected error (e.g. "affects N% of swaps in the pinned window").

## Handoff notes for your PR body

- Whether multi-tick swap apportionment is exact or approximated, and the ADR link.
- The `FeeGrowthState` shape (T14 checkpoints it).
- `reconcile`'s output fields (T13 renders them).

## Process (mandatory)

Branch → implement → tests green → `code-reviewer` → fix → commit → push → PR → update `STATUS.md`.
