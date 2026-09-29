# ADR 011 — Baseline policies encode ranges through the factored `Action`; full range is the widest snapped band

**Status:** accepted
**Affects:** `src/undertow/sim/policies/baselines.py` · sim `CONTRACTS.md` §1/§10 · tasks S10, S11, S12
**Raised by:** S10 (`feature-sim-baselines`), from the code-review gate

## Context

`CONTRACTS.md` §10 defines the baseline ladder. For the passive and τ-reset policies it
describes ranges in terms of the factored action, e.g. PassiveNarrow is "±5 % around entry price"
and TauReset's docstring says "New range = [current_tick − τ, current_tick + τ] where τ = the
entry half-width in tick-spacings". But for `FullRangeV2Policy` it says:

> Full-range (Uniswap V2 equivalent) position: `tick_lower = MIN_TICK`, `tick_upper = MAX_TICK`.

`CONTRACTS.md` §1 freezes `Action` to exactly two fields:

```python
@dataclass(frozen=True, slots=True)
class Action:
    action_type: Literal["hold", "rebalance"]
    center_offset: int   # tick-spacings from current tick
    width: int           # tick-spacings per side
```

and `Action.lower_tick` / `upper_tick` resolve a **symmetric** band around the *current* tick and snap
to the tick spacing.

## Problem

For the pinned pool the tick spacing is 60 and the protocol bounds are
`MIN_TICK = −887272`, `MAX_TICK = 887272`. Neither bound is a multiple of 60, so **no** snapped
`Action` can resolve to `MIN_TICK`/`MAX_TICK` exactly. Worse, an early implementation returned
`center_offset=0, width=(MAX_TICK−MIN_TICK)/(2·60)=14787`; at the realistic entry tick 196242 that
resolves to `[−691020, 1083480]` — `upper > MAX_TICK`, an out-of-protocol position. `Position` only
checks `lower < upper`, so the error is silent (S10 brief test #6 did not exercise the consumer path).

Two secondary encoding facts:

1. The baseline widths (±5 % → 8 spacings, ±20 % → 33 spacings, full range → 14786 spacings) are
   **not** members of the agent action grid `{1, 2, 5, 10, 25, 50}` (`ActionGridConfig`). Baselines are
   evaluation references, not agent actions, so this is intended but must be explicit.
2. `half_width_ticks` is named in ticks, while §10's TauReset docstring calls τ a tick-spacing width.
   The S10 brief resolves this: it is ticks.

## Decision

**Baselines return ordinary `Action`s; the consumer always resolves them through
`Action.lower_tick`/`upper_tick`.**

- Passive narrow/wide set `center_offset=0` and compute `width` from the entry price via data's
  `price_to_tick` (ADR-009), floored to whole spacings (matches the S10 brief formula). The deployed
  band is therefore the snapped ±5 %/±20 % band centred on the current tick.
- `TauResetPolicy` stores the band in ticks (snapped via the returned `Action`) and recenters when
  `obs.tick` leaves it; τ is fixed for the episode.
- `FullRangeV2Policy` returns `center_offset = round(−current_tick / spacing)` and
  `width = floor(MAX_TICK / spacing) − 1`. The offset places the band centre near tick 0, and the
  one-spacing margin guarantees the resolved `[lower, upper]` stay inside
  `[MIN_TICK, MAX_TICK]` for every current tick while spanning all but ≤2 spacings of the full range.
  This is the V2-equivalent position to within one tick spacing. The policy asserts the bounds.
- **Baseline `Action`s are exempt from the agent `ActionGridConfig`.** They are comparison strategies,
  not policy-network outputs; S12 masks only the agent's action head.

## Consequences

- **S11 (backtester)** must not special-case `FullRangeV2Policy` by name: it already derives
  `tick_lower`/`tick_upper` from the returned `Action`, which now yields a valid near-full range.
- **S12 (env)** must not enforce `ActionGridConfig.widths` on baseline policies; it only needs to mask
  the agent's discrete action head.
- **`CONTRACTS.md` §10** should note that `FullRangeV2Policy` deploys the widest *snapped* full range
  (not byte-exact `MIN_TICK`/`MAX_TICK`) and that `half_width_ticks` is in ticks. The §1 `Action`
  definition itself stays unchanged.
- **Rejected — a sentinel `width` plus `tick_lower`/`tick_upper` properties on the policy.** A generic
  backtester cannot reach policy-specific attributes, so this leaves the silent out-of-bounds bug in
  place.
- **Rejected — extending `Action` with absolute `tick_lower`/`tick_upper` fields.** Allowed in
  principle by PLAN §0.2 (add a defaulted field), but `types.py` is owned by S01 and already merged;
  S10 must not edit it. This is the cleanest long-term fix if a future task needs byte-exact bounds and
  can coordinate the `types.py` change.
