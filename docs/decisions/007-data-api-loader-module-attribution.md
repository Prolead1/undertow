# ADR 007 — `undertow.data` sim-API loader attribution and the T16-pending `tick_to_sqrt_price` stub

**Status:** accepted, **superseded in part by ADR-008**
**Affects:** S02 `src/undertow/data/__init__.py` · `src/undertow/data/loader.py` · CONTRACTS.md §3
**Raised by:** S02 (`feature-sim-data-api`)

> **Correction (ADR-008).** The premise that T16 had not shipped is **false** — T16 (PR #25) is
> merged, and it never contained a `tick_to_sqrt_price` name. The `TICK_BASE`-from-`.config`
> attribution decision below stands. The `tick_to_sqrt_price` `NotImplementedError` stub, its
> `strict=True` xfail, and the "T16 pending" framing are **withdrawn**; ADR-008 drops the phantom
> name and settles the sim-facing surface. Read this ADR for the module-attribution decision only.

## Context

CONTRACTS.md §3 (`docs/plans/sim/CONTRACTS.md`) fixes the names `undertow.data` must re-export for the
sim, and attributes each group to a source module:

```python
# Re-exported from undertow.data.fixedpoint:
from undertow.data.fixedpoint import (
    Q96, Q128, TICK_BASE, MIN_TICK, MAX_TICK,
    sqrt_price_x96_to_price, price_to_sqrt_price_x96,
    price_to_tick, tick_to_price, tick_to_sqrt_price,
)

# New exports (S02 writes these adapters in loader.py or directly in __init__.py):
from undertow.data.loader import (
    load_dataset, load_tape, load_reference_feed, load_gas_feed, load_regime_labels,
)
```

CONTRACTS §0 requires that a *genuine* deviation be recorded as an ADR, flagged in the PR, and not
made silently:

> If something here is genuinely wrong, write `docs/decisions/NNN-<slug>.md`, flag it in your PR, and
> notify the orchestrator — do not unilaterally change it, because other agents are already coding
> against it.

The S02 brief also requires that, where a data internal does not yet exist because T16 has not shipped,
S02 ships a `NotImplementedError("Pending T16 — data public API not yet complete")` stub and xfails its
test. The S02 definition of done repeats this.

## Problem

Two attributions in CONTRACTS §3 do not match the as-built data package, and both would break a
literal implementation:

1. **`tick_to_sqrt_price` does not exist in `undertow.data.fixedpoint`.** The module exposes only the
   exact-integer `tick_to_sqrt_price_x96(tick: int) -> int` (`src/undertow/data/fixedpoint.py:182`).
   The sim contract asks for a human-units `sqrt(1.0001 ** tick)` returning a float, and T16 owns the
   final fixedpoint surface. Re-implementing it now would guess units/return type and collide with T16.

2. **`TICK_BASE` is defined in `undertow.data.config`, not `.fixedpoint`.** It is an exact `Decimal`
   constant `Decimal("1.0001")` at `src/undertow/data/config.py:36`. Importing it from
   `undertow.data.fixedpoint` (as §3 writes) raises `ImportError`; the value is correct where it lives.

## Decision

S02 makes the minimum additive change that preserves §3's *public names* while attributing each name to
its real source module — the contract's intent is the export surface, not the import path:

1. `tick_to_sqrt_price` is a thin adapter in the new `undertow.data.loader` module. It raises
   `NotImplementedError("Pending T16 — data public API not yet complete")`. Its test is `xfail`ed with
   that message and `strict=True`. T16 replaces the body, not the signature.
2. `TICK_BASE` is imported from `undertow.data.config` and re-exported from `undertow.data`. The value
   is the correct one; only the module attribution in §3 is wrong.
3. The new `load_*` bridge adapters live in `undertow.data.loader` (the "loader.py or directly in
   `__init__.py`" option §3 explicitly allows). They delegate to `pipeline.build` via `load_dataset`
   and are the sole sanctioned sim-facing seam.

A code comment at the `TICK_BASE` import in `src/undertow/data/__init__.py` points back to this ADR so
the deviation is discoverable from the code.

## Consequences

- `src/undertow/data/__init__.py`: `TICK_BASE` imported from `.config` (not `.fixedpoint`);
  `tick_to_sqrt_price` re-exported from `.loader`. Both remain in `__all__` exactly as §3 lists them,
  so S03/S04/S07/S11 code against the same public names.
- `src/undertow/data/loader.py`: owns `tick_to_sqrt_price` (T16-pending stub) and the five `load_*`
  adapters. The sim never imports this module directly.
- Existing names in the data package are untouched (scope is strictly additive, per the S02 brief).
- T16 must know: when it ships the final fixedpoint surface it should move/replace
  `tick_to_sqrt_price` and decide whether `TICK_BASE` belongs in `fixedpoint`; this ADR documents the
  interim state. No other running sim task is affected — none import `undertow.data.fixedpoint`
  directly.
- CONTRACTS.md §3 needs a follow-up edit to change the `TICK_BASE` and `tick_to_sqrt_price`
  attributions once T16 lands. Per project rule this ADR flags it; the code task does not edit the
  frozen contract.
- **Vault note flagged to the human** (never edited by a code task): the thesis appendix may cite
  CONTRACTS §3 attributions; the as-built module paths differ as described above.

### Rejected alternatives

- **Silently import `TICK_BASE` from `.config` without an ADR.** Fixes the `ImportError` but violates
  CONTRACTS §0 and hides the deviation from T16 and the orchestrator.
- **Re-implement `tick_to_sqrt_price` from `tick_to_sqrt_price_x96`.** Would guess the human-units
  float conversion T16 owns; risks a second, conflicting implementation.
- **Move `TICK_BASE` into `fixedpoint` to satisfy §3 literally.** Non-additive — touches a data module
  the S02 brief forbids changing and files T16 owns.

---

**Process:** ADR raised on `feature-sim-data-api`; listed in `docs/plans/sim/STATUS.md` under "ADRs
raised" and linked in the S02 PR body.
