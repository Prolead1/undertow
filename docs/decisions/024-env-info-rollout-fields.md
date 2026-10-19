# ADR 024 — Additive env `info` fields for the U06 rollout recorder

**Status:** accepted (implementation)
**Affects:** `src/undertow/sim/env/lp_env.py` · `src/undertow/sim/train/rollouts.py` · ui `CONTRACTS.md` §2.1
**Raised by:** U06

## Context

`CONTRACTS.md` §2.1 freezes the rollout parquet columns `RolloutRecorder` must
persist for the Policy Replay view: `timestamp`, `price`, `tick`,
`position_value`, `in_range`, `regime`, `gas_price_wei` and `action_index`
(plus `tick_lower`/`tick_upper`/`equity`, which `info` already carried).
U06's boundary rule is that the recorder translates the environment `info` dict
into the parquet schema and nothing else — it must not read `LpEnvironment`
private state or index into the raw observation vector by position.

At the time U06 started, `LpEnvironment._make_info` did **not** expose those
seven fields. The values already existed on the `Observation` object that the
env builds each step, and `_observe` immediately collapsed it to
`observation.to_vector()` before returning, discarding the labels.

## Problem

The recorder needed a stable, named source for each §2.1 column. The two
options without an `info` change were both unacceptable:

* index the observation vector by hard-coded offset — couples the recorder to
  `OBSERVATION_VECTOR_LENGTH` and breaks silently when the feature layout
  changes; or
* reach into `LpEnvironment` internals (`env._position`, `env._builder`, …) —
  violates the "translate `info` only" boundary and the no-god-module rule.

`LpEnvironment` therefore had to carry the labels in `info`. That is nominally
outside U06's stated files, but it is a purely additive interface extension,
and the orchestrator has **authorised U06 to edit `src/undertow/sim/env/lp_env.py`
for exactly this change**.

## Decision

1. **`info` gains eight additive keys**, all sourced from the already-built
   `Observation` (or from values already computed in `step`/`reset`):
   `timestamp`, `price`, `tick`, `position_value`, `in_range`, `regime`,
   `gas_price_wei`, `action_index`.
   * `_observe` now returns the `Observation` object; `reset`/`step` call
     `observation.to_vector()` themselves. The public Gymnasium return types are
     unchanged: `reset -> (np.ndarray, dict)` and
     `step -> (np.ndarray, float, bool, bool, dict)`.
   * `in_range` is the observation's `position_in_range`; the recorder still
     recomputes it from the recorded ticks so the parquet invariant
     `tick_lower <= tick < tick_upper` holds by construction.
   * `_make_info` takes `at_time` and `action_index` explicitly because they are
     not derivable from the `Action` alone; `reset` passes `action_index=0` and
     the zero `RewardBreakdown` for the deployed initial position.
2. **All pre-existing `info` keys and values are byte-for-byte unchanged.**
   This is an additive change only (the frozen contract allows adding an
   optional field with a default / extra keys; readers tolerate extras).
3. **Cost columns stay on `reward_breakdown_raw`.** §2.1 requires USDC for
   `fees`/`gas`/`slippage`/`il`/`risk_penalty`; `rows_from_env` reads those from
   `info["reward_breakdown_raw"]` (USDC, per §0) and only `reward` from the
   capital-normalised `info["reward_breakdown"]`.

## Consequences

* `RolloutRecorder` has one named source per §2.1 column and no dependency on
  env internals or vector offsets.
* **Safe by construction:**
  * *No look-ahead.* Every added value comes from the same backward-looking
    `ObservationBuilder.build_at` (`close_time <= at_time`) the policy already
    sees; nothing reads future tape rows.
  * *No RNG / state change.* `_observe` still builds exactly the same
    `Observation`; the change only stops discarding it before `to_vector()` and
    copies already-computed scalars into a dict. No extra `PoolEngine` step, no
    RNG draw, no state mutation.
  * *Parity/backtest untouched.* `undertow.sim.validation.parity` and
    `undertow.sim.evaluate.protocol` read only `reward_breakdown_raw` (and
    `equity`), all of which are unchanged.
* `CONTRACTS.md` itself needs **no edit**: the env `info` layout is not part of
  the UI contract, and the change is additive. The thesis text does not describe
  the `info` key set, so no vault correction is required; any future vault note
  is flagged to the human, never edited from a code task.
* **Rejected:** positional observation-vector indexing (fragile, undocumented);
  a recorder-only env subclass (duplicates the builder and risks drifting from
  the training env); widening `reward_breakdown` to carry raw costs (would mix
  units in one dict and break the normalized PPO signal).
