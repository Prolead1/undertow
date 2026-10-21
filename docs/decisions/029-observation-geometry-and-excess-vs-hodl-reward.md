# ADR 029 — Band geometry in the observation and an excess-vs-HODL reward

**Status:** accepted (implementation)
**Affects:** `src/undertow/sim/env/observations.py` · `src/undertow/sim/env/lp_env.py` · `src/undertow/sim/backtest/engine.py` · `src/undertow/sim/evaluate/protocol.py` · `src/undertow/sim/validation/lookahead.py` · `src/undertow/sim/cli.py` · sim `CONTRACTS.md` §9/§11 · `docs/plans/sim/PLAN.md` §7 · sim `STATUS.md`
**Raised by:** RL kick-off diagnostics (degenerate hold policy)

## Context

The first real PPO kick-off (`runs/experiment2`, five seeds × 1M steps) produced
degenerate policies. Every seed collapsed to a single deterministic action, and
the rollouts recorded from the training environment showed two failure modes:
some seeds always held, others always picked one rebalance action. A diagnostic
rollout-buffer pass measured the per-action advantages as nearly
indistinguishable — the correlation between the action index and its advantage
was ≈ 0.02, well below the advantage noise — so the policy had no usable
gradient to prefer rebalancing.

The frozen interface the agent was trained against is `CONTRACTS.md` §11
(observation) and §9 (reward). Two defects in that setup were identified.

## Problem

### 1. The observation cannot see the position's band

`CONTRACTS.md` §11 exposed `position_in_range: bool`, `position_value` and
`il_vs_hodl`, but **not the position's bounds or any distance to them**. The
action space is defined in tick-spacings — `rebalance(center_offset, width)`
recenters a band `width` spacings either side of the *current* tick — but the
state never exposes how far the current tick is from either edge of the live
band, nor how wide that band is.

The project's own baselines are written in exactly those terms:
`TauResetPolicy` recenters when price exits a band of a known half-width, and
`CostAwareRebalancePolicy` compares the estimated fee gain to the gas cost of a
rebalance. A stateless MLP policy with no frame stacking cannot reconstruct the
band from the remaining features: `il_vs_hodl` conflates price drift with band
width, and `position_in_range` is a one-bit summary with no magnitude or
duration. The agent was therefore solving a strictly harder, partially observed
version of the baseline task, and the only action whose value does not depend on
band geometry — hold — was the one it could learn reliably.

### 2. The training reward diverged from the evaluated metric after a rebalance

The simulator measured IL against a **rolling entry basket**: `_il_offset`
accumulated each closed position's `il_vs_hodl` against its own entry, and the
entry was rebased at every rebalance. The backtester's ledger does the opposite:
`engine.py` computes `il_now = position_value - hodl_now` where `hodl_now` is the
**fixed initial deposit's** token mix held for the whole window, and the ledger
documents `net = fees + il + gas + slippage` as exactly `excess_vs_hodl`
(`ledger.py`). So a policy that rebalanced was rewarded against a different
benchmark than the one the thesis reports, and the per-step reward was not the
change in the strategy's excess over buy-and-hold.

The S14 parity harness never caught this because it deliberately runs the no-op
HOLD policy (`validation/parity.py`); with no rebalance, the rolling and fixed
benchmarks coincide.

## Decision

### 1. Append band geometry to the observation vector (ADR-029)

`Observation`, `ObservationBuilder`, the backtester's structural
`BacktestObservation`, and `evaluate.observation_from_vector` gain three
features, in tick-spacing units (the action grid's own units):

| Field | Definition |
| --- | --- |
| `distance_to_lower_bound_spacings` | `(tick - tick_lower) / tick_spacing` |
| `distance_to_upper_bound_spacings` | `(tick_upper - tick) / tick_spacing` |
| `position_width_spacings` | `(tick_upper - tick_lower) / tick_spacing` |

Distances are positive inside the band (so `position_in_range` is a threshold on
them) and negative outside. They are appended at indices 18–20, after the time
features, and `OBSERVATION_VECTOR_LENGTH` becomes 21.

Appending keeps the ordinal meaning of the original 18 features stable, so UI
code that reads `observation[dim]` and older rollout parquet files (which carry
18-entry vectors) remain interpretable: `StateInspector` already sizes itself
from the longest vector it sees and pads missing dims with zero. Raw
`tick_lower`/`tick_upper` were rejected because they are ~2·10⁵ and would add to
the existing feature-scale problem; spacings are `O(1–50)`.

### 2. Make the reward the step change in excess over HODL

`LpEnvironment` now keeps the deposit deployed at `reset()` as
`_initial_position` and never rebases its entry price. The reward's `il_change`
term is measured on the **pre-action** position as

```
position.value(sqrt, price) - initial_position.hodl_value(entry_sqrt, entry_price, price)
```

which is the backtester's `il` column step-for-step. `_il_offset` is deleted.
The per-step reward therefore equals the change in `equity - HODL_initial`; its
episode sum is the reported `excess_vs_hodl`, and `pnl_net` matches the
backtester's `net`. The observation's `il_vs_hodl` uses the same fixed basket
via a new `hodl_position` argument on `ObservationBuilder.build_at`.

The band geometry and the fixed benchmark are independent fixes: the first makes
the rebalance decision representable, the second makes the reward measure the
quantity the evaluation reports.

### 3. Fail stale checkpoints loudly

`_CheckpointPolicy` validates the loaded model's `observation_space.shape`
against `OBSERVATION_VECTOR_LENGTH` and raises a `CliError` telling the user to
retrain, instead of failing later with a shape mismatch on an 18-dim vector.

## Consequences

- Old checkpoints are incompatible; every run must be retrained. The CLI guard
  reports this clearly.
- Rollout parquet observation arrays grow from 18 to 21 entries. The schema
  stores a list, so no column change; the UI's state inspector is
  length-agnostic.
- The S14 HOLD-only parity numbers are unchanged (the benchmarks coincide with
  no rebalance). A future parity test that exercises a rebalance now compares
  like-for-like.
- The risk-penalty volatility term (`_pnl_volatility`, disabled by default,
  `λ = 0`) still tracks absolute equity changes, not excess changes. It is left
  unchanged to keep this ADR scoped; if the penalty is ever enabled, its σ
  should be switched to the excess series.
- `CONTRACTS.md` §9 and §11 and `PLAN.md` §7 are updated in the same change so
  the contracts match the implementation. The thesis text in the vault was not
  edited from this task; if it quotes the §11 field list or the reward
  benchmark, it needs a human update.

## Rejected alternatives

- **Raw bounds (`tick_lower`/`tick_upper`) in the observation.** Large absolute
  values that (a) worsen the scale problem and (b) are not the quantity the
  action grid is expressed in.
- **Inserting the geometry next to the other position fields.** It would shift
  the regime ordinal and every subsequent index, silently misreading the 18-dim
  rolling-window vectors already on disk.
- **A configurable IL benchmark** (`rolling` vs `fixed`). The fixed initial
  basket is the definition of `excess_vs_hodl`, not an RQ1 ablation; exposing it
  as a flag would create two differently-measured "rewards" with no research
  question attached.
- **A recurrent policy or frame stacking** instead of geometry features. A
  larger change to the training stack that still leaves the reward benchmark
  wrong; the band is a sufficient statistic the environment already has.
