# ADR 041 — Observation regime-feature ablation gate

**Status:** accepted (implementation)
**Affects:** `src/undertow/sim/config.py` · `env/observations.py` · `env/lp_env.py` · `backtest/engine.py` · `evaluate/protocol.py` · `validation/parity.py` · `validation/lookahead.py` · `tests/sim/test_config.py` · `tests/sim/test_env.py` · `tests/sim/test_backtester.py` · `tests/sim/test_evaluation.py` · `configs/sim_default.toml` · `CONTRACTS.md` §2/§11
**Raised by:** ADR-030's binding ablation list (regime features on/off) and the S23 ablation program

## Context

ADR-030 pins the observation's 24-dim vector (CONTRACTS §11); slot 15 carries
the live regime ordinal. The thesis ablation program needs an arm trained with
that information removed and evaluated under the same protocol, without
changing the default behaviour of any shipped config, checkpoint or parity
fixture.

## Decision

Add `[observation] regime_features` to `SimConfig` (`ObservationConfig`,
default `true`). When false, `ObservationBuilder` emits the `REGIME_ABLATED`
sentinel instead of the as-of label, and `Observation.to_vector()` maps that
slot to `0.0`, so the ablation arm sees a constant in place of the regime
signal. `LpEnvironment` and the exact backtester both construct the
observation builder with the config flag, so the arm is fed the same constant
in training and evaluation. The default leaves the vector byte-identical, so
existing checkpoints, tests and parity fixtures are unaffected.

## Consequences

- The ablation arm differs from the main arm in exactly one input dimension:
  constant `0.0` instead of the live ordinal.
- A checkpoint trained with the gate off must be evaluated with the gate off;
  the config travels with the run manifest, and a mismatched config would feed
  a different observation than training did.
- `Observation.regime` is empty only in the ablation arm; the `Regime` labels
  themselves are untouched, and nothing in the reward or metrics reads the
  field. Telemetry keeps the live label by falling back to the market view, so
  rollout grouping is unaffected.
- Unit tests pin both paths: the default builder follows the fixture's four
  labels, the gated builder stays at `0.0` across them.

## Rejected alternatives

- **Rewriting the dataset's regime stream to one constant label.** It would
  also corrupt the per-regime stratification of the ablation arm's evaluation.
- **Masking the regime input at inference only.** It measures sensitivity to a
  missing input rather than the policy learned without it.
- **Removing the field from `Observation`.** It breaks the frozen CONTRACTS §11
  surface for every existing consumer and checkpoint.
