# S20b — Agent-selected hedge ratio: the joint (LP, hedge) head

**Wave 7 · size M · depends on: S20 · blocks: S21, S23**
**Branch:** `feature-sim-hedge-ratio-head`

## Why this task exists

S20 shipped the hedge **instrument** as a deterministic overlay: the short is
always `HedgeConfig.ratio * LP ETH delta`, set by configuration. ADR-030 C3 and
`report/sections/04_problem_formulation.tex` describe the research action as
`a_t = (kappa_t, c_t, w_t, h_t)` with `h_t` in `{0, 0.25, 0.5, 0.75, 1.0}` —
"fraction of the LP's ETH delta carried by the short" — and the shipped
configuration has no hedge head at all. Without `h_t` the agent cannot trade
directional risk against funding, which is the second control axis the thesis
claims.

This task promotes `h_t` to a learned head and makes the evaluation count the
hedge leg:

1. `Action` gains an additive `hedge_ratio: float | None = None`. `None` keeps
   the S20 deterministic semantics (resolve through `HedgeConfig.ratio`); a
   float in `[0, 1]` is the policy's target.
2. `HedgeConfig.agent_ratio = false` (new, defaulted). When `true` the overlay
   must be `enabled` and the environment exposes
   `MultiDiscrete([n_lp, len(ratios)])`; the agent chooses the LP head and the
   ratio head jointly every step, `reset` starts **flat**, and each step
   re-hedges to `ratios[j] * LP ETH delta` at the decision mark (ADR-031).
3. The backtester applies the action's ratio through the same overlay so sim /
   backtest parity holds for a joint policy, and records the chosen ratio.
4. Evaluation counts the hedge leg: episode metrics use
   `equity + hedge_equity` so Sharpe / return / PnL are the joint strategy's,
   with the hedge flows exposed in the decomposition. Without this the learned
   policy's hedge PnL would disappear from every reported number.
5. A `HedgedHODLPolicy` baseline (beta-matched short of the initial basket) is
   added so the learned policy can be compared against the same risk profile,
   as the measurement protocol requires.

## Files you own

```
src/undertow/sim/types.py                    (Action.hedge_ratio additive field)
src/undertow/sim/config.py                   (HedgeConfig.agent_ratio + validation)
src/undertow/sim/env/lp_env.py               (MultiDiscrete head, decode, reset, step)
src/undertow/sim/backtest/engine.py          (action ratio, decision-log ratio, reset flat)
src/undertow/sim/cli.py                      (checkpoint adapter decode + validation)
src/undertow/sim/evaluate/protocol.py        (joint equity, action_to_index decode)
src/undertow/sim/policies/baselines.py       (HedgedHODLPolicy)
src/undertow/sim/policies/__init__.py        (export)
src/undertow/sim/evaluate/ablation.py        (baseline registry)
tests/sim/test_hedge_policy.py               (new)
tests/sim/test_env.py · test_config.py · test_backtester.py · test_evaluation.py
tests/sim/test_validation.py · test_cli.py · test_baselines.py
docs/plans/sim/CONTRACTS.md                  (§1, §12, §13)
docs/plans/sim/STATUS.md                     (row)
configs/sim_eth_usdc_500_hedged.toml         (new: primary + joint head)
```

## What to build

### 1. `Action.hedge_ratio`

Append one field after `width`:

```python
hedge_ratio: float | None = None   # S20b; None = no hedge opinion (config ratio)
```

A non-`None` value must be finite and in `[0, 1]` (raise `ValueError` otherwise).
The LP fields, `is_hold`, `lower_tick`/`upper_tick` and equality are unchanged
for actions that never inspect the new field, so every existing call site keeps
its exact behaviour.

### 2. `HedgeConfig.agent_ratio`

```python
agent_ratio: bool = False
```

`__post_init__` rejects `agent_ratio=True` with `enabled=False`, an empty
`ratios` sequence, or a `ratios` head without a `0.0` entry, so a config cannot
silently ask for a head that the env will never expose and a ratio-less policy
always has a representable unhedged choice.

### 3. Environment

- `action_space` stays `Discrete(n_lp)` unless `enabled and agent_ratio`; then it
  is `MultiDiscrete([n_lp, len(ratios)])`. The LP head's index order is
  unchanged, so `hold_bias` still targets logit `0`.
- `action_for_index` accepts an int (LP index; `hedge_ratio=None`) or a
  two-element sequence (LP index, ratio index) and returns the resolved
  `Action`. Out-of-range components raise `EnvError`.
- `reset`: in agent mode the account starts flat (no initial trade); the first
  decision chooses the ratio. In deterministic mode the S20 initial trade is
  unchanged.
- `step`: `ratio = action.hedge_ratio if action.hedge_ratio is not None else
  config.hedge.ratio`; re-hedge `ratio * position.amount1(decision_sqrt)` at the
  decision mark. Funding accrual is unchanged.
- `info` gains `hedge_ratio` (the resolved ratio, `None` when the overlay is
  off); `action_index` remains the LP head index so the frozen rollout schema
  and UI keep working.

### 4. Backtester

- Initial deployment only when the overlay is deterministic
  (`agent_ratio=False`), matching the env's reset.
- Each decision re-hedges to the action's ratio (falling back to the config
  ratio when `None`), and the decision log gains additive
  `hedge_ratio`/`hedge_short_size` columns when the overlay is enabled.

### 5. Evaluation joint equity

- A single `resolve_hedge_ratio` helper in `config.py` is the precedence rule
  the env, backtester and adapter all call: an explicit action ratio always
  wins; `None` is the config ratio for the deterministic overlay and `0.0`
  (unhedged) for the joint head.
- `action_to_index` understands the `MultiDiscrete` space (LP index + ratio
  index). Without the head it returns the LP index: a ratio is inert when the
  overlay is disabled and must equal the config ratio under the deterministic
  overlay (the Discrete index cannot carry a different one).
- `EpisodeRecord` metrics are computed on the joint equity:
  `ledger.equity_curve["equity"] + cumsum(hedge_pnl + hedge_funding +
  hedge_fee)` for the backtest backend, `info["equity"] + info["hedge_equity"]`
  for the replay-sim backend. A disabled overlay adds 0 so LP-only numbers are
  byte-identical.
- The decomposition gains additive `total_hedge_pnl`, `total_hedge_funding`,
  `total_hedge_fee`, `total_hedge` keys (0 when the overlay is off). `net`
  stays the LP cash identity; `total = net + total_hedge`.

### 6. Baseline

`HedgedHODLPolicy(hedge_ratio=1.0)`: returns `Action("hold", hedge_ratio=r)` on
every step; it holds the initial basket and shorts its delta. Registered in the
policy package and the ablation baseline registry.

### 7. Config

`configs/sim_eth_usdc_500_hedged.toml` copies the primary 0.05 % config and sets
`[hedge] enabled=true, agent_ratio=true` with the five-ratio head. The primary
`configs/sim_eth_usdc_500.toml` is untouched: the shipped LP-only results and
the ADR-036 recipe stay reproducible.

## Tests

1. `Action.hedge_ratio` validation, `None` default, LP-only equality.
2. `HedgeConfig` validation (agent_ratio needs enabled + non-empty ratios).
3. Env agent mode: `MultiDiscrete` shape; `action_for_index` both arities and
   error paths; reset flat; a `hold + ratio=1` step offsets a price move;
   `hedge_ratio` in info; a disabled overlay keeps the 21 LP dims and
   `Discrete` space byte-identical.
4. Backtester: a joint policy's hedge series match the env on the same window
   and the decision log records the ratio.
5. Evaluation: joint equity includes hedge flows (backtest + replay-sim);
   disabled overlay unchanged.
6. CLI: checkpoint adapter decodes `MultiDiscrete` and rejects a mismatched
   ratio count; LP-only checkpoints still load.
7. `HedgedHODLPolicy`: constant ratio, default 1.0, registered as a baseline.

## Acceptance criteria

- Full suite `uv run pytest -q -n 4` green with the 90 % coverage gate.
- `enabled=false` (all shipped configs) → action space, observation, reward and
  every metric byte-identical to today.
- The joint-headed config trains a non-degenerate policy (the greedy rollout
  does not hold every step with a constant ratio) and reports hedge PnL,
  funding and perp cost per evaluation window, with the hedged HODL baseline
  over the same windows.

## Out of scope

- Risk-sensitive reward shaping (S21) — the `h_t` head is exposed and trained
  under the existing reward; the CVaR/Sharpe objective stays a later task.
- Per-step rollout schema changes — `action_index` stays the LP index.
