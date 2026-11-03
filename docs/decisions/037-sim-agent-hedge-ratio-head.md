# ADR 037 — Agent-selected hedge ratio: a joint (LP, hedge) action head and joint evaluation

**Status:** accepted (implementation)
**Affects:** `src/undertow/sim/types.py` · `config.py` · `env/lp_env.py` · `backtest/engine.py` · `evaluate/protocol.py` · `policies/baselines.py` · `cli.py` · `CONTRACTS.md` §1/§2/§12/§13 · `configs/sim_eth_usdc_500_hedged.toml`
**Raised by:** ADR-030 C3 and `report/sections/04_problem_formulation.tex` (the joint action `a_t = (κ_t, c_t, w_t, h_t)`)

## Context

S20 shipped the hedge as a **deterministic** overlay: the short is always
`HedgeConfig.ratio * LP ETH delta`, set by config, with funding, perp fees and
mark PnL accounted in the reward. ADR-030's C3 claim and the thesis report both
describe the research action as a *joint* choice over timing, width **and hedge
ratio** `h_t ∈ {0, 0.25, 0.5, 0.75, 1.0}`; the shipped configuration has no
hedge head, so the agent cannot trade directional risk against funding — the
second control axis is missing.

Two facts force the promotion now:

1. The primary 0.05 % LP-only policy converges to pass-hold and ties `hodl`
   (ADR-036). The LP head alone has a weak, cost-dominated signal; the hedge
   head adds a genuinely different trade-off (funding carry vs directional
   variance) the agent can learn from.
2. Evaluation reports the LP equity only (`BacktestLedger.equity_curve` is the
   LP leg; `CONTRACTS §13`). Once the agent chooses the ratio, a policy's hedge
   PnL would silently disappear from every reported Sharpe / PnL number, so the
   measurement layer must count the hedge leg.

## Decision

1. **`Action` gains an additive `hedge_ratio: float | None = None`** appended
   after `width`. `None` is the policy's "no hedge opinion"; a finite value in
   `[0, 1]` is the policy-chosen ratio (invalid values raise). The LP fields and
   all existing call sites are unchanged.
2. **`HedgeConfig.agent_ratio = false`** (defaulted) promotes `ratios` to the
   agent head. It requires `enabled=true`, a non-empty `ratios` sequence, and a
   `0.0` entry — the unhedged choice a ratio-less action resolves to, so the
   environment and backtester can never silently disagree.
3. **Environment action space.** With the head active the space is
   `MultiDiscrete([n_lp, len(ratios)])`; the LP index order is unchanged (so
   `hold_bias` still targets logit 0). `action_for_index` accepts an int (LP
   head, ratio `None`) or a two-element `(lp_index, ratio_index)` sequence.
   `reset` starts **flat** (the first decision chooses the ratio) and every step
   re-hedges to `ratio * position.amount1(decision_sqrt)` at the decision mark
   (ADR-031). `info["hedge_ratio"]` carries the resolved ratio;
   `info["action_index"]` stays the LP index so the frozen rollout schema and
   the UI are untouched.
4. **Backtester.** The same resolution: a joint config starts flat, each
   decision re-hedges to the action's ratio (`None` = unhedged), and the
   decision log gains an additive `hedge_ratio` column next to the S20 columns.
5. **Joint evaluation and one resolver.** `resolve_hedge_ratio` in
   `undertow.sim.config` is the single rule all three layers call: an explicit
   action ratio always wins; `None` resolves to the config ratio under the
   deterministic overlay and to `0.0` (unhedged) under the joint head.
   `action_to_index` understands the `MultiDiscrete` space; without the head it
   returns the LP index, so an explicit ratio is inert when the overlay is off
   and must equal the config ratio under the deterministic overlay (the
   `Discrete` index cannot carry a different one). Episode metrics are computed on the **joint**
   equity — `equity + cumsum(hedge flows)` for a ledger, `info["equity"] +
   info["hedge_equity"]` for the replay simulator — and the decomposition gains
   `total_hedge_pnl`, `total_hedge_funding`, `total_hedge_fee`, `total_hedge`
   (the LP `net` identity is preserved; `total = net + total_hedge`). A
   disabled overlay adds 0, so every LP-only number is byte-identical.
6. **`HedgedHODLPolicy`** (beta-matched benchmark, `hedge_ratio=1.0` default)
   returns `hold` with a constant ratio. It is registered in the policy package,
   the ablation baseline registry and the CLI policy list, so `evaluate` reports
   the same risk profile the learned policy claims to find.
7. **`configs/sim_eth_usdc_500_hedged.toml`** is the primary config plus
   `enabled=true, agent_ratio=true`. The shipped LP-only config is untouched so
   its ADR-036 results stay reproducible.

## Consequences

- `enabled=false` (all shipped configs and the default) keeps the `Discrete`
  space, the 24-dim observation, the reward and every metric exactly as today.
- `enabled=true, agent_ratio=false` is the S20 deterministic overlay path, and
  the S20 parity tests are unchanged; the deterministic backtester's
  `hedge_curve` is identical to a constant joint policy's (test
  `test_agent_ratio_head_matches_the_deterministic_ratio`).
- The joint policy is scored on joint equity in both evaluation backends, and
  `evaluate` can compare it against the scripted baselines (unhedged) and
  `hedged_hodl` (full short) on the same windows.
- A checkpoint's action space is now part of its contract: the CLI adapter
  rejects a `Discrete` checkpoint under a joint config and a `MultiDiscrete`
  checkpoint under an LP-only config (SB3's action-head shape guard would
  otherwise be cryptic).
- `hold_bias` continues to work with the joint head: the first categorical
  head's logit 0 is the LP `hold`.
- The rollout parquet schema is unchanged (`action_index` is the LP index);
  per-step ratio telemetry lives in the environment `info`/backtester decision
  log rather than a schema bump.

## Rejected alternatives

- **Flatten the joint action into one `Discrete(n_lp × n_ratio)`.** It makes
  `hold × ratio` five distinct actions and `hold_bias` a five-logit edit, and
  loses the factored structure SB3 supports natively; `MultiDiscrete` keeps the
  LP head bit-identical.
- **Keep the config ratio when an action has none, in agent mode.** A scripted
  baseline under a joint config would then inherit the deterministic overlay
  (e.g. `hodl` silently becomes a full short), turning an unhedged comparator
  into a hedged one. Resolving `None` to the unhedged entry is the honest
  reading.
- **Report LP-only metrics and add hedge terms separately.** The reward the
  agent optimises includes the hedge leg, so an LP-only Sharpe would rank
  policies by a different objective than the one they learned.
- **Duplicate the `None`-ratio rule in the env, the backtester and the
  evaluation adapter.** Three copies can drift; the shared resolver is one
  tested function with a documented precedence order.
- **Change `equity_curve["equity"]` to include the hedge in the ledger.** It
  would break the frozen §13 column/identity and the LP-only byte-exactness;
  the joint series is derived at the evaluation boundary instead.
