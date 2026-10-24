# S20 — Hedge overlay: perp short with funding, costs and beta-neutral reporting

**Wave 7 · size L · depends on: S18, S19, S22 · blocks: S20b, S21, S23**
**Branch:** `feature-sim-hedge-overlay`

## Why this task exists

ADR-030 C3: the agent's value comes from *joint* LP + hedge control. This task
builds the hedge **instrument** — a delta overlay on the LP's ETH exposure, with
real funding and perp costs — and wires it into the environment, the reward and
the observation. It is deliberately the **deterministic delta-neutral overlay**
(the agent chooses the ratio in S20b); this task makes the mechanics, the
accounting and the parity correct first.

Measured inputs (S19): ETHUSDT perp funding mean **+0.0067 %/8 h (+7.3 %/yr)**,
positive in 84 % of periods (+13.0 %/yr in 2024); a delta hedge's total-variation
trading cost is 0.25–1.4 %/month at 5 bps. The hedge is roughly self-financing
carry, not free alpha.

## Files you own

```
src/undertow/sim/hedge.py                    (new: HedgeAccount, HedgeState, HedgeTrade)
src/undertow/sim/config.py                   (HedgeConfig + [hedge] parsing + RewardConfig.hedge_enabled)
src/undertow/sim/marketview.py               (additive funding frame + funding_at lookup)
src/undertow/sim/env/lp_env.py               (wire the hedge into reset/step/reward/observation)
src/undertow/sim/env/observations.py         (append hedge features; OBSERVATION_VECTOR_LENGTH)
src/undertow/sim/env/reward.py               (hedge terms in RewardBreakdown/compute_reward)
src/undertow/sim/backtest/engine.py          (the same overlay so parity holds)
src/undertow/sim/validation/parity.py        (hedge-aware series/report fields)
src/undertow/sim/train/rollouts.py           (record hedge fields)
src/undertow/sim/train/telemetry.py          (optional hedge metric)
tests/sim/test_hedge.py                      (new)
tests/sim/test_env.py · test_backtester.py · test_validation.py · test_reward.py
docs/plans/sim/CONTRACTS.md                  (§2, §9, §11, §12)
configs/sim_default.toml                     ([hedge] section, disabled by default)
```

## What to build

### 1. `hedge.py` — `HedgeAccount`

A short ETH-perp overlay. State: `short_size` (ETH), `entry_mark` (USDC/ETH),
`realized_pnl`, `funding_pnl`, `fees_paid`. Methods (all return signed USDC where
applicable):

- `mark_pnl(mark) = short_size * (entry_mark - mark)` — a short gains when the
  mark falls.
- `trade_to(target_short, mark) -> float` — trade `|target - short_size|` at
  `mark`, charge `fee = |Δ| * mark * fee_bps / 1e4` (returned as a **positive
  magnitude**), realize the mark PnL into `realized_pnl`, set
  `entry_mark = mark`, `short_size = target_short`.
- `accrue_funding(rate, mark) -> float` — `short_size * mark * rate` (a short
  **receives** when the rate is positive; signed).
- `equity(mark) = realized_pnl + mark_pnl(mark) + funding_pnl - fees_paid`.

Zero/negative sizes and non-finite marks raise. Unit-test every branch.

### 2. `MarketView.funding` + `funding_at`

Add an **additive** defaulted field `funding: pl.DataFrame | None = None` and
`funding_at(timestamp) -> tuple[float, float] | None` returning the as-of
(backward) `(funding_rate, mark_price)` — `None` before the first settlement.
`build_market_view` loads `stream=funding` when the snapshot has it (public data
API only); a view without funding is valid and accrues 0. `mark_price` may be
null in 2022–23 (2005/3288 rows) — **the hedge marks against the sim's own
reference price**, not the stream mark; the stream supplies the funding rate
only. Document that.

### 3. `HedgeConfig` + reward flag

`[hedge]`: `enabled = false`, `fee_bps = 5.0`, `funding_enabled = true`,
`ratio = 1.0` (the deterministic overlay's target), `ratios = [0, 0.25, 0.5, 0.75, 1.0]`
(S20b). `RewardConfig.hedge_enabled = true` gates the hedge terms for the RQ1
ablation, exactly like the existing flags.

### 4. Env wiring (`lp_env.py`)

- `reset()`: build/reset the hedge account; the initial target short is
  `ratio * position.amount1(sqrt)` at the entry mark, traded at reset (fee
  charged). A disabled hedge leaves it at 0.
- `step()`: after the market advances, re-hedge to
  `ratio * position.amount1(sqrt)` at the current mark, accrue any 8 h funding
  settlements in `(t_{k-1}, t_k]` via `funding_at`, and record the step's
  `hedge_pnl` (Δ equity excluding funding/fees), `hedge_funding`, `hedge_fee`.
- The hedge is **not** rebased at an LP rebalance — the target tracks the live
  delta, so a rebalance changes the target and the next step re-hedges.
- `info` carries the hedge breakdown under the existing `reward_breakdown_raw`.

### 5. Observation (append; ADR-029 pattern)

Append three fields in this order: `hedge_short_ratio` (`short_size /
max(eth_delta, eps)`; 0 when disabled), `hedge_equity_frac` (`hedge equity /
initial capital`), `funding_rate_recent` (as-of 8 h rate; 0 when absent).
`OBSERVATION_VECTOR_LENGTH` 21 → 24; the stale-checkpoint guard message updates.
Appending keeps existing indices stable.

### 6. Reward

`RewardBreakdown` gains defaulted `hedge_pnl`, `hedge_funding`, `hedge_fee`
(≤ 0) fields appended after `equity`; `compute_reward` gains keyword-only
defaulted args for them. `reward` sums the gated hedge terms;
`pnl_net` is the ungated economic total. The existing sign conventions and
ablation semantics are preserved.

### 7. Backtester + parity

`BacktestEngine` applies the same deterministic overlay (same `HedgeConfig`) so
the sim-vs-backtester comparison is like-for-like; the parity report gains
defaulted hedge drift fields and `passed` includes them. The backtester's
decision log records the hedge trade per decision.

## Tests

1. `test_hedge.py`: mark PnL signs (short gains on a fall), trade fee on
   `|Δ|`, funding sign (positive rate pays a short), equity continuity across a
   trade, zero/disabled paths, error paths.
2. Env: with `hedge.enabled=true, ratio=1`, a pure price move leaves total
   equity ≈ flat (LP + hedge) up to fees/funding; funding accrues only on the 8 h
   grid; the hedge trades at the decision price (ADR-031); a disabled hedge is
   byte-identical to today's reward/observation for the first 21 dims.
3. Reward: hedge terms are gated by `hedge_enabled` and `pnl_net` is ungated;
   normalization scales them.
4. Backtester: the same overlay produces the same hedge series as the env on the
   identical window; parity `passed` with hedge drifts ≈ 0.
5. Observation: length 24, the three appended fields present and zero when the
   hedge is disabled/absent.

## Acceptance criteria

- Full suite `uv run pytest -q -n 4` green with the 90 % coverage gate.
- `[hedge] enabled = false` in `configs/sim_default.toml` → all existing
  behaviour and tests unchanged (byte-identical reward for the LP-only path).
- With `enabled = true`, report: the hedged vs unhedged equity std over ≥5 eval
  windows (the variance reduction), the funding received, and the perp cost — as
  numbers in the PR.

## Out of scope

- The agent-chosen hedge ratio (S20b) — `HedgeConfig.ratios` is defined here but
  unused.
- Risk-sensitive metrics (S21).
