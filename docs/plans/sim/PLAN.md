# Undertow — `undertow.sim` Implementation Plan

**Plan id:** `sim-v1`
**Scope:** the AMM / concentrated-liquidity simulator, RL environment, backtester, and training harness
**Depends on:** `undertow.data` (completed waves 0–7)
**Source requirement:** `~/Documents/fyp/lesson_plan/10_data_evaluation_roadmap.md` §10.4–§10.6

> **Roadmap quote that defines "done":** *"A working PPO agent that learns to provide liquidity in
> a concentrated-liquidity AMM, evaluated against passive LPs and the HODL baseline across multiple
> market regimes."* — §10.6, Weeks 2–4.

---

## 0. Read this first (all agents)

1. This plan is **normative**. Do not redesign it. If you believe a task's design is wrong, write an
   ADR note in `docs/decisions/NNN-slug.md` and flag it in your PR description — do not silently
   deviate, because other agents are coding against your interface *as written here*.
2. **`CONTRACTS.md` in this directory is the interface contract.** Every type signature, default value,
   and unit convention in it is a cross-task contract. Implement exactly that. Extending is allowed
   (add a keyword-only argument with a default); renaming or retyping is not.
3. **ADRs live in `docs/decisions/`.** Write the full decision record to
   `docs/decisions/NNN-slug.md` and cite *that* path from any code comment or docstring.
4. Obey `undertow/AGENTS.md` without exception: branch first → write code + tests → `uv run pytest`
   green → **`code-reviewer` subagent review** → address findings → commit → push → open PR against
   `main`. **Never** commit to `main`, **never** merge your own PR, **never** commit `.pi/`.
5. One task = one branch = one PR. Your task file lists the **files you own**.
6. Update `STATUS.md` (in this directory) when you start and when you finish. It is the handoff board.

---

## 1. What is being built

`undertow.sim` is the research engine: a self-contained AMM simulator, RL environment, and training
harness. It is **strictly separated** from `undertow.data` — the sim never imports the data package,
and vice versa. Data flows from `undertow.data` into `undertow.sim` as plain arrays via the caller.

| Module | Purpose | Owning task |
|---|---|---|
| `math` | Fixed-point, tick, and liquidity math | S00 |
| `config` | Frozen `SimConfig` tree + TOML loader | S00 |
| `pool` | Concentrated-liquidity pool state machine | S00 |
| `price` | Price processes (GBM, regime-switching, replay) | S00 |
| `env` | Gymnasium environment wrapping pool + price + friction | S00 |
| `metrics` | Sharpe, Sortino, MDD, VaR/CVaR, PnL decomposition | S00 |
| Training harness | PPO trainer with stable-baselines3 | S01 |
| Backtester | Multi-episode evaluation runner | S02 |
| Baselines | Passive LP strategies (full-range, fixed-width, HODL) | S03 |
| Analysis | Regime-conditional metrics, position behavior visualisation | S04 |

---

## 2. Target architecture (file ownership map)

```
src/undertow/sim/
├── __init__.py          S00   public API surface
├── math.py              S00   self-contained tick/liquidity/fixed-point
├── config.py            S00   frozen SimConfig tree + load_sim_config TOML
├── pool.py              S00   ConcentratedLiquidityPool (swap/mint/burn/collect)
├── price.py             S00   GBMPrice, RegimeSwitchingPrice, ReplayPrice
├── env.py               S00   UndertowEnv (gymnasium.Env)
├── metrics.py           S00   Sharpe, Sortino, MDD, VaR/CVaR
├── train/
│   ├── __init__.py      S01   training package
│   └── ppo_runner.py    S01   PPO training loop (stable-baselines3)
├── backtest/
│   ├── __init__.py      S02   backtest package
│   └── runner.py        S02   episode runner + decomposition
├── baselines/
│   ├── __init__.py      S03   baseline strategies
│   ├── full_range.py    S03   full-range LP
│   ├── fixed_width.py   S03   fixed-width LP
│   └── hodl.py          S03   buy-and-hold
└── analysis/
    ├── __init__.py      S04   analysis package
    └── viz.py           S04   regime-conditioned plots + tables

tests/sim/
├── conftest.py           S00   shared fixtures
├── test_math.py          S00
├── test_config.py        S00
├── test_pool.py          S00
├── test_price.py         S00
├── test_env.py           S00
├── test_metrics.py       S00
├── test_ppo_runner.py    S01
├── test_runner.py        S02
├── test_baselines.py     S03
└── test_viz.py           S04

configs/
└── sim_default.toml      S00   default SimConfig

docs/plans/sim/
├── README.md             S00
├── PLAN.md               S00   this file
├── CONTRACTS.md          S00   frozen interfaces
├── STATUS.md             S00   handoff board
└── tasks/                Sxx   one per task
```

---

## 3. Task graph

```
wave 0   S00  scaffold (math, config, pool, price, env, metrics)
              │
wave 1   S01  RL training harness (PPO runner)
              │
wave 2   S02  backtester + episode runner
              │
wave 3   S03  baselines (full-range, fixed-width, HODL)
              │
wave 4   S04  analysis + visualisation
```

### 3.1 Dependency table (authoritative)

| Task | Title | Depends on | Blocks | Wave | Est. size |
|---|---|---|---|---|---|
| **S00** | sim scaffold | — | all | 0 | L |
| **S01** | PPO training harness | S00 | S02 | 1 | L |
| **S02** | backtester + episode runner | S00,S01 | S03,S04 | 2 | M |
| **S03** | baseline LP strategies | S00,S02 | S04 | 3 | M |
| **S04** | analysis + visualisation | S00,S02,S03 | — | 4 | M |

Waves are serial here (unlike the data pipeline) because later tasks genuinely depend on earlier ones.

---

## 4. Conventions every task must follow

- **No `undertow.data` imports.** The scaffold test `test_sim_does_not_import_data` enforces this.
- **Type hints** on all public functions; `from __future__ import annotations` at the top.
- **Big integers stay integers.** No float coercion on tick math or liquidity arithmetic.
- **Config/constants separate from logic.** `config.py` holds all defaults and look-up tables.
- **Unit tests are non-negotiable.** Every new module ships with `test_*.py` that asserts real behavior.
- **Gymnasium API compatibility.** `UndertowEnv` conforms to the Gymnasium `Env` protocol; `check_env` passes.
- **Self-contained price processes.** Price processes carry their own RNG and don't depend on external data.

---

## 5. Definition of done (per task)

1. Files listed in "Files you own" exist, are typed, and contain no TODO stubs.
2. Tests exist and assert **real behavior** — values, invariants, error paths, boundary cases.
3. `uv run pytest` is green.
4. `code-reviewer` has reviewed the change and returned **APPROVE**, or findings are fixed.
5. The public interface matches `CONTRACTS.md`.
6. A PR is open against `main`; `STATUS.md` is updated.

---

## 6. Pinned decisions

| Decision | Value | Rationale |
|---|---|---|
| RL library | `stable-baselines3 >= 2.9` | ADR-004; community standard, Gymnasium-compatible, battle-tested PPO |
| Primary pool | USDC/WETH 0.30% (tick spacing 60) | matches `undertow.data`'s primary pool |
| Default price | $3000 ETH | approximate mid-range for the study window |
| Agent capital | $100,000 USDC | realistic retail-sized LP |
| Marginal agent | `True` (default) | agent liquidity doesn't move pool price — simpler, validated in literature |
| step_minutes | 60 | hourly decision granularity; 720 steps per 30-day episode |
| Reward components | fees − gas − IL − risk_penalty | eq. 2 from roadmap; each component individually ablatable |
| Action space | discrete grid: 9 center offsets × 4 widths + hold = 37 actions | matches the anchor paper's action discretisation |

---

## 7. Risk register

| Risk | Owner task | Mitigation |
|---|---|---|
| Tick ordering bug (higher price = lower tick) | S00 | explicit docs + tests; `reset()` and `_rebalance()` tested |
| Fee-growth math diverges from EVM | S00 | exact integer Q128 math, wrapping arithmetic, test vectors |
| Agent learns to avoid gas costs | S01 | gas components individually ablatable in reward config |
| Overfitting to one regime | S02,S04 | regime-switching price process + regime-conditional evaluation |
| Look-ahead bias in backtest | S02 | step-by-step price replay; `Observation` built before `step()` |
| Empty position after rebalance | S00 | fallback to minimum-width range when ticks collapse |