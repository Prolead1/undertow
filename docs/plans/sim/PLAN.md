# Undertow — `undertow.sim` Simulator, Backtester & RL Environment: Multi-Agent Execution Plan

**Plan id:** `sim-v1`
**Scope:** the Week-2-to-6 critical path of the thesis roadmap — a *correct* backtester, a *fast,
controllable* training simulator, a friction-realistic RL environment, the baseline ladder, and the
evaluation harness that produces the RQ1–RQ3 deliverables.
**Source requirement:** `~/Documents/fyp/lesson_plan/10_data_evaluation_roadmap.md` §10.2 (the two
artifacts and their four components), §10.2.3 (backtest loop), §10.3 (walk-forward / multi-seed /
per-regime / cost-honest evaluation), §10.4 (metrics eqs 7–10), §10.5 (baseline ladder), §10.6 (Weeks
2–6 milestones), §10.7 (risks). Supporting: `09_research_gap_problem_statement.md` §9.2 (G1–G4, reward
eq (1)), §9.3 (RQ1–RQ3 experiments); `05_concentrated_liquidity_lp_problem.md` §5.3–5.6 (position math,
eqs 8–11, IL), §5.8 (rebalancing costs); `06_stochastic_control.md` §6.1–6.3 (state/action/impulse
formulation); `07_reinforcement_learning.md` §7.8–7.9 (env design, pitfalls);
`08_rl_market_making_sota.md` §8.4 (anchor setup + PPO hyperparameters), §8.6 (baselines), §8.10
(honest reward, eq 6).

> **Roadmap quotes that define "done":** Week 2 — *"A correct backtester with the three
> passive/HODL/V2 baselines reproduced; a fee-accounting unit test."* Week 3 — *"A simulator with a
> config for each price mode; a working RL training loop."* Week 4 — *"An ablation table (full vs
> each partial friction, per metric) — the RQ1 deliverable."* Week 5 — *"A per-regime results matrix
> + train→test shift analysis — the RQ2 deliverable."* Week 6 — *"An on-chain backtest + a quantified
> simulation-to-reality gap — the RQ3 deliverable."* — §10.6.

---

## 0. Read this first (all agents)

1. This plan is **normative**. Do not redesign it. If you believe a task's design is wrong, write an
   ADR note in `docs/decisions/NNN-slug.md` and flag it in your PR description — do not silently
   deviate, because other agents are coding against your interface *as written here*.
2. **`CONTRACTS.md` in this directory is frozen.** Every dataclass field, function signature, unit,
   sign convention and protocol method in it is a cross-task contract. Implement exactly that.
   Extending is allowed (add a field with a default, add a keyword-only argument with a default);
   renaming, retyping, or reordering is not.
3. **ADRs live in `docs/decisions/`** (shared with the data plan; numbering continues from the highest
   existing number). Cite the ADR path from any code comment or docstring that depends on it.
4. Obey `undertow/AGENTS.md` without exception: branch first → write code + tests → `uv run pytest`
   green → **`code-reviewer` subagent review** → address findings → commit → push → open PR against
   `main`. **Never** commit to `main`, **never** merge your own PR, **never** commit `.pi/`.
5. One task = one branch = one PR. **Work in your own git worktree**, so parallel agents never edit
   the same files on disk; when your branch is pushed and the PR is open, remove that worktree. Your
   task file lists the **files you own**. Do not touch files owned by another task.
6. Update `STATUS.md` (in this directory) when you start and when you finish. It is the handoff board.
7. **The `undertow.data` ↔ `undertow.sim` boundary is law.** `undertow.sim` imports **only** names
   exported from the top-level `undertow.data` public API (`undertow/data/__init__.py`). Reaching into
   `undertow.data.transforms.feegrowth` or any other internal module is a **Critical** review finding.
   S02 exists precisely to widen that public surface where the sim needs it.

---

## 1. What is being built

Two artifacts, **strictly separated** (roadmap §10.2.1 — "conflating them is the classic failure that
produces unrealistically-positive results"), plus the harness that turns them into thesis results:

- **The backtester** (`undertow.sim.backtest`) replays a **frozen** policy over the *real* block-ordered
  event tape from `undertow.data`, with exact integer fee accrual (the data module's fee-growth
  engine), real per-block gas, and real reference prices. It is a *measurement instrument*: correct,
  not fast. This is where **G3** lives.
- **The training simulator** (`undertow.sim.env`) is a fast, seeded, Gymnasium environment composed of
  four swappable models (roadmap §10.2.2): a **price process** (replay or calibrated regime-switching
  jump-diffusion, eq (1) of §10.2.2), a **pool model** (CPMM-in-range mechanics of ch. 5), a **friction
  model** (gas + slippage), and the **reward** (eq (1) of §9.2). This is where **G1/G2** live.

The components and who owns them:

| # | Component | Why it exists (source §) | Owning task |
|---|---|---|---|
| 1 | market view + walk-forward split | one look-ahead-safe window onto the data (§10.3.1) | S03 |
| 2 | position & valuation math | eqs (8)–(11) of §5.3, IL vs HODL of §5.6 | S04 |
| 3 | performance metrics | eqs (7)–(10) of §10.4, PnL decomposition, per-regime slicing | S05 |
| 4 | price processes | replay + calibrated MRSJD — G2 (§10.2.2, §9.2) | S06 |
| 5 | pool engine | tick-lattice fee accrual & range mechanics (§5.4, §10.2.2) | S07 |
| 6 | friction models | gas `G_t` + slippage `S_t` — G1 (§5.8, §9.2) | S08 |
| 7 | reward | eq (1) of §9.2 with ablation flags — the RQ1 instrument | S09 |
| 8 | baseline ladder | HODL / passive / V2 / τ-reset / cost-aware (§10.5, §8.6) | S10 |
| 9 | backtester | frozen-policy replay on real events — G3 (§10.2.3) | S11 |
| 10 | Gymnasium environment | the MDP of §6.2 wired from #2–#7 | S12 |
| 11 | training harness | PPO, ≥5 seeds, checkpoints (§7.8, §8.4.4, §10.3) | S13 |
| 12 | parity & look-ahead validation | sim vs backtester on identical inputs (§10.7) | S14 |
| 13 | evaluation runner | ablation table, regime matrix, sim-to-reality gap (RQ1–RQ3) | S15 |
| 14 | CLI + public API + docs | `undertow-sim train|backtest|evaluate|ablate` | S16 |

**Out of scope for this plan** (do not build): re-implementations of published RL baselines (ZeroSwap,
Chionas et al. — §10.5 item 4; they plug into the S10 `Policy` protocol and get their own plan),
interpretability surrogates (RQ4), multi-pool routing, MEV, hyperparameter search beyond the pinned
modest grid, and any change to `undertow.data` beyond S02's additive export extension.

**The walk-forward split enters here.** The data plan deliberately shipped the full contiguous window
and left the split to this plan (data `PLAN.md` §1). S03 owns it: the pinned boundaries (§7 below) are
recorded through `WindowConfig.train_end_utc` / `eval_start_utc` so the split is provenance-tracked in
the dataset manifest, and `MarketView` makes it *impossible* for a consumer to read evaluation-window
rows while building a training view.

---

## 2. Target architecture (file ownership map)

```
undertow/
├── pyproject.toml                          S00 (adds sim deps) · S16 (adds undertow-sim script)
├── src/undertow/data/__init__.py           S02   (additive export extension ONLY)
├── src/undertow/sim/
│   ├── __init__.py                         S16   (public API surface; stub exists on main)
│   ├── types.py                            S01   enums, aliases, exception hierarchy
│   ├── config.py                           S01   SimConfig tree: episode, action grid, frictions,
│   │                                             reward, split, training, backtest
│   ├── marketview.py                       S03   Dataset -> look-ahead-safe MarketView + split
│   ├── core/
│   │   ├── __init__.py                     S04
│   │   ├── position.py                     S04   amounts/value/IL math (eqs 8–11), Position
│   │   └── pool.py                         S07   tick-lattice pool state + agent fee accrual
│   ├── prices/
│   │   ├── __init__.py                     S06
│   │   ├── base.py                         S06   PriceProcess protocol
│   │   ├── replay.py                       S06   reference-feed replay source
│   │   └── regime_jump.py                  S06   calibrated Markov-regime-switching jump-diffusion
│   ├── frictions/
│   │   ├── __init__.py                     S08
│   │   ├── gas.py                          S08   GasModel: replay / flat / spike
│   │   └── slippage.py                     S08   SlippageModel for rebalancing trades
│   ├── policies/
│   │   ├── __init__.py                     S10
│   │   ├── base.py                         S10   Policy protocol + Action types
│   │   └── baselines.py                    S10   HODL, passive ranges, full-range V2, τ-reset,
│   │                                             cost-aware rebalance
│   ├── env/
│   │   ├── __init__.py                     S09
│   │   ├── reward.py                       S09   eq (1) of §9.2 + RewardBreakdown + ablation flags
│   │   ├── observations.py                 S12   observation builder (§6.1.1 state)
│   │   └── lp_env.py                       S12   Gymnasium Env wiring prices+pool+frictions+reward
│   ├── backtest/
│   │   ├── __init__.py                     S11
│   │   ├── engine.py                       S11   frozen-policy event replay (§10.2.3 skeleton)
│   │   └── ledger.py                       S11   equity curve, decision log, cost ledger
│   ├── metrics/
│   │   ├── __init__.py                     S05
│   │   ├── performance.py                  S05   Sharpe/Sortino/MaxDD/VaR/CVaR (eqs 7–10)
│   │   ├── decompose.py                    S05   PnL = ΣF − ΣΔIL − Σ(G+S) decomposition
│   │   └── regimes.py                      S05   per-regime metric slicing
│   ├── train/
│   │   ├── __init__.py                     S13
│   │   ├── ppo_runner.py                   S13   PPO training loop (SB3 or vendored; ADR at S00)
│   │   └── runs.py                         S13   multi-seed orchestration, checkpoints, run manifest
│   ├── validation/
│   │   ├── __init__.py                     S14
│   │   ├── parity.py                       S14   sim vs backtester on identical inputs
│   │   └── lookahead.py                    S14   look-ahead probes for env + backtester
│   ├── evaluate/
│   │   ├── __init__.py                     S15
│   │   ├── protocol.py                     S15   walk-forward, multi-seed evaluation runner
│   │   ├── ablation.py                     S15   RQ1 friction-ablation matrix
│   │   ├── gap.py                          S15   RQ3 sim-to-reality gap
│   │   └── report.py                       S15   results tables (markdown/CSV artifacts)
│   └── cli.py                              S16   `undertow-sim train|backtest|evaluate|ablate|info`
├── configs/sim_default.toml                S00 skeleton → S01 fills (shared, §2.1)
├── tests/sim/conftest.py                   S00   seeded-rng fixtures, tiny MarketView factory hook
├── tests/sim/...                           each task ships its own tests + fixtures
├── docs/running_experiments.md             S16
├── docs/decisions/NNN-*.md                 whoever raises the ADR (§0.3); S00 owns the RL-library ADR
└── docs/plans/sim/                         this directory (PLAN, CONTRACTS, STATUS, tasks/)
```

Sub-package `__init__.py` files belong to the first task that creates the package, as annotated above
(`env/` → S09, since reward lands one wave before the env).

Rules the reviewer will enforce: no god modules (an env that also implements price processes or
metrics is a rejection), config/constants out of logic, type hints on every public function,
`numpy.random.Generator` injected — never global seeding — and **no `undertow.data` internal imports**.

### 2.1 The deliberately-shared paths

| File | Created by | Also edited by | Rule |
|---|---|---|---|
| `pyproject.toml` | exists | S00 (adds sim deps), S16 (adds `undertow-sim` script entry) | S00's diff touches `[project].dependencies` (and optionally a `[project.optional-dependencies]` group) only; S16's diff touches `[project.scripts]` only. |
| `src/undertow/sim/__init__.py` | exists (stub on `main`) | S16 (replaces with the real surface) | Nobody between S00 and S16 adds an export. Modules are importable by full path until then. |
| `src/undertow/data/__init__.py` | exists (data T16) | S02 (appends exports) | S02 appends names to `__all__` + import lines. It must not reorder or remove anything — the data contract allows extension only. |
| `configs/sim_default.toml` | S00 (empty section skeleton) | S01 (fills the real field set) | S01 owns field semantics; S00 only creates the sections. |
| `tests/sim/conftest.py` | S00 | S03 (adds the shared `tiny_market_view()` factory) | S00 ships seeds/markers; S03 appends the factory fixture other tasks reuse. Neither edits the other's fixtures. |
| `README.md` | exists | S16 (Status + sim Quickstart) | Status-checklist lines and the sim Quickstart section only. |

Everything else: if a path is not in your task's "Files you own", do not create or edit it. Need a
change there? Ask the orchestrator, or write an ADR.

---

## 3. Task graph

Read this as "an arrow means *needs the interface of*". Waves are the parallel batches; §9 has the
launch order.

```
wave 0   S00  scaffold + RL-library ADR
              │
wave 1   S01  types/config                 S02  data public-API extension
              │                                 │
wave 2   S03 marketview   S04 position   S05 metrics   S06 prices
              │               │              │             │
wave 3   S07 pool engine  S08 frictions  S09 reward   S10 baselines
              │               │              │             │
wave 4        ├──────► S11 backtester ◄──────┤   S12 gym env ◄─┘
              │               │              │        │
wave 5        │        S13 training ◄────────┴────────┤
              │               │        S14 parity ◄───┤
              │               │              │        │
wave 6        └──────► S15 evaluation ◄──────┴────────┘
                              │
wave 7                 S16 CLI + public API + docs
```

Edges the diagram flattens, spelled out:

- S02 → S04, S07, S11 (everyone touching fixed-point/liquidity math or the fee-growth engine)
- S03 → S06 (calibration input *data*), S08 (gas stream *data*), S11, S12, S15
- S04 → S07, S10, S11, S12 (everyone valuing a position)
- S05 → S13 (eval callbacks), S15
- S09 → S12 (reward is computed inside `step()`)
- S10 → S11 (backtester runs policies), S13 (eval-time comparison policies), S15
- S14 → S15 (evaluation may not run until parity holds — it *gates*, it does not export code)

### 3.1 Dependency table (authoritative)

| Task | Title | Depends on | Blocks | Wave | Est. size |
|---|---|---|---|---|---|
| **S00** | sim scaffold: deps, test markers, config skeleton, RL-library ADR | — | all | 0 | S |
| **S01** | `types.py` + `config.py` (episode, action grid, frictions, reward, split, training) | S00 | S03–S16 | 1 | M |
| **S02** | `undertow.data` public-API extension (additive exports) | S00 | S03,S04,S07,S11 | 1 | S |
| **S03** | MarketView + walk-forward split | S01,S02 | S11,S12,S15 (+data for S06,S08) | 2 | M |
| **S04** | Position & valuation math (eqs 8–11, IL) | S01,S02 | S07,S10,S11,S12 | 2 | M |
| **S05** | Metrics (eqs 7–10, decomposition, regime slicing) | S01 | S13,S15 | 2 | M |
| **S06** | Price processes (replay + calibrated MRSJD) | S01 (+S03 *data*) | S12 | 2 | L |
| **S07** | Pool engine (tick lattice, agent fee accrual) | S02,S04 | S12,S14 | 3 | L |
| **S08** | Friction models (gas + slippage) | S01 (+S03 *data*) | S11,S12 | 3 | M |
| **S09** | Reward + `RewardBreakdown` + ablation flags | S01 | S12 | 3 | S |
| **S10** | Policy protocol + baseline ladder | S01,S04 | S11,S13,S15 | 3 | M |
| **S11** | Backtester (frozen-policy event replay) | S02,S03,S04,S08,S10 | S14,S15 | 4 | L |
| **S12** | Gymnasium environment | S03,S04,S06,S07,S08,S09 | S13,S14 | 4 | L |
| **S13** | Training harness (PPO, multi-seed, checkpoints) | S05,S10,S12 | S15 | 5 | L |
| **S14** | Parity + look-ahead validation | S11,S12 | S15 (gate) | 5 | M |
| **S15** | Evaluation runner (RQ1 ablation, RQ2 regime matrix, RQ3 gap) | S05,S10,S11,S13,S14 | S16 | 6 | L |
| **S16** | CLI + public API + docs | S11,S13,S15 | — | 7 | M |

**Code dependency vs data dependency** (same distinction as the data plan §3.1): S06 and S08 *consume
tables* (`REFERENCE_SCHEMA`, `GAS_SCHEMA` data reached through the `MarketView` interface) but code
against fixtures built to those schemas — so they sit in waves 2–3 without waiting for real data at
test time. Consequences: **your fixtures are load-bearing** (trim values from real snapshot data where
possible; synthesize only what you must, and say which), and a consumer in an earlier wave than its
data producer owns its own fixture (`CONTRACTS.md` §10 assigns every fixture).

### 3.2 Stacked-branch rule (how to work before a dependency is merged)

Identical to the data plan: dependency merged → branch off `main`; dependency PR still open → branch
off the dependency branch and open your PR with `Stacked on #<dep PR> (branch <dep-branch>). Merge
that first.` at the top of the body. Never copy a dependency's files; never re-implement a
dependency's function "temporarily" — if blocked, say so in `STATUS.md` and stop.

---

## 4. Conventions every task must follow

- **Boundary discipline.** `undertow.sim` imports only top-level `undertow.data` exports (§0.7).
  `undertow.data` never imports `undertow.sim`. Grep-able rule: the string
  `from undertow.data.` (with a trailing dot) must not appear anywhere in `src/undertow/sim/`.
- **Two numeric worlds, never mixed.** The **backtester's fee path is exact integer math** (the data
  module's `FeeGrowthTracker`, Q128 ints) — converting an accumulator to float before the final
  division is Critical, exactly as in the data plan. The **simulator is float64** end to end for
  speed (millions of steps, §10.2.1). S14 quantifies the drift between the two; nobody hides it.
- **Determinism.** Every stochastic component takes an explicit `numpy.random.Generator` (or an
  integer seed at the config boundary that constructs one). No `np.random.*` module-level calls, no
  global seeding, no wall-clock in results. Same config + same seed ⇒ bit-identical episode streams
  and training data order. Run artifacts record config hash, seed, git commit, and library versions.
- **No look-ahead, ever.** An observation or decision at step *t* may only be built from data at
  blocks/bars ≤ *t*. All rolling features are backward-looking, closed on the right. `MarketView` is
  the only door to historical data and it enforces the train/eval wall. Each of S06, S07, S11, S12
  ships an explicit look-ahead unit test; S14 attacks the composed system.
- **Frozen means frozen.** The backtester takes a `Policy` and calls only its `act()`; there is no
  learning API on the protocol at all, so "the policy updated during backtest" is unrepresentable.
- **Cost honesty (§10.3.4).** Every reported PnL/equity number is net of gas and slippage. Gross
  numbers exist only inside `RewardBreakdown` / the decomposition ledger, clearly labelled.
- **Units and numeraire.** Wealth, PnL, fees, costs are in **token0 units (USDC)**. Gas is computed in
  wei and converted at that block's reference ETH price. Prices follow the data module's orientation
  (token1 per token0 as delivered by `price_reference` / `price_pool` — see data `CONTRACTS.md` §3.0);
  never re-derive an orientation.
- **Timestamps** are timezone-aware UTC. Sim time is indexed by `seq` (tape) or bar index (replay);
  wall-clock never enters the state.
- **Network and GPUs are forbidden in tests.** Training tests run tiny CPU configs (hundreds of
  steps). Anything slow is marked `@pytest.mark.slow` (S00 registers the marker; default-on but
  budgeted); anything live-network reuses the existing `network` marker and skip hook.
- **Logging** via `logging.getLogger("undertow.sim.<module>")`. No `print` in library code. Training
  progress goes through the run-manifest/metrics writer, not stdout.
- **Type hints** on all public functions; `from __future__ import annotations` everywhere.

---

## 5. Definition of done (per task)

Identical to the data plan, restated: (1) owned files exist, typed, no TODO stubs on the public path;
(2) tests assert real behavior — values, invariants, error paths, golden numbers from the lesson-plan
worked examples — not "does not raise"; (3) `uv run pytest` green, summary pasted in the PR body;
(4) `code-reviewer` APPROVE (after fixes if needed), summarized in the PR body; (5) public interface
matches `CONTRACTS.md` exactly; (6) PR open against `main` from a `feature-sim-*` branch;
`STATUS.md` updated.

---

## 6. Definition of done (whole plan)

- **Week-2 bar (backtester):** `uv run undertow-sim backtest --config configs/sim_default.toml
  --policy passive_narrow` (and `hodl`, `full_range_v2`, …) replays the real event tape over the
  evaluation window and emits an equity curve, decision log, and full cost/PnL decomposition. Its fee
  accrual agrees with the data module's Collect reconciliation within the tolerance that pipeline
  already established (backtest fees come from the same fee-growth engine — S11 asserts this wiring).
- **Week-3 bar (simulator + training):** `undertow-sim train` trains PPO on the environment in both
  price modes (replay and calibrated MRSJD), across the pinned ≥5 seeds, writing reproducible
  checkpoints + a run manifest. A re-run with the same config and seeds reproduces the learning curves.
- **Week-4 bar (RQ1):** `undertow-sim ablate` produces the ablation table — {full, no_gas,
  static_fee, no_il} × {PnL, Sharpe, MaxDD, turnover, decomposition} — from §9.3.3's experiment spec.
- **Week-5 bar (RQ2):** the evaluation runner produces the per-regime results matrix (agent + every
  baseline × four regimes × metrics), walk-forward, mean ± std over seeds.
- **Week-6 bar (RQ3):** the gap report quantifies `Δ = sim_PnL − onchain_PnL` for the frozen policy on
  identical evaluation windows, decomposed by cost term.
- **Parity:** S14's report shows the simulator and backtester, fed the identical price path, positions
  and gas series, agree on fees/IL/PnL within a stated, justified tolerance — and that tolerance with
  its justification appears in the report.
- Boundary intact: no `undertow.data` internal import in sim, no `undertow.sim` import in data.
- `docs/running_experiments.md` walks a fresh clone from `undertow-data snapshot` to a filled
  ablation table.

---

## 7. Pinned decisions (so agents don't have to choose)

| Decision | Value | Rationale |
|---|---|---|
| Env API | Gymnasium (`gymnasium.Env`, `reset(seed=…)`, 5-tuple `step`) | de-facto standard; SB3-compatible |
| Algorithm | **PPO with clipping**; DQN out of scope | roadmap §10.7 "prefer clarity over SOTA"; anchor uses PPO |
| PPO defaults | 2×256 MLP, γ=0.99, GAE λ=0.95, clip 0.15 | anchor's setup (§8.4.4) — a defensible, citable starting point; sweepable in config |
| RL library | `stable-baselines3` **if** it installs on the project venv (Python 3.14); else a vendored single-file PPO in `train/ppo_runner.py` | S00 verifies installability and records ADR either way |
| Seeds | `(0, 1, 2, 3, 4)`, pinned in config | §10.3.2's ≥5-seed floor; fixed so results are reproducible |
| Decision cadence | every **10 minutes** (10 reference bars in sim; nearest tape block boundary in backtest), configurable | matches §10.2.3's "decision boundaries every K blocks"; identical cadence in both artifacts so parity (S14) is meaningful |
| Action space | discrete, factored: `{hold} ∪ {rebalance(center_offset c, width w)}`; `c ∈ {−4,…,+4}` tick-spacings around current tick, `w ∈ {1, 2, 5, 10, 25, 50}` tick-spacings per side; ranges snapped to tick spacing | §7.8's discrete-menu recommendation, PPO-friendly; hybrid continuous-width head is future work |
| Exit action | not in the action space v1 | keeps the MDP episodic-by-window; §8.6.3 optimal-exit noted as future work |
| Episode | replay: one contiguous 30-day window sampled (seeded) from the **train** split, 10-min steps; MRSJD: same length from the fitted model | long enough for regime persistence, short enough for return estimates (§7.8 item 5) |
| Agent capital | 100,000 USDC notional at episode start | small vs pool TVL — keeps the marginal assumption honest |
| Marginal-agent assumption | agent liquidity never moves the price path; fee share = `L_agent / (L_agent + L_pool_active)` per step, using the tape's recorded active liquidity | standard replay assumption; recorded in every run manifest as `agent_is_marginal = true` |
| ΔIL definition | change in (position value − HODL value) marked at the **reference** price, per step | §10.1.7: IL/LVR must be measured against the external price, not the pool's echo |
| Risk penalty | `λ = 0` default; `σ_PnL` = rolling std of per-step PnL over the trailing day when enabled | eq (1) §9.2 marks it optional; ablatable dial (§7.10) |
| Reward scale | raw USDC per step, normalized by initial capital at the env boundary (`reward = ΔPnL_net / W_0`) | keeps PPO value targets O(1) without hiding costs |
| Gas units per action | mint 460k · burn 215k · collect 130k · rebalance swap 150k (constants in `config.py`, each with a source comment; sweepable) | representative NFT-position-manager costs; exact per-tx receipts are future refinement |
| Gas → USD | `base_fee_per_gas × (1 + tip_surcharge_pct/100) × gas_units`, converted at that block's reference ETH price | ADR-005 flat tip surcharge; `tip_surcharge_pct` defaults to 3 (sweepable via S15 ablation). Uses the tape's joined gas columns; §10.2.3 "that block's gas price" |
| Slippage | proportional model: `S = notional × (pool_fee_tier + fixed_impact_bps)`, `fixed_impact_bps = 5` default | rebalance trades pay the pool fee plus impact; honest floor, sweepable |
| Walk-forward split | train `2022-01-01 → 2023-12-31`, eval `2024-01-01 → 2024-12-31` UTC, recorded via `WindowConfig.train_end_utc`/`eval_start_utc` | uses the hook the data plan reserved; eval spans distinct regimes |
| Price replay source | reference feed 1-minute closes (`REFERENCE_SCHEMA`) | §10.2.2 mode (a); the pool price is an echo (§10.1.7) |
| Calibrated model | Markov-regime-switching jump-diffusion, eq (1) §10.2.2: 4 latent states matching the regime labels, Student-t jump sizes, fit on the **train** split only | G2's named fix; fitting on eval data would be look-ahead |
| Regime labels | consumed from the data module's regime table only; never recomputed in sim | single source of truth; pre-committed thresholds live in `undertow.data.config` |
| Baseline ladder | `hodl` · `passive_narrow` (±5% around entry) · `passive_wide` (±20%) · `full_range_v2` · `tau_reset` (recenter when price exits, τ = entry width) · `cost_aware` (recenter at most daily, only if est. fee gain > gas) | §10.5 items 1–3 + 5; §8.6's DeployNarrow/DeployWide and τ-reset made concrete |
| Numeraire | token0 (USDC) | matches the data module's price orientation |
| Sim numeric type | float64 (simulator) / exact int (backtester fee path) | §4; parity quantifies the difference |

---

## 8. Risk register (mapped to §10.7 / §9.7)

| Risk | Owner task | Mitigation baked into the plan |
|---|---|---|
| Simulator unrealism → agent monetizes artifacts | S14 | parity vs backtester on identical inputs is a *blocking* gate for S15; the residual gap is reported, not hidden (it *is* the RQ3 finding) |
| Look-ahead bias | S03, S14 | MarketView is the only data door; backward-only windows; explicit look-ahead probes per component + composed |
| RL instability / irreproducibility | S13 | PPO+clipping, pinned seeds, seeded envs, run manifests with config hash + commit; modest grid only |
| Fee numbers wrong in backtest | S11 | fees come from the data module's already-reconciled `FeeGrowthTracker`, not a re-implementation; S11 tests pin the wiring |
| "RL doesn't beat passive after costs" | S15 | evaluation runner also emits the RQ-B break-even view (edge as a function of gas level) so either headline is a result |
| Sim too slow to train on | S07, S12 | minute-bar stepping (not per-swap), vectorized env support, float64 fast path; a rollout-throughput benchmark test with a budget |
| Two numeric worlds diverge silently | S14 | drift quantified with stated tolerance; tolerance justified in the parity report |
| RL library doesn't support the venv's Python | S00 | installability check + vendored-PPO fallback decided by ADR before any dependent task starts |
| Two agents editing one file | all | per-agent worktrees + file-ownership map (§2) + stacked-branch rule (§3.2) |

---

## 9. Launch instructions (for the orchestrating agent / human)

Each task file `tasks/S<nn>_*.md` is a **self-contained brief** — a fresh agent needs nothing but that
file, `CONTRACTS.md`, and `PLAN.md`. Launch pattern per task:

```
Agent(subagent_type="general-purpose",
      isolation="worktree",
      prompt="Read docs/plans/sim/PLAN.md and docs/plans/sim/CONTRACTS.md, then execute
              docs/plans/sim/tasks/S07_pool_engine.md end to end in the repo checkout, including
              the mandatory code-reviewer gate and the PR. Once the branch is pushed, remove the
              temporary worktree. Report the branch, PR, pytest summary and reviewer verdict.")
```

Sequencing:

- Wave 0: S00 alone (blocks everything — run to completion first; the RL-library ADR gates S13's design).
- Wave 1: S01, S02 in parallel.
- Wave 2: S03, S04, S05, S06 in parallel (4 agents; disjoint files).
- Wave 3: S07, S08, S09, S10 in parallel (4 agents; disjoint files).
- Wave 4: S11, S12 in parallel.
- Wave 5: S13, S14 in parallel.
- Wave 6: S15.
- Wave 7: S16.

Do not start a wave until every task in the previous wave has an open, reviewer-approved PR. If the
human has not merged yet, dependent agents use the stacked-branch rule (§3.2).
