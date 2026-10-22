# ADR 030 — Thesis research framing: risk-sensitive RL for joint LP + hedge management

**Status:** accepted
**Affects:** `docs/plans/sim/PLAN.md` §9.3 (RQ restructure) · `docs/plans/sim/tasks/S17–S23` · `CONTRACTS.md` §9/§11/§12/§15 · `src/undertow/data` (funding stream) · `src/undertow/sim/{env,train,evaluate,backtest}`
**Raised by:** RL kick-off diagnostics (`runs/experiment2`, `runs/experiment3`) + literature review

## Context

`PLAN.md` §9.3 commits the thesis to three deliverables:

- **RQ1** — friction ablation table (full vs each partial friction).
- **RQ2** — per-regime results matrix + train→test shift analysis.
- **RQ3** — on-chain backtest + quantified simulation-to-reality gap.

The first real PPO kick-off (`runs/experiment2`, five seeds × 1M steps) produced
degenerate policies: every seed collapsed to a single deterministic action, the
learning curve was flat, and the agent learned to **hold**. ADR-029 fixed two
fidelity defects (band geometry in the observation; the IL benchmark), but
`runs/experiment3` on the ADR-029 code reproduced the same collapse exactly:
`ep_rew_mean` constant at −1.0211 from 20k to 1M steps, all rollouts `hold`.

Diagnosis (measured, not inferred):

1. **Economics.** The reward is `fees − LVR − costs` (ADR-029: excess vs the fixed
   initial HODL basket). On the pinned ETH/USDC 0.3 % pool, fee income is
   comparable to or below LVR: 15 eval windows give HOLD mean excess
   **−1,739 USDC** (fees +5,217, IL −6,956). A static wide band (w50) is the best
   strategy at −416; every recentring strategy loses to it. This is the
   loss-versus-rebalancing result, independently documented for real Uniswap v3
   pools (`arXiv:2404.05803`: LVR exceeds fees across many large pools; v2 is
   more profitable for passive LPs than v3).
2. **Training bug.** `LpEnvironment.reset(seed=None)` reuses the previous
   `self._seed`, and SB3 auto-resets call `reset()` with no seed. Every training
   episode therefore replays the **same** 30-day window for all 1M steps; the
   agent never sees regime diversity.
3. **Cost-model defect.** Both `LpEnvironment._rebalance` and
   `BacktestEngine._apply_rebalance` pass the **whole position value** as the
   slippage model's `notional_usdc`. A V3 rebalance only swaps the inventory
   imbalance (≤ ~50 % of notional, → 0 for a balanced in-range recenter), so the
   fee/impact is overcharged ~2× in the worst case and arbitrarily for small
   recentres. The model and `CONTRACTS.md` §8 are fine; the call sites are wrong.

Literature review (all verified against the papers):

| Work | Finding | Left open |
|---|---|---|
| Milionis et al., LVR (`2208.06046`); `2404.05803` | Fees < LVR on many large v3 pools; passive v2 > v3 | No policy; no risk management |
| RAmmStein (`2602.19419`) | Greedy rebalancing −8.4 %; regime-aware laziness +1.6 % net ROI; 85 % fewer rebalances | Net ROI **excludes IL**; single pair; off-chain data |
| Zhang, Chen, Yang (`2309.10129`) | Hedged DDQN beats baselines 9–69 %; unhedged PnL is negative in uptrends, positive in downtrends | **Frictionless** hedge (their Assumptions 2–3); no regime features; 4 test periods |
| Chionas et al. (`2608.19389`) | Higher vol/gas → wider positions, lower rebalancing; risk aversion compresses the left tail | Simulation only; no on-chain data; no hedge |
| ZeroSwap (`2310.09413`) | RL market maker sets the **spread** to track the external price (zero-profit); static `x·y=k` converges as 1/T² vs e^(−kT) | Orthogonal lever (spread, not range); no IL/HODL metric |

## Problem

The current RQ1–RQ3 are **validation** questions. They are necessary for
credibility but they are not a research contribution, and a phase-boundary or
sweep study built on them is **not RL** — a finite grid of hand-written policies
can answer it. The thesis needs an RL-centred claim that a grid search cannot
reach.

## Decision

### 1. Research framing

> **Risk-sensitive reinforcement learning for joint concentrated-liquidity and
> hedge management on Uniswap v3.**

The LP problem is formulated as a **partially-observed, risk-sensitive
impulse-control MDP**: the agent decides *when* to rebalance, *where* to allocate
(width), and *how much market risk to carry* (hedge ratio), under frictions
(gas, swap fee, perp cost, funding) and uncertainty (price, volume, regime).

### 2. Claims (in order of presentation)

- **C1 — Limits (negative, robust).** Active rebalancing is economically
  dominated by passive on raw returns: fees < LVR on the pinned pool/period. We
  quantify the decomposition (fees, LVR, gas, slippage, funding) and the cost of
  greedy rebalancing. Replicates and extends the LVR literature on an exact
  simulator.
- **C2 — RL methods / failure modes (methods contribution).** Naive RL fails on
  this MDP; the fixes are necessary and each is ablated: band geometry in the
  observation (ADR-029), a beta-neutral reward benchmark (ADR-029), and
  **environment diversity** (S17). We document the failure modes and the design
  principles.
- **C3 — Learned joint policy (positive).** A single risk-sensitive policy over
  the joint (timing, width, hedge-ratio) space outperforms the best hand-tuned
  heuristic family out-of-sample, and its decision boundaries are interpretable
  (they recover the LVR/regime structure). The joint space is the reason RL is
  required: a scalar-threshold heuristic cannot cover it.
- **C4 — Active/passive boundary (supporting).** Map the region where active
  management beats passive as a function of (σ, volume/TVL, fee tier, gas,
  capital); validate against the analytical LVR threshold. This is the
  *interpretation* of C3, not the thesis spine.

### 3. Measurement protocol (binding)

- **Primary metric: risk-adjusted** — Sharpe (per-episode, annualised) and CVaR
  at 5 %. Raw PnL and excess-vs-HODL are secondary and always reported.
- **Benchmarks**: the existing six-policy ladder **plus** the best static width
  (chosen on train, evaluated on eval) and a **hedged HODL** (beta-matched).
  Beating only the width-2 `hodl` baseline is not a result.
- **Design**: ≥5 seeds, walk-forward episodes, paired comparisons with
  confidence intervals, regime-stratified reporting.
- **Ablations**: band geometry on/off; window resampling on/off; hedge on/off;
  risk penalty on/off; regime features on/off.
- **Parity**: extend S14 to a **rebalancing** policy. The current parity harness
  runs the no-op HOLD policy only, so the fee/LVR path that C1–C3 depend on has
  never been validated against exact accounting. This is a blocking prerequisite
  for the hedge work.

### 4. Mechanism

- **Hedge leg**: a Binance USDT-perp short of the position's ETH delta, with
  realistic **funding** and perp taker costs. Measured feasibility: delta total
  variation implies 0.25–1.4 %/month of perp cost at 5 bps; short funding income
  is +7.3 %/yr average (2022–24), +13.0 %/yr in 2024, positive in 84 % of
  periods — so the hedge is roughly self-financing carry. The hedge does **not**
  create alpha (the reward is already beta-neutral); it removes directional
  variance, makes reported PnL comparable to a beta-matched benchmark, and adds
  a genuine second control axis.
- **Risk-sensitive objective**: keep the beta-neutral excess-vs-HODL reward
  (ADR-029) and add the hedge PnL − hedge costs + funding, with an optional
  CVaR/risk penalty (`RewardConfig.risk_penalty_*` already exists, λ = 0).
- **Joint action space**: extend the discrete grid with a hedge-ratio head
  (e.g. {0, 0.25, 0.5, 0.75, 1.0}); the LP head is unchanged.
- **Observation**: add hedge state (current short notional, funding rate, basis)
  to the §11 vector; `regime`, `realized_vol_24h` and gas are already present.

### 5. Scope

**In scope:** S17 (window resampling), S18 (swap-imbalance cost), S19 (funding
stream), S20 (hedge overlay + joint action + observation), S21 (risk-sensitive
objective + metrics), S22 (rebalancing parity), S23 (experiment runner + report).

**Out of scope** (unchanged from `PLAN.md`): reimplementing published RL
baselines, multi-pool routing, MEV, dynamic fee markets, interpretability
surrogates, hyperparameter search beyond the pinned modest grid.

### 6. Task map

| Task | Branch | Depends on | Deliverable |
|---|---|---|---|
| S17 episode window resampling | `bug-sim-episode-window-resampling` | — | New window per no-seed reset; determinism tests |
| S18 swap-imbalance rebalance cost | `bug-sim-rebalance-swap-cost` | — | Correct notional at both call sites; tests |
| S19 funding stream | `feature-data-funding-stream` | — | `stream=funding` in `undertow.data`; snapshot rebuild |
| S20 hedge overlay + joint action | `feature-sim-hedge-overlay` | S17, S18, S19 | `sim/hedge` module; env/backtester wiring; observation; action head |
| S21 risk-sensitive objective + metrics | `feature-sim-risk-metrics` | S20 | CVaR/Sharpe primary metric; reward integration; evaluation |
| S22 rebalancing parity | `feature-sim-rebalance-parity` | S18 | S14 extended to a rebalancing policy |
| S23 experiment runner + report | `feature-sim-thesis-experiments` | S17–S22 | Sweep harness, multi-seed runs, claims C1–C4 |

S17, S18, S19 are independent and can run in parallel worktrees.

## Consequences

- `CONTRACTS.md` §8 gains one sentence: `notional_usdc` is the **swapped
  imbalance**, not the position value. §9/§11/§12/§15 are amended by S20/S21.
- `PLAN.md` §9.3's RQ1–RQ3 are retained as the validation layer but are no longer
  the thesis contribution; C1–C4 supersede them as the results chapter.
- The vault thesis text (`~/Documents/fyp/lesson_plan/`) that quotes RQ1–RQ3 or
  the §11 field list needs a **human** update — flagged here, not edited from a
  code task.
- Rejected: a pure phase-boundary sweep as the thesis (not RL; grid-searchable);
  raw-PnL outperformance of HODL as the headline (structurally hard, weak); a
  full RL hedge head from day one (build the deterministic overlay first, promote
  only if C3 needs it).
