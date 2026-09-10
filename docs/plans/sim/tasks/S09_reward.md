# S09 — Reward function + RewardBreakdown + ablation flags

**Wave 3 · size S · depends on: S01 · blocks: S12**  
**Branch:** `feature-sim-reward`

## Why this task exists

The reward function is the objective the agent optimizes, and it is also the research instrument for
RQ1 (friction ablation). Equation (1) of the problem statement (§9.2) defines:

```
R_t = F_t − G_t − S_t − ΔIL_t − λ·σ_PnL,t
```

Each term is gated by an enable flag in `RewardConfig`, making the ablation table
{full, no_gas, static_fee, no_il, no_risk_penalty} a config change rather than a code change. The
`RewardBreakdown` tracks every term even when disabled, so the evaluation runner can decompose any
run.

This is the smallest wave-3 task. Do it carefully — a sign error in the reward propagates to every
result.

## Files you own

```
src/undertow/sim/env/__init__.py           (creates the env package; re-exports from reward.py)
src/undertow/sim/env/reward.py             (compute_reward, RewardBreakdown)
tests/sim/test_reward.py
```

## What to build

### 1. `env/reward.py` — CONTRACTS.md §9

**`RewardBreakdown`** dataclass (frozen, slots):
```python
@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    fees: float = 0.0              # ≥ 0
    gas: float = 0.0               # ≤ 0 (cost, stored as negative)
    slippage: float = 0.0          # ≤ 0 (cost, stored as negative)
    il_change: float = 0.0         # ΔIL; ≤ 0 most steps
    risk_penalty: float = 0.0      # −λ·σ_PnL; ≤ 0
    reward: float = 0.0            # sum of the above
    pnl_net: float = 0.0           # ΔPnL after all costs (ungated — the true economic change)
    equity: float = 0.0            # current equity (for reference)
```
Note: `gas` and `slippage` are stored as **negative** values (they are costs). `fees` is positive.
`reward` is the sum of all gated terms. `pnl_net` is the ungated net PnL change regardless of
ablation flags — this is what the ledger records.

**`compute_reward(fees_earned, gas_cost, slippage_cost, il_change, pnl_volatility, config)`**
→ `RewardBreakdown`:

```python
def compute_reward(
    fees_earned: float,        # gross fees in USDC (≥ 0)
    gas_cost: float,           # gas cost in USDC (≥ 0, from GasModel)
    slippage_cost: float,      # slippage cost in USDC (≥ 0, from SlippageModel)
    il_change: float,          # ΔIL in USDC (≤ 0 typically)
    pnl_volatility: float,     # trailing σ_PnL for risk penalty
    config: RewardConfig,
) -> RewardBreakdown:
    # Gate each term by its config flag
    fees = fees_earned if config.fees_enabled else 0.0
    gas = -gas_cost if config.gas_enabled else 0.0          # stored negative
    slippage = -slippage_cost if config.slippage_enabled else 0.0
    il = il_change if config.il_enabled else 0.0
    risk = -config.risk_penalty_lambda * pnl_volatility if config.risk_penalty_enabled else 0.0

    reward = fees + gas + slippage + il + risk
    pnl_net = fees_earned - gas_cost - slippage_cost + il_change  # always ungated

    return RewardBreakdown(
        fees=fees,
        gas=gas,
        slippage=slippage,
        il_change=il,
        risk_penalty=risk,
        reward=reward,
        pnl_net=pnl_net,
        equity=0.0,  # filled by caller
    )
```

### 2. Sign convention (critical)

- `fees_earned` is **entered as positive** (the agent earned fees)
- `gas_cost` and `slippage_cost` are **entered as positive** (cost magnitude from the models)
- The reward function **negates** them: `gas = -gas_cost`
- `il_change` is typically negative (IL is a loss) and is passed through as-is
- `reward` = `fees + gas + slippage + il_change + risk_penalty`, where `gas` ≤ 0, `slippage` ≤ 0,
  `il_change` ≤ 0, `risk_penalty` ≤ 0

This matters: the caller passes in positive cost numbers; the reward function makes them negative.
Do not double-negate.

### 3. Normalization

If `config.normalize_by_capital` is True, divide the reward by `initial_capital`. However, since
`compute_reward` doesn't have `initial_capital` as a parameter, normalization is done at the
env level (S12), not here. The `RewardBreakdown` stores raw USDC values. Add a note in the
docstring that S12 applies normalization.

## Tests you must write

1. Full reward with all flags enabled: nonzero fees, gas, slippage, IL → correct breakdown
2. `reward = fees + gas + slippage + il_change + risk_penalty` matches manual sum
3. `gas` is negative in the breakdown when gas_cost > 0 and `gas_enabled=True`
4. `slippage` is negative in the breakdown when slippage_cost > 0 and `slippage_enabled=True`
5. `gas_enabled=False` → `gas=0.0` in breakdown
6. `slippage_enabled=False` → `slippage=0.0` in breakdown
7. `fees_enabled=False` → `fees=0.0` in breakdown
8. `il_enabled=False` → `il_change=0.0` in breakdown
9. `risk_penalty_enabled=True` with `lambda=0.1` and `vol=100` → `risk_penalty = -10.0`
10. `risk_penalty_enabled=False` → `risk_penalty=0.0`
11. `pnl_net` is always the ungated version: `fees_earned - gas_cost - slippage_cost + il_change`,
    regardless of ablation flags
12. All-zero inputs → reward=0, pnl_net=0
13. **Golden test from CONTRACTS.md §19**: "Full IL/LVR reward with all terms ← vs no-LVR → net
    negative vs net positive." Construct an example where fees < costs + IL → net negative reward
    with all terms enabled. With `il_enabled=False`, the same inputs should give net positive.
14. Each field in `RewardBreakdown` is accessible by name and has the correct type

## Definition of done

- All files exist, typed, no TODO stubs
- `uv run pytest tests/sim/test_reward.py -q` green
- `code-reviewer` APPROVE
- PR open against `main` from `feature-sim-reward`
- `STATUS.md` updated