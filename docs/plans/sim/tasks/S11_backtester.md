# S11 — Backtester (frozen-policy event replay)

**Wave 4 · size L · depends on: S02, S03, S04, S08, S10 · blocks: S14, S15**  
**Branch:** `feature-sim-backtester`

## Why this task exists

The backtester is the measurement instrument. It replays a frozen policy over the *real* block-ordered
event tape from `undertow.data`, with exact integer fee accrual (via the data module's
`FeeGrowthTracker`), real per-block gas prices, and real reference prices. This is where G3 lives:
the backtester cannot be "improved" by simulator artifacts. Its output is the ground truth against
which the simulator is validated (S14) and the on-chain reality against which the gap is measured
(RQ3).

The backtester is deliberately not fast. It replays one block at a time, computing exact fees.
It is a measurement instrument, not a training loop.

## Files you own

```
src/undertow/sim/backtest/__init__.py      (re-exports run_backtest, summarize_backtest,
                                            BacktestLedger, BacktestResult)
src/undertow/sim/backtest/engine.py        (run_backtest)
src/undertow/sim/backtest/ledger.py        (BacktestLedger, BacktestResult, summarize_backtest)
tests/sim/test_backtester.py
tests/sim/conftest.py                      (appends tiny_ledger fixture)
```

## What to build

### 1. `backtest/ledger.py` — CONTRACTS.md §13

**`BacktestLedger`** dataclass (frozen, slots):
- `equity_curve: pl.DataFrame` — columns: `step, equity, pnl, fees, gas, slippage, il`
- `decision_log: pl.DataFrame` — columns: `step, action_type, center_offset, width, tick_lower,
  tick_upper, gas_paid, slippage_paid, fees_collected`
- `cost_ledger: pl.DataFrame` — columns: `step, cost_type, amount_usdc`
- `pnl_decomposition: dict[str, float]` — `total_fees, total_gas, total_slippage, total_il, net`
- `config_hash: str` — SHA256 of the config JSON
- `git_commit: str` — current git HEAD
- `policy_name: str`
- `split: str` — `"train"` or `"eval"`

**`BacktestResult`** dataclass (frozen, slots):
- `total_pnl: Wealth`, `annualized_return: float`, `sharpe: float`, `max_drawdown: float`
- `fee_income: float`, `gas_paid: float`, `slippage_paid: float`, `il_realized: float`
- `hodl_return: float`, `excess_vs_hodl: float`

**`summarize_backtest(ledger)` → `BacktestResult`**: Compute all summary metrics from the ledger
using S05's metric functions. `hodl_return` is computed by simulating a HODL position over the
same equity curve: initial capital just held as the initial token mix, valued at each step's
reference price.

### 2. `backtest/engine.py` — CONTRACTS.md §13

**`run_backtest(policy, market_view, config)` → `BacktestLedger`**:

The backtest loop (§10.2.3 of the roadmap):

```
1. Load the event tape for the active split from market_view
2. Initialize: entry price = reference_close at first tape event time
3. Deploy initial position: call policy.act(first_observation) to get the initial range,
   then initial_deposit(capital, tick_lower, tick_upper, ...) → Position
4. For each tape event (block) at step t:
   a. Get reference_close at this event's block time → current price
   b. Get gas data for this block → gas cost parameters
   c. Advance FeeGrowthTracker with this event's swap/mint/burn data
      - The FeeGrowthTracker is imported from undertow.data (via S02's public API)
      - Pass the event's swap volume, fee tier
      - Per-position fee growth is tracked
   d. Compute position value at current price
   e. Accrue fees: uncollected_fees since last step
   f. Record step in ledger: equity, fees, IL, gas (0 if no action), slippage (0 if no action)
   g. At decision boundaries (every K blocks, matching config.episode.step_minutes cadence):
      - Build Observation from current state
      - Call policy.act(observation)
      - If action is "rebalance":
        * Compute gas cost via GasModel
        * Compute slippage via SlippageModel
        * Close current position (collect fees)
        * Open new position at new range
        * Record gas, slippage, and fees in decision_log
      - If action is "hold": record in decision_log
5. Return BacktestLedger
```

**Key implementation decisions:**

- **Fee accrual**: The backtester uses the data module's `FeeGrowthTracker` for exact integer fee
  math. This means the backtester's fee numbers are in the **same numeric world** as the data
  pipeline's Collect reconciliation. Import it from `undertow.data` (if S02 exports it; if not,
  this task must request the export via an ADR or a note in S02's PR).

  If `FeeGrowthTracker` is not yet exported by S02, **do not import it from an internal module**.
  Instead, implement a float64 fee accrual path in the backtester that matches the pool engine's
  logic, and file an ADR noting that the exact-integer path is gated on S02 adding the export.
  Mark the float64 path with `# TODO(exact): switch to FeeGrowthTracker when S02 exports it`.

- **Decision cadence**: The backtester makes decisions at the nearest tape event to each
  `step_minutes` boundary. Not every tape event — only at decision boundaries.
  Between decisions, it still records per-block equity, fees, and IL in the equity_curve.

- **Gas**: Gas is only paid when an action is taken (mint, burn, collect, rebalance). No gas cost
  for `hold`. The gas model uses the block's actual gas data.

- **Slippage**: Only paid on rebalance trades. No slippage for mint/burn/collect (those are
  Uniswap V3 NFT operations, not swaps through the pool — the gas cost covers them).

- **Config hash**: `hashlib.sha256(json.dumps(dataclasses.asdict(config), sort_keys=True, default=str).encode()).hexdigest()[:16]`

- **Git commit**: `subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()`

### 3. `tests/sim/conftest.py` — `tiny_ledger` fixture

Append a `tiny_ledger` fixture: a `BacktestLedger` from a 100-step replay of a `dummy_policy`
on the `tiny_market_view`. Built deterministically from the shared fixtures.

## Tests you must write

1. `run_backtest(HODLPolicy, tiny_market_view)` completes without error
2. Backtest ledger has non-empty equity_curve, decision_log, cost_ledger
3. Equity_curve has the expected columns
4. Decision_log records hold actions at every decision boundary for HODL
5. Backtest with `PassiveNarrow` records a rebalance in the first decision and holds after
6. `BacktestResult` from `summarize_backtest` has non-None fields
7. HODL backtest result: `hodl_return` matches manual computation (compute HODL value at
   each step from the initial token mix and reference price)
8. `excess_vs_hodl` = `total_pnl - hodl_return`
9. Backtest is deterministic: same config + seed → identical ledger
10. Config hash in the ledger is consistent across runs with same config
11. **Fee accrual test**: run a backtest where the position is in-range for a known swap, verify
    that fees accrue proportionally to `L_agent / L_active` (or use the FeeGrowthTracker's known
    correctness as the golden)
12. Decision cadence: backtest with `step_minutes=60` on a 2-hour episode has ~2 decisions
    (not 120)
13. Gas is only recorded on action steps (non-zero only for rebalance actions)
14. Slippage is only recorded on rebalance steps
15. `tiny_ledger` fixture is importable

## Definition of done

- All files exist, typed, no TODO stubs (the FeeGrowthTracker TODO is acceptable if ADR'd)
- `uv run pytest tests/sim/test_backtester.py -q` green
- `code-reviewer` APPROVE
- PR open against `main` from `feature-sim-backtester`
- `STATUS.md` updated