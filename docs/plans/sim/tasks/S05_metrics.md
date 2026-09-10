# S05 — Performance metrics, PnL decomposition & regime slicing

**Wave 2 · size M · depends on: S01 · blocks: S13, S15**  
**Branch:** `feature-sim-metrics`

## Why this task exists

Every evaluation — backtest summarization, training callbacks, ablation tables, regime matrices —
needs the same set of performance metrics and the same PnL decomposition formula. Implementing them
once, correctly, avoids drift between the backtester's and simulator's reported numbers.

The metrics implement equations (7)–(10) from the roadmap §10.4: Sharpe, Sortino, MaxDD, VaR, CVaR,
annualized return, and the PnL decomposition `PnL = ΣF − ΣΔIL − Σ(G+S)`.

## Files you own

```
src/undertow/sim/metrics/__init__.py       (re-exports all public functions)
src/undertow/sim/metrics/performance.py    (Sharpe, Sortino, MaxDD, VaR, CVaR, annualized_return,
                                            all_metrics)
src/undertow/sim/metrics/decompose.py      (pnl_decomposition)
src/undertow/sim/metrics/regimes.py        (per_regime_metrics)
tests/sim/test_metrics.py
```

## What to build

### 1. `metrics/performance.py` — CONTRACTS.md §14

All functions operate on `pl.Series` (float64). Assume equally-spaced steps at the caller's cadence.

- `sharpe_ratio(equity_curve, risk_free_rate=0.0, periods_per_year=365*24*6)` → `float`:
  Annualized. `mean(period_returns - rf/periods_per_year) / std(period_returns) * sqrt(periods_per_year)`.
  Returns 0.0 if std is 0. Period returns are computed as `equity_curve.pct_change()` (first element
  is NaN → drop).

- `sortino_ratio(equity_curve, risk_free_rate=0.0, periods_per_year=365*24*6)` → `float`:
  Same numerator as Sharpe, denominator = std of only negative period returns.

- `max_drawdown(equity_curve)` → `float`: Maximum peak-to-trough decline as a negative fraction.
  E.g., -0.25 means a 25% drawdown. Track the running maximum and compute
  `min((equity - running_max) / running_max)`.

- `value_at_risk(returns, confidence=0.95)` → `float`: Historical VaR — the quantile of the return
  distribution at the given confidence level. Returns a negative number (loss).

- `cvar(returns, confidence=0.95)` → `float`: Conditional VaR — the mean of returns below the VaR
  threshold.

- `annualized_return(equity_curve, periods_per_year=365*24*6)` → `float`:
  `(equity[-1] / equity[0])^(periods_per_year / n) − 1`.

- `all_metrics(equity_curve, returns, periods_per_year=365*24*6)` → `dict[str, float]`:
  Convenience wrapper returning all six metrics in one dict.

### 2. `metrics/decompose.py` — CONTRACTS.md §14

- `pnl_decomposition(ledger)` → `dict[str, float]`: Takes a ledger DataFrame with columns
  `fees`, `il_change`, `gas`, `slippage` (per-step values in USDC). Returns:
  ```python
  {
      "total_fees": float,       # Σ fees
      "total_il_change": float,  # Σ IL (negative = loss)
      "total_gas": float,        # Σ gas (negative = cost)
      "total_slippage": float,   # Σ slippage (negative = cost)
      "net_pnl": float,          # Σ fees + Σ il_change + Σ gas + Σ slippage
  }
  ```
  Verify: `net_pnl ≈ total_fees + total_il_change + total_gas + total_slippage`.

### 3. `metrics/regimes.py` — CONTRACTS.md §14

- `per_regime_metrics(equity_curve, regime_column, metric_fns)` → `pl.DataFrame`:
  Takes a DataFrame with an equity column, plus a regime label column (string), plus a dict of
  metric-name → callable. Groups by regime, computes each metric, returns a table
  `regime × metric_name → value`. Each row is one regime, each column (after `regime`) is one metric.

  The input DataFrame has per-step rows. Group by `regime_column`, extract the equity sub-series for
  each group, and apply each metric function. The metric functions must accept a `pl.Series` and
  return a float.

## Tests you must write

1. `sharpe_ratio` on a flat equity curve returns 0.0 (zero std)
2. `sharpe_ratio` on a linearly increasing equity curve returns positive value
3. `sharpe_ratio` disregards risk-free rate correctly (test with `rf = 0.02`)
4. `sortino_ratio` ≈ `sharpe_ratio` for symmetric returns
5. `sortino_ratio` < `sharpe_ratio` for negatively-skewed returns (synthesize)
6. `max_drawdown` on a flat curve returns 0.0
7. `max_drawdown` on [100, 80, 60, 90, 110] returns -0.40 (peak 100 → trough 60)
8. `max_drawdown` on a monotonic up curve returns 0.0
9. `value_at_risk` at 95% on N(0,1) returns ≈ −1.645 * σ (synthesize 10k samples)
10. `cvar` ≤ `value_at_risk` (always: the conditional mean of the tail is worse than the
    quantile)
11. `annualized_return` on a flat curve returns 0.0
12. `annualized_return` on +10% over 1 year returns ≈ 0.10
13. `pnl_decomposition` on a DataFrame with known values returns correct sums
14. `pnl_decomposition` net_pnl matches the sum of components
15. `per_regime_metrics` returns a row per unique regime
16. `per_regime_metrics` values match direct computation on each regime's sub-series
17. `all_metrics` returns all six keys
18. Edge: single-row equity curve → Sharpe/Sortino/VaR/CVaR return 0.0, MaxDD returns 0.0

## Definition of done

- All files exist, typed, no TODO stubs
- `uv run pytest tests/sim/test_metrics.py -q` green
- `code-reviewer` APPROVE
- PR open against `main` from `feature-sim-metrics`
- `STATUS.md` updated