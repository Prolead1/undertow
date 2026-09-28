"""Behavioural tests for ``undertow.sim.metrics`` (S05).

Every expected value is either hand-computed in the comment above the assertion or
reproduced with an independent ``numpy`` formula, so a regression in the metric
definition fails the test rather than being masked by copying the implementation.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from undertow.sim.metrics import (
    DEFAULT_PERIODS_PER_YEAR,
    all_metrics,
    annualized_return,
    cvar,
    max_drawdown,
    per_regime_metrics,
    periods_per_year_from_step_minutes,
    pnl_decomposition,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)


def _series(values: list[float]) -> pl.Series:
    return pl.Series("equity", values, dtype=pl.Float64)


def _equity_from_returns(returns: list[float], start: float = 100.0) -> pl.Series:
    """Build an equity curve whose ``pct_change`` reproduces ``returns`` exactly."""
    levels = [start]
    for r in returns:
        levels.append(levels[-1] * (1.0 + r))
    return _series(levels)


# ---------------------------------------------------------------------------
# Sharpe
# ---------------------------------------------------------------------------


def test_sharpe_flat_curve_is_zero() -> None:
    assert sharpe_ratio(_series([100.0] * 5)) == 0.0


def test_sharpe_linearly_increasing_is_positive() -> None:
    equity = _series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    assert sharpe_ratio(equity) > 0.0


def test_sharpe_matches_independent_numpy_formula() -> None:
    # returns = [0.1, -0.1, 0.1]; mean = 1/30; sample std = sqrt(0.04/3)
    # sharpe = (1/30) / sqrt(0.04/3) * sqrt(3) = 0.5 exactly.
    equity = _equity_from_returns([0.1, -0.1, 0.1])
    expected = float(np.mean([0.1, -0.1, 0.1]) / np.std([0.1, -0.1, 0.1], ddof=1) * np.sqrt(3))
    assert sharpe_ratio(equity, periods_per_year=3) == pytest.approx(0.5, abs=1e-12)
    assert sharpe_ratio(equity, periods_per_year=3) == pytest.approx(expected, abs=1e-12)


def test_sharpe_risk_free_rate_reduces_ratio() -> None:
    equity = _equity_from_returns([0.1, -0.1, 0.1])
    # rf_step = 0.03/3 = 0.01; excess = [0.09, -0.11, 0.09].
    with_rf = sharpe_ratio(equity, risk_free_rate=0.03, periods_per_year=3)
    without_rf = sharpe_ratio(equity, risk_free_rate=0.0, periods_per_year=3)
    assert with_rf == pytest.approx(0.35, abs=1e-12)
    assert with_rf < without_rf


def test_sharpe_single_observation_is_zero() -> None:
    assert sharpe_ratio(_series([100.0])) == 0.0


# ---------------------------------------------------------------------------
# Sortino
# ---------------------------------------------------------------------------


def test_sortino_matches_independent_numpy_formula() -> None:
    # returns = [0.4, -0.1, -0.2, 0.1]; mean = 0.05.
    # downside deviation = sqrt(mean(min(r, 0)^2)) = sqrt(0.0125) = 0.11180339.
    # sortino = 0.05 / sqrt(0.0125) * sqrt(4) = 0.89442719...
    returns = np.array([0.4, -0.1, -0.2, 0.1])
    equity = _equity_from_returns(returns.tolist())
    dd = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2))
    expected = float(np.mean(returns) / dd * np.sqrt(4))
    assert sortino_ratio(equity, periods_per_year=4) == pytest.approx(0.8944271909999159, abs=1e-12)
    assert sortino_ratio(equity, periods_per_year=4) == pytest.approx(expected, abs=1e-12)


def test_sortino_equals_sharpe_for_zero_mean_symmetric_returns() -> None:
    # Symmetric, zero-mean returns => numerator is ~0 for both ratios.
    symmetric = [0.05, 0.03, 0.01, -0.01, -0.03, -0.05]
    equity = _equity_from_returns(symmetric)
    sharpe = sharpe_ratio(equity, periods_per_year=6)
    sortino = sortino_ratio(equity, periods_per_year=6)
    assert sharpe == pytest.approx(0.0, abs=1e-9)
    assert sortino == pytest.approx(0.0, abs=1e-9)
    assert abs(sortino - sharpe) < 1e-9


def test_sortino_worse_than_sharpe_for_negatively_skewed_returns() -> None:
    # A fat left tail inflates downside deviation relative to the full std.
    skewed = [-0.10, -0.02, -0.01, 0.01, 0.02, 0.03]
    equity = _equity_from_returns(skewed)
    sharpe = sharpe_ratio(equity, periods_per_year=6)
    sortino = sortino_ratio(equity, periods_per_year=6)
    assert sharpe < 0.0 and sortino < 0.0
    assert sortino < sharpe


def test_sortino_all_positive_returns_is_zero() -> None:
    equity = _equity_from_returns([0.01, 0.02, 0.03, 0.05])
    assert sortino_ratio(equity) == 0.0


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------


def test_max_drawdown_flat_is_zero() -> None:
    assert max_drawdown(_series([100.0] * 5)) == 0.0


def test_max_drawdown_golden_series() -> None:
    # peak 100 -> trough 60 => -0.40
    assert max_drawdown(_series([100.0, 80.0, 60.0, 90.0, 110.0])) == pytest.approx(-0.40)


def test_max_drawdown_monotonic_up_is_zero() -> None:
    assert max_drawdown(_series([100.0, 110.0, 120.0, 130.0])) == 0.0


def test_max_drawdown_empty_is_zero() -> None:
    assert max_drawdown(_series([])) == 0.0


# ---------------------------------------------------------------------------
# VaR / CVaR
# ---------------------------------------------------------------------------


def test_value_at_risk_matches_normal_quantile() -> None:
    rng = np.random.default_rng(7)
    sample = rng.standard_normal(10_000)
    var = value_at_risk(pl.Series("r", sample), confidence=0.95)
    # Independent formula: 5% empirical quantile ~= -1.645 for N(0, 1).
    assert var == pytest.approx(-1.645, abs=0.05)
    assert var == pytest.approx(float(np.quantile(sample, 0.05)), abs=1e-12)


def test_cvar_at_most_value_at_risk() -> None:
    rng = np.random.default_rng(11)
    sample = rng.standard_normal(10_000)
    rets = pl.Series("r", sample)
    var = value_at_risk(rets, confidence=0.95)
    es = cvar(rets, confidence=0.95)
    tail = sample[sample <= np.quantile(sample, 0.05)]
    assert es == pytest.approx(float(np.mean(tail)), abs=1e-12)
    assert es <= var


def test_var_cvar_empty_are_zero() -> None:
    empty = pl.Series("r", [], dtype=pl.Float64)
    assert value_at_risk(empty) == 0.0
    assert cvar(empty) == 0.0


def test_var_rejects_bad_confidence() -> None:
    with pytest.raises(ValueError):
        value_at_risk(pl.Series("r", [0.1, -0.1]), confidence=1.0)


def test_cvar_rejects_bad_confidence() -> None:
    with pytest.raises(ValueError):
        cvar(pl.Series("r", [0.1, -0.1]), confidence=0.0)


# ---------------------------------------------------------------------------
# Annualized return
# ---------------------------------------------------------------------------


def test_annualized_return_flat_is_zero() -> None:
    assert annualized_return(_series([100.0] * 6), periods_per_year=12) == 0.0


def test_annualized_return_golden_one_year() -> None:
    # 12 monthly periods, +10% total: (110/100)^(12/12) - 1 = 0.10.
    equity = _series([100.0 * 1.10 ** (i / 12.0) for i in range(13)])
    assert annualized_return(equity, periods_per_year=12) == pytest.approx(0.10, abs=1e-12)


def test_annualized_return_total_loss_is_minus_one() -> None:
    assert annualized_return(_series([100.0, 0.0]), periods_per_year=12) == -1.0


def test_annualized_return_non_positive_start_is_zero() -> None:
    assert annualized_return(_series([0.0, 50.0]), periods_per_year=12) == 0.0


def test_periods_per_year_from_step_minutes() -> None:
    assert periods_per_year_from_step_minutes(10) == DEFAULT_PERIODS_PER_YEAR == 365 * 24 * 6
    assert periods_per_year_from_step_minutes(1) == 365 * 24 * 60
    assert periods_per_year_from_step_minutes(5) == 105_120
    with pytest.raises(ValueError):
        periods_per_year_from_step_minutes(0)


# ---------------------------------------------------------------------------
# PnL decomposition
# ---------------------------------------------------------------------------


def test_pnl_decomposition_known_values_with_contract_il_column() -> None:
    # CONTRACTS §13 names the column ``il`` on BacktestLedger.equity_curve.
    ledger = pl.DataFrame(
        {
            "step": [0, 1],
            "fees": [10.0, 5.0],
            "il": [-2.0, -3.0],
            "gas": [-1.0, -1.0],
            "slippage": [-0.5, -0.5],
        }
    )
    decomposition = pnl_decomposition(ledger)
    assert set(decomposition) == {
        "total_fees",
        "total_il_change",
        "total_gas",
        "total_slippage",
        "net_pnl",
    }
    assert decomposition["total_fees"] == pytest.approx(15.0)
    assert decomposition["total_il_change"] == pytest.approx(-5.0)
    assert decomposition["total_gas"] == pytest.approx(-2.0)
    assert decomposition["total_slippage"] == pytest.approx(-1.0)
    # 15 - 5 - 2 - 1 = 7
    assert decomposition["net_pnl"] == pytest.approx(7.0)


def test_pnl_decomposition_accepts_il_change_alias() -> None:
    # The S05 brief spells the same column ``il_change``; both must work identically.
    ledger = pl.DataFrame(
        {
            "fees": [10.0, 5.0],
            "il_change": [-2.0, -3.0],
            "gas": [-1.0, -1.0],
            "slippage": [-0.5, -0.5],
        }
    )
    decomposition = pnl_decomposition(ledger)
    assert decomposition["total_il_change"] == pytest.approx(-5.0)
    assert decomposition["net_pnl"] == pytest.approx(7.0)


def test_pnl_decomposition_both_spellings_agree() -> None:
    base = {"fees": [10.0, 5.0], "gas": [-1.0, -1.0], "slippage": [-0.5, -0.5]}
    il = pnl_decomposition(pl.DataFrame({**base, "il": [-2.0, -3.0]}))
    il_change = pnl_decomposition(pl.DataFrame({**base, "il_change": [-2.0, -3.0]}))
    assert il == il_change


def test_pnl_decomposition_prefers_contract_il_when_both_present() -> None:
    ledger = pl.DataFrame(
        {
            "fees": [0.0],
            "il": [-2.0],
            "il_change": [-999.0],
            "gas": [0.0],
            "slippage": [0.0],
        }
    )
    decomposition = pnl_decomposition(ledger)
    assert decomposition["total_il_change"] == pytest.approx(-2.0)


def test_pnl_decomposition_sign_convention_pinned() -> None:
    # Costs/gains are already signed on the ledger; the decomposition must not re-sign
    # them. Fees positive, IL/gas/slippage negative pass through unchanged, and net is
    # exactly their sum. Pinning this guards against an accidental ``abs``/negation.
    ledger = pl.DataFrame(
        {
            "fees": [100.0],
            "il": [-30.0],
            "gas": [-5.0],
            "slippage": [-2.0],
        }
    )
    decomposition = pnl_decomposition(ledger)
    assert decomposition["total_fees"] == 100.0
    assert decomposition["total_il_change"] == -30.0
    assert decomposition["total_gas"] == -5.0
    assert decomposition["total_slippage"] == -2.0
    assert decomposition["net_pnl"] == pytest.approx(100.0 - 30.0 - 5.0 - 2.0)
    assert decomposition["net_pnl"] == pytest.approx(
        decomposition["total_fees"]
        + decomposition["total_il_change"]
        + decomposition["total_gas"]
        + decomposition["total_slippage"]
    )


def test_pnl_decomposition_net_is_sum_of_components() -> None:
    ledger = pl.DataFrame(
        {
            "fees": [1.0, 2.0, 3.0],
            "il": [-1.0, -0.5, -2.0],
            "gas": [-0.1, -0.2, -0.3],
            "slippage": [-0.01, -0.02, -0.03],
        }
    )
    decomposition = pnl_decomposition(ledger)
    component_sum = sum(
        decomposition[k] for k in ("total_fees", "total_il_change", "total_gas", "total_slippage")
    )
    assert decomposition["net_pnl"] == pytest.approx(component_sum)


def test_pnl_decomposition_missing_il_column_raises() -> None:
    ledger = pl.DataFrame({"fees": [1.0], "gas": [-1.0], "slippage": [0.0]})
    with pytest.raises(KeyError, match="il"):
        pnl_decomposition(ledger)


def test_pnl_decomposition_empty_ledger_is_zero() -> None:
    ledger = pl.DataFrame(
        schema={
            "fees": pl.Float64,
            "il_change": pl.Float64,
            "gas": pl.Float64,
            "slippage": pl.Float64,
        }
    )
    decomposition = pnl_decomposition(ledger)
    assert all(value == 0.0 for value in decomposition.values())


# ---------------------------------------------------------------------------
# Per-regime slicing
# ---------------------------------------------------------------------------


def _regime_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "step": list(range(8)),
            "regime": ["bull", "bull", "bear", "bear", "bull", "sideways", "sideways", "bear"],
            "equity": [100.0, 110.0, 99.0, 90.0, 120.0, 118.0, 130.0, 125.0],
        }
    )


def test_per_regime_metrics_one_row_per_regime() -> None:
    result = per_regime_metrics(
        _regime_frame(), "regime", {"maxdd": max_drawdown, "sharpe": sharpe_ratio}
    )
    assert result["regime"].to_list() == ["bull", "bear", "sideways"]
    assert result.columns == ["regime", "maxdd", "sharpe"]
    assert result.height == 3


def test_per_regime_metrics_values_match_direct_computation() -> None:
    frame = _regime_frame()
    metrics = {"maxdd": max_drawdown, "sharpe": sharpe_ratio}
    result = per_regime_metrics(frame, "regime", metrics)
    lookup = {row["regime"]: row for row in result.to_dicts()}

    for regime in ("bull", "bear", "sideways"):
        sub = frame.filter(pl.col("regime") == regime)["equity"]
        assert lookup[regime]["maxdd"] == pytest.approx(max_drawdown(sub))
        assert lookup[regime]["sharpe"] == pytest.approx(sharpe_ratio(sub))


def test_per_regime_metrics_custom_equity_column() -> None:
    frame = _regime_frame().rename({"equity": "nav"})
    result = per_regime_metrics(frame, "regime", {"maxdd": max_drawdown}, equity_column="nav")
    lookup = {row["regime"]: row for row in result.to_dicts()}
    # bear equity = [99, 90, 125] -> peak 99, trough 90 -> (90 - 99) / 99
    assert lookup["bear"]["maxdd"] == pytest.approx(-9.0 / 99.0)


def test_per_regime_metrics_drops_null_labels() -> None:
    frame = pl.DataFrame(
        {
            "regime": ["bull", None, "bull"],
            "equity": [100.0, 50.0, 80.0],
        }
    )
    result = per_regime_metrics(frame, "regime", {"maxdd": max_drawdown})
    assert result["regime"].to_list() == ["bull"]
    assert result["maxdd"][0] == pytest.approx(-0.20)


def test_per_regime_metrics_empty_frame_has_schema() -> None:
    empty = pl.DataFrame(schema={"regime": pl.Utf8, "equity": pl.Float64})
    result = per_regime_metrics(empty, "regime", {"maxdd": max_drawdown})
    assert result.columns == ["regime", "maxdd"]
    assert result.height == 0


def test_per_regime_metrics_missing_column_raises() -> None:
    with pytest.raises(KeyError, match="nope"):
        per_regime_metrics(_regime_frame(), "nope", {"maxdd": max_drawdown})
    with pytest.raises(KeyError, match="nav"):
        per_regime_metrics(_regime_frame(), "regime", {"maxdd": max_drawdown}, equity_column="nav")


# ---------------------------------------------------------------------------
# all_metrics + edge cases
# ---------------------------------------------------------------------------


def test_all_metrics_returns_all_six_keys() -> None:
    equity = _equity_from_returns([0.1, -0.1, 0.1, 0.05])
    returns = equity.pct_change().drop_nulls()
    result = all_metrics(equity, returns)
    assert set(result) == {
        "sharpe",
        "sortino",
        "maxdd",
        "var_95",
        "cvar_95",
        "annualized_return",
    }
    assert result["annualized_return"] == pytest.approx(
        annualized_return(equity, DEFAULT_PERIODS_PER_YEAR)
    )


def test_single_row_equity_edge_cases() -> None:
    equity = _series([100.0])
    returns = pl.Series("r", [], dtype=pl.Float64)
    assert sharpe_ratio(equity) == 0.0
    assert sortino_ratio(equity) == 0.0
    assert max_drawdown(equity) == 0.0
    assert value_at_risk(returns) == 0.0
    assert cvar(returns) == 0.0
    assert annualized_return(equity) == 0.0
