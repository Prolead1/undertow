"""U02 — parsing ``results/`` markdown and ``summary.json`` into :class:`EvalMetrics`."""

from __future__ import annotations

import json
from pathlib import Path

from undertow.ui.results import (
    load_eval_metrics,
    parse_regime_pnl,
    parse_results_summary_markdown,
    parse_summary_json,
)

_COMPARISON_MD = """# RQ2 — Policy comparison by regime

Mean ± std over walk-forward episodes and seeds.

| Policy | Regime | PnL | Sharpe | MaxDD |
| --- | --- | --- | --- | --- |
| ppo | bull | 1,234.56 ± 12.3 | 1.2000 ± 0.1 | -0.0500 ± 0.01 |
| ppo | bear | -500.00 ± 5.0 | -0.3000 ± 0.1 | -0.2000 ± 0.02 |
| ppo | sideways | 200.00 ± 1.0 | 0.1000 ± 0.0 | -0.1000 ± 0.0 |
"""

_MATRIX_MD = """# RQ2 — Per-regime results: ppo

Mean ± std over walk-forward episodes and seeds.

| Regime | PnL | Sharpe | MaxDD | Fee Income | Gas Paid | IL |
| --- | --- | --- | --- | --- | --- | --- |
| bull | 1,000.00 ± 10.0 | 1.0 ± 0.0 | -0.05 ± 0.0 | 10.0 | -1.0 | -0.5 |
| bear | -250.00 ± 4.0 | -0.5 ± 0.1 | -0.20 ± 0.0 | 5.0 | -1.0 | -0.5 |
"""

_SUMMARY_JSON = {
    "policy_name": "ppo",
    "result": {
        "total_pnl": 1234.5,
        "annualized_return": 0.25,
        "sharpe": 1.75,
        "max_drawdown": -0.12,
        "fee_income": 900.0,
    },
}


def test_regime_comparison_markdown_parses_expected_pnl() -> None:
    per_regime = parse_regime_pnl(_COMPARISON_MD)

    assert per_regime == {"bull": 1234.56, "bear": -500.0, "sideways": 200.0}


def test_regime_matrix_markdown_parses_expected_pnl() -> None:
    per_regime = parse_regime_pnl(_MATRIX_MD)

    assert per_regime == {"bull": 1000.0, "bear": -250.0}


def test_load_eval_metrics_combines_regime_and_summary(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "regime_comparison.md").write_text(_COMPARISON_MD, encoding="utf-8")
    (results / "summary.json").write_text(json.dumps(_SUMMARY_JSON), encoding="utf-8")

    metrics = load_eval_metrics(results)

    assert metrics is not None
    assert metrics.per_regime == {"bull": 1234.56, "bear": -500.0, "sideways": 200.0}
    assert metrics.worst_regime_pnl == -500.0
    assert metrics.sharpe == 1.75
    assert metrics.total_pnl == 1234.5
    assert metrics.annualized_return == 0.25
    assert metrics.max_drawdown == -0.12
    assert metrics.source == "regime_comparison.md"


def test_unrecognised_markdown_returns_none_fields(tmp_path: Path) -> None:
    assert parse_regime_pnl("# hello\n\nno table here\n") == {}
    assert parse_regime_pnl("") == {}

    results = tmp_path / "results"
    results.mkdir()
    (results / "regime_comparison.md").write_text("# nothing useful\n", encoding="utf-8")

    metrics = load_eval_metrics(results)

    assert metrics is not None
    assert metrics.per_regime == {}
    assert metrics.worst_regime_pnl is None
    assert metrics.sharpe is None
    assert metrics.total_pnl is None
    assert metrics.source == "regime_comparison.md"


def test_regime_pnl_policy_filter_selects_one_policy() -> None:
    multi = """| Policy | Regime | PnL | Sharpe | MaxDD |
| --- | --- | --- | --- | --- |
| baseline | bull | 10.00 ± 0.1 | 0.1 ± 0.0 | -0.1 ± 0.0 |
| ppo | bull | 999.00 ± 1.0 | 2.0 ± 0.1 | -0.01 ± 0.0 |
| ppo | bear | -100.00 ± 1.0 | -1.0 ± 0.1 | -0.05 ± 0.0 |
"""

    assert parse_regime_pnl(multi) == {"bull": 10.0, "bear": -100.0}
    assert parse_regime_pnl(multi, policy="ppo") == {"bull": 999.0, "bear": -100.0}


def test_load_eval_metrics_prefers_matrix_for_policy(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "regime_comparison.md").write_text(_COMPARISON_MD, encoding="utf-8")
    (results / "regime_matrix_ppo.md").write_text(_MATRIX_MD, encoding="utf-8")

    metrics = load_eval_metrics(results, policy="ppo")

    assert metrics is not None
    assert metrics.source == "regime_matrix_ppo.md"
    assert metrics.per_regime == {"bull": 1000.0, "bear": -250.0}


def test_load_eval_metrics_survives_undecodable_artifact(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "regime_comparison.md").write_bytes(b"\xff\xfe not utf-8")
    (results / "summary.json").write_text(json.dumps(_SUMMARY_JSON), encoding="utf-8")

    metrics = load_eval_metrics(results)

    assert metrics is not None
    assert metrics.per_regime == {}
    assert metrics.sharpe == 1.75


def test_summary_json_maps_nested_result_keys() -> None:
    parsed = parse_summary_json(json.dumps(_SUMMARY_JSON))

    assert parsed == {
        "sharpe": 1.75,
        "total_pnl": 1234.5,
        "annualized_return": 0.25,
        "max_drawdown": -0.12,
    }


def test_summary_json_accepts_flat_payload_and_rejects_garbage() -> None:
    assert parse_summary_json(json.dumps({"sharpe": 2.0}))["sharpe"] == 2.0

    garbage = parse_summary_json("{not json")
    assert garbage == {
        "sharpe": None,
        "total_pnl": None,
        "annualized_return": None,
        "max_drawdown": None,
    }


def test_results_summary_markdown_parses_writer_output() -> None:
    text = (
        "# Evaluation summary (RQ1–RQ3)\n\n"
        "## RQ1 — friction ablation\n\n"
        "* Best PnL: **full** (1,234.56 USDC)\n\n"
        "\n| Ablation | PnL (USDC) | Sharpe |\n"
        "| --- | --- | --- |\n"
        "| full | 1,234.56 | 1.7500 |\n"
        "| no_gas | 900.00 | 1.2000 |\n"
    )

    parsed = parse_results_summary_markdown(text)

    assert parsed["total_pnl"] == 1234.56
    assert parsed["sharpe"] == 1.75
    assert parsed["max_drawdown"] is None
    assert parsed["annualized_return"] is None


def test_results_summary_markdown_uses_best_pnl_bullet_without_table() -> None:
    text = "# Evaluation summary\n\n* Best PnL: **full** (2,500.00 USDC)\n"

    assert parse_results_summary_markdown(text)["total_pnl"] == 2500.0


def test_load_eval_metrics_falls_back_to_results_summary(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "results_summary.md").write_text(
        "# Evaluation summary\n\n* Best PnL: **full** (500.00 USDC)\n", encoding="utf-8"
    )

    metrics = load_eval_metrics(results)

    assert metrics is not None
    assert metrics.total_pnl == 500.0
    assert metrics.source == "results_summary.md"


def test_load_eval_metrics_missing_dir_is_none(tmp_path: Path) -> None:
    assert load_eval_metrics(tmp_path / "nope") is None


def test_load_eval_metrics_merges_partial_summary_with_markdown(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "summary.json").write_text(
        json.dumps({"result": {"total_pnl": 10.0}}), encoding="utf-8"
    )
    (results / "results_summary.md").write_text(
        "| Ablation | PnL (USDC) | Sharpe |\n"
        "| --- | --- | --- |\n"
        "| full | 999.00 | 2.5 |\n",
        encoding="utf-8",
    )

    metrics = load_eval_metrics(results)

    assert metrics is not None
    assert metrics.total_pnl == 10.0  # summary.json wins
    assert metrics.sharpe == 2.5  # filled from the markdown fallback
    assert metrics.source == "summary.json"


def test_load_eval_metrics_empty_dir_is_none(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()

    assert load_eval_metrics(results) is None
