"""Results writers: markdown/CSV artifacts for the RQ1–RQ3 deliverables (S15).

:func:`write_results` is the CONTRACTS §17 surface.  Every markdown table is a
GitHub-flavored pipe table (header, separator, at least one data row) so the
files drop straight into the thesis, and every CSV is written through polars so
``polars.read_csv`` round-trips it.

Artifacts
---------
* ``ablation_table.md`` / ``ablation_table.csv`` — the RQ1 friction ablation.
* ``regime_matrix_<policy>.md`` (one per policy) + ``regime_comparison.md`` —
  the RQ2 per-regime matrix and the policy-comparison view.
* ``gap_report.md`` / ``gap_report.csv`` — the RQ3 sim-to-reality gap.
* ``results_summary.md`` — a single-page summary of the key numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from undertow.sim.evaluate.ablation import AblationResult, RegimeMatrix
from undertow.sim.evaluate.gap import GapReport
from undertow.sim.evaluate.protocol import METRIC_NAMES, REGIME_LABELS

__all__ = ["write_results"]

#: metric -> display column for the per-regime tables.
_REGIME_COLUMNS: tuple[tuple[str, str], ...] = (
    ("total_pnl", "PnL"),
    ("sharpe", "Sharpe"),
    ("max_drawdown", "MaxDD"),
    ("fee_income", "Fee Income"),
    ("gas_paid", "Gas Paid"),
    ("il_realized", "IL"),
)

_ABLATION_HEADERS: tuple[str, ...] = (
    "Ablation",
    "PnL (USDC)",
    "Sharpe",
    "MaxDD",
    "Fee Income",
    "Gas Paid",
    "Slippage Paid",
    "IL Realized",
)


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a GitHub-flavored pipe table.

    String cells are written verbatim; callers are responsible for not embedding
    a ``|``.  The separator row uses ``---`` in every column.
    """
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    # A GitHub pipe table is only valid with at least one data row, so an empty
    # input gets an explicit placeholder rather than a bare header+separator.
    if not rows:
        rows = [tuple("n/a" for _ in headers)]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines) + "\n"


def _fmt_usdc(value: float) -> str:
    return f"{value:,.2f}"


def _fmt_ratio(value: float) -> str:
    return f"{value:.4f}"


def _fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.6g} ± {std:.4g}"


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    """Write a CSV with a guaranteed header, even when there are no rows."""
    if rows:
        frame = pl.DataFrame(list(rows))
        present = [column for column in columns if column in frame.columns]
        remaining = [column for column in frame.columns if column not in present]
        frame = frame.select(present + remaining)
        frame.write_csv(path)
    else:
        path.write_text(",".join(columns) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Ablation (RQ1)
# ---------------------------------------------------------------------------
def _ablation_markdown(ablations: Sequence[AblationResult]) -> str:
    rows = [
        (
            result.ablation_name,
            _fmt_usdc(result.metrics.get("total_pnl", 0.0)),
            _fmt_ratio(result.metrics.get("sharpe", 0.0)),
            _fmt_ratio(result.metrics.get("max_drawdown", 0.0)),
            _fmt_usdc(result.metrics.get("fee_income", 0.0)),
            _fmt_usdc(result.metrics.get("gas_paid", 0.0)),
            _fmt_usdc(result.metrics.get("slippage_paid", 0.0)),
            _fmt_usdc(result.metrics.get("il_realized", 0.0)),
        )
        for result in ablations
    ]
    body = (
        "# RQ1 — Friction ablation table\n\n"
        "Backtester (ground-truth) evaluation, mean over walk-forward episodes and "
        "seeds. Costs are signed (gas/slippage/IL ≤ 0); every PnL is net of gas and "
        "slippage.\n\n"
    )
    body += _md_table(_ABLATION_HEADERS, rows)
    return body


def _write_ablation(ablations: Sequence[AblationResult], output_dir: Path) -> None:
    (output_dir / "ablation_table.md").write_text(
        _ablation_markdown(ablations), encoding="utf-8"
    )
    columns = [
        "ablation_name",
        "fees",
        "gas",
        "slippage",
        "il",
        *METRIC_NAMES,
    ]
    rows: list[dict[str, object]] = []
    for result in ablations:
        row: dict[str, object] = {"ablation_name": result.ablation_name}
        for flag in ("fees", "gas", "slippage", "il"):
            row[flag] = bool(result.flags.get(flag, False))
        for metric in METRIC_NAMES:
            row[metric] = float(result.metrics.get(metric, 0.0))
        rows.append(row)
    _write_csv(output_dir / "ablation_table.csv", columns, rows)


# ---------------------------------------------------------------------------
# Regimes (RQ2)
# ---------------------------------------------------------------------------
def _regime_lookup(matrix: RegimeMatrix) -> dict[tuple[str, str], tuple[float, float]]:
    lookup: dict[tuple[str, str], tuple[float, float]] = {}
    frame = matrix.metrics_per_regime
    for row in frame.iter_rows(named=True):
        lookup[(str(row["regime"]), str(row["metric"]))] = (
            float(row["mean"]),
            float(row["std"]),
        )
    return lookup


def _ordered_regimes(matrix: RegimeMatrix) -> list[str]:
    present = [str(v) for v in matrix.metrics_per_regime["regime"].unique().to_list()]
    ordered = [label for label in REGIME_LABELS if label in present]
    ordered += [label for label in present if label not in REGIME_LABELS]
    return ordered


def _regime_markdown(matrix: RegimeMatrix) -> str:
    lookup = _regime_lookup(matrix)
    rows = []
    for regime in _ordered_regimes(matrix):
        cells = [regime]
        for metric, _label in _REGIME_COLUMNS:
            mean, std = lookup.get((regime, metric), (0.0, 0.0))
            cells.append(_fmt_mean_std(mean, std))
        rows.append(cells)
    body = f"# RQ2 — Per-regime results: {matrix.policy_name}\n\n"
    body += "Mean ± std over walk-forward episodes and seeds.\n\n"
    body += _md_table(["Regime", *[label for _m, label in _REGIME_COLUMNS]], rows)
    return body


def _write_regime(matrices: Sequence[RegimeMatrix], output_dir: Path) -> None:
    comparison_rows = []
    for matrix in matrices:
        lookup = _regime_lookup(matrix)
        (output_dir / f"regime_matrix_{matrix.policy_name}.md").write_text(
            _regime_markdown(matrix), encoding="utf-8"
        )
        for regime in _ordered_regimes(matrix):
            pnl = lookup.get((regime, "total_pnl"), (0.0, 0.0))
            sharpe = lookup.get((regime, "sharpe"), (0.0, 0.0))
            maxdd = lookup.get((regime, "max_drawdown"), (0.0, 0.0))
            comparison_rows.append(
                (
                    matrix.policy_name,
                    regime,
                    _fmt_mean_std(*pnl),
                    _fmt_mean_std(*sharpe),
                    _fmt_mean_std(*maxdd),
                )
            )
    body = "# RQ2 — Policy comparison by regime\n\n"
    body += "Mean ± std over walk-forward episodes and seeds.\n\n"
    body += _md_table(
        ["Policy", "Regime", "PnL", "Sharpe", "MaxDD"], comparison_rows
    )
    (output_dir / "regime_comparison.md").write_text(body, encoding="utf-8")


# ---------------------------------------------------------------------------
# Gap (RQ3)
# ---------------------------------------------------------------------------
def _gap_markdown(report: GapReport) -> str:
    rows = [
        (name, _fmt_usdc(report.gap_by_cost_type.get(name, 0.0)))
        for name in ("fees", "gas", "slippage", "il")
    ]
    rows.append(("**Total PnL**", _fmt_usdc(report.gap)))
    body = (
        f"# RQ3 — Sim-to-reality gap: {report.policy_name}\n\n"
        f"* Simulator PnL (replay, mean over episodes): **{_fmt_usdc(report.sim_pnl)} USDC**\n"
        f"* On-chain/backtester PnL (mean over episodes): "
        f"**{_fmt_usdc(report.onchain_pnl)} USDC**\n"
        f"* Gap (`sim_pnl - onchain_pnl`): **{_fmt_usdc(report.gap)} USDC**\n\n"
        "Component rows are the mean per-term difference (`sim - on-chain`); the "
        "total additionally carries the HODL-path residual. The gap is reported "
        "honestly even when large.\n\n"
    )
    body += _md_table(["Component", "Gap (USDC)"], rows)
    return body


def _write_gap(report: GapReport, output_dir: Path) -> None:
    (output_dir / "gap_report.md").write_text(
        _gap_markdown(report), encoding="utf-8"
    )
    columns = ["policy_name", "component", "gap", "sim_pnl", "onchain_pnl"]
    rows: list[dict[str, object]] = []
    for name in ("fees", "gas", "slippage", "il"):
        rows.append(
            {
                "policy_name": report.policy_name,
                "component": name,
                "gap": float(report.gap_by_cost_type.get(name, 0.0)),
                "sim_pnl": None,
                "onchain_pnl": None,
            }
        )
    rows.append(
        {
            "policy_name": report.policy_name,
            "component": "total",
            "gap": float(report.gap),
            "sim_pnl": float(report.sim_pnl),
            "onchain_pnl": float(report.onchain_pnl),
        }
    )
    _write_csv(output_dir / "gap_report.csv", columns, rows)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _summary_markdown(
    ablations: Sequence[AblationResult],
    matrices: Sequence[RegimeMatrix],
    gap: GapReport,
) -> str:
    lines = ["# Evaluation summary (RQ1–RQ3)\n"]
    lines.append("## RQ1 — friction ablation\n")
    if ablations:
        best = max(ablations, key=lambda item: item.metrics.get("total_pnl", 0.0))
        worst = min(ablations, key=lambda item: item.metrics.get("total_pnl", 0.0))
        lines.append(
            f"* Variants: {', '.join(item.ablation_name for item in ablations)}\n"
            f"* Best PnL: **{best.ablation_name}** "
            f"({_fmt_usdc(best.metrics.get('total_pnl', 0.0))} USDC)\n"
            f"* Worst PnL: **{worst.ablation_name}** "
            f"({_fmt_usdc(worst.metrics.get('total_pnl', 0.0))} USDC)\n"
        )
        rows = [
            (
                item.ablation_name,
                _fmt_usdc(item.metrics.get("total_pnl", 0.0)),
                _fmt_ratio(item.metrics.get("sharpe", 0.0)),
            )
            for item in ablations
        ]
        lines.append("\n" + _md_table(["Ablation", "PnL (USDC)", "Sharpe"], rows))
    else:
        lines.append("\n_No ablation results._\n")

    lines.append("\n## RQ2 — per-regime robustness\n")
    if matrices:
        for matrix in matrices:
            lookup = _regime_lookup(matrix)
            pnls = {
                regime: lookup.get((regime, "total_pnl"), (0.0, 0.0))[0]
                for regime in _ordered_regimes(matrix)
            }
            best_regime = max(pnls, key=lambda key: pnls[key]) if pnls else "n/a"
            lines.append(
                f"* **{matrix.policy_name}**: best regime **{best_regime}** "
                f"({_fmt_usdc(pnls.get(best_regime, 0.0))} USDC)\n"
            )
    else:
        lines.append("\n_No policy matrices._\n")

    lines.append("\n## RQ3 — sim-to-reality gap\n")
    lines.append(
        f"* Policy **{gap.policy_name}**: sim {_fmt_usdc(gap.sim_pnl)} USDC, "
        f"on-chain {_fmt_usdc(gap.onchain_pnl)} USDC, "
        f"gap **{_fmt_usdc(gap.gap)} USDC**\n"
    )
    lines.append(
        "\n*Component gaps (sim − on-chain):*\n\n"
    )
    lines.append(
        _md_table(
            ["Component", "Gap (USDC)"],
            [
                (name, _fmt_usdc(gap.gap_by_cost_type.get(name, 0.0)))
                for name in ("fees", "gas", "slippage", "il")
            ],
        )
    )
    return "\n".join(lines)


def write_results(
    ablation: list[AblationResult],
    regime: list[RegimeMatrix],
    gap: GapReport,
    output_dir: Path,
) -> None:
    """Write every RQ1–RQ3 artifact into ``output_dir`` (CONTRACTS §17)."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_ablation(ablation, output)
    _write_regime(regime, output)
    _write_gap(gap, output)
    (output / "results_summary.md").write_text(
        _summary_markdown(ablation, regime, gap), encoding="utf-8"
    )
