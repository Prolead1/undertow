"""Generate the report's appendix tables from the evaluation artefacts.

Every number in the report's appendices is rendered here, directly from the
CSVs under ``results/``; the fragments are written to ``report/generated/`` and
``\\input`` by the appendix sections.  Run from the repository root:

    uv run python scripts/report/make_appendix_tables.py

The module keeps the aggregation and bootstrap helpers free of file I/O so the
test suite can exercise them on small frames.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections.abc import Iterable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT_DIR = ROOT / "report" / "generated"

JOINT = RESULTS / "eth_usdc_500_joint_rl" / "analysis"
CONTRAST = RESULTS / "eth_usdc_3000_contrast" / "analysis"
NOREGIME = RESULTS / "eth_usdc_500_joint_noregime" / "analysis"

#: Display names for the artefact policy keys.
POLICY_LABELS: dict[str, str] = {
    "hodl": r"\textsf{hodl}",
    "passive_narrow": r"\textsf{passive\_narrow}",
    "passive_wide": r"\textsf{passive\_wide}",
    "full_range_v2": r"\textsf{full\_range\_v2}",
    "hedged_hodl_1.0": r"\textsf{hedged\_hodl}",
    "tau_reset": r"\textsf{tau\_reset}",
    "cost_aware": r"\textsf{cost\_aware}",
    "static_w8": r"static $8$",
    "static_w42": r"static $42$",
    "static_w50": r"static $50$",
    "static_w167": r"static $167$",
    "static_w250": r"static $250$",
    "static_w1000": r"static $1000$",
}
for _seed in range(5):
    POLICY_LABELS[f"ppo_s{_seed}"] = rf"\textsf{{seed {_seed}}}"

REGIME_LABELS = {
    "bull": "bull",
    "bear": "bear",
    "sideways": "sideways",
    "high_vol": "high volatility",
    "unknown": "unknown",
    "": "unknown",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of string dictionaries."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values: Iterable[float]) -> float:
    """Arithmetic mean of a non-empty sequence."""
    materialised = list(values)
    if not materialised:
        raise ValueError("mean() requires at least one value")
    return statistics.fmean(materialised)


def num(value: float, *, decimals: int = 0, signed: bool = True) -> str:
    """Format a number for the appendix tables (thin-space thousands)."""
    rounded = round(value, decimals)
    text = f"{rounded:,.{decimals}f}" if decimals else f"{rounded:,.0f}"
    text = text.replace(",", r"\,")
    if signed and not text.startswith("-"):
        text = "+" + text
    return "$" + text + "$"


def policy_label(key: str) -> str:
    """LaTeX label for an artefact policy key."""
    return POLICY_LABELS.get(key, rf"\textsf{{{key.replace('_', r'\_')}}}")


def _aggregate(rows: Sequence[dict[str, str]], metric: str) -> float:
    return mean(float(row[metric]) for row in rows)


def _mean_table(
    rows: Sequence[dict[str, str]],
    *,
    policies: Sequence[str],
    metrics: Sequence[tuple[str, str, int, bool]],
) -> str:
    """Render a policy-by-metric mean table body.

    Each metric is ``(title, column, decimals, signed)``; ``signed`` controls
    the leading ``+`` on non-negative values.
    """
    lines = []
    for policy in policies:
        subset = [row for row in rows if row["policy"] == policy]
        if not subset:
            raise ValueError(f"no rows for expected policy {policy!r}")
        values = " & ".join(
            num(_aggregate(subset, column), decimals=decimals, signed=signed)
            for _, column, decimals, signed in metrics
        )
        lines.append(f"{policy_label(policy)} & {values} \\\\")
    return "\n".join(lines)


def make_summary_table(
    rows: Sequence[dict[str, str]],
    *,
    caption: str,
    label: str,
    policies: Sequence[str],
    metrics: Sequence[tuple[str, str, int, bool]],
) -> str:
    """A complete booktabs table of per-policy means."""
    headers = " & ".join(title for title, _, _, _ in metrics)
    body = _mean_table(rows, policies=policies, metrics=metrics)
    return (
        "\\begin{table}[tbp]\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\centering\n\\small\n"
        f"\\begin{{tabular}}{{@{{}}L{{0.16\\linewidth}}{'r' * len(metrics)}@{{}}}}\n"
        "\\toprule\n"
        f"\\textbf{{Policy}} & {headers} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def make_policy_windows_table(
    rows: Sequence[dict[str, str]],
    policy: str,
    *,
    label: str,
) -> str:
    """One per-window table for a single policy."""
    subset = sorted(
        (row for row in rows if row["policy"] == policy),
        key=lambda row: int(row["window"]),
    )
    lines = []
    for row in subset:
        lines.append(
            f"{int(row['window'])} & {REGIME_LABELS.get(row['regime'], row['regime'])}"
            f" & {num(float(row['total_pnl']))}"
            f" & {num(float(row['sharpe']), decimals=2, signed=False)}"
            f" & {num(float(row['max_drawdown']) * 100, decimals=1, signed=False)}"
            f" & {num(float(row['cvar_95']), decimals=1)}"
            f" & {num(float(row['net']))} \\\\"
        )
    return (
        "\\begin{table}[tbp]\n"
        f"\\caption{{Per-window evaluation of {policy_label(policy)}. PnL, CVaR\n"
        "and net are in USDC; drawdown is a percentage of capital; net is the\n"
        "liquidity leg's excess over the opening basket.}\n"
        f"\\label{{{label}}}\n"
        "\\centering\n\\small\n"
        "\\begin{tabular}{@{}rrrrrrr@{}}\n"
        "\\toprule\n"
        "\\textbf{Window} & \\textbf{Regime} & \\textbf{PnL} & \\textbf{Sharpe}"
        " & \\textbf{maxDD (\\%)} & \\textbf{CVaR} & \\textbf{Net} \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def make_lvr_table(rows: Sequence[dict[str, str]]) -> str:
    """Per-window realised volatility, volume and the analytical LVR."""
    seen: dict[int, dict[str, str]] = {}
    for row in rows:
        seen.setdefault(int(row["window"]), row)
    lines = []
    for index in sorted(seen):
        row = seen[index]
        lines.append(
            f"{index} & {REGIME_LABELS.get(row['regime'], row['regime'])}"
            f" & {float(row['sigma']):.3f}"
            f" & {num(float(row['volume']), signed=False)}"
            f" & {num(float(row['lvr_usdc']), decimals=1)} \\\\"
        )
    return (
        "\\begin{table}[tbp]\n"
        "\\caption{Recorded market state per evaluation window: annualised\n"
        "realised volatility from the reference series, swap notional, and the\n"
        "analytical loss-versus-rebalancing floor $0.125\\,\\sigma^2Vt$ at\n"
        "\\usdc{100000} for a full month; the final window is partial, so its\n"
        "fee comparison rescales the floor to the window length.}\n"
        "\\label{tab:appendix-lvr}\n"
        "\\centering\n\\small\n"
        "\\begin{tabular}{@{}rrrrr@{}}\n"
        "\\toprule\n"
        "\\textbf{Window} & \\textbf{Regime} & $\\sigma$ & \\textbf{Volume (USDC)}"
        " & \\textbf{LVR floor (USDC)} \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )




def make_friction_table(
    rows: Sequence[dict[str, str]], *, caption: str, label: str
) -> str:
    """Render one five-variant friction-ablation table."""
    order = ["full", "no_gas", "no_slippage", "no_il", "static_fee"]
    labels = {
        "full": "full",
        "no_gas": "no gas",
        "no_slippage": "no slippage",
        "no_il": "no IL",
        "static_fee": "flat gas",
    }
    by_name = {row["ablation_name"]: row for row in rows}
    lines = []
    for name in order:
        row = by_name.get(name)
        if row is None:
            raise ValueError(f"missing friction variant {name!r}")
        lines.append(
            f"{labels[name]}"
            f" & {num(float(row['total_pnl']))}"
            f" & {num(float(row['sharpe']), decimals=3)}"
            f" & {num(float(row['fee_income']), signed=False)}"
            f" & {num(float(row['il_realized']))}"
            f" & {num(float(row['gas_paid']))}"
            f" & {num(float(row['slippage_paid']))} \\\\"
        )
    return (
        "\\begin{table}[tbp]\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\centering\n\\small\n"
        "\\begin{tabular}{@{}L{0.14\\linewidth}rrrrrr@{}}\n"
        "\\toprule\n"
        "\\textbf{Variant} & \\textbf{PnL} & \\textbf{Sharpe}"
        " & \\textbf{Fee income} & \\textbf{IL} & \\textbf{Gas}"
        " & \\textbf{Slippage} \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def generate(out_dir: Path = OUT_DIR) -> list[Path]:
    """Write every appendix fragment; return the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    joint = read_csv(JOINT / "windows.csv")
    extra = read_csv(JOINT / "extra_benchmarks.csv")

    # -- Appendix E: full per-window results -----------------------------
    parts = [
        "% Generated by scripts/report/make_appendix_tables.py; do not edit.\n"
    ]
    for policy in (
        "hodl",
        "passive_narrow",
        "passive_wide",
        "full_range_v2",
        "hedged_hodl_1.0",
    ):
        parts.append(
            make_policy_windows_table(
                joint, policy, label=f"tab:appendix-{policy.replace('.', '-')}"
            )
        )
    for seed in range(5):
        parts.append(
            make_policy_windows_table(joint, f"ppo_s{seed}", label=f"tab:appendix-ppo-s{seed}")
        )
    for policy in ("tau_reset", "cost_aware"):
        parts.append(
            make_policy_windows_table(extra, policy, label=f"tab:appendix-{policy}")
        )
    path = out_dir / "appendix_windows.tex"
    path.write_text("\n".join(parts), encoding="utf-8")
    written.append(path)

    # -- Appendix F: ablation summaries ----------------------------------
    static = read_csv(JOINT / "static_widths.csv")
    static_train = [row for row in static if row["split"] == "train"]
    static_eval = [row for row in static if row["split"] == "eval"]
    metrics = (
        ("PnL", "total_pnl", 0, True),
        ("Sharpe", "sharpe", 3, True),
        ("CVaR", "cvar_95", 2, True),
        ("Fees", "total_fees", 0, False),
        ("IL", "total_il", 0, True),
        ("Net", "net", 0, True),
    )
    static_parts = [
        "% Generated by scripts/report/make_appendix_tables.py; do not edit.\n",
        make_summary_table(
            static_train,
            caption=(
                "Static one-shot widths on the training split, mean over the "
                "twenty-five training windows. The width with the best training "
                "Sharpe is frozen as the comparator."
            ),
            label="tab:appendix-static-train",
            policies=("static_w50", "static_w250", "static_w1000"),
            metrics=metrics,
        ),
        make_summary_table(
            static_eval,
            caption=(
                "The same static widths on the thirteen evaluation windows, "
                "mean. Width $50$ is the frozen comparator chosen on the "
                "training split."
            ),
            label="tab:appendix-static-eval",
            policies=("static_w50", "static_w250", "static_w1000"),
            metrics=metrics,
        ),
    ]
    path = out_dir / "appendix_static.tex"
    path.write_text("\n".join(static_parts), encoding="utf-8")
    written.append(path)

    path = out_dir / "appendix_extra.tex"
    path.write_text(
        "% Generated by scripts/report/make_appendix_tables.py; do not edit.\n"
        + make_summary_table(
            extra,
            caption=(
                "The two scripted recentring rules completed for the study, "
                "mean over the thirteen evaluation windows, with the same "
                "metrics as the main ladder."
            ),
            label="tab:appendix-extra",
            policies=("tau_reset", "cost_aware"),
            metrics=(
                ("PnL", "total_pnl", 0, True),
                ("Sharpe", "sharpe", 3, True),
                ("CVaR", "cvar_95", 2, True),
                ("Rebalances", "n_rebalances", 2, False),
                ("Fees", "total_fees", 0, False),
                ("Net", "net", 0, True),
            ),
        ),
        encoding="utf-8",
    )
    written.append(path)

    boundary = read_csv(JOINT / "boundary_sweep.csv")
    boundary_parts = [
        "% Generated by scripts/report/make_appendix_tables.py; do not edit.\n",
        make_lvr_table(boundary),
    ]
    for level in ("gas_0.25x", "gas_4x", "capital_10k", "capital_1m"):
        subset = [row for row in boundary if row["level"] == level]
        boundary_parts.append(
            make_summary_table(
                subset,
                caption=(
                    f"Boundary sweep at {level.replace('_', ' ')}, mean over "
                    "the thirteen evaluation windows. PnL and CVaR are in USDC."
                ),
                label=f"tab:appendix-{level.replace('.', '-')}",
                policies=("hodl", "passive_wide", "tau_reset", "cost_aware", "ppo_s3"),
                metrics=(
                    ("PnL", "total_pnl", 0, True),
                    ("Sharpe", "sharpe", 2, True),
                    ("CVaR", "cvar_95", 1, True),
                    ("Rebalances", "n_rebalances", 2, False),
                ),
            )
        )
    path = out_dir / "appendix_boundary.tex"
    path.write_text("\n".join(boundary_parts), encoding="utf-8")
    written.append(path)

    contrast = read_csv(CONTRAST / "windows.csv")
    path = out_dir / "appendix_contrast.tex"
    path.write_text(
        "% Generated by scripts/report/make_appendix_tables.py; do not edit.\n"
        + make_summary_table(
            contrast,
            caption=(
                "The rule ladder on the $0.30\\,\\%$ contrast pool, mean over "
                "the thirteen evaluation windows. The static widths are the "
                "price-equivalent of the primary pool's ladder at this pool's "
                "spacing of $60$."
            ),
            label="tab:appendix-contrast",
            policies=(
                "hodl",
                "passive_narrow",
                "passive_wide",
                "full_range_v2",
                "static_w8",
                "static_w42",
                "static_w167",
                "tau_reset",
                "cost_aware",
            ),
            metrics=(
                ("PnL", "total_pnl", 0, True),
                ("Sharpe", "sharpe", 3, True),
                ("CVaR", "cvar_95", 1, True),
                ("Fees", "total_fees", 0, False),
                ("IL", "total_il", 0, True),
                ("Net", "net", 0, True),
            ),
        ),
        encoding="utf-8",
    )
    written.append(path)


    friction_files = (
        (
            "the primary pool's one-shot wide band",
            JOINT / "friction_passive_wide" / "ablation_table.csv",
            "tab:appendix-friction-wide",
        ),
        (
            "the primary pool's recentring rule",
            JOINT / "friction_tau_reset" / "ablation_table.csv",
            "tab:appendix-friction-tau",
        ),
        (
            "the contrast pool's one-shot wide band",
            CONTRAST / "friction_passive_wide" / "ablation_table.csv",
            "tab:appendix-friction-contrast",
        ),
    )
    friction_parts = [
        "% Generated by scripts/report/make_appendix_tables.py; do not edit.\n"
    ]
    for description, path_csv, label in friction_files:
        friction_parts.append(
            make_friction_table(
                read_csv(path_csv),
                caption=(
                    "Friction ablation on " + description + ", mean over the "
                    "thirteen evaluation windows. The variant switches one cost "
                    "term off or re-prices gas at a flat 30 gwei base fee plus "
                    "a 3\\,\\% tip."
                ),
                label=label,
            )
        )
    friction_path = out_dir / "appendix_friction.tex"
    friction_path.write_text("\n".join(friction_parts), encoding="utf-8")
    written.append(friction_path)

    noregime = read_csv(NOREGIME / "windows.csv")
    path = out_dir / "appendix_noregime.tex"
    path.write_text(
        "% Generated by scripts/report/make_appendix_tables.py; do not edit.\n"
        + make_summary_table(
            noregime,
            caption=(
                "The regime-feature ablation arm, mean over the thirteen "
                "evaluation windows. The regime input is frozen to a constant "
                "for training and evaluation."
            ),
            label="tab:appendix-noregime",
            policies=tuple(f"ppo_s{seed}" for seed in range(5)),
            metrics=(
                ("PnL", "total_pnl", 0, True),
                ("Sharpe", "sharpe", 3, True),
                ("CVaR", "cvar_95", 2, True),
                ("Rebalances", "n_rebalances", 2, False),
            ),
        ),
        encoding="utf-8",
    )
    written.append(path)

    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=OUT_DIR, help="fragment output directory"
    )
    args = parser.parse_args(argv)
    for path in generate(args.out_dir):
        shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        print(f"wrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
