"""Tests for the report appendix-table generator (``scripts/report``)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "report" / "make_appendix_tables.py"


def _load_module():  # noqa: ANN202 - test helper
    spec = importlib.util.spec_from_file_location("make_appendix_tables", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tables = _load_module()


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_mean_rejects_empty_and_computes() -> None:
    assert tables.mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="at least one"):
        tables.mean([])


def test_num_formats_signed_thin_space_numbers() -> None:
    assert tables.num(1622.95) == "$+1\\,623$"
    assert tables.num(-17680.17) == "$-17\\,680$"
    assert tables.num(1.5264, decimals=2, signed=False) == "$1.53$"
    assert tables.num(-0.1012 * 100, decimals=1, signed=False) == "$-10.1$"


def test_num_never_emits_a_plus_on_negative_zero() -> None:
    assert tables.num(-0.0001, decimals=2) == "$-0.00$"
    assert tables.num(-0.4) == "$-0$"


def test_policy_label_escapes_unknown_keys() -> None:
    assert tables.policy_label("hodl") == r"\textsf{hodl}"
    assert tables.policy_label("passive_narrow") == r"\textsf{passive\_narrow}"
    assert tables.policy_label("new_rule") == r"\textsf{new\_rule}"


def test_summary_table_renders_booktabs_rows() -> None:
    rows = [
        {"policy": "hodl", "total_pnl": "10.0", "sharpe": "0.5"},
        {"policy": "cost_aware", "total_pnl": "-20.0", "sharpe": "-1.0"},
    ]
    text = tables.make_summary_table(
        rows,
        caption="caption",
        label="tab:x",
        policies=("hodl", "cost_aware"),
        metrics=(("PnL", "total_pnl", 0, True), ("Sharpe", "sharpe", 2, True)),
    )
    assert r"\begin{table}" in text and r"\bottomrule" in text
    assert r"\textsf{hodl} & $+10$ & $+0.50$ \\" in text
    assert r"\textsf{cost\_aware} & $-20$ & $-1.00$ \\" in text


def test_summary_table_raises_for_a_missing_policy() -> None:
    rows = [{"policy": "hodl", "total_pnl": "10.0", "sharpe": "0.5"}]
    with pytest.raises(ValueError, match="passive_wide"):
        tables.make_summary_table(
            rows,
            caption="caption",
            label="tab:x",
            policies=("hodl", "passive_wide"),
            metrics=(("PnL", "total_pnl", 0, True),),
        )


def test_policy_windows_table_orders_windows_and_labels_regimes() -> None:
    rows = [
        {
            "policy": "hodl",
            "window": "3",
            "regime": "nonsense",
            "total_pnl": "100.0",
            "sharpe": "2.0",
            "max_drawdown": "-0.10",
            "cvar_95": "-5.0",
            "net": "12.0",
        },
        {
            "policy": "hodl",
            "window": "1",
            "regime": "bull",
            "total_pnl": "-50.0",
            "sharpe": "-1.0",
            "max_drawdown": "-0.05",
            "cvar_95": "-2.0",
            "net": "-4.0",
        },
    ]
    text = tables.make_policy_windows_table(rows, "hodl", label="tab:hodl")
    first = text.index("1 & bull")
    second = text.index("3 & nonsense")
    assert first < second, "windows must be rendered in ascending order"
    # Unknown regime labels pass through, and drawdown scales to percent.
    assert "-10.0" in text and "-5.0" in text
    assert "$+100$" in text and "$-50$" in text


def test_lvr_table_deduplicates_windows() -> None:
    rows = [
        {
            "window": "1",
            "regime": "bull",
            "sigma": "0.5",
            "volume": "1000000",
            "lvr_usdc": "300.0",
        },
        {
            "window": "1",
            "regime": "bull",
            "sigma": "0.5",
            "volume": "1000000",
            "lvr_usdc": "300.0",
        },
        {
            "window": "2",
            "regime": "bear",
            "sigma": "0.7",
            "volume": "2000000",
            "lvr_usdc": "500.0",
        },
    ]
    text = tables.make_lvr_table(rows)
    assert text.count("bull") == 1
    assert text.count("bear") == 1


def test_friction_table_renders_all_variants_in_order() -> None:
    rows = [
        {
            "ablation_name": name,
            "total_pnl": "1.0",
            "sharpe": "0.5",
            "fee_income": "10.0",
            "il_realized": "-2.0",
            "gas_paid": "-0.1",
            "slippage_paid": "-0.01",
        }
        for name in ("full", "no_il", "static_fee", "no_gas", "no_slippage")
    ]
    text = tables.make_friction_table(rows, caption="caption", label="tab:f")
    assert text.index("full") < text.index("no gas") < text.index("no slippage")
    assert text.index("no slippage") < text.index("no IL") < text.index("flat gas")


def test_friction_table_raises_when_a_variant_is_missing() -> None:
    present = {
        "full": "0",
        "no_gas": "0",
        "no_slippage": "0",
        "static_fee": "0",
    }
    rows = [
        {
            "ablation_name": name,
            "total_pnl": pnl,
            "sharpe": "0",
            "fee_income": "0",
            "il_realized": "0",
            "gas_paid": "0",
            "slippage_paid": "0",
        }
        for name, pnl in present.items()
    ]
    with pytest.raises(ValueError, match="no_il"):
        tables.make_friction_table(rows, caption="caption", label="tab:f")


def test_generate_writes_every_fragment(tmp_path, monkeypatch) -> None:
    """The full generator runs against a minimal set of CSVs."""
    joint = tmp_path / "joint"
    contrast = tmp_path / "contrast"
    noregime = tmp_path / "noregime"
    monkeypatch.setattr(tables, "JOINT", joint)
    monkeypatch.setattr(tables, "CONTRAST", contrast)
    monkeypatch.setattr(tables, "NOREGIME", noregime)

    joint_policies = [
        "hodl",
        "passive_narrow",
        "passive_wide",
        "full_range_v2",
        "hedged_hodl_1.0",
        "ppo_s0",
        "ppo_s1",
        "ppo_s2",
        "ppo_s3",
        "ppo_s4",
    ]
    window_header = "policy,window,regime,total_pnl,sharpe,max_drawdown,cvar_95,net"
    _write_csv(
        joint / "windows.csv",
        window_header,
        [f"{p},1,bull,1.0,0.5,-0.1,-2.0,0.5" for p in joint_policies],
    )
    _write_csv(
        joint / "extra_benchmarks.csv",
        "policy,window,regime,total_pnl,sharpe,max_drawdown,cvar_95,n_rebalances,total_fees,net",
        [
            "tau_reset,1,bull,1.0,0.5,-0.1,-2.0,3.0,4.0,0.5",
            "cost_aware,1,bull,1.0,0.5,-0.1,-2.0,3.0,4.0,0.5",
        ],
    )
    _write_csv(
        joint / "static_widths.csv",
        "split,policy,width,window,regime,total_pnl,sharpe,max_drawdown,cvar_95,total_fees,total_il,net",
        [
            f"{split},{policy},50,1,bull,1.0,0.5,-0.1,-2.0,4.0,-1.0,0.5"
            for split in ("train", "eval")
            for policy in ("static_w50", "static_w250", "static_w1000")
        ],
    )
    boundary_policies = ["hodl", "passive_wide", "tau_reset", "cost_aware", "ppo_s3"]
    _write_csv(
        joint / "boundary_sweep.csv",
        "axis,level,policy,window,regime,total_pnl,sharpe,cvar_95,n_rebalances,sigma,volume,lvr_usdc",
        [
            f"gas,{level},{policy},1,bull,1.0,0.5,-2.0,1.0,0.5,1000.0,300.0"
            for level in ("gas_0.25x", "gas_4x", "capital_10k", "capital_1m")
            for policy in boundary_policies
        ],
    )
    friction_header = (
        "ablation_name,total_pnl,sharpe,fee_income,il_realized,gas_paid,slippage_paid"
    )
    friction_rows = [
        f"{name},1.0,0.5,10.0,-2.0,-0.1,-0.01"
        for name in ("full", "no_gas", "no_slippage", "no_il", "static_fee")
    ]
    for base in (joint, contrast):
        _write_csv(
            base / "friction_passive_wide" / "ablation_table.csv",
            friction_header,
            friction_rows,
        )
    _write_csv(joint / "friction_tau_reset" / "ablation_table.csv", friction_header, friction_rows)
    _write_csv(
        contrast / "windows.csv",
        "policy,window,regime,total_pnl,sharpe,cvar_95,total_fees,total_il,net",
        [
            f"{p},1,bull,1.0,0.5,-2.0,4.0,-1.0,0.5"
            for p in (
                "hodl", "passive_narrow", "passive_wide", "full_range_v2",
                "static_w8", "static_w42", "static_w167", "tau_reset", "cost_aware",
            )
        ],
    )
    _write_csv(
        noregime / "windows.csv",
        "policy,window,regime,total_pnl,sharpe,cvar_95,n_rebalances",
        [f"ppo_s{i},1,bull,1.0,0.5,-2.0,1.0" for i in range(5)],
    )

    out = tmp_path / "generated"
    written = tables.generate(out)
    assert len(written) == 7
    for path in written:
        assert path.exists() and path.read_text(encoding="utf-8").startswith("%")
    assert "tau\\_reset" in (out / "appendix_extra.tex").read_text(encoding="utf-8")
