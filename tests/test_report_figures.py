"""Tests for ``report/figures/make_figures.py``.

The generator is loaded by path (``report/figures`` is not a package). The
tests check the position math against the v3 identities the figures rely on,
the data-shaping helpers, and that every figure is written from stub
artefacts.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "report" / "figures" / "make_figures.py"


def _load_module():  # noqa: ANN202 - test helper
    spec = importlib.util.spec_from_file_location("make_figures", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


figures = _load_module()

ENTRY = 3000.0
LOW = ENTRY / 1.05
HIGH = ENTRY * 1.05
CAPITAL = 100_000.0


def test_position_value_at_entry_equals_capital() -> None:
    assert figures.position_value(ENTRY, LOW, HIGH, ENTRY) == pytest.approx(CAPITAL, rel=1e-9)


def test_position_value_above_band_matches_all_usdc_amount() -> None:
    liquidity = figures.liquidity_for_capital(ENTRY, LOW, HIGH)
    s_a, s_b = figures.sqrt_price(HIGH), figures.sqrt_price(LOW)
    all_usdc = liquidity * (1.0 / s_a - 1.0 / s_b) / 1e6
    assert figures.position_value(3300.0, LOW, HIGH, ENTRY) == pytest.approx(all_usdc, rel=1e-12)
    assert figures.position_value(3600.0, LOW, HIGH, ENTRY) == pytest.approx(all_usdc, rel=1e-12)


def test_position_value_below_band_matches_all_weth_amount() -> None:
    liquidity = figures.liquidity_for_capital(ENTRY, LOW, HIGH)
    s_a, s_b = figures.sqrt_price(HIGH), figures.sqrt_price(LOW)
    slope = liquidity * (s_b - s_a) / 1e18
    assert figures.position_value(2400.0, LOW, HIGH, ENTRY) == pytest.approx(
        slope * 2400.0, rel=1e-12
    )
    assert figures.position_value(2000.0, LOW, HIGH, ENTRY) == pytest.approx(
        slope * 2000.0, rel=1e-12
    )


def test_entry_amounts_and_hodl_value_agree() -> None:
    amount0, amount1 = figures.entry_amounts(ENTRY, LOW, HIGH)
    assert amount0 + amount1 * ENTRY == pytest.approx(CAPITAL, rel=1e-9)
    assert figures.hodl_value(ENTRY, ENTRY, LOW, HIGH) == pytest.approx(CAPITAL, rel=1e-9)
    q = 3400.0
    assert figures.hodl_value(q, ENTRY, LOW, HIGH) == pytest.approx(
        amount0 + amount1 * q, rel=1e-12
    )
    # A multiplicatively symmetric band enters exactly half of its value in WETH.
    assert amount1 * ENTRY == pytest.approx(0.5 * CAPITAL, rel=1e-9)


def test_eth_weight_moves_from_one_to_zero() -> None:
    assert figures.eth_weight(ENTRY, LOW, HIGH, ENTRY) == pytest.approx(0.5, rel=1e-9)
    assert figures.eth_weight(4000.0, LOW, HIGH, ENTRY) == pytest.approx(0.0, abs=1e-12)
    assert figures.eth_weight(2000.0, LOW, HIGH, ENTRY) == pytest.approx(1.0, abs=1e-12)


def test_impermanent_loss_matches_known_values() -> None:
    assert figures.impermanent_loss(1.0) == pytest.approx(0.0, abs=1e-12)
    assert figures.impermanent_loss(1.2) == pytest.approx(-0.004141, abs=1e-6)
    # Symmetric around 1 in the ratio r <-> 1/r sense.
    assert figures.impermanent_loss(0.8) == pytest.approx(
        figures.impermanent_loss(1.25), rel=1e-9
    )


def test_lvr_rate_is_quadratic_in_volatility() -> None:
    assert figures.lvr_rate(0.5, 100_000.0) == pytest.approx(3125.0, rel=1e-12)
    assert figures.lvr_rate(1.0, 100_000.0) == pytest.approx(
        4 * figures.lvr_rate(0.5, 100_000.0)
    )


def test_read_rows_reads_csv(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("policy,sharpe\na,1.0\nb,2.0\n", encoding="utf-8")
    rows = figures.read_rows(path)
    assert rows == [{"policy": "a", "sharpe": "1.0"}, {"policy": "b", "sharpe": "2.0"}]


def test_mean_metric_averages_and_rejects_empty() -> None:
    rows = [{"policy": "a", "sharpe": "1.0"}, {"policy": "a", "sharpe": "3.0"}]
    assert figures.mean_metric(rows, "sharpe") == pytest.approx(2.0)
    with pytest.raises(ValueError, match="no rows"):
        figures.mean_metric([], "sharpe")


def test_policy_means_groups_per_policy() -> None:
    rows = [
        {"policy": "b", "cvar_95": "-2.0"},
        {"policy": "a", "cvar_95": "-4.0"},
        {"policy": "b", "cvar_95": "-6.0"},
    ]
    means = figures.policy_means(rows, "cvar_95")
    assert list(means) == ["b", "a"]
    assert means == {"b": pytest.approx(-4.0), "a": pytest.approx(-4.0)}


def test_telemetry_curve_skips_null_rewards(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {"timestep": 10_000, "ep_rew_mean": None},
                {"timestep": 20_000, "ep_rew_mean": -0.9},
                {"timestep": 30_000, "ep_rew_mean": -0.6},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    assert figures.telemetry_curve(path) == [(20_000, -0.9), (30_000, -0.6)]


def test_telemetry_curve_empty_file_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    assert figures.telemetry_curve(path) == []


def test_action_map_data_builds_band_matrix_and_normalised_mix() -> None:
    rows = [
        {
            "policy": "ppo_s0",
            "window": "2",
            "band": "250",
            "ratio_0": "0",
            "ratio_025": "0",
            "ratio_05": "2",
            "ratio_075": "0",
            "ratio_10": "0",
        },
        {
            "policy": "ppo_s0",
            "window": "1",
            "band": "-1",
            "ratio_0": "1",
            "ratio_025": "0",
            "ratio_05": "1",
            "ratio_075": "0",
            "ratio_10": "0",
        },
        {
            "policy": "ppo_s1",
            "window": "1",
            "band": "50",
            "ratio_0": "0",
            "ratio_025": "1",
            "ratio_05": "0",
            "ratio_075": "0",
            "ratio_10": "0",
        },
        # Benchmarks are ignored by the action map.
        {
            "policy": "hodl",
            "window": "1",
            "band": "-1",
            "ratio_0": "9",
            "ratio_025": "0",
            "ratio_05": "0",
            "ratio_075": "0",
            "ratio_10": "0",
        },
    ]
    windows, bands, mix = figures.action_map_data(rows, ["ppo_s0", "ppo_s1"])
    assert windows == [1, 2]
    assert bands.tolist() == [[-1.0, 250.0], [50.0, -1.0]]
    assert mix[0].tolist() == pytest.approx([0.25, 0.0, 0.75, 0.0, 0.0])
    assert mix[1].tolist() == pytest.approx([0.0, 1.0, 0.0, 0.0, 0.0])


def test_action_map_data_rejects_a_seed_without_decisions() -> None:
    rows = [
        {
            "policy": "ppo_s0",
            "window": "1",
            "band": "50",
            "ratio_0": "0",
            "ratio_025": "0",
            "ratio_05": "0",
            "ratio_075": "0",
            "ratio_10": "0",
        }
    ]
    with pytest.raises(ValueError, match="no hedge-ratio"):
        figures.action_map_data(rows, ["ppo_s0"])


def test_text_color_for_share_uses_contrast_at_both_ends() -> None:
    # Viridis is near-black at the bottom of the map and bright at the top, so
    # the annotations flip from white to dark grey.
    assert figures.text_color_for_share(0.0, 1.0) == "white"
    assert figures.text_color_for_share(1.0, 1.0) == "0.1"


def test_text_color_for_share_scales_with_a_small_maximum() -> None:
    # A share at the maximum is the top of the map, whatever the maximum is.
    assert figures.text_color_for_share(0.1, 0.1) == figures.text_color_for_share(1.0, 1.0)
    assert figures.text_color_for_share(0.0, 0.1) == "white"


def test_fee_lvr_points_filters_to_the_passive_gas_arm() -> None:
    rows = [
        {"level": "gas_0.25x", "policy": "hodl"},
        {"level": "gas_4x", "policy": "hodl"},
        {"level": "gas_0.25x", "policy": "passive_wide"},
    ]
    assert figures.fee_lvr_points(rows) == [{"level": "gas_0.25x", "policy": "hodl"}]


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def _stub_artefacts(root: Path) -> None:
    """Write the minimal result CSVs the five result figures read."""
    main = root / "results" / "eth_usdc_500_joint_rl" / "analysis"
    contrast = root / "results" / "eth_usdc_3000_contrast" / "analysis"
    window_policies = (
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
    )
    contrast_policies = (
        "hodl",
        "passive_narrow",
        "passive_wide",
        "full_range_v2",
        "cost_aware",
        "tau_reset",
        "static_w8",
        "static_w42",
        "static_w167",
    )
    window_header = (
        "policy,window,regime,total_pnl,sharpe,max_drawdown,cvar_95,net,band,"
        "ratio_0,ratio_025,ratio_05,ratio_075,ratio_10"
    )
    _write_csv(
        main / "windows.csv",
        window_header,
        [
            f"{policy},1,bull,1.0,0.5,-0.1,-2.0,0.5,-1,1,0,0,0,0"
            for policy in window_policies
        ],
    )
    _write_csv(
        main / "extra_benchmarks.csv",
        "policy,sharpe,cvar_95,net",
        ["tau_reset,0.4,-3.0,0.4", "cost_aware,0.3,-2.5,0.6"],
    )
    _write_csv(
        main / "static_widths.csv",
        "split,policy,sharpe,cvar_95",
        ["eval,static_w50,0.6,-4.0", "train,static_w50,0.7,-5.0"],
    )
    _write_csv(
        main / "boundary_sweep.csv",
        "level,policy,total_fees,lvr_usdc,sigma",
        [
            "gas_0.25x,hodl,100.0,120.0,0.5",
            "gas_0.25x,hodl,80.0,200.0,0.9",
        ],
    )
    _write_csv(
        contrast / "windows.csv",
        "policy,net",
        [f"{policy},0.5" for policy in contrast_policies],
    )
    for seed in range(5):
        telemetry = (
            root
            / "results"
            / "eth_usdc_500_joint_rl"
            / f"seed_{seed}"
            / "telemetry.jsonl"
        )
        telemetry.parent.mkdir(parents=True, exist_ok=True)
        telemetry.write_text(
            json.dumps({"timestep": 0, "ep_rew_mean": None})
            + "\n"
            + json.dumps({"timestep": 1_000_000, "ep_rew_mean": -0.5 - seed * 0.1})
            + "\n",
            encoding="utf-8",
        )


def test_make_all_writes_every_figure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every figure function runs end to end against stub result artefacts."""
    _stub_artefacts(tmp_path)
    results = tmp_path / "results"
    monkeypatch.setattr(figures, "RESULTS", results)
    monkeypatch.setattr(figures, "MAIN_ANALYSIS", results / "eth_usdc_500_joint_rl" / "analysis")
    monkeypatch.setattr(
        figures, "CONTRAST_ANALYSIS", results / "eth_usdc_3000_contrast" / "analysis"
    )
    output = tmp_path / "out"
    output.mkdir()
    paths = figures.make_all(output)
    assert [path.stem for path in paths] == [
        "position_payoff",
        "il_lvr",
        "decision_path",
        "testbed",
        "learning_curves",
        "risk_return",
        "action_map",
        "fee_lvr",
        "fee_tier",
    ]
    for path in paths:
        assert path.is_file(), f"missing figure: {path.name}"
        assert path.stat().st_size > 1000, f"suspiciously small figure: {path.name}"

    # The argparse entry point resolves --output-dir and reports success.
    cli_dir = tmp_path / "cli"
    cli_dir.mkdir()
    assert figures.main(["--output-dir", str(cli_dir)]) == 0
    assert (cli_dir / "fee_tier.pdf").is_file()


def test_make_fee_lvr_rejects_empty_artefacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis = tmp_path / "analysis"
    _write_csv(analysis / "boundary_sweep.csv", "level,policy,total_fees,lvr_usdc,sigma", [])
    monkeypatch.setattr(figures, "MAIN_ANALYSIS", analysis)
    with pytest.raises(ValueError, match="no gas"):
        figures.make_fee_lvr(tmp_path / "out")
