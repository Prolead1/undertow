"""Unit tests for ``report/figures/make_figures.py``.

The generator is loaded by path (``report/figures`` is not a package). The
tests check the position math against the v3 identities the figures rely on,
and that the CLI actually writes the four committed figures.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "report" / "figures" / "make_figures.py"


def _load_figures() -> object:
    spec = importlib.util.spec_from_file_location("make_figures", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_figures"] = module
    spec.loader.exec_module(module)
    return module


figs = _load_figures()

ENTRY = 3000.0
LOW = ENTRY / 1.05
HIGH = ENTRY * 1.05
CAPITAL = 100_000.0


def test_position_value_at_entry_equals_capital() -> None:
    assert figs.position_value(ENTRY, LOW, HIGH, ENTRY) == pytest.approx(CAPITAL, rel=1e-9)


def test_position_value_above_band_matches_all_usdc_amount() -> None:
    liquidity = figs.liquidity_for_capital(ENTRY, LOW, HIGH)
    s_a, s_b = figs.sqrt_price(HIGH), figs.sqrt_price(LOW)
    all_usdc = liquidity * (1.0 / s_a - 1.0 / s_b) / 1e6
    assert figs.position_value(3300.0, LOW, HIGH, ENTRY) == pytest.approx(all_usdc, rel=1e-12)
    assert figs.position_value(3600.0, LOW, HIGH, ENTRY) == pytest.approx(all_usdc, rel=1e-12)


def test_position_value_below_band_matches_all_weth_amount() -> None:
    liquidity = figs.liquidity_for_capital(ENTRY, LOW, HIGH)
    s_a, s_b = figs.sqrt_price(HIGH), figs.sqrt_price(LOW)
    slope = liquidity * (s_b - s_a) / 1e18
    assert figs.position_value(2400.0, LOW, HIGH, ENTRY) == pytest.approx(slope * 2400.0, rel=1e-12)
    assert figs.position_value(2000.0, LOW, HIGH, ENTRY) == pytest.approx(slope * 2000.0, rel=1e-12)


def test_entry_amounts_and_hodl_value_agree() -> None:
    amount0, amount1 = figs.entry_amounts(ENTRY, LOW, HIGH)
    assert amount0 + amount1 * ENTRY == pytest.approx(CAPITAL, rel=1e-9)
    assert figs.hodl_value(ENTRY, ENTRY, LOW, HIGH) == pytest.approx(CAPITAL, rel=1e-9)
    q = 3400.0
    assert figs.hodl_value(q, ENTRY, LOW, HIGH) == pytest.approx(amount0 + amount1 * q, rel=1e-12)
    # A multiplicatively symmetric band enters exactly half of its value in WETH.
    assert amount1 * ENTRY == pytest.approx(0.5 * CAPITAL, rel=1e-9)


def test_eth_weight_moves_from_one_to_zero() -> None:
    assert figs.eth_weight(ENTRY, LOW, HIGH, ENTRY) == pytest.approx(0.5, rel=1e-9)
    assert figs.eth_weight(4000.0, LOW, HIGH, ENTRY) == pytest.approx(0.0, abs=1e-12)
    assert figs.eth_weight(2000.0, LOW, HIGH, ENTRY) == pytest.approx(1.0, abs=1e-12)


def test_impermanent_loss_matches_known_values() -> None:
    assert figs.impermanent_loss(1.0) == pytest.approx(0.0, abs=1e-12)
    assert figs.impermanent_loss(1.2) == pytest.approx(-0.004141, abs=1e-6)
    # Symmetric around 1 in the ratio r <-> 1/r sense.
    assert figs.impermanent_loss(0.8) == pytest.approx(figs.impermanent_loss(1.25), rel=1e-9)


def test_lvr_rate_is_quadratic_in_volatility() -> None:
    assert figs.lvr_rate(0.5, 100_000.0) == pytest.approx(3125.0, rel=1e-12)
    assert figs.lvr_rate(1.0, 100_000.0) == pytest.approx(4 * figs.lvr_rate(0.5, 100_000.0))


def test_cli_writes_all_four_figures(tmp_path: Path) -> None:
    assert figs.main(["--output-dir", str(tmp_path)]) == 0
    for name in ("position_payoff", "il_lvr", "decision_path", "testbed"):
        path = tmp_path / f"{name}.pdf"
        assert path.is_file(), f"missing figure: {name}"
        assert path.stat().st_size > 1000
