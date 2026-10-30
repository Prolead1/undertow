"""Generate the report's story figures with Python and the QB dark theme.

Each function here builds one figure and writes it next to this file as a
vector PDF. The figures illustrate the mechanics the report describes:

* ``position_payoff``   -- what a concentrated position is worth versus a
  passive holding, and how its asset mix rotates as the price moves.
* ``il_lvr``            -- the two losses: impermanent loss against a price
  move, and the LVR rate against volatility and position size.
* ``decision_path``     -- an illustrative price path with band exits and the
  lumpy cost of each recentring.
* ``testbed``           -- the two engines and the parity check that keeps
  them honest.

Usage::

    uv run python report/figures/make_figures.py            # writes the PDFs
    uv run python report/figures/make_figures.py --png      # also writes PNGs
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import qbstyles  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

#: Decimal factor for the USDC/WETH pair (10^(18-6)).
DECIMALS = 1e12
#: Illustrative starting ETH price, USDC per WETH.
ENTRY_PRICE = 3000.0
#: Illustrative capital, USDC.
CAPITAL = 100_000.0
#: Illustrative rebalancing costs (see the report caption).
GAS_USD = 18.0
SLIPPAGE_BPS = 5.0

_STYLE_APPLIED = False


def apply_style() -> None:
    """Apply the QuantumBlack dark theme once, before any figure is made.

    The theme's style files are applied directly: ``qbstyles.mpl_style``
    monkey-patches figure creation for minor ticks in a way that corrupts
    axes on current matplotlib, so the grid is enabled here instead.
    """
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    style_dir = Path(qbstyles.__file__).resolve().parent / "styles"
    plt.style.use(
        [str(style_dir / "qb-common.mplstyle"), str(style_dir / "qb-dark.mplstyle")]
    )
    plt.rcParams.update(
        {
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.30,
            "grid.linewidth": 0.5,
            "legend.framealpha": 0.25,
        }
    )
    _STYLE_APPLIED = True


def sqrt_price(price: float) -> float:
    """Raw Uniswap sqrt price for a human USDC-per-WETH price."""
    return float(np.sqrt(DECIMALS / price))


def liquidity_for_capital(
    q_entry: float, q_low: float, q_high: float, capital: float = CAPITAL
) -> float:
    """Raw liquidity that makes the position worth ``capital`` at entry."""
    s0 = sqrt_price(q_entry)
    s_a = sqrt_price(q_high)
    s_b = sqrt_price(q_low)
    value_per_liquidity = (1.0 / s0 - 1.0 / s_b) / 1e6 + q_entry * (s0 - s_a) / 1e18
    return capital / value_per_liquidity


def position_value(
    q: float, q_low: float, q_high: float, q_entry: float, capital: float = CAPITAL
) -> float:
    """USDC value of a concentrated position at human price ``q``.

    Above the band the position holds only USDC; below it, only WETH; in
    between the value follows the v3 formula in the report.
    """
    liquidity = liquidity_for_capital(q_entry, q_low, q_high, capital)
    s = sqrt_price(q)
    s_a = sqrt_price(q_high)
    s_b = sqrt_price(q_low)
    if q >= q_high:
        return liquidity * (1.0 / s_a - 1.0 / s_b) / 1e6
    if q <= q_low:
        return liquidity * (s_b - s_a) * q / 1e18
    return liquidity * ((1.0 / s - 1.0 / s_b) / 1e6 + q * (s - s_a) / 1e18)


def entry_amounts(
    q_entry: float, q_low: float, q_high: float, capital: float = CAPITAL
) -> tuple[float, float]:
    """Human token amounts of the opening deposit (USDC, WETH)."""
    liquidity = liquidity_for_capital(q_entry, q_low, q_high, capital)
    s0 = sqrt_price(q_entry)
    s_a = sqrt_price(q_high)
    s_b = sqrt_price(q_low)
    return (
        liquidity * (1.0 / s0 - 1.0 / s_b) / 1e6,
        liquidity * (s0 - s_a) / 1e18,
    )


def hodl_value(
    q: float, q_entry: float, q_low: float, q_high: float, capital: float = CAPITAL
) -> float:
    """USDC value of holding the opening basket at price ``q``."""
    amount0, amount1 = entry_amounts(q_entry, q_low, q_high, capital)
    return amount0 + amount1 * q


def eth_weight(
    q: float, q_low: float, q_high: float, q_entry: float, capital: float = CAPITAL
) -> float:
    """Share of the position's value held in WETH, in [0, 1]."""
    value = position_value(q, q_low, q_high, q_entry, capital)
    if value <= 0.0 or q >= q_high:
        return 0.0
    if q <= q_low:
        return 1.0
    liquidity = liquidity_for_capital(q_entry, q_low, q_high, capital)
    s = sqrt_price(q)
    s_a = sqrt_price(q_high)
    weth = liquidity * (s - s_a) / 1e18
    return float(np.clip(weth * q / value, 0.0, 1.0))


def impermanent_loss(price_ratio: float) -> float:
    """IL of a full-range constant-product position at ``r = q_t / q_0``."""
    return 2.0 * np.sqrt(price_ratio) / (1.0 + price_ratio) - 1.0


def lvr_rate(sigma: float, value: float = CAPITAL) -> float:
    """Annual LVR of a position worth ``value`` at annualised volatility ``sigma``."""
    return 0.125 * sigma**2 * value


def _save(fig: plt.Figure, output_dir: Path, name: str, png: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{name}.pdf"
    fig.savefig(pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
    if png:
        fig.savefig(output_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf


def make_position_payoff(output_dir: Path, *, png: bool = False) -> Path:
    """Value versus a passive holding, for a narrow and a wide band."""
    apply_style()
    prices = np.linspace(0.75 * ENTRY_PRICE, 1.25 * ENTRY_PRICE, 400)
    narrow = (ENTRY_PRICE / 1.05, ENTRY_PRICE * 1.05)
    wide = (ENTRY_PRICE / 1.20, ENTRY_PRICE * 1.20)

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.16, 2.7))
    left.plot(prices, [hodl_value(q, ENTRY_PRICE, *narrow) for q in prices],
              color="#ffb74d", linewidth=1.6, label="Hold the opening basket")
    left.plot(prices, [position_value(q, *narrow, ENTRY_PRICE) for q in prices],
              color="#4dd0e1", linewidth=1.8, label="Band $\\pm5\\%$")
    left.plot(prices, [position_value(q, *wide, ENTRY_PRICE) for q in prices],
              color="#81c784", linewidth=1.8, label="Band $\\pm20\\%$")
    left.axvline(ENTRY_PRICE, color="0.7", linewidth=0.7, linestyle=":")
    left.set_xlabel("ETH price, USDC per WETH")
    left.set_ylabel("Position value, USDC")
    left.set_title("What the position is worth", fontsize=9)
    left.legend(fontsize=7, loc="upper left")
    left.tick_params(labelsize=7)

    right.plot(prices, [eth_weight(q, *narrow, ENTRY_PRICE) for q in prices],
               color="#4dd0e1", linewidth=1.8, label="Band $\\pm5\\%$")
    right.plot(prices, [eth_weight(q, *wide, ENTRY_PRICE) for q in prices],
               color="#81c784", linewidth=1.8, label="Band $\\pm20\\%$")
    right.axhline(0.5, color="0.7", linewidth=0.7, linestyle=":")
    right.set_xlabel("ETH price, USDC per WETH")
    right.set_ylabel("Share held in WETH")
    right.set_title("How the asset mix rotates", fontsize=9)
    right.legend(fontsize=7, loc="lower right")
    right.tick_params(labelsize=7)

    fig.suptitle("A concentrated position stops following the price at the band edge", fontsize=9)
    fig.tight_layout()
    return _save(fig, output_dir, "position_payoff", png)


def make_il_lvr(output_dir: Path, *, png: bool = False) -> Path:
    """Impermanent loss versus a price move, and LVR versus volatility."""
    apply_style()
    ratios = np.linspace(0.5, 1.5, 400)
    sigmas = np.linspace(0.2, 1.5, 300)

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.5, 4.4))
    top.plot(ratios, 100.0 * impermanent_loss(ratios), color="#e57373", linewidth=1.8)
    top.axhline(0.0, color="0.6", linewidth=0.7)
    top.axvline(1.2, color="0.7", linewidth=0.7, linestyle=":")
    top.annotate(
        "$-0.41$% at $+20$%",
        xy=(1.2, 100 * float(impermanent_loss(1.2))),
        xytext=(1.24, -3.4),
        fontsize=7,
        arrowprops={"arrowstyle": "->", "color": "0.7", "lw": 0.8},
    )
    top.set_xlabel("Price ratio $r = q_t / q_0$", fontsize=8)
    top.set_ylabel("IL, % of deposit", fontsize=8)
    top.set_title("Impermanent loss is second order", fontsize=9)
    top.tick_params(labelsize=7)

    bottom.plot(sigmas, [lvr_rate(s) / 1000.0 for s in sigmas], color="#ba68c8", linewidth=1.8,
                label=r"$\$100k$ position")
    bottom.plot(sigmas, [lvr_rate(s, 2 * CAPITAL) / 1000.0 for s in sigmas],
                color="#4dd0e1", linewidth=1.3, linestyle="--", label=r"$\$200k$ position")
    bottom.annotate(r"$\$3.1k/yr$ at $50$%", xy=(0.5, lvr_rate(0.5) / 1000.0),
                    xytext=(0.56, 1.6), fontsize=7,
                    arrowprops={"arrowstyle": "->", "color": "0.7", "lw": 0.8})
    bottom.set_xlabel("Annualised volatility $\\sigma$", fontsize=8)
    bottom.set_ylabel("LVR, thousand USDC per year", fontsize=8)
    bottom.set_title("LVR grows with volatility squared", fontsize=9)
    bottom.legend(fontsize=7, loc="upper left")
    bottom.tick_params(labelsize=7)

    fig.tight_layout()
    return _save(fig, output_dir, "il_lvr", png)


def _illustrative_path(days: int = 90, seed: int = 7) -> np.ndarray:
    """A fixed-seed price path used only to illustrate the decisions."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 365.0
    steps = rng.normal(0.0, 0.65, size=days) * np.sqrt(dt)
    return ENTRY_PRICE * np.exp(np.cumsum(steps))


def make_decision_path(output_dir: Path, *, png: bool = False) -> Path:
    """An illustrative path with band exits and the cost of each recentring."""
    apply_style()
    prices = _illustrative_path()
    days = np.arange(len(prices))
    half_width = 0.05
    low_factor = 1.0 / (1.0 + half_width)
    high_factor = 1.0 + half_width
    gas = GAS_USD
    slippage = SLIPPAGE_BPS / 1e4 * CAPITAL

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.16, 2.9), gridspec_kw={"height_ratios": [2.4, 1.0]}
    )
    top.plot(days, prices, color="white", linewidth=1.2, label="ETH price")

    costs: list[float] = []
    running = 0.0
    band_low, band_high = ENTRY_PRICE * low_factor, ENTRY_PRICE * high_factor
    exits: list[int] = []
    for day, price in enumerate(prices):
        if day > 0:
            top.fill_between([day - 1, day], band_low, band_high, color="#4dd0e1", alpha=0.10)
        if price < band_low or price > band_high:
            exits.append(day)
            band_low, band_high = price * low_factor, price * high_factor
            running += gas + slippage
        costs.append(running)
    for day in exits:
        top.axvline(day, color="#e57373", linewidth=0.7, linestyle="--", alpha=0.8)
        top.scatter([day], [prices[day]], color="#e57373", s=18, zorder=5)
    top.set_ylabel("ETH price, USDC", fontsize=8)
    top.set_title("Each exit forces a recentring; each recentring costs (illustrative path)",
                  fontsize=9)
    top.legend(fontsize=7, loc="upper left")
    top.tick_params(labelsize=7)
    top.set_xlim(0, len(prices) - 1)

    bottom.step(days, costs, where="post", color="#ffd54f", linewidth=1.6)
    bottom.set_xlabel("Days", fontsize=8)
    bottom.set_ylabel("Cumulative\ncosts, USDC", fontsize=8)
    bottom.tick_params(labelsize=7)
    bottom.set_xlim(0, len(prices) - 1)
    bottom.text(0.02, 0.66,
                f"{len(exits)} recentres, ${costs[-1]:.0f} paid in gas and slippage",
                transform=bottom.transAxes, fontsize=7, color="#ffd54f")

    fig.tight_layout()
    return _save(fig, output_dir, "decision_path", png)


def _box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str, color: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.0, edgecolor=color, facecolor=color, alpha=0.22,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7, color="white",
            linespacing=1.35)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, linewidth=1.0,
                        color="0.75", shrinkA=1, shrinkB=1)
    )


def make_testbed(output_dir: Path, *, png: bool = False) -> Path:
    """A single-column flow diagram of the testbed and the parity check."""
    apply_style()
    fig, ax = plt.subplots(figsize=(3.5, 4.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1.0)
    ax.axis("off")

    _box(ax, 0.10, 0.87, 0.80, 0.10, "On-chain events\nswaps, mints, burns, collects", "#4dd0e1")
    _box(ax, 0.10, 0.71, 0.80, 0.10, "Event tape and feeds\nprices, gas, regimes, funding",
         "#4dd0e1")
    _box(ax, 0.03, 0.51, 0.44, 0.12, "Exact backtester\ninteger fee accounting", "#81c784")
    _box(ax, 0.53, 0.51, 0.44, 0.12, "Fast simulator\nfloat64, millions of steps", "#81c784")
    _box(ax, 0.20, 0.32, 0.60, 0.10, "Parity check\nsame inputs, same accounting", "#ffb74d")
    _box(ax, 0.20, 0.15, 0.60, 0.10, "PPO policy\ntiming, width, hedge", "#ba68c8")
    _box(ax, 0.10, 0.00, 0.80, 0.09, "Evaluation\nSharpe, CVaR, cost ledger", "#e57373")

    _arrow(ax, (0.50, 0.87), (0.50, 0.81))
    _arrow(ax, (0.35, 0.71), (0.25, 0.63))
    _arrow(ax, (0.65, 0.71), (0.75, 0.63))
    _arrow(ax, (0.25, 0.51), (0.40, 0.42))
    _arrow(ax, (0.75, 0.51), (0.60, 0.42))
    _arrow(ax, (0.50, 0.32), (0.50, 0.25))
    _arrow(ax, (0.50, 0.15), (0.50, 0.09))

    fig.tight_layout()
    return _save(fig, output_dir, "testbed", png)


def make_all(output_dir: Path, *, png: bool = False) -> list[Path]:
    """Build every story figure and return the written paths."""
    return [
        make_position_payoff(output_dir, png=png),
        make_il_lvr(output_dir, png=png),
        make_decision_path(output_dir, png=png),
        make_testbed(output_dir, png=png),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the report's story figures.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--png", action="store_true", help="also write PNG copies for inspection")
    args = parser.parse_args(argv)
    for path in make_all(args.output_dir, png=args.png):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
