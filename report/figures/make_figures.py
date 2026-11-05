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
* ``learning_curves``   -- the training reward of every seed in the main run.
* ``risk_return``       -- Sharpe against CVaR for the evaluation ladder.
* ``action_map``        -- the band and hedge ratio each seed chose per window.
* ``fee_lvr``           -- fee income against the analytical LVR floor.
* ``fee_tier``          -- net excess on the primary and contrast fee tiers.

Usage::

    uv run python report/figures/make_figures.py            # writes the PDFs
    uv run python report/figures/make_figures.py --png      # also writes PNGs
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import overload

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
import qbstyles  # type: ignore[import-untyped]  # noqa: E402
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


@overload
def impermanent_loss(price_ratio: float) -> float: ...


@overload
def impermanent_loss(price_ratio: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...


def impermanent_loss(
    price_ratio: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
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


# ---------------------------------------------------------------------------
# Result figures: read the evaluation artefacts written by scripts/report/.
# ---------------------------------------------------------------------------

#: Repository root (this file lives at report/figures/make_figures.py).
ROOT = Path(__file__).resolve().parents[2]
#: Analysis artefacts produced by the reproduction programs.
RESULTS = ROOT / "results"
MAIN_ANALYSIS = RESULTS / "eth_usdc_500_joint_rl" / "analysis"
CONTRAST_ANALYSIS = RESULTS / "eth_usdc_3000_contrast" / "analysis"


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a result CSV into a list of string dictionaries."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_metric(rows: Sequence[dict[str, str]], metric: str) -> float:
    """Mean of one numeric column over the rows."""
    values = [float(row[metric]) for row in rows]
    if not values:
        raise ValueError(f"mean_metric: no rows for {metric!r}")
    return float(np.mean(values))


def policy_means(rows: Sequence[dict[str, str]], metric: str) -> dict[str, float]:
    """Mean of one metric per policy, preserving first-seen order."""
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["policy"], []).append(float(row[metric]))
    return {name: float(np.mean(values)) for name, values in grouped.items()}


def telemetry_curve(path: Path) -> list[tuple[int, float]]:
    """``(timestep, mean episode return)`` pairs from one seed's telemetry."""
    curve: list[tuple[int, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        reward = record.get("ep_rew_mean")
        if reward is not None:
            curve.append((int(record["timestep"]), float(reward)))
    return curve


def action_map_data(
    rows: Sequence[dict[str, str]], seeds: Sequence[str]
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Band matrix and normalised hedge-ratio mix for the action map.

    Returns the sorted window numbers, the ``(seeds, windows)`` band matrix
    with ``-1`` for a hold, and the ``(seeds, 5)`` share of decision steps at
    each hedge ratio. Raises ``ValueError`` when a seed has no hedge-ratio
    observations, since its mix would be undefined.
    """
    windows = sorted({int(row["window"]) for row in rows})
    ratio_keys = ("ratio_0", "ratio_025", "ratio_05", "ratio_075", "ratio_10")
    bands = np.full((len(seeds), len(windows)), -1.0)
    ratios = np.zeros((len(seeds), len(ratio_keys)))
    for row in rows:
        if row["policy"] not in seeds:
            continue
        index = seeds.index(row["policy"])
        bands[index, windows.index(int(row["window"]))] = float(row["band"])
        ratios[index] += np.array([float(row[key]) for key in ratio_keys])
    totals = ratios.sum(axis=1, keepdims=True)
    if np.any(totals == 0.0):
        raise ValueError("action_map_data: a seed has no hedge-ratio observations")
    return windows, bands, ratios / totals


def fee_lvr_points(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    """Passive-band rows of the gas x0.25 arm, in the file's window order."""
    return [
        row for row in rows if row["level"] == "gas_0.25x" and row["policy"] == "hodl"
    ]


def make_learning_curves(output_dir: Path, *, png: bool = False) -> Path:
    """Training reward of the five seeds of the main joint run."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.16, 2.7))
    colors = ["#4dd0e1", "#81c784", "#ffb74d", "#ba68c8", "#e57373"]
    grid = np.linspace(0.0, 1.0, 100)
    curves: list[np.ndarray] = []
    for seed in range(5):
        path = RESULTS / "eth_usdc_500_joint_rl" / f"seed_{seed}" / "telemetry.jsonl"
        curve = telemetry_curve(path)
        steps = np.array([step for step, _ in curve]) / 1e6
        rewards = np.array([reward for _, reward in curve])
        ax.plot(steps, rewards, color=colors[seed], linewidth=0.9, alpha=0.8,
                label=f"seed {seed}")
        curves.append(np.interp(grid, steps, rewards))
    ax.plot(grid, np.mean(curves, axis=0), color="white", linewidth=2.0,
            label="mean over seeds")
    ax.axhline(0.0, color="0.6", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Training steps, millions", fontsize=8)
    ax.set_ylabel("Mean episode\nreturn", fontsize=8)
    ax.set_title("Every seed learns the same thing: stop losing", fontsize=9)
    ax.legend(fontsize=7, loc="lower right", ncol=3)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return _save(fig, output_dir, "learning_curves", png)


def make_risk_return(output_dir: Path, *, png: bool = False) -> Path:
    """Sharpe against CVaR for the ladder, the learned seeds and the hedge."""
    apply_style()
    rows = read_rows(MAIN_ANALYSIS / "windows.csv") + read_rows(
        MAIN_ANALYSIS / "extra_benchmarks.csv"
    )
    sharpes = policy_means(rows, "sharpe")
    cvars = policy_means(rows, "cvar_95")
    static = [
        row
        for row in read_rows(MAIN_ANALYSIS / "static_widths.csv")
        if row["split"] == "eval" and row["policy"] == "static_w50"
    ]
    static_sharpe = mean_metric(static, "sharpe")
    static_cvar = mean_metric(static, "cvar_95")
    fig, ax = plt.subplots(figsize=(7.16, 3.4))

    def point(
        name: str,
        label: str,
        color: str,
        marker: str,
        *,
        size: int = 36,
        offset: tuple[float, float] = (6, 4),
        align: str = "left",
    ) -> None:
        ax.scatter(sharpes[name], cvars[name], color=color, marker=marker, s=size, zorder=5)
        ax.annotate(label, (sharpes[name], cvars[name]), textcoords="offset points",
                    xytext=offset, fontsize=7, color=color, ha=align)

    for name, label, offset in (("hodl", "hodl", (6, 4)), ("passive_narrow", "narrow", (6, 4)),
                                ("passive_wide", "wide", (6, 4)),
                                ("full_range_v2", "full range", (-6, 4)),
                                ("hedged_hodl_1.0", "hedged HODL", (6, 4))):
        point(name, label, "#ffb74d", "s", offset=offset,
              align="right" if offset[0] < 0 else "left")
    point("tau_reset", "recentring", "#e57373", "v")
    point("cost_aware", "cost-aware", "#e57373", "v", offset=(-6, -12), align="right")
    for seed, offset in ((0, (6, 4)), (1, (6, 4)), (2, (6, -12)), (3, (6, 4)),
                         (4, (-6, -12))):
        point(f"ppo_s{seed}", f"seed {seed}", "#4dd0e1", "o", offset=offset,
              align="right" if offset[0] < 0 else "left")
    ax.scatter(static_sharpe, static_cvar, color="#81c784", marker="*", s=110, zorder=6)
    ax.annotate(
        "frozen width",
        (static_sharpe, static_cvar),
        textcoords="offset points",
        xytext=(6, -10),
        fontsize=7,
        color="#81c784",
    )
    ax.axvline(0.0, color="0.6", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Annualised Sharpe ratio", fontsize=8)
    ax.set_ylabel(r"CVaR at $5\,\%$, USDC", fontsize=8)
    ax.set_title("The learned seeds trade return for a smaller tail", fontsize=9)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return _save(fig, output_dir, "risk_return", png)


def text_color_for_share(share: float, share_max: float) -> str:
    """Dark or light annotation text for a viridis cell at ``share``.

    ``share_max`` must be positive; the action map normalises every seed's
    row, so it always is. Returns a dark grey on bright cells and white on
    dark ones, keeping the percentage readable on either end of the map.
    """
    red, green, blue, _ = matplotlib.colormaps["viridis"](share / share_max)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "0.1" if luminance > 0.5 else "white"


def make_action_map(output_dir: Path, *, png: bool = False) -> Path:
    """Band choice per seed and window, and each seed's hedge-ratio mix.

    Both panels are labelled cell grids: the left one is categorical (the band
    each seed chose per window) and the right one is a sequential share matrix
    whose cells carry the percentage, so neither relies on a cramped legend.
    """
    apply_style()
    seeds = [f"ppo_s{seed}" for seed in range(5)]
    windows, bands, ratios = action_map_data(read_rows(MAIN_ANALYSIS / "windows.csv"), seeds)

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.16, 3.0), gridspec_kw={"width_ratios": [1.5, 1.05]}
    )
    cmap = matplotlib.colors.ListedColormap(["#555555", "#4dd0e1", "#81c784", "#ffb74d"])
    norm = matplotlib.colors.BoundaryNorm([-1.5, -0.5, 50.5, 250.5, 1000.5], cmap.N)
    left.imshow(bands, cmap=cmap, norm=norm, aspect="auto")
    left.set_xticks(range(len(windows)), [str(window) for window in windows], fontsize=7)
    left.set_yticks(range(len(seeds)), [f"seed {seed}" for seed in range(5)], fontsize=8)
    left.set_xlabel("Evaluation window", fontsize=8)
    left.set_title("Band chosen, tick spacings", fontsize=9)
    for row in range(len(seeds)):
        for column in range(len(windows)):
            label = "hold" if bands[row, column] < 0 else f"{int(bands[row, column])}"
            left.text(column, row, label, ha="center", va="center", fontsize=7,
                      color="white")

    ratio_labels = ("0", "0.25", "0.5", "0.75", "1.0")
    # Each seed's row is normalised to sum to one, so the maximum is positive.
    share_max = float(ratios.max())
    right.imshow(ratios.T, cmap="viridis", vmin=0.0, vmax=share_max, aspect="auto")
    right.set_xticks(range(len(seeds)), [str(seed) for seed in range(5)], fontsize=7)
    right.set_yticks(range(len(ratio_labels)), ratio_labels, fontsize=7)
    right.set_xlabel("Seed", fontsize=8)
    right.set_ylabel("Hedge ratio", fontsize=8)
    right.set_title("Share of steps at each ratio, %", fontsize=9)
    for row in range(len(ratio_labels)):
        for column in range(len(seeds)):
            share = float(ratios[column, row])
            right.text(column, row, f"{100 * share:.0f}", ha="center", va="center",
                       fontsize=7, color=text_color_for_share(share, share_max))
    fig.suptitle("Four seeds deploy the same band; every seed hedges partially", fontsize=9)
    fig.tight_layout()
    return _save(fig, output_dir, "action_map", png)


def make_fee_lvr(output_dir: Path, *, png: bool = False) -> Path:
    """Fee income of the opening band against the analytical LVR floor."""
    apply_style()
    rows = fee_lvr_points(read_rows(MAIN_ANALYSIS / "boundary_sweep.csv"))
    if not rows:
        raise ValueError("make_fee_lvr: no gas x0.25 passive rows to plot")
    fees = np.array([float(row["total_fees"]) for row in rows])
    lvr = np.array([float(row["lvr_usdc"]) for row in rows])
    sigma = np.array([float(row["sigma"]) for row in rows])
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    scatter = ax.scatter(lvr, fees, c=sigma, cmap="viridis", s=38, zorder=4)
    limit = max(fees.max(), lvr.max()) * 1.08
    ax.plot([0.0, limit], [0.0, limit], color="0.7", linewidth=0.8, linestyle="--")
    ax.set_xlim(0.0, limit)
    ax.set_ylim(0.0, limit)
    ax.set_xlabel("Analytical LVR floor, USDC", fontsize=8)
    ax.set_ylabel("Fee income, USDC", fontsize=8)
    ax.set_title("Fees clear the full-range floor", fontsize=9)
    ax.tick_params(labelsize=7)
    bar = fig.colorbar(scatter, ax=ax)
    bar.set_label("Realised $\\sigma$", fontsize=7)
    bar.ax.tick_params(labelsize=6)
    ax.annotate("partial window\n(floor rescales)", xy=(lvr[-1], fees[-1]),
                xytext=(lvr[-1] + 160, fees[-1] - 260), fontsize=6, color="0.85",
                arrowprops={"arrowstyle": "->", "color": "0.7", "lw": 0.7})
    fig.tight_layout()
    return _save(fig, output_dir, "fee_lvr", png)


def make_fee_tier(output_dir: Path, *, png: bool = False) -> Path:
    """Net excess over the opening basket on both fee tiers."""
    apply_style()
    primary_rows = read_rows(MAIN_ANALYSIS / "windows.csv") + read_rows(
        MAIN_ANALYSIS / "extra_benchmarks.csv"
    )
    contrast_rows = read_rows(CONTRAST_ANALYSIS / "windows.csv")
    policies = ("hodl", "passive_narrow", "passive_wide", "full_range_v2",
                "cost_aware", "tau_reset")
    labels = ("hodl", "narrow", "wide", "full", "cost-aware", "recentring")
    primary = policy_means(primary_rows, "net")
    contrast = policy_means(contrast_rows, "net")
    x = np.arange(len(policies))
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.16, 2.9), sharey=True)
    for ax, means, title, color in (
        (left, primary, "Primary pool, $0.05\\,\\%$", "#4dd0e1"),
        (right, contrast, "Contrast pool, $0.30\\,\\%$", "#81c784"),
    ):
        ax.bar(x, [means[policy] for policy in policies], color=color, width=0.62)
        ax.axhline(0.0, color="0.7", linewidth=0.8)
        ax.set_xticks(x, labels, fontsize=6.5, rotation=35, ha="right")
        ax.set_title(title, fontsize=9)
        ax.set_yscale("symlog", linthresh=1000.0)
        ax.tick_params(labelsize=7)
    left.set_ylabel("Net excess over the\nopening basket, USDC", fontsize=8)
    fig.suptitle("The higher fee tier turns the passive bands positive; recentring loses on both",
                 fontsize=9)
    fig.tight_layout()
    return _save(fig, output_dir, "fee_tier", png)


def make_all(output_dir: Path, *, png: bool = False) -> list[Path]:
    """Build every story figure and return the written paths."""
    return [
        make_position_payoff(output_dir, png=png),
        make_il_lvr(output_dir, png=png),
        make_decision_path(output_dir, png=png),
        make_testbed(output_dir, png=png),
        make_learning_curves(output_dir, png=png),
        make_risk_return(output_dir, png=png),
        make_action_map(output_dir, png=png),
        make_fee_lvr(output_dir, png=png),
        make_fee_tier(output_dir, png=png),
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
