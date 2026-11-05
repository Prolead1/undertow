"""Fee-tier contrast: the full scripted ladder plus the three static widths on
the 0.30 % (3000 pips) ETH/USDC pool, exact backtester, 2024 eval windows.

The static one-shot widths are the price-equivalent of the primary pool's
{50, 250, 1000} spacing-10 arms at the contrast pool's spacing of 60:
ticks are ~500, ~2500 and ~10000, so widths are 8, 42 and 167 spacings.
"""

import csv
import time
from collections import Counter
from pathlib import Path

from undertow.sim.backtest import run_backtest
from undertow.sim.cli import build_policy
from undertow.sim.config import load_sim_config
from undertow.sim.evaluate.protocol import (
    _joint_accounting,
    dominant_regime,
    walk_forward_windows,
)
from undertow.sim.marketview import load_market_view
from undertow.sim.metrics import cvar_of_equity_changes, max_drawdown, sharpe_ratio
from undertow.sim.policies.base import HOLDAction
from undertow.sim.types import Action

SIM = "configs/sim_eth_usdc_3000.toml"
DATA = "configs/eth_usdc_3000.toml"
OUT = Path("results/eth_usdc_3000_contrast/analysis/windows.csv")


class StaticWidth:
    def __init__(self, width: int) -> None:
        self.width = int(width)
        self._name = f"static_w{self.width}"
        self._deployed = False

    @property
    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        self._deployed = False

    def act(self, observation, rng=None):  # noqa: ANN001, ANN201
        if self._deployed:
            return HOLDAction
        self._deployed = True
        return Action(action_type="rebalance", center_offset=0, width=self.width)


t0 = time.time()
config = load_sim_config(SIM)
view = load_market_view(DATA, config.split, split="eval", episode_config=config.episode)
windows = walk_forward_windows(view, config)
print(f"loaded view; windows={len(windows)} t={time.time() - t0:.0f}s", flush=True)

policies: list[tuple[str, object]] = [
    (name, build_policy(name, config))
    for name in (
        "hodl",
        "passive_narrow",
        "passive_wide",
        "full_range_v2",
        "tau_reset",
        "cost_aware",
    )
]
policies += [(f"static_w{w}", StaticWidth(w)) for w in (8, 42, 167)]

rows: list[dict[str, object]] = []
for name, policy in policies:
    t_pol = time.time()
    for index, (start, end) in enumerate(windows):
        ledger = run_backtest(policy, view.slice(start, end), config)
        equity, dec, _curve = _joint_accounting(ledger)
        capital = float(ledger.initial_capital)
        decision = ledger.decision_log
        rebalance = (
            decision.filter(decision["action_type"] == "rebalance")
            if decision.height
            else None
        )
        band = (
            int(rebalance["width"][0])
            if rebalance is not None and rebalance.height
            else -1
        )
        rows.append(
            {
                "policy": name,
                "window": index + 1,
                "regime": dominant_regime(view, start, end),
                "total_pnl": float(equity[-1]) - capital,
                "sharpe": sharpe_ratio(equity, periods_per_year=int(ledger.periods_per_year)),
                "max_drawdown": max_drawdown(equity),
                "cvar_95": cvar_of_equity_changes(equity),
                "n_rebalances": int(rebalance.height) if rebalance is not None else 0,
                "band": band,
                **{k: float(v) for k, v in dec.items()},
            }
        )
        print(
            f"{name:<14} w{index + 1:2d} pnl={rows[-1]['total_pnl']:9.1f} "
            f"sharpe={rows[-1]['sharpe']:7.3f} cvar={rows[-1]['cvar_95']:8.2f}",
            flush=True,
        )
    print(
        f"policy {name:<14} done t={time.time() - t0:.0f}s "
        f"(+{time.time() - t_pol:.0f}s)",
        flush=True,
    )

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


print("\n=== 0.30 % contrast, mean over 13 eval windows ===", flush=True)
print(
    "policy          pnl     sharpe   maxdd   cvar95  nreb  band  fees      il       net",
    flush=True,
)
for name, _ in policies:
    sub = [r for r in rows if r["policy"] == name]
    bands = Counter(r["band"] for r in sub)
    print(
        f"{name:<14} {mean([float(r['total_pnl']) for r in sub]):8.1f} "
        f"{mean([float(r['sharpe']) for r in sub]):7.3f} "
        f"{mean([float(r['max_drawdown']) for r in sub]):7.4f} "
        f"{mean([float(r['cvar_95']) for r in sub]):8.2f} "
        f"{mean([float(r['n_rebalances']) for r in sub]):5.2f} {dict(bands)} "
        f"{mean([float(r['total_fees']) for r in sub]):8.0f} "
        f"{mean([float(r['total_il']) for r in sub]):8.0f} "
        f"{mean([float(r['net']) for r in sub]):8.0f}",
        flush=True,
    )
print(f"\nDONE t={time.time() - t0:.0f}s -> {OUT}", flush=True)
