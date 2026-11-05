"""Static one-shot band-width arms from the learned grid, on train and eval.

Three static policies deploy once at the first decision with half-width
{50, 250, 1000} tick spacings and hold forever; exactly like a learned
band-selection policy that never changes its mind. Evaluated with the exact
backtester on both splits under the joint config; a ratio-less action resolves
to the unhedged entry under the agent-ratio head, so these arms are unhedged
passive LP benchmarks.
"""

import csv
import time
from collections.abc import Sequence
from pathlib import Path

from undertow.sim.backtest import run_backtest
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

OUT = Path("results/eth_usdc_500_joint_rl/analysis/static_widths.csv")


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


config = load_sim_config("configs/sim_eth_usdc_500_joint_rl.toml")
rows = []
t0 = time.time()
for split in ("train", "eval"):
    view = load_market_view(
        "configs/eth_usdc_500.toml",
        config.split,
        split=split,
        episode_config=config.episode,
    )
    windows = walk_forward_windows(view, config)
    print(f"{split}: windows={len(windows)} t={time.time()-t0:.0f}s", flush=True)
    for width in (50, 250, 1000):
        policy = StaticWidth(width)
        for index, (start, end) in enumerate(windows):
            ledger = run_backtest(policy, view.slice(start, end), config)
            equity, dec, _ = _joint_accounting(ledger)
            capital = float(ledger.initial_capital)
            rows.append(
                {
                    "split": split,
                    "policy": policy.name,
                    "width": width,
                    "window": index + 1,
                    "regime": dominant_regime(view, start, end),
                    "total_pnl": float(equity[-1]) - capital,
                    "sharpe": sharpe_ratio(equity, periods_per_year=int(ledger.periods_per_year)),
                    "max_drawdown": max_drawdown(equity),
                    "cvar_95": cvar_of_equity_changes(equity),
                    "total_fees": float(dec.get("total_fees", 0.0)),
                    "total_il": float(dec.get("total_il", 0.0)),
                    "total_gas": float(dec.get("total_gas", 0.0)),
                    "total_slippage": float(dec.get("total_slippage", 0.0)),
                    "net": float(dec.get("net", 0.0)),
                }
            )
        print(f"  {policy.name} done t={time.time()-t0:.0f}s", flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


print("\n=== static width arms (mean over windows) ===", flush=True)
for split in ("train", "eval"):
    for width in (50, 250, 1000):
        sub = [r for r in rows if r["split"] == split and r["width"] == width]
        print(
            f"{split:<5} w{width:<5} n={len(sub):2d} "
            f"pnl={mean([r['total_pnl'] for r in sub]):9.0f} "
            f"sharpe={mean([r['sharpe'] for r in sub]):7.3f} "
            f"mdd={mean([r['max_drawdown'] for r in sub]):7.4f} "
            f"cvar={mean([r['cvar_95'] for r in sub]):8.2f} "
            f"fees={mean([r['total_fees'] for r in sub]):8.0f} "
            f"il={mean([r['total_il'] for r in sub]):8.0f} "
            f"net={mean([r['net'] for r in sub]):8.0f}",
            flush=True,
        )
print(f"\nDONE t={time.time()-t0:.0f}s -> {OUT}", flush=True)
