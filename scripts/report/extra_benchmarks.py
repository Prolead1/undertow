"""tau_reset and cost_aware under the joint eval config (exact backtester).

The main joint analysis covered hodl, passive_narrow, passive_wide,
full_range_v2 and hedged_hodl; these two scripted recentring baselines complete
the pre-registered ladder on the same 13 eval windows.
"""

import csv
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

OUT = Path("results/eth_usdc_500_joint_rl/analysis/extra_benchmarks.csv")
config = load_sim_config("configs/sim_eth_usdc_500_joint_rl.toml")
view = load_market_view(
    "configs/eth_usdc_500.toml",
    config.split,
    split="eval",
    episode_config=config.episode,
)
windows = walk_forward_windows(view, config)

rows: list[dict[str, object]] = []
for name in ("tau_reset", "cost_aware"):
    policy = build_policy(name, config)
    for index, (start, end) in enumerate(windows):
        ledger = run_backtest(policy, view.slice(start, end), config)
        equity, dec, _curve = _joint_accounting(ledger)
        capital = float(ledger.initial_capital)
        decision = ledger.decision_log
        rebalance = (
            decision.filter(decision["action_type"] == "rebalance") if decision.height else None
        )
        row: dict[str, object] = {
            "policy": name,
            "window": index + 1,
            "regime": dominant_regime(view, start, end),
            "total_pnl": float(equity[-1]) - capital,
            "sharpe": sharpe_ratio(equity, periods_per_year=int(ledger.periods_per_year)),
            "max_drawdown": max_drawdown(equity),
            "cvar_95": cvar_of_equity_changes(equity),
            "n_rebalances": int(rebalance.height) if rebalance is not None else 0,
            **{k: float(v) for k, v in dec.items()},
        }
        rows.append(row)
        print(
            f"{name:<11} w{index + 1:2d} pnl={row['total_pnl']:9.1f} "
            f"sharpe={row['sharpe']:7.3f} cvar={row['cvar_95']:8.2f} nreb={row['n_rebalances']}",
            flush=True,
        )

with OUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


for name in ("tau_reset", "cost_aware"):
    sub = [r for r in rows if r["policy"] == name]
    print(
        f"\n{name}: pnl={mean([float(r['total_pnl']) for r in sub]):.2f} "
        f"sharpe={mean([float(r['sharpe']) for r in sub]):.3f} "
        f"mdd={mean([float(r['max_drawdown']) for r in sub]):.4f} "
        f"cvar={mean([float(r['cvar_95']) for r in sub]):.2f} "
        f"nreb={mean([float(r['n_rebalances']) for r in sub]):.2f}",
        flush=True,
    )
print(f"-> {OUT}")
