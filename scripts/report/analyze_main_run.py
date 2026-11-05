"""Full per-window analysis of the joint band x hedge run (seed 3 + all seeds).

Exact backtester, joint (LP + hedge) equity. Writes per-window rows to
results/eth_usdc_500_joint_rl/analysis/windows.csv and prints summary tables.
"""

import csv
import json
import time
from collections import Counter
from pathlib import Path

from undertow.sim.backtest import run_backtest
from undertow.sim.cli import _CheckpointPolicy, _load_checkpoint
from undertow.sim.config import load_sim_config
from undertow.sim.evaluate.protocol import (
    _joint_accounting,
    dominant_regime,
    walk_forward_windows,
)
from undertow.sim.marketview import load_market_view
from undertow.sim.metrics import cvar_of_equity_changes, max_drawdown, sharpe_ratio
from undertow.sim.policies import (
    FullRangeV2Policy,
    HODLPolicy,
    HedgedHODLPolicy,
    PassiveNarrowPolicy,
    PassiveWidePolicy,
)

CONFIG_PATH = "configs/sim_eth_usdc_500_joint_rl.toml"
CHECKPOINT_DIR = Path("results/eth_usdc_500_joint_rl")
OUT_DIR = CHECKPOINT_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

t_start = time.time()
config = load_sim_config(CONFIG_PATH)
view = load_market_view(
    "configs/eth_usdc_500.toml",
    config.split,
    split="eval",
    episode_config=config.episode,
)
windows = walk_forward_windows(view, config)
print(f"loaded view; windows={len(windows)} t={time.time()-t_start:.0f}s", flush=True)

tick = int(config.episode.tick_spacing)
policies: list[tuple[str, object]] = [
    ("hodl", HODLPolicy()),
    ("passive_narrow", PassiveNarrowPolicy(tick_spacing=tick)),
    ("passive_wide", PassiveWidePolicy(tick_spacing=tick)),
    ("full_range_v2", FullRangeV2Policy(tick_spacing=tick)),
    ("hedged_hodl_1.0", HedgedHODLPolicy(1.0)),
]
for seed in range(5):
    model = _load_checkpoint(CHECKPOINT_DIR / f"seed_{seed}" / f"ppo_seed_{seed}.zip")
    policies.append((f"ppo_s{seed}", _CheckpointPolicy(model, config, name=f"ppo_s{seed}")))

rows: list[dict[str, object]] = []
for name, policy in policies:
    t_pol = time.time()
    for index, (start, end) in enumerate(windows):
        ledger = run_backtest(policy, view.slice(start, end), config)
        equity, dec, _curve = _joint_accounting(ledger)
        capital = float(ledger.initial_capital)
        decision = ledger.decision_log
        rebalance = decision.filter(decision["action_type"] == "rebalance") if decision.height else None
        band = int(rebalance["width"][0]) if rebalance is not None and rebalance.height else None
        ratios = Counter(float(v) for v in decision["hedge_ratio"].to_list()) if decision.height else Counter()
        row = {
            "policy": name,
            "window": index + 1,
            "start_seq": int(start),
            "end_seq": int(end),
            "regime": dominant_regime(view, start, end),
            "total_pnl": float(equity[-1]) - capital,
            "sharpe": sharpe_ratio(equity, periods_per_year=int(ledger.periods_per_year)),
            "max_drawdown": max_drawdown(equity),
            "cvar_95": cvar_of_equity_changes(equity),
            "n_rebalances": int(rebalance.height) if rebalance is not None else 0,
            "band": band if band is not None else -1,
            "steps": int(equity.len()),
            "ratio_0": ratios.get(0.0, 0),
            "ratio_025": ratios.get(0.25, 0),
            "ratio_05": ratios.get(0.5, 0),
            "ratio_075": ratios.get(0.75, 0),
            "ratio_10": ratios.get(1.0, 0),
            **{k: float(v) for k, v in dec.items()},
        }
        rows.append(row)
    print(f"policy {name:<15} done t={time.time()-t_start:.0f}s (+{time.time()-t_pol:.0f}s)", flush=True)

columns = list(rows[0].keys())
with (OUT_DIR / "windows.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)


def mean(values):
    return sum(values) / len(values)


def boot_ci(values, reps=10000, seed=7):
    import random

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(reps):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * reps)], means[int(0.975 * reps)]


metrics = ("total_pnl", "sharpe", "max_drawdown", "cvar_95", "n_rebalances")
print("\n=== aggregate over 13 windows (mean) ===", flush=True)
print("policy           total_pnl    sharpe    maxdd   cvar95  nreb  bands", flush=True)
for name, _ in policies:
    sub = [r for r in rows if r["policy"] == name]
    bands = Counter(r["band"] for r in sub)
    print(
        f"{name:<15} {mean([r['total_pnl'] for r in sub]):10.2f} "
        f"{mean([r['sharpe'] for r in sub]):9.4f} "
        f"{mean([r['max_drawdown'] for r in sub]):8.4f} "
        f"{mean([r['cvar_95'] for r in sub]):8.3f} "
        f"{mean([r['n_rebalances'] for r in sub]):5.2f}  {dict(bands)}",
        flush=True,
    )

print("\n=== paired per-window differences vs ppo_s3 (95% bootstrap CI over windows) ===", flush=True)
base = {r["window"]: r for r in rows if r["policy"] == "ppo_s3"}
for name, _ in policies:
    if name == "ppo_s3":
        continue
    sub = {r["window"]: r for r in rows if r["policy"] == name}
    for metric in ("sharpe", "cvar_95"):
        diffs = [base[w][metric] - sub[w][metric] for w in base]
        lo, hi = boot_ci(diffs)
        star = "*" if lo > 0 or hi < 0 else " "
        print(
            f"ppo_s3 - {name:<15} {metric:<10} mean={mean(diffs):+8.4f} "
            f"CI=[{lo:+8.4f},{hi:+8.4f}]{star}",
            flush=True,
        )

print("\n=== per-seed action mix (all windows) ===", flush=True)
for seed in range(5):
    name = f"ppo_s{seed}"
    sub = [r for r in rows if r["policy"] == name]
    ratio = Counter()
    for r in sub:
        ratio[0.0] += r["ratio_0"]
        ratio[0.25] += r["ratio_025"]
        ratio[0.5] += r["ratio_05"]
        ratio[0.75] += r["ratio_075"]
        ratio[1.0] += r["ratio_10"]
    total = sum(ratio.values())
    mix = {f"{k:.2f}": round(v / total, 3) for k, v in sorted(ratio.items())}
    print(f"{name}: bands={dict(Counter(r['band'] for r in sub))} ratio_mix={mix}", flush=True)

print("\n=== per-regime PnL/Sharpe/CVaR (mean over windows and policies) ===", flush=True)
regimes = sorted({r["regime"] for r in rows})
for name, _ in policies:
    sub = [r for r in rows if r["policy"] == name]
    parts = []
    for regime in regimes:
        rs = [r for r in sub if r["regime"] == regime]
        if not rs:
            continue
        parts.append(
            f"{regime}: n={len(rs)} pnl={mean([r['total_pnl'] for r in rs]):8.0f} "
            f"sharpe={mean([r['sharpe'] for r in rs]):6.2f} cvar={mean([r['cvar_95'] for r in rs]):7.1f}"
        )
    print(f"{name:<15} " + " | ".join(parts), flush=True)

print("\n=== decomposition means (excess-vs-HODL terms; USDC) ===", flush=True)
keys = (
    "total_fees",
    "total_il",
    "total_gas",
    "total_slippage",
    "total_hedge_pnl",
    "total_hedge_funding",
    "total_hedge_fee",
    "total_hedge",
    "net",
)
for name, _ in policies:
    sub = [r for r in rows if r["policy"] == name]
    values = " ".join(f"{k.replace('total_', '')}={mean([r[k] for r in sub]):+8.0f}" for k in keys)
    print(f"{name:<15} {values}", flush=True)

print(f"\nDONE t={time.time()-t_start:.0f}s -> {OUT_DIR/'windows.csv'}", flush=True)
