"""C4 boundary sweep on the primary 0.05 % pool (evaluation-only).

Axes
----
* realised volatility sigma and swap notional relative to active liquidity,
  measured per evaluation window from the recorded tape and reference feed;
* gas price, proxied by the config's per-action gas units scaled x0.25 and x4
  (the x1 level is the main run already in windows.csv; the friction ablation
  covers gas x0);
* capital, x0.1 and x10 (x1 is the main run);
* fee tier, from the 0.30 % contrast run (results/eth_usdc_3000_contrast).

Analytical overlay: LVR = 0.125 * sigma^2 * V * T (Milionis et al.), with
V = 100k USDC and T the 30-day window fraction; compared with the default
band's fee income per window.

Writes results/eth_usdc_500_joint_rl/analysis/boundary_sweep.csv.
"""

import csv
import dataclasses
import statistics
import time
from pathlib import Path

import numpy as np
import polars as pl

from undertow.sim.backtest import run_backtest
from undertow.sim.cli import _CheckpointPolicy, _load_checkpoint, build_policy
from undertow.sim.config import load_sim_config
from undertow.sim.evaluate.protocol import (
    _joint_accounting,
    dominant_regime,
    walk_forward_windows,
)
from undertow.sim.marketview import load_market_view
from undertow.sim.metrics import cvar_of_equity_changes, max_drawdown, sharpe_ratio

OUT = Path("results/eth_usdc_500_joint_rl/analysis/boundary_sweep.csv")
POLICIES = ("hodl", "passive_wide", "tau_reset", "cost_aware", "ppo_s3")
LEARNED = Path("results/eth_usdc_500_joint_rl/seed_3/ppo_seed_3.zip")
CAPITAL = 100_000.0
DAYS_PER_WINDOW = 30.0

t0 = time.time()
config = load_sim_config("configs/sim_eth_usdc_500_joint_rl.toml")
view = load_market_view(
    "configs/eth_usdc_500.toml",
    config.split,
    split="eval",
    episode_config=config.episode,
)
windows = walk_forward_windows(view, config)
model = _load_checkpoint(LEARNED)
print(
    f"loaded view; windows={len(windows)} learned={LEARNED.is_file()} "
    f"t={time.time() - t0:.0f}s",
    flush=True,
)

# --- window features -------------------------------------------------------
# The tape's swap rows do not carry active liquidity; the eval-start boundary
# snapshot does (constant across the initialized-tick rows).  Use it as the
# denominator for the volume-relative-to-liquidity axis.
snapshot = view.pool_state
liq_ref = 0.0
if snapshot is not None and "current_liquidity" in snapshot.columns:
    values = snapshot["current_liquidity"].cast(pl.Float64, strict=False).drop_nulls()
    if values.len():
        liq_ref = float(values.median())
print(f"liquidity reference (eval start) = {liq_ref:.3e}", flush=True)
ref = view.reference.sort("close_time")
features: dict[int, dict[str, float]] = {}
for index, (start, end) in enumerate(windows):
    window = view.window(start, end)
    swaps = window.filter(pl.col("event_type") == "swap")
    volume = float(
        swaps["amount0"].cast(pl.Float64, strict=False).abs().sum() / 1e6
    )
    liquidity = liq_ref
    start_t = window["block_timestamp"].min()
    end_t = window["block_timestamp"].max()
    bars = ref.filter((pl.col("close_time") >= start_t) & (pl.col("close_time") <= end_t))
    closes = bars["close"].to_numpy()
    times_us = bars["close_time"].dt.epoch("us").to_numpy()
    dt_s = float(np.median(np.diff(times_us))) / 1e6
    log_returns = np.diff(np.log(closes))
    sigma = float(np.std(log_returns) * np.sqrt(365.25 * 24.0 * 3600.0 / dt_s))
    lvr = 0.125 * sigma**2 * CAPITAL * (DAYS_PER_WINDOW / 365.25)
    features[index + 1] = {
        "sigma": sigma,
        "volume": volume,
        "liquidity": liquidity,
        "volume_per_liquidity": volume / liquidity if liquidity else float("nan"),
        "lvr_usdc": lvr,
    }
    print(
        f"w{index + 1:2d} sigma={sigma:6.3f} volume={volume:12.0f} "
        f"liq={liquidity:12.0f} v/liq={features[index + 1]['volume_per_liquidity']:8.4f} "
        f"lvr={lvr:8.1f} t={time.time() - t0:.0f}s",
        flush=True,
    )

# --- gas and capital sweeps -------------------------------------------------
gas_levels = {"gas_0.25x": 0.25, "gas_4x": 4.0}
capital_levels = {"capital_10k": 10_000.0, "capital_1m": 1_000_000.0}

rows: list[dict[str, object]] = []
for axis, levels in (("gas", gas_levels), ("capital", capital_levels)):
    for level_name, value in levels.items():
        if axis == "gas":
            variant = dataclasses.replace(
                config,
                gas=dataclasses.replace(
                    config.gas,
                    mint_units=int(config.gas.mint_units * value),
                    burn_units=int(config.gas.burn_units * value),
                    collect_units=int(config.gas.collect_units * value),
                    rebalance_swap_units=int(config.gas.rebalance_swap_units * value),
                ),
            )
        else:
            variant = dataclasses.replace(
                config,
                episode=dataclasses.replace(config.episode, agent_capital_usdc=value),
            )
        for name in POLICIES:
            if name == "ppo_s3":
                policy = _CheckpointPolicy(model, variant, name="ppo_s3")
            else:
                policy = build_policy(name, variant)
            for index, (start, end) in enumerate(windows):
                ledger = run_backtest(policy, view.slice(start, end), variant)
                equity, dec, _curve = _joint_accounting(ledger)
                capital = float(ledger.initial_capital)
                decision = ledger.decision_log
                rebalance = (
                    decision.filter(decision["action_type"] == "rebalance")
                    if decision.height
                    else None
                )
                rows.append(
                    {
                        "axis": axis,
                        "level": level_name,
                        "policy": name,
                        "window": index + 1,
                        "regime": dominant_regime(view, start, end),
                        "capital": capital,
                        "total_pnl": float(equity[-1]) - capital,
                        "sharpe": sharpe_ratio(
                            equity, periods_per_year=int(ledger.periods_per_year)
                        ),
                        "cvar_95": cvar_of_equity_changes(equity),
                        "max_drawdown": max_drawdown(equity),
                        "n_rebalances": int(rebalance.height) if rebalance is not None else 0,
                        "total_fees": float(dec.get("total_fees", 0.0)),
                        "total_il": float(dec.get("total_il", 0.0)),
                        "total_gas": float(dec.get("total_gas", 0.0)),
                        "total_slippage": float(dec.get("total_slippage", 0.0)),
                        "net": float(dec.get("net", 0.0)),
                        **features[index + 1],
                    }
                )
            print(
                f"{axis:<8} {level_name:<12} {name:<12} done t={time.time() - t0:.0f}s",
                flush=True,
            )

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def mean(rs: list[dict[str, object]], key: str) -> float:
    return statistics.mean(float(r[key]) for r in rs)


print("\n=== boundary sweeps (mean over 13 windows) ===", flush=True)
for axis in ("gas", "capital"):
    for level in sorted({r["level"] for r in rows if r["axis"] == axis}):
        parts = []
        for name in POLICIES:
            sub = [r for r in rows if r["axis"] == axis and r["level"] == level and r["policy"] == name]
            parts.append(
                f"{name}: pnl={mean(sub, 'total_pnl'):9.0f} "
                f"sharpe={mean(sub, 'sharpe'):6.2f} net={mean(sub, 'net'):8.0f}"
            )
        print(f"{axis:<8} {level:<12} | " + " | ".join(parts), flush=True)

# LVR versus default-band fee income, from the main run's decomposition.
main = list(csv.DictReader(open("results/eth_usdc_500_joint_rl/analysis/windows.csv")))
print("\n=== window LVR vs default-band fee income ===", flush=True)
for index in range(1, 14):
    hodl = next(r for r in main if r["policy"] == "hodl" and int(r["window"]) == index)
    f = features[index]
    ratio = float(hodl["total_fees"]) / f["lvr_usdc"] if f["lvr_usdc"] else float("nan")
    print(
        f"w{index:2d} sigma={f['sigma']:6.3f} lvr={f['lvr_usdc']:8.1f} "
        f"fees={float(hodl['total_fees']):8.0f} fees/lvr={ratio:6.2f}",
        flush=True,
    )
print(f"\nDONE t={time.time() - t0:.0f}s -> {OUT}", flush=True)
