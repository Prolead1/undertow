# ADR 034 — Log-compress the heavy-tailed features of the §11 observation vector

**Status:** accepted (implementation)
**Affects:** `src/undertow/sim/env/observations.py` · `src/undertow/sim/backtest/engine.py` · `src/undertow/sim/evaluate/protocol.py` · `configs/sim_default.toml` (no key change) · sim `CONTRACTS.md` §11 · ADR-029, ADR-033

**Raised by:** RL collapse diagnostics (a dead value function on the 0.05 % run)

## Context

The first 0.05 % primary-pool PPO run (`results/eth_usdc_500_full`, 5 × 1M)
collapsed to a deterministic `hold` policy. The telemetry shows the mechanism:
`train/explained_variance` stayed at `0.0` (later `5.96e-08`) from the first
update to 1 M steps, while the value loss fell to `~6.8e-05` — the critic had
converged to predicting the mean return, i.e. it learned nothing about the
state. With a constant value baseline, PPO's advantages are noise, the entropy
collapses from `ln(91) ≈ 4.50` to `0.19`, and the policy deterministically
freezes on whichever action started ahead: `hold`.

The input layer is the prime suspect. `Observation.to_vector()` casts raw
protocol quantities straight into the network:

| field | raw magnitude |
|---|---|
| `pool_active_liquidity` | `~3e19` |
| `uncollected_fees_token1` | up to `~1e18` |
| `gas_price_recent_wei` | `~1.4e10` |
| `price` / `eth_usd_price_recent` | `~1e3` |
| `sqrt_price` / `tick` / `position_value` | `~1e4`–`1e5` |
| returns, volatility, `il_vs_hodl` | `~1e-2` |

ADR-029 already named "the existing feature-scale problem" but only avoided
adding raw ticks. The MLP's first-layer pre-activations are dominated by the
`1e10`–`1e19` columns; their `tanh` units saturate, gradients vanish, and the
small, informative features (returns, volatility, geometry) never influence the
fit. The problem is representational, not economic: the `passive_narrow`
backtest on the same 0.05 % eval split returns **+8,329 USDC excess vs HODL**
(fees +31,257, IL −22,903), so there is a learnable, profitable policy.

## Decision

Log-compress the heavy-tailed features of the vector with the odd, strictly
increasing map

```
scale(v) = sign(v) * log1p(|v|),     unscale(v) = sign(v) * expm1(|v|)
```

applied at exactly these §11 positions:

```
SCALED_VECTOR_INDICES = (0, 1, 2, 7, 8, 9, 11, 13, 14)
# price, sqrt_price, tick, position_value, fees0, fees1,
# pool_active_liquidity, gas_price_recent_wei, eth_usd_price_recent
```

* Both vector producers — `Observation.to_vector()` (simulator) and
  `BacktestObservation.to_vector()` (exact replay) — call the shared
  `scale_vector`, so a checkpoint trained in the simulator sees the same scale
  when the backtester replays it.
* `observation_from_vector()` calls `unscale_vector` first, so the structured
  `Observation` handed back to baseline policies keeps its raw units. Baselines
  never see compressed values.
* The structured dataclass fields are unchanged; only the flattened vector the
  network consumes is compressed.

## Consequences

- The vector's scaled positions are now `O(1)` (the largest is
  `log1p(3e19) ≈ 45`), while everything else is byte-identical. The
  observation space stays `Box(-inf, inf, (24,))`.
- **Existing 24-dim checkpoints are incompatible.** The shape guard in
  `_CheckpointPolicy` cannot detect this (the length did not change); every
  checkpoint must be retrained, and a pre-change run's results are not
  comparable with post-change ones. The run manifest's `git_commit` is the
  audit trail.
- The UI's state inspector displays the compressed values it reads from vectors;
  the raw per-step ledgers (rollout parquets, telemetry) are untouched.
- The transform is a bijection defined at zero and for negative ticks, and it is
  monotone, so no ordinal or geometric information is lost.

## Rejected alternatives

- **`VecNormalize` in the training harness only.** It would leave the
  training-time representation dependent on learned running statistics that must
  be saved, loaded and applied inside both the simulator and the exact
  backtester; the deterministic transform needs no state and cannot drift
  between the two engines.
- **Scaling the dataclass fields at build time.** It would silently change the
  units the baselines, the UI and the rollout ledger read, breaking far more
  than the network input.
- **Dropping the redundant price features (`price`/`sqrt_price`/`tick`).** A
  contract change (24 → fewer) that removes information the geometry and regime
  analysis rely on; compression keeps all three.
