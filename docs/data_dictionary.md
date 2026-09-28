<!-- GENERATED FILE — do not edit by hand. Regenerate with `uv run python -m undertow.data.dictionary`. -->

# Undertow data dictionary

This dictionary is generated from the canonical schemas in `src/undertow/data/schemas.py` (`SCHEMA_REGISTRY`) and their per-column unit/convention metadata. It is committed as generated output and tested for freshness; do not edit it by hand. Timestamps are `timestamp[us, tz=UTC]` values; big integers are decimal strings (see *Fixed-point and big integers* below).

## Streams

### `swap`

**Source route:** The Graph subgraph (Route A, the pull-time default) or RPC ``eth_getLogs`` (Route C); the two routes are cross-checked for exact agreement by T13

**Sort key:** `(block_number, log_index)` · **Pool-scoped:** yes

| column | type | unit / convention | nullable |
| --- | --- | --- | --- |
| `block_number` | `int64` | block height (int64) | no |
| `log_index` | `int32` | index of the log within the block (int32) | no |
| `block_timestamp` | `timestamp[us, tz=UTC]` | block time, UTC (timestamp[us]) | no |
| `tx_hash` | `string` | 0x-prefixed lowercase transaction hash | no |
| `pool_address` | `string` | lowercase 0x-prefixed pool address | no |
| `event_type` | `string` | stream discriminator: swap\|mint\|burn\|collect\|flash | no |
| `amount0` | `string` | raw token0 units, signed int256, positive=flowed INTO the pool | no |
| `amount1` | `string` | raw token1 units, signed int256, positive=flowed INTO the pool | no |
| `sqrt_price_x96` | `string` | uint160 pool sqrt price after the swap, Q64.96 | no |
| `liquidity` | `string` | uint128 active liquidity after the swap; RPC-only (null on The Graph route) | yes |
| `tick` | `int32` | pool tick after the swap (int32) | no |
| `sender` | `string` | lowercase 0x-prefixed address | no |
| `recipient` | `string` | lowercase 0x-prefixed address | no |

### `mint`

**Source route:** The Graph subgraph (Route A, the pull-time default) or RPC ``eth_getLogs`` (Route C); cross-checked by T13

**Sort key:** `(block_number, log_index)` · **Pool-scoped:** yes

| column | type | unit / convention | nullable |
| --- | --- | --- | --- |
| `block_number` | `int64` | block height (int64) | no |
| `log_index` | `int32` | index of the log within the block (int32) | no |
| `block_timestamp` | `timestamp[us, tz=UTC]` | block time, UTC (timestamp[us]) | no |
| `tx_hash` | `string` | 0x-prefixed lowercase transaction hash | no |
| `pool_address` | `string` | lowercase 0x-prefixed pool address | no |
| `event_type` | `string` | stream discriminator: swap\|mint\|burn\|collect\|flash | no |
| `owner` | `string` | lowercase 0x-prefixed address (NFT position manager for retail positions) | no |
| `tick_lower` | `int32` | lower tick, on the spacing grid (int32) | no |
| `tick_upper` | `int32` | upper tick, on the spacing grid, > tick_lower (int32) | no |
| `liquidity_amount` | `string` | uint128 event amount — the position's L | no |
| `amount0` | `string` | raw token0 units, unsigned uint256 | no |
| `amount1` | `string` | raw token1 units, unsigned uint256 | no |
| `sender` | `string` | lowercase 0x-prefixed address; null on burn rows (Burn has no sender) | yes |

### `burn`

**Source route:** The Graph subgraph (Route A, the pull-time default) or RPC ``eth_getLogs`` (Route C); cross-checked by T13

**Sort key:** `(block_number, log_index)` · **Pool-scoped:** yes

| column | type | unit / convention | nullable |
| --- | --- | --- | --- |
| `block_number` | `int64` | block height (int64) | no |
| `log_index` | `int32` | index of the log within the block (int32) | no |
| `block_timestamp` | `timestamp[us, tz=UTC]` | block time, UTC (timestamp[us]) | no |
| `tx_hash` | `string` | 0x-prefixed lowercase transaction hash | no |
| `pool_address` | `string` | lowercase 0x-prefixed pool address | no |
| `event_type` | `string` | stream discriminator: swap\|mint\|burn\|collect\|flash | no |
| `owner` | `string` | lowercase 0x-prefixed address (NFT position manager for retail positions) | no |
| `tick_lower` | `int32` | lower tick, on the spacing grid (int32) | no |
| `tick_upper` | `int32` | upper tick, on the spacing grid, > tick_lower (int32) | no |
| `liquidity_amount` | `string` | uint128 event amount — the position's L | no |
| `amount0` | `string` | raw token0 units, unsigned uint256 | no |
| `amount1` | `string` | raw token1 units, unsigned uint256 | no |
| `sender` | `string` | lowercase 0x-prefixed address; null on burn rows (Burn has no sender) | yes |

### `collect`

**Source route:** The Graph subgraph (Route A, the pull-time default) or RPC ``eth_getLogs`` (Route C); cross-checked by T13. ``recipient`` is RPC-only

**Sort key:** `(block_number, log_index)` · **Pool-scoped:** yes

| column | type | unit / convention | nullable |
| --- | --- | --- | --- |
| `block_number` | `int64` | block height (int64) | no |
| `log_index` | `int32` | index of the log within the block (int32) | no |
| `block_timestamp` | `timestamp[us, tz=UTC]` | block time, UTC (timestamp[us]) | no |
| `tx_hash` | `string` | 0x-prefixed lowercase transaction hash | no |
| `pool_address` | `string` | lowercase 0x-prefixed pool address | no |
| `event_type` | `string` | stream discriminator: swap\|mint\|burn\|collect\|flash | no |
| `owner` | `string` | lowercase 0x-prefixed address | no |
| `recipient` | `string` | lowercase 0x-prefixed address; RPC-only (null on The Graph route) | yes |
| `tick_lower` | `int32` | lower tick, on the spacing grid (int32) | no |
| `tick_upper` | `int32` | upper tick, on the spacing grid, > tick_lower (int32) | no |
| `amount0` | `string` | raw token0 units withdrawn, unsigned uint128 | no |
| `amount1` | `string` | raw token1 units withdrawn, unsigned uint128 | no |

### `flash`

**Source route:** RPC ``eth_getLogs`` — the pinned subgraph does not index flash events, so Route C is the sole source

**Sort key:** `(block_number, log_index)` · **Pool-scoped:** yes

| column | type | unit / convention | nullable |
| --- | --- | --- | --- |
| `block_number` | `int64` | block height (int64) | no |
| `log_index` | `int32` | index of the log within the block (int32) | no |
| `block_timestamp` | `timestamp[us, tz=UTC]` | block time, UTC (timestamp[us]) | no |
| `tx_hash` | `string` | 0x-prefixed lowercase transaction hash | no |
| `pool_address` | `string` | lowercase 0x-prefixed pool address | no |
| `event_type` | `string` | stream discriminator: swap\|mint\|burn\|collect\|flash | no |
| `sender` | `string` | lowercase 0x-prefixed address | no |
| `recipient` | `string` | lowercase 0x-prefixed address | no |
| `amount0` | `string` | raw token0 units borrowed, unsigned uint256 | no |
| `amount1` | `string` | raw token1 units borrowed, unsigned uint256 | no |
| `paid0` | `string` | raw token0 units repaid, unsigned uint256 (exceeds amount0 by the fee) | no |
| `paid1` | `string` | raw token1 units repaid, unsigned uint256 (exceeds amount1 by the fee) | no |

### `fee_growth`

**Source route:** RPC ``eth_call`` snapshots of ``ticks()`` / ``slot0()`` / ``liquidity()`` at sampled blocks (T06); no block-range fetch, and not in the default pull

**Sort key:** `(block_number, tick)` · **Pool-scoped:** yes

| column | type | unit / convention | nullable |
| --- | --- | --- | --- |
| `block_number` | `int64` | block the eth_call was made at; state after the block (int64) | no |
| `tick` | `int32` | tick whose Tick struct was read; GLOBAL_TICK_SENTINEL marks global-only rows (int32) | no |
| `fee_growth_outside_0_x128` | `string` | uint256 Q128; null on global rows | yes |
| `fee_growth_outside_1_x128` | `string` | uint256 Q128; null on global rows | yes |
| `liquidity_gross` | `string` | uint128 gross liquidity at the tick | no |
| `liquidity_net` | `string` | signed int128 net liquidity at the tick | no |
| `initialized` | `bool` | tick struct initialized flag (bool) | no |
| `fee_growth_global_0_x128` | `string` | uint256 Q128; present on every row (denormalized) | no |
| `fee_growth_global_1_x128` | `string` | uint256 Q128; present on every row | no |
| `current_tick` | `int32` | pool slot0.tick at that block (int32) | no |
| `current_liquidity` | `string` | pool liquidity() at that block, uint128 | no |
| `source` | `string` | 'rpc_call' (exact) or 'interpolated' (flagged, never silently) | no |
| `pool_address` | `string` | lowercase 0x-prefixed pool address | no |
| `fee_protocol` | `int32` | packed uint8 from slot0: bits 0-3 = fp0, bits 4-7 = fp1; 0 when off | no |

### `gas`

**Source route:** RPC ``eth_feeHistory``, base-fee only (ADR-006, ADR-005)

**Sort key:** `(block_number,)` · **Pool-scoped:** no

| column | type | unit / convention | nullable |
| --- | --- | --- | --- |
| `block_number` | `int64` | block height; one row per block, no gaps (int64) | no |
| `base_fee_per_gas` | `string` | wei, EIP-1559 base fee (uint256) | no |
| `gas_used` | `int64` | block gas used (int64) | no |
| `gas_limit` | `int64` | block gas limit (int64) | no |
| `priority_fee_p50_wei` | `string` | median effective priority fee of txs in the block (uint256) | no |
| `priority_fee_p90_wei` | `string` | 90th percentile effective priority fee — the spike cost (uint256) | no |

### `reference`

**Source route:** Binance 1m klines — REST API and bulk monthly archive (T08)

**Sort key:** `(symbol, open_time)` · **Pool-scoped:** no

| column | type | unit / convention | nullable |
| --- | --- | --- | --- |
| `open_time` | `timestamp[us, tz=UTC]` | kline open, left-closed interval, UTC (timestamp[us]) | no |
| `close_time` | `timestamp[us, tz=UTC]` | kline close, UTC (timestamp[us]) | no |
| `open` | `double` | open price, ETH in USD-stable terms (float64) | no |
| `high` | `double` | high price (float64) | no |
| `low` | `double` | low price (float64) | no |
| `close` | `double` | close price (float64) | no |
| `volume_base` | `double` | ETH volume (float64) | no |
| `quote_volume` | `double` | quote volume (float64) | no |
| `trades` | `int64` | number of trades (int64) | no |
| `symbol` | `string` | 'ETHUSDT' / 'ETHUSDC' | no |
| `is_gap_filled` | `bool` | true if the bar was forward-filled over an exchange outage (bool) | no |

### `regime`

**Source route:** Derived in T12 from the cached ``reference`` feed (no network)

**Sort key:** `(timestamp,)` · **Pool-scoped:** no

| column | type | unit / convention | nullable |
| --- | --- | --- | --- |
| `timestamp` | `timestamp[us, tz=UTC]` | end of the backward-looking window; the label applies at this time, UTC | no |
| `sigma_rv` | `double` | annualized realized vol over the lookback (float64) | no |
| `mu` | `double` | total log return over the lookback: log(S_end/S_start) (float64) | no |
| `regime` | `string` | label: bull\|bear\|sideways\|high_vol\|unknown | no |
| `window_complete` | `bool` | false => regime == 'unknown' (bool) | no |
| `symbol` | `string` | reference symbol the labels were computed from | no |

### `event_tape`

**Source route:** T11 aligned union of the log streams plus backward as-of joins of the reference, regime, gas and fee-growth side-streams

**Sort key:** `(block_number, log_index)` · **Pool-scoped:** yes

| column | type | unit / convention | nullable |
| --- | --- | --- | --- |
| `block_number` | `int64` | block height (int64) | no |
| `log_index` | `int32` | index of the log within the block (int32) | no |
| `block_timestamp` | `timestamp[us, tz=UTC]` | block time, UTC (timestamp[us]) | no |
| `tx_hash` | `string` | 0x-prefixed lowercase transaction hash | no |
| `pool_address` | `string` | lowercase 0x-prefixed pool address | no |
| `event_type` | `string` | stream discriminator: swap\|mint\|burn\|collect\|flash | no |
| `amount0` | `string` | signed on swap rows, non-negative on mint/burn/collect rows; null where inapplicable | yes |
| `amount1` | `string` | signed on swap rows, non-negative on mint/burn/collect rows; null where inapplicable | yes |
| `sqrt_price_x96` | `string` | uint160 pool sqrt price, Q64.96; swap rows only | yes |
| `liquidity` | `string` | uint128 active liquidity; swap rows only | yes |
| `tick` | `int32` | pool tick (int32); swap rows only | yes |
| `sender` | `string` | lowercase 0x-prefixed address; swap/mint rows only | yes |
| `recipient` | `string` | lowercase 0x-prefixed address; swap/collect rows only | yes |
| `owner` | `string` | lowercase 0x-prefixed address; mint/burn/collect rows only | yes |
| `tick_lower` | `int32` | lower tick on the spacing grid (int32); mint/burn/collect rows only | yes |
| `tick_upper` | `int32` | upper tick on the spacing grid (int32); mint/burn/collect rows only | yes |
| `liquidity_amount` | `string` | uint128 position liquidity; mint/burn rows only | yes |
| `seq` | `int64` | 0..N-1 dense canonical step index (int64) | no |
| `price_pool` | `double` | pool price derived from sqrt_price_x96, decimals-adjusted (float64) | yes |
| `price_reference` | `double` | as-of join: last reference close with close_time <= block_timestamp (float64) | yes |
| `regime` | `string` | as-of join from the regime table (bull\|bear\|sideways\|high_vol\|unknown) | yes |
| `base_fee_per_gas` | `string` | joined from gas on block_number, wei (uint256) | yes |
| `priority_fee_p50_wei` | `string` | joined from gas on block_number, wei (uint256) | yes |
| `fee_growth_global_0_x128` | `string` | forward-filled from nearest prior snapshot, Q128 (uint256) | yes |
| `fee_growth_global_1_x128` | `string` | forward-filled from nearest prior snapshot, Q128 (uint256) | yes |
| `fee_growth_source` | `string` | 'exact' \| 'stale_prior' — never 'interpolated_future' | yes |


## Semantics and conventions

### Swap signs and which side pays the fee

`swap.amount0` / `swap.amount1` are **signed** `int256` values in raw token units.
**Sign rule:** positive = flowed into the pool; negative = flowed out. A real swap has
opposite signs and neither amount is zero (T13's `check_swap_sign_convention`). The fee is
taken on the **input** (positive) side: `fee_paid = amount_in * fee_tier / 1_000_000`, in the
input token. Never assume which token is the input — read the sign. On the `event_tape`,
`amount0`/`amount1` are signed on swap rows and non-negative on mint/burn/collect rows; the
`event_type` discriminator tells a consumer which convention applies.

### Fixed-point and big integers

On-chain fixed-point values are integers, not decimals: `Q64.96` means scaled by `2**96`
(`sqrt_price_x96 = sqrt(price_raw) * 2**96`) and `Q128` means scaled by `2**128` (fee growth
per unit of liquidity). `tick` indexes the `1.0001**tick` grid. Values that can exceed 64 bits
— `uint256`, `int256`, `uint128`, `uint160` — are stored in Parquet as `pa.string()` holding a
plain **decimal string** and handled in Python as `int`. Parquet has no native 256-bit integer
type, and `float64` preserves only 53 bits of mantissa (`2**128` is ~70 bits wider), so a Q128
accumulator round-tripped through a float would silently destroy exactly the low bits this
pipeline exists to preserve. `schemas.encode_uint` / `schemas.decode_uint` are the only
sanctioned conversions; signed values keep their leading `-`, and hex, exponent and `+`-signed
forms are rejected.

### Price orientation

`price_pool` and `price_reference` are in the **same units**: the price of token1 denominated
in token0, in human (decimals-adjusted) units — for the pinned USDC/WETH pools, **USDC per
WETH**, i.e. ~2000–4000 over the study window. This is T02's `sqrt_price_x96_to_price`
orientation, quoted verbatim: *“Human price of token1 denominated in token0 — USDC per WETH
for the pinned pools (~3000).”* USDC is token0 (6 dp) and WETH is token1 (18 dp), and the raw
tick is `1.0001**tick` in raw token1-per-token0 terms, so `price_pool` is **decreasing** in
`tick` for the pinned pools (a higher tick means fewer USDC per WETH). The conversion is
`10**(dec1 - dec0) * 2**192 / sqrt_price_x96**2`, evaluated exactly as a `Decimal`.

### `Collect` is not accrual; paired `Burn` separates principal

`collect.amount0` / `collect.amount1` are the amounts **withdrawn**, not the fees accrued in
the current window: a `Collect` includes fees that accrued *before* the window began. A `Burn`
immediately followed by a `Collect` in the same transaction withdraws **principal + fees
together**, so reconciliation must separate them using the paired `Burn(amount0, amount1)`
principal: the fee component is `Collect.amountX - Burn.amountX` when the two are paired in one
transaction, and `Collect.amountX` for a fee-only collect that stands alone. This is why T13's
`reconcile_fees_against_collect` tolerance is defined on the **fee** component, not the total.
`burn.liquidity_amount` is unsigned in the event; the sign of the liquidity *delta* is applied
in T11, not stored in this stream.

### `fee_growth_source` and why there is no forward-interpolated option

`event_tape.fee_growth_source` is either `"exact"` (a fee-growth snapshot exists at this exact
block) or `"stale_prior"` (the value was forward-filled from the nearest snapshot at a **prior**
block). There is deliberately **no** `"interpolated_future"` value: the only values knowable at
block *n* come from blocks ≤ *n*, and a value that needs a later snapshot is look-ahead bias.
All as-of joins (`price_reference`, `regime`, the fee-growth globals) are backward-only. Inside
the `fee_growth` stream itself, `source` is `"rpc_call"` (an exact `eth_call`) or
`"interpolated"`, which is always flagged and never silent.

### Regime labels and the ADR-001 `μ` definition

`regime.regime` is one of `bull`, `bear`, `sideways`, `high_vol` or `unknown`, computed from a
30-day **backward-looking** rolling window of reference minute closes. The thresholds are
pre-committed in `RegimeConfig` and swept by `sensitivity_sweep`: `vol_threshold = 0.80`
(annualized `sigma_rv`) and `drift_threshold = 0.05`. The decision order is fixed and the
comparisons are strict: (1) `sigma_rv > 0.80` → `high_vol`; (2) else `mu > 0.05` → `bull`;
(3) else `mu < -0.05` → `bear`; (4) else `sideways`. The first `lookback_days` of output have
`window_complete = false` and `regime = "unknown"`; they are never dropped. Per **ADR-001**,
`mu` is the **total** window log return `μ₃₀ = Σ log(S_t/S_{t−1}) = log(S_end/S_start)`, not the
per-step mean the roadmap's formula literally wrote — a per-minute mean thresholded at ±5% is
dimensionally inconsistent and would label every real window `sideways`. `sigma_rv` is
`std(log returns, ddof=1) * sqrt(365*1440)`.

### Reference-feed gaps and the USDT proxy assumption

`reference` is Binance 1-minute klines: `ETHUSDT` primary, `ETHUSDC` as the cross-check. Gaps
(exchange outages, thin pairs, a kline that never printed) are **forward-filled and flagged**,
never interpolated and never silently dropped: a missing bar after at least one observed bar
copies the prior close into OHLC, zeroes volume and trades, and sets `is_gap_filled = true`.
A missing **leading** edge is left absent — back-filling it would be look-ahead. `close_time` is
`open_time + 59.999 s`, and every as-of join uses `close_time <= block_timestamp`. Prices are
ETH in USD-stable terms, and this pipeline treats **USDT as a USD proxy (1 USDT ≙ 1 USD)** — an
explicit assumption, not a fact: USDT can deviate from parity under stress (tens of bps within
the study window), which is second-order for IL/LVR measurement and 30-day regime labels. The
symbol set is a constructor argument, so the feed can be switched without code changes.

### Known limitations

- **Position attribution** uses `owner` as it appears on-chain. Retail Uniswap V3 positions are
  owned by the NFT position manager (`0xC364…FE88`), so `mint.owner`, `burn.owner` and
  `collect.owner` identify the manager, not the end user. Per-user attribution needs a separate
  NFT-transfer join and is out of scope (`PLAN.md` §7).
- **Fee growth is sampled, not per-block.** `fee_growth` rows are `eth_call` snapshots at
  selected blocks; between samples the pipeline replays swaps and liquidity events with T10's
  `FeeGrowthTracker` and verifies that replay against the observed snapshots (T13's
  `reconcile_fees_against_collect`), rather than paying for an `eth_call` at every block.
  The lifecycle-boundary-plus-stride sampling strategy is not wired into the default pull;
  `fee_growth` has no block-range fetch and is not in the default pull set.
- **Gas is base-fee only** (ADR-006): per-block timestamps and per-block priority-fee resolution
  were removed. The tape carries `block_timestamp` from the event logs and converts gas to USD
  via its `price_reference`; `gas_used`, `gas_limit`, `priority_fee_p50_wei` and `priority_fee_p90_wei`
  are written as `0` (ADR-005).
- **BigQuery is not implemented.** `Route.BIGQUERY` is an escape hatch reserved by the contract,
  not a v1 code path.
- **No `adr/002` exists** in this repository, so there is no documented swap-segment-apportionment
  residual; T13's fee reconciliation targets exact integer agreement within a bounded
  tolerance (raw 2 or relative 1e-6) as a documented fallback.
