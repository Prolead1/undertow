# CONTRACTS.md — frozen cross-task interfaces for `undertow.data`

**Status: FROZEN.** Every signature, column name, type, unit and sign convention below is a contract
between tasks. Implement exactly this. You may *add* (a new column, a new keyword-only argument with a
default, a new helper); you may not *rename*, *retype*, *reorder* or *remove*. If something here is
genuinely wrong, write `adr/NNN-<slug>.md`, flag it in your PR, and notify the orchestrator — do not
unilaterally change it, because other agents are already coding against it.

---

## 0. Notation

- `int` in a schema column means an arbitrary-precision Python integer serialized as a **decimal string**
  in Parquet (`pa.string()`). Rationale: `uint256` does not fit in any Arrow integer type, and float64
  loses precision above 2^53. Column names for these end in nothing special — the type table is the
  authority. Helpers live in `schemas.py` (`encode_uint`, `decode_uint`).
- `Q128` = value scaled by `2**128`. `Q64.96` = value scaled by `2**96`.
- All timestamps: `pa.timestamp("us", tz="UTC")`.
- "block-ordered" = sorted ascending by `(block_number, log_index)`.

---

## 1. `undertow.data.types` (T01)

```python
from __future__ import annotations
from enum import Enum
from typing import NewType

Address    = NewType("Address", str)   # lowercase 0x-prefixed, 42 chars, checksum-stripped
BlockNumber = NewType("BlockNumber", int)
Tick        = NewType("Tick", int)

class EventType(str, Enum):
    SWAP    = "swap"
    MINT    = "mint"
    BURN    = "burn"
    COLLECT = "collect"
    FLASH   = "flash"        # fetched for completeness; contributes fee growth, no liquidity change

class FeeTier(int, Enum):
    BPS_1   = 100     # 0.01%, tick spacing 1
    BPS_5   = 500     # 0.05%, tick spacing 10
    BPS_30  = 3000    # 0.30%, tick spacing 60
    BPS_100 = 10000   # 1.00%, tick spacing 200

TICK_SPACING: dict[FeeTier, int] = {
    FeeTier.BPS_1: 1, FeeTier.BPS_5: 10, FeeTier.BPS_30: 60, FeeTier.BPS_100: 200,
}

class Regime(str, Enum):
    BULL     = "bull"
    BEAR     = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"
    UNKNOWN  = "unknown"     # window not yet full (first 30 days) — never silently dropped

class Route(str, Enum):
    THEGRAPH  = "thegraph"
    RPC       = "rpc"
    BIGQUERY  = "bigquery"    # escape hatch, not implemented in v1
    REFERENCE = "reference"

# ---------------------------------------------------------------------------
# CheckResult lives HERE, in types.py, not in validation/ — on purpose.
# T08 (wave 3) returns one from compare_symbols, but validation/ is owned by
# T13 (wave 5). A result type shared across waves must be defined no later than
# the earliest wave that produces one. T13 IMPORTS it from here and must not
# redefine it.
# ---------------------------------------------------------------------------
Severity = Literal["critical", "warning", "info"]

@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    passed: bool
    severity: Severity
    detail: str
    metrics: Mapping[str, float | int | str] = field(default_factory=dict)

# Exception hierarchy — every fetcher raises only these.
class UndertowDataError(Exception): ...
class ConfigError(UndertowDataError): ...
class FetchError(UndertowDataError): ...            # base for all I/O failures
class RateLimitError(FetchError): ...               # retryable
class TransientNetworkError(FetchError): ...        # retryable
class PermanentFetchError(FetchError): ...          # not retryable (4xx other than 429, bad query)
class SchemaViolationError(UndertowDataError): ...
class ValidationError(UndertowDataError): ...
class ReorgDetectedError(UndertowDataError): ...
```

`ConfigError` and `SchemaViolationError` must carry the offending value in `args[0]`'s message.

---

## 2. `undertow.data.config` (T01)

```python
@dataclass(frozen=True, slots=True)
class PoolConfig:
    address: Address
    token0_symbol: str
    token1_symbol: str
    token0_decimals: int
    token1_decimals: int
    fee_tier: FeeTier
    tick_spacing: int          # must equal TICK_SPACING[fee_tier]; validated in __post_init__
    deployment_block: BlockNumber

@dataclass(frozen=True, slots=True)
class WindowConfig:
    start_block: BlockNumber
    end_block: BlockNumber     # inclusive
    # exactly one of block or date bounds is given; dates resolved to blocks by T07's block index
    start_utc: datetime | None = None
    end_utc: datetime | None = None
    # Walk-forward split hook (§10.3). undertow.data NEVER acts on these — it only records them in the
    # manifest, so that a chosen split is provenance-tracked instead of living in a notebook. The split
    # itself belongs to undertow.sim's plan (PLAN.md §1). Validation: when both are given they must be
    # UTC-aware, inside the window, and train_end_utc <= eval_start_utc.
    train_end_utc: datetime | None = None
    eval_start_utc: datetime | None = None

@dataclass(frozen=True, slots=True)
class RegimeConfig:
    lookback_days: int = 30
    vol_threshold: float = 0.80        # annualized σ_rv above this ⇒ HIGH_VOL
    drift_threshold: float = 0.05      # |μ₃₀| above this ⇒ BULL / BEAR
    periods_per_year: int = 365 * 1440 # minute bars

@dataclass(frozen=True, slots=True)
class EndpointConfig:
    graph_url: str
    rpc_url: str                       # archive node required
    reference_base_url: str
    max_concurrency: int = 4
    request_timeout_s: float = 30.0
    max_retries: int = 5

@dataclass(frozen=True, slots=True)
class DataConfig:
    pool: PoolConfig
    window: WindowConfig
    regime: RegimeConfig
    endpoints: EndpointConfig
    cache_dir: Path
    output_dir: Path

def load_config(path: str | Path) -> DataConfig: ...   # TOML; env-var expansion for secrets only
def default_pools() -> dict[str, PoolConfig]: ...      # keyed "USDC_WETH_3000", "USDC_WETH_500"
```

Secrets: `endpoints.graph_url` / `rpc_url` may contain `${ENV_VAR}` placeholders expanded at load time.
A missing env var raises `ConfigError`. The expanded value must never be logged or written to a manifest;
log the *host* only.

Pinned constants also live here: `Q96 = 2**96`, `Q128 = 2**128`, `TICK_BASE = Decimal("1.0001")`,
`MIN_TICK = -887272`, `MAX_TICK = 887272`.

---

## 3. `undertow.data.fixedpoint` (T02)

### 3.0 The price orientation — read this before writing a line

The pool's raw price is `raw_token1_per_raw_token0 = (sqrt_price_x96 / 2**96)**2`, in **raw base units**
(wei-like integers), not human units. For USDC(6dp)/WETH(18dp) at ETH ≈ $3000 that raw ratio is ~3.3e8 and means
"raw WETH units per raw USDC unit". Neither number is human-meaningful, so a decimal adjustment is
mandatory, and its **direction is the single most consequential convention in this pipeline**.

**The pinned convention:** `sqrt_price_x96_to_price` returns the price of **token1 denominated in
token0, in human units** — for the pinned pools, **USDC per WETH**, i.e. the familiar "ETH price in
dollars", ~2000–4000 across the study window.

The formula that produces it:

```
price_token1_in_token0 = 10**(dec1 - dec0) * 2**192 / sqrt_price_x96**2
```

**Worked check** (transcribe this into T02's test): at ETH = $3000 exactly, `sqrt_price_x96 =
1446501726624926496477173928747177`; the formula above returns `3000.000000…`, and the naive
`(sqrt_price_x96 / 2**96)**2 * 10**(dec0 - dec1)` returns `3.333…e-4` — its reciprocal, i.e. WETH per
USDC. Both are "correct" arithmetic; only one is the convention, and the wrong one silently inverts or
1e12-scales every impermanent-loss number in the thesis.

Compute it as an exact `Decimal` (integer numerator and denominator, no intermediate float), and note
the division is by `sqrt_price_x96**2`, so guard `sqrt_price_x96 == 0` with a `ValueError`.

**Reference anchors for the pinned pools** (transcribe into T02's tests; T13 uses them as sanity bounds).
The pool tick for USDC/WETH is **positive, around +190k to +210k** — if you see negative ticks on this
pool, your decode or your orientation is wrong:

| tick | USDC per WETH |
|---|---|
| 190 000 | ~5 608 |
| 194 000 | ~3 759 |
| 196 242 | ~3 004 |
| 200 000 | ~2 063 |
| 207 000 | ~1 025 |

```python
def sqrt_price_x96_to_price(sqrt_price_x96: int, dec0: int, dec1: int) -> Decimal:
    """Human price of token1 denominated in token0 — USDC per WETH for the pinned pools (~3000).

    price = 10**(dec1 - dec0) * 2**192 / sqrt_price_x96**2, as an exact Decimal. See CONTRACTS §3.0.
    """

def price_to_sqrt_price_x96(price: Decimal, dec0: int, dec1: int) -> int:
    """Inverse of sqrt_price_x96_to_price. Round-trips to within 1 unit of sqrt_price_x96."""

def tick_to_sqrt_price_x96(tick: int) -> int:
    """Exact integer reimplementation of TickMath.getSqrtRatioAtTick. Must match on-chain bit-for-bit."""

def sqrt_price_x96_to_tick(sqrt_price_x96: int) -> int:
    """Exact integer reimplementation of TickMath.getTickAtSqrtRatio (floor semantics)."""

def tick_to_price(tick: int, dec0: int, dec1: int) -> Decimal:
    """Same orientation as sqrt_price_x96_to_price (token1 in token0, human units).

    NOTE the consequence of that orientation: the raw tick price 1.0001**tick is
    raw-token1-per-raw-token0, so this function is DECREASING in tick for the pinned pools.
    A higher tick means a higher raw token1/token0 ratio, i.e. a LOWER USDC-per-WETH price.
    Assert this direction in a test; it surprises everyone exactly once.
    """

def price_to_tick(price: Decimal, dec0: int, dec1: int) -> int:
    """Inverse of tick_to_price, floor semantics (see §3 note), NOT round."""

def align_tick_down(tick: int, spacing: int) -> int: ...   # floor to grid, correct for negatives
def align_tick_up(tick: int, spacing: int) -> int: ...

def wrapping_sub_256(a: int, b: int) -> int:
    """(a - b) mod 2**256 — Solidity unchecked subtraction. Required for all feeGrowth deltas."""

def wrapping_add_256(a: int, b: int) -> int: ...

def q128_to_decimal(value_q128: int) -> Decimal:
    """value / 2**128 as exact Decimal. Never use float here."""

# Liquidity ↔ amounts (Uniswap V3 LiquidityAmounts / SqrtPriceMath, integer, floor/ceil per protocol)
def get_amount0_delta(sqrt_a: int, sqrt_b: int, liquidity: int, round_up: bool) -> int: ...
def get_amount1_delta(sqrt_a: int, sqrt_b: int, liquidity: int, round_up: bool) -> int: ...
def liquidity_for_amounts(sqrt_price: int, sqrt_a: int, sqrt_b: int, amount0: int, amount1: int) -> int: ...
def amounts_for_liquidity(sqrt_price: int, sqrt_a: int, sqrt_b: int, liquidity: int) -> tuple[int, int]: ...
```

Non-negotiable properties (these become T02's tests):
- `sqrt_price_x96_to_tick(tick_to_sqrt_price_x96(t)) == t` for all `t` in
  `{MIN_TICK, MIN_TICK+1, -887000, -100000, -60, -1, 0, 1, 60, 100000, 887000, MAX_TICK}`.
- `tick_to_sqrt_price_x96` matches the hardcoded on-chain values in
  `tests/data/fixtures/tickmath_vectors.json` (extracted from the Solidity library) **exactly**.
- `wrapping_sub_256(0, 1) == 2**256 - 1`.
- `sqrt_price_x96_to_price(1446501726624926496477173928747177, 6, 18)` == `3000` to 20 significant
  digits — the §3.0 worked check.
- No function in this module returns or accepts `float`. Anywhere.

---

## 4. Canonical Parquet schemas (T03) — the data contract

### 4.0 Two families of stream — the key-column rule applies to one of them

| Family | Streams | Keyed by | Pool-scoped? |
|---|---|---|---|
| **Log streams** | `swap`, `mint`, `burn`, `collect`, `flash`, `event_tape` | `(block_number, log_index)` | yes |
| **Non-log streams** | `fee_growth`, `gas`, `reference`, `regime` | see below | `fee_growth` yes; `gas` no; `reference`/`regime` no |

The six **key columns** below are mandatory, first, and in this order **for log streams only**. Non-log
streams are enumerated in full in §4.4–4.7 and carry only what they actually have: a Binance kline has no
block number and no `log_index`, and inventing `-1` sentinels for it would be a lie in the data.

Per-stream sort keys (T09 sorts by these before writing; T03 records them in
`SORT_KEYS: dict[str, tuple[str, ...]]`):

| Stream | Sort key |
|---|---|
| `swap`, `mint`, `burn`, `collect`, `flash`, `event_tape` | `(block_number, log_index)` |
| `fee_growth` | `(block_number, tick)` |
| `gas` | `(block_number,)` |
| `reference` | `(symbol, open_time)` |
| `regime` | `(timestamp,)` |

Pool scoping (T09 partitions on `pool_address` only for pool-scoped streams; `reference` and `regime`
are **global** and must not be duplicated under each pool directory):

- pool-scoped: `swap`, `mint`, `burn`, `collect`, `flash`, `fee_growth`, `event_tape`
- global: `gas` (chain-wide), `reference` (exchange feed), `regime` (derived from `reference`)

T03 exposes this as `POOL_SCOPED: frozenset[str]` so T09 and T14 never have to guess.

### 4.0.1 Key columns (log streams)

These six come first, in this order:

| Column | Arrow type | Meaning |
|---|---|---|
| `block_number` | `int64` | block height |
| `log_index` | `int32` | index of the log within the block; non-null, never a sentinel (non-log streams simply do not have this column — see §4.0) |
| `block_timestamp` | `timestamp[us,UTC]` | block time |
| `tx_hash` | `string` | `0x`-prefixed, lowercase |
| `pool_address` | `string` | lowercase |
| `event_type` | `string` (`EventType`) | discriminator |

### 4.1 `swap`

| Column | Type | Unit / convention |
|---|---|---|
| `amount0` | `string`→`int` | **signed** `int256`, raw token units. **Positive = flowed INTO the pool.** |
| `amount1` | `string`→`int` | signed `int256`, raw token units, same convention |
| `sqrt_price_x96` | `string`→`int` | `uint160`, pool sqrt price **after** the swap |
| `liquidity` | `string`→`int` | `uint128`, active liquidity **after** the swap |
| `tick` | `int32` | pool tick **after** the swap |
| `sender` | `string` | lowercase address |
| `recipient` | `string` | lowercase address |

`amount0`/`amount1` always have opposite signs for a real swap. The fee is taken on the **input**
(positive) side: `fee_paid = amount_in * fee_tier / 1_000_000`, in the *input* token. Never assume the
input token.

### 4.2 `mint` / `burn`

| Column | Type | Notes |
|---|---|---|
| `owner` | `string` | for retail positions this is the NFT position manager, not the human |
| `tick_lower` | `int32` | on the spacing grid |
| `tick_upper` | `int32` | on the spacing grid, `> tick_lower` |
| `liquidity_amount` | `string`→`int` | `uint128`; the event's `amount` field — the position's `L` |
| `amount0` | `string`→`int` | `uint256`, **unsigned**, raw units, token0 moved |
| `amount1` | `string`→`int` | `uint256`, unsigned, raw units |
| `sender` | `string` | **nullable**: the `Mint` event's non-indexed `sender`; **null** on `burn` rows (the `Burn` event has no `sender`) |

`burn.liquidity_amount` is unsigned in the event; the sign convention for *liquidity delta* is applied in
T11, not stored here. Do not negate it in the fetcher.

**Null, not `""`.** Inapplicable fields are null in every stream and in the tape — an absence is not an
empty string and not a zero. T11's rule (§6.2) is the same rule; there is deliberately no sentinel-value
exception anywhere in this contract.

### 4.3 `collect`

`owner`, `recipient`, `tick_lower`, `tick_upper`, `amount0` (`uint128`), `amount1` (`uint128`).
Semantics note that T13 depends on: `Collect.amount0/1` are the amounts **withdrawn**, which include
fees that accrued *before* this window. A `Burn` immediately followed by a `Collect` in the same tx
withdraws principal + fees together — reconciliation must use the `Burn(amount0,amount1)` to separate
principal from fee. This is why T13's tolerance is defined on *fee* not *total*.

### 4.3.1 `flash`

`Flash(address indexed sender, address indexed recipient, uint256 amount0, uint256 amount1,
uint256 paid0, uint256 paid1)`. Columns beyond the key set: `sender`, `recipient`, `amount0`,
`amount1`, `paid0`, `paid1` (all `string`→`int`, unsigned raw units).

Why it is in scope at all: a flash swap's `paid` amounts exceed the borrowed `amount`, and the
difference **is fee growth** — it moves `feeGrowthGlobal` without any liquidity change. T10's replay
misses that fee source if this stream is absent, which shows up as a small negative bias in
reconstructed fees. Low volume on the pinned pools, so if T06 finds zero flash events in the window it
may ship an empty table — but the schema and the registry key exist either way, because T09 and T14
index `SCHEMA_REGISTRY[stream]` for every stream they touch.

### 4.4 `fee_growth` (T06 fetches, T10 consumes)

**The stream key is `fee_growth`** — that exact string in `SCHEMA_REGISTRY`, `SORT_KEYS`,
`POOL_SCOPED`, `FetchRequest.stream`, the on-disk `stream=` partition, and `Dataset.fee_growth`. The
schema constant is `FEE_GROWTH_SCHEMA`. There is no `fee_growth_snapshot` key anywhere.

One row per `(block_number, tick)` observation, plus global rows.

| Column | Type | Notes |
|---|---|---|
| `block_number` | `int64` | the block the `eth_call` was made **at** (state after the block) |
| `tick` | `int32` | the tick whose `Tick` struct was read; the sentinel `GLOBAL_TICK_SENTINEL` (defined in `config.py` as the int32 minimum) marks a global-only row |
| `fee_growth_outside_0_x128` | `string`→`int` | `uint256`, Q128; null on global rows |
| `fee_growth_outside_1_x128` | `string`→`int` | `uint256`, Q128; null on global rows |
| `liquidity_gross` | `string`→`int` | `uint128` |
| `liquidity_net` | `string`→`int` | **signed** `int128` |
| `initialized` | `bool` | tick struct initialized flag |
| `fee_growth_global_0_x128` | `string`→`int` | Q128; present on **every** row (denormalized) |
| `fee_growth_global_1_x128` | `string`→`int` | Q128; present on every row |
| `current_tick` | `int32` | pool `slot0.tick` at that block — decides the (4)/(5) branch |
| `current_liquidity` | `string`→`int` | pool `liquidity()` at that block |
| `source` | `string` | `"rpc_call"` (exact) or `"interpolated"` (flagged, never silently) |
| `pool_address` | `string` | lowercase; pool-scoped stream |

### 4.5 `gas` (T07) — complete column list

| Column | Type | Notes |
|---|---|---|
| `block_number` | `int64` | one row **per block** in the window — no gaps allowed; the sort key |
| `block_timestamp` | `timestamp[us,UTC]` | |
| `base_fee_per_gas` | `string`→`int` | wei, EIP-1559 |
| `gas_used` | `int64` | block gas used |
| `gas_limit` | `int64` | |
| `priority_fee_p50_wei` | `string`→`int` | median effective priority fee of txs in the block |
| `priority_fee_p90_wei` | `string`→`int` | 90th percentile — the "get included during a spike" cost |
| `eth_usd_price` | `float64` | **nullable, always null as written by T07.** Populated by T11 (§6.2) via a backward as-of join from `reference.close`, so that a gas cost can be expressed in USD at the block's prevailing price. T07 does not fetch prices. |

No `log_index`, `tx_hash`, `pool_address` or `event_type`: `gas` is a **non-log, chain-wide** stream
(§4.0). It is not partitioned per pool — both pinned pools share one gas table.

Gas cost of an action at block `n`: `(base_fee_per_gas[n] + priority_fee_p50_wei[n]) * gas_units`.
Gas-unit constants for mint/burn/collect live in `config.py`, not hardcoded in transforms.

### 4.6 `reference` (T08) — complete column list

| Column | Type | Notes |
|---|---|---|
| `open_time` | `timestamp[us,UTC]` | kline open, **left-closed** interval |
| `close_time` | `timestamp[us,UTC]` | kline close |
| `open`,`high`,`low`,`close` | `float64` | price of ETH in USD-stable terms |
| `volume_base` | `float64` | ETH volume |
| `quote_volume` | `float64` | |
| `trades` | `int64` | |
| `symbol` | `string` | `"ETHUSDT"` / `"ETHUSDC"` |
| `is_gap_filled` | `bool` | true if the bar was forward-filled over an exchange outage |

No block columns at all — an exchange bar has no block. Sort key `(symbol, open_time)`; **global**, not
pool-partitioned. Rule: gaps are **forward-filled and flagged**, never interpolated, never dropped
silently.

### 4.7 `regime` (T12) — complete column list

| Column | Type | Notes |
|---|---|---|
| `timestamp` | `timestamp[us,UTC]` | end of the backward-looking window (the label applies *at* this time) |
| `sigma_rv` | `float64` | annualized realized vol over the lookback |
| `mu` | `float64` | total log return over the lookback (see §6) |
| `regime` | `string` (`Regime`) | label |
| `window_complete` | `bool` | false ⇒ `regime == "unknown"` |
| `symbol` | `string` | which reference symbol the labels were computed from |

No block columns; sort key `(timestamp,)`; **global**, not pool-partitioned.

### 4.8 `event_tape` (T11) — the handoff artifact

**Column order is contractual** (§4 makes `validate_table(strict=True)` order-sensitive), so the union is
enumerated here in full rather than left to T11's discretion. Three blocks, in this order:

1. **The six key columns** of §4.0.1, in that order.
2. **The union of log-stream payload columns**, in this exact order — null where inapplicable to the row's
   `event_type`:
   `amount0`, `amount1`, `sqrt_price_x96`, `liquidity`, `tick`, `sender`, `recipient`,
   `owner`, `tick_lower`, `tick_upper`, `liquidity_amount`
   (note `amount0`/`amount1` are shared: **signed** on swap rows, non-negative on mint/burn/collect rows
   — the `event_type` discriminator tells a consumer which convention applies, and T13's
   `check_swap_sign_convention` only inspects swap rows.)
3. **The derived columns**, in this exact order:

| Column | Type | Notes |
|---|---|---|
| `seq` | `int64` | 0..N−1, dense, the canonical step index |
| `price_pool` | `float64` | derived from `sqrt_price_x96`, decimals-adjusted (convenience only) |
| `price_reference` | `float64` | **as-of** join: last reference close with `close_time <= block_timestamp` |
| `regime` | `string` | as-of join from the regime table |
| `base_fee_per_gas` | `string`→`int` | joined from `gas` on `block_number` |
| `priority_fee_p50_wei` | `string`→`int` | joined from `gas` |
| `fee_growth_global_0_x128` | `string`→`int` | forward-filled from nearest **prior** snapshot; `fee_growth_source` marks it |
| `fee_growth_global_1_x128` | `string`→`int` | same |
| `fee_growth_source` | `string` | `"exact"` \| `"stale_prior"` — never `"interpolated_future"` |

**Every as-of join is backward-only.** `pandas.merge_asof(direction="backward")` /
`polars.join_asof(strategy="backward")`. A forward or nearest join is look-ahead bias and is a
**Critical** review finding.

### 4.9 Schema API

```python
SWAP_SCHEMA: pa.Schema
MINT_SCHEMA: pa.Schema
BURN_SCHEMA: pa.Schema
COLLECT_SCHEMA: pa.Schema
FLASH_SCHEMA: pa.Schema                      # see §4.3.1
FEE_GROWTH_SCHEMA: pa.Schema
GAS_SCHEMA: pa.Schema
REFERENCE_SCHEMA: pa.Schema
REGIME_SCHEMA: pa.Schema
EVENT_TAPE_SCHEMA: pa.Schema

SCHEMA_REGISTRY: dict[str, pa.Schema]        # stream name -> schema; keys are exactly:
                                             #   swap, mint, burn, collect, flash,
                                             #   fee_growth, gas, reference, regime, event_tape
SORT_KEYS: dict[str, tuple[str, ...]]        # stream name -> sort key columns (§4.0)
POOL_SCOPED: frozenset[str]                  # streams partitioned by pool_address (§4.0)
SCHEMA_VERSION: str = "1.0.0"                # bumped on any change; written into the manifest

def validate_table(table: pa.Table, schema: pa.Schema, *, strict: bool = True) -> None:
    """Raise SchemaViolationError listing every mismatch (missing, extra, wrong type, wrong order)."""

def encode_uint(value: int) -> str: ...
def decode_uint(value: str) -> int: ...
def empty_table(schema: pa.Schema) -> pa.Table: ...
```

---

## 5. Fetcher interfaces

### 5.1 Base (T04)

```python
@dataclass(frozen=True, slots=True)
class FetchRequest:
    stream: str                 # "swap" | "mint" | ... | "gas" | "reference"
    pool: PoolConfig | None
    start_block: BlockNumber | None
    end_block: BlockNumber | None
    start_utc: datetime | None = None
    end_utc: datetime | None = None

@dataclass(frozen=True, slots=True)
class FetchResult:
    table: pa.Table             # conforms to SCHEMA_REGISTRY[request.stream]
    route: Route
    request: FetchRequest
    n_requests: int             # network calls actually made (0 ⇒ full cache hit)
    from_cache: bool
    warnings: tuple[str, ...]

# ---------------------------------------------------------------------------
# Layering rule (why T04 can be built in the same wave as T03):
#   BaseHttpFetcher knows HTTP, block ranges, retries and bytes. It does NOT
#   import schemas.py and does NOT validate tables. Its template method returns
#   raw rows; each concrete fetcher (T05-T08) owns `_rows_to_table` and calls
#   `schemas.validate_table` itself before wrapping the result.
#   So T04 depends only on types.py/config.py, and the SCHEMA_REGISTRY reference
#   in FetchResult above is a documentation constraint honored by subclasses.
# ---------------------------------------------------------------------------

class Fetcher(Protocol):
    route: Route
    def supported_streams(self) -> frozenset[str]: ...
    def fetch(self, request: FetchRequest) -> FetchResult: ...

class BaseHttpFetcher:
    """Provides: session with pooling, exponential backoff + jitter on RateLimitError /
    TransientNetworkError (max_retries from EndpointConfig), block-range chunking with adaptive
    shrink on 'query returned too many results', and an on-disk raw-response cache keyed by a
    deterministic hash of (route, stream, params).

    Imports nothing from schemas.py (see the layering rule above). Subclasses implement:
      _fetch_chunk(request, start, end) -> list[dict]     # raw decoded rows
      _rows_to_table(rows, stream) -> pa.Table            # + validate_table, in the subclass
    """

    def chunks(self, start: int, end: int, size: int) -> Iterator[tuple[int, int]]: ...
    def cache_key(self, route: Route, stream: str, params: Mapping[str, object]) -> str: ...
    def fetch_rows(self, request: FetchRequest) -> tuple[list[dict], int, bool, list[str]]:
        """(rows, n_requests, from_cache, warnings). The schema-free core of fetch()."""
```

Retry policy is a contract: retry only `RateLimitError` and `TransientNetworkError`; never retry
`PermanentFetchError`; cap total wall-clock per request; surface the final failure with the request
params (minus secrets) in the message.

### 5.2 The Graph (T05) and RPC (T06)

```python
class TheGraphFetcher(BaseHttpFetcher):
    route = Route.THEGRAPH
    def supported_streams(self) -> frozenset[str]:   # {"swap","mint","burn","collect"}
    def fetch(self, request: FetchRequest) -> FetchResult: ...

class RpcFetcher(BaseHttpFetcher):
    route = Route.RPC
    def supported_streams(self) -> frozenset[str]:   # {"swap","mint","burn","collect","flash","fee_growth"}
    def fetch(self, request: FetchRequest) -> FetchResult: ...
    def fee_growth_at(self, pool: PoolConfig, block: BlockNumber,
                      ticks: Sequence[int]) -> pa.Table: ...   # FEE_GROWTH_SCHEMA
```

Both must return **identical** `swap`/`mint`/`burn`/`collect` tables for the same block range — that is
literally T13's blocking test. Pagination for the subgraph uses `(block_number, log_index)` cursor
pagination, **not** `skip` (the hosted service caps `skip` at 5000 and silently drops rows).

### 5.3 Gas (T07) and Reference (T08)

```python
class GasFetcher(BaseHttpFetcher):
    route = Route.RPC
    def fetch(self, request: FetchRequest) -> FetchResult: ...
    def block_for_timestamp(self, ts: datetime) -> BlockNumber: ...   # binary search, cached
    def timestamp_for_block(self, block: BlockNumber) -> datetime: ...

class ReferenceFetcher(BaseHttpFetcher):
    route = Route.REFERENCE
    def fetch(self, request: FetchRequest) -> FetchResult: ...
```

`block_for_timestamp` is what turns `WindowConfig`'s dates into blocks; it must return the **last block
with `timestamp <= ts`** and be exact at boundaries.

---

## 6. Transforms

### 6.1 Fee growth (T10)

```python
def fee_growth_below(current_tick: int, tick_lower: int, g_global: int, g_out_lower: int) -> int:
    """eq (4). Integer, wrapping."""

def fee_growth_above(current_tick: int, tick_upper: int, g_global: int, g_out_upper: int) -> int:
    """eq (5). Integer, wrapping."""

def fee_growth_inside(current_tick: int, tick_lower: int, tick_upper: int,
                      g_global: int, g_out_lower: int, g_out_upper: int) -> int:
    """eq (3). Integer, wrapping."""

def uncollected_fees(liquidity: int, g_inside_now: int, g_inside_last: int) -> int:
    """eq (6): liquidity * wrapping_sub_256(now, last) >> 128, floor. Raw token units, integer."""

@dataclass(frozen=True, slots=True)
class PositionKey:
    owner: Address
    tick_lower: int
    tick_upper: int

@dataclass(frozen=True, slots=True)
class FeeAccrual:
    fees0: int            # raw token0 units
    fees1: int
    g_inside_0_last: int  # new snapshot, Q128
    g_inside_1_last: int
    block_number: BlockNumber
    exact: bool           # False if any input fee-growth value was interpolated

@dataclass(frozen=True, slots=True)
class TickState:
    fee_growth_outside_0_x128: int
    fee_growth_outside_1_x128: int
    liquidity_gross: int
    liquidity_net: int            # signed
    initialized: bool

@dataclass(frozen=True, slots=True)
class FeeGrowthState:
    """Serializable tracker state — T14 checkpoints this to resume a long replay."""
    block_number: BlockNumber
    fee_growth_global_0_x128: int
    fee_growth_global_1_x128: int
    current_tick: int
    current_liquidity: int
    ticks: Mapping[int, TickState]

@dataclass(frozen=True, slots=True)
class Mismatch:
    block_number: BlockNumber
    tick: int | None              # None for a global-accumulator mismatch
    field_name: str
    replayed: int
    observed: int
    abs_delta: int
    rel_delta: float

@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    """Returned by FeeGrowthTracker.reconcile; rendered by T13. Reports, never raises."""
    n_compared: int
    n_exact: int
    max_abs_delta_g0: int
    max_abs_delta_g1: int
    max_rel_delta: float
    mismatches: tuple[Mismatch, ...]

class FeeGrowthTracker:
    """Replays swaps + tick crossings to maintain feeGrowthGlobal and per-tick feeGrowthOutside,
    so fee accrual can be evaluated at any block without an eth_call per block."""
    def __init__(self, pool: PoolConfig, initial: FeeGrowthState) -> None: ...
    def apply_swap(self, row: Mapping[str, object]) -> None: ...
    def apply_liquidity_event(self, row: Mapping[str, object]) -> None: ...
    def cross_tick(self, tick: int, upward: bool) -> None: ...
    def accrue(self, key: PositionKey, liquidity: int,
               g_inside_0_last: int, g_inside_1_last: int) -> FeeAccrual: ...
    def snapshot(self) -> FeeGrowthState: ...
    def reconcile(self, observed: pa.Table) -> ReconciliationReport: ...
```

Semantics that MUST hold (tests): a range entirely below the current tick accrues **zero new** interior
growth as the global accumulator grows (roadmap Q3); `fee_growth_inside` of `[MIN_TICK, MAX_TICK]`
equals `fee_growth_global`; all arithmetic wraps mod 2^256; a position at exactly `current_tick ==
tick_lower` is **in range** and at `current_tick == tick_upper` is **out of range** (protocol
half-open `[lower, upper)` convention).

### 6.2 Alignment (T11)

```python
@dataclass(frozen=True, slots=True)
class Dataset:
    """The handoff artifact. Fields are Optional so T16's `load_dataset(streams=[...])`
    fast path can return a partial dataset; a field the caller did not request is None
    (NOT an empty table — absent and empty must stay distinguishable)."""
    tape: pa.Table | None
    gas: pa.Table | None
    reference: pa.Table | None
    regime: pa.Table | None
    fee_growth: pa.Table | None
    manifest: DatasetManifest
    # Per-route samples over a small verification window, so T13's crosscheck_routes has
    # both routes' tables to compare. Keyed (route, stream) -> table; empty when no
    # verification window was pulled. Populated by T14's route policy, not by T11's merge.
    route_samples: Mapping[tuple[str, str], pa.Table] = field(default_factory=dict)

    def require(self, name: str) -> pa.Table:
        """Return a stream or raise ValidationError naming the stream and the pull command.
        Use this instead of asserting non-None at every call site."""

def build_event_tape(streams: Mapping[str, pa.Table], *, pool: PoolConfig,
                     gas: pa.Table, reference: pa.Table, regime: pa.Table,
                     fee_growth: pa.Table) -> pa.Table: ...

def assert_no_lookahead(table: pa.Table, time_col: str = "block_timestamp") -> None:
    """Raise ValidationError if any as-of-joined column could only be known from a later row."""
```

Intra-block ordering when `log_index` ties are impossible (they aren't within a block, but across
sources they are): the merge key is `(block_number, log_index)` and it must be **unique** in the tape.
Duplicates ⇒ `ValidationError`, never a silent `drop_duplicates`.

### 6.3 Regimes (T12)

```python
def realized_volatility(closes: np.ndarray, periods_per_year: int) -> float:
    """std of log returns * sqrt(periods_per_year). Sample std (ddof=1)."""

def window_drift(closes: np.ndarray) -> float:
    """Total log return over the window: log(closes[-1] / closes[0])."""

def label_regime(sigma_rv: float, mu: float, cfg: RegimeConfig) -> Regime: ...

def label_series(reference: pa.Table, cfg: RegimeConfig) -> pa.Table:
    """REGIME_SCHEMA. Backward-looking rolling windows only. Rows whose window is incomplete get
    Regime.UNKNOWN with window_complete=False — they are NOT dropped."""

def sensitivity_sweep(reference: pa.Table, cfg: RegimeConfig,
                      vol_grid: Sequence[float], drift_grid: Sequence[float]) -> pa.Table:
    """Regime-share table over the threshold grid — the §10.3 pre-commitment sweep."""
```

**Why `mu` is the total window log return, not the per-step mean:** §10.3 writes
`μ = (1/T) Σ log(S_t/S_{t−1})`, i.e. the *mean per-minute* log return, then thresholds it at ±5%. A
mean per-minute return of 5% is a ~10^900× move over 30 days — the threshold and the statistic are
dimensionally inconsistent. `μ₃₀ = Σ log-returns = log(S_end/S_start)` is the quantity ±5% is meant
for. This is recorded in `adr/001-regime-drift-definition.md`; the thesis text should be corrected to
match.

---

## 7. Storage (T09)

```python
@dataclass(frozen=True, slots=True)
class DatasetManifest:
    schema_version: str
    dataset_id: str                    # deterministic hash of (pool, window, route set, schema_version)
    pool: PoolConfig
    window: WindowConfig
    streams: dict[str, StreamManifest] # per-stream row count, block range, content hash, route
    created_at_utc: datetime           # wall clock lives HERE, not in the data
    git_commit: str
    library_versions: dict[str, str]
    warnings: tuple[str, ...]

def write_stream(table: pa.Table, root: Path, stream: str, *, pool: PoolConfig) -> StreamManifest:
    """Partition by (pool_address, stream, block_bucket=block_number // 100_000).
    Sort by (block_number, log_index) before writing. Deterministic bytes."""

def read_stream(root: Path, stream: str, *, pool: PoolConfig,
                start_block: BlockNumber | None = None,
                end_block: BlockNumber | None = None) -> pa.Table:
    """Predicate-pushdown on block_number. Returns schema-validated table."""

def write_manifest(manifest: DatasetManifest, root: Path) -> Path: ...
def read_manifest(root: Path) -> DatasetManifest: ...
def content_hash(table: pa.Table) -> str: ...     # stable across runs and machines
```

---

## 8. Validation (T13)

`CheckResult` and `Severity` are **defined in `types.py` (§1, T01)**, not here — T08 needs them in wave 3.
T13 imports them: `from undertow.data.types import CheckResult, Severity`.

```python
def run_all_checks(dataset: Dataset) -> list[CheckResult]:
    """Run every check the dataset has the inputs for.

    Checks whose inputs are absent (e.g. crosscheck_routes with no route samples in
    Dataset.route_samples, or the Dune comparison when Dune was unavailable) return an
    `info` CheckResult saying they were skipped and why — never a silent omission and
    never a spurious failure.
    """
def render_report(results: Sequence[CheckResult], path: Path) -> None: ...   # markdown

# individual checks (each independently testable)
def check_block_ordering(table: pa.Table) -> CheckResult: ...
def check_key_uniqueness(table: pa.Table) -> CheckResult: ...
def check_no_block_gaps(gas: pa.Table, window: WindowConfig) -> CheckResult: ...
def check_swap_sign_convention(swaps: pa.Table) -> CheckResult: ...
def check_tick_price_consistency(swaps: pa.Table, pool: PoolConfig) -> CheckResult: ...
def check_ticks_on_spacing_grid(mints: pa.Table, burns: pa.Table, pool: PoolConfig) -> CheckResult: ...
def check_liquidity_conservation(swaps: pa.Table, mints: pa.Table, burns: pa.Table,
                                fee_growth: pa.Table, start_block: BlockNumber) -> CheckResult:
    """Delta-based, seeded from fee_growth's current_liquidity at/just before start_block.

    Positions minted before start_block are invisible in a windowed dataset, so absolute
    liquidity can never be reproduced from in-window events alone — only CHANGES can.
    Severity: warning. See T13's brief.
    """
def check_reference_coverage(reference: pa.Table, window: WindowConfig) -> CheckResult: ...
def check_regime_labels_complete(regime: pa.Table) -> CheckResult: ...

def crosscheck_routes(graph: pa.Table, rpc: pa.Table, stream: str) -> CheckResult:
    """Field-level equality. Any mismatch is critical."""

def reconcile_fees_against_collect(tape: pa.Table, fee_growth: pa.Table,
                                  pool: PoolConfig, *, rel_tolerance: float = 1e-6) -> CheckResult:
    """THE acceptance test for §10.2.4: reconstruct uncollected fees for real positions and compare
    against actual Collect amounts (principal separated out via the paired Burn)."""
```

`run_all_checks` never raises on a failed check — it returns results. Only the CLI decides to exit
non-zero (on any `critical`).

---

## 9. CLI (T14) and public API (T16)

```
undertow-data pull     --config PATH [--stream S ...] [--route thegraph|rpc] [--force]
undertow-data verify   --config PATH [--sample-windows N] [--fail-on warning|critical]
undertow-data snapshot --config PATH [--out DIR]        # pull + build tape + verify + manifest
undertow-data info     --config PATH                    # manifest summary, row counts, coverage
```

Exit codes: `0` ok, `1` critical check failed, `2` config error, `3` fetch error (after retries).
All commands are **idempotent and resumable**; `pull` re-run with a warm cache makes zero network calls
and must print `n_requests=0`.

```python
# undertow/data/__init__.py — the only surface undertow.sim may use
__all__ = [
    "load_dataset", "Dataset", "DataConfig", "load_config", "default_pools",
    "PoolConfig", "WindowConfig", "RegimeConfig", "EventType", "FeeTier", "Regime",
    "SCHEMA_REGISTRY", "SCHEMA_VERSION", "DatasetManifest",
    "UndertowDataError", "ConfigError", "FetchError", "ValidationError",
]

def load_dataset(config: DataConfig | str | Path, *,
                 streams: Sequence[str] | None = None,
                 validate: bool = True) -> Dataset:
    """Load a cached snapshot from disk. Never hits the network — that is `pull`'s job.
    Raises ValidationError if validate=True and a critical check fails."""
```

`undertow.data` imports nothing from `undertow.sim`. Ever. The reviewer checks this.

---

## 10. Shared test fixtures (who creates what)

**The ownership rule: a fixture is owned by the EARLIEST task that needs it.** A consumer in an earlier
wave than the producer cannot wait for the producer's fixture, so it owns a minimal one of its own. Two
small overlapping fixtures beat one shared fixture that does not exist yet.

| Fixture path | Owned by | Also used by | Note |
|---|---|---|---|
| `tests/data/fixtures/tickmath_vectors.json` | T02 (w1) | T06, T11, T13 | transcribed from v3-core test vectors |
| `tests/data/fixtures/feegrowth_replay.json` | **T10 (w2)** | T13 | T10's own ~20-event hand-built replay sequence **and** a handful of synthetic `FEE_GROWTH_SCHEMA` rows. T10 must NOT wait for T06's fixture (wave 3) — it writes its own to the schema in §4.4. |
| `tests/data/fixtures/graph_swaps_page.json` | T05 (w3) | T13 | |
| `tests/data/fixtures/graph_mints_page.json` | T05 (w3) | T13 | |
| `tests/data/fixtures/graph_burns_page.json` | T05 (w3) | T13 | |
| `tests/data/fixtures/graph_collects_page.json` | T05 (w3) | T13 | only if the chosen subgraph indexes `Collect` |
| `tests/data/fixtures/rpc_logs_{swap,mint,burn,collect}.json` | T06 (w3) | T13 | must cover the SAME block range as T05's pages so `crosscheck_routes` is testable offline |
| `tests/data/fixtures/rpc_ticks_call.json` | T06 (w3) | T13 | real `ticks()`/`slot0()` returns |
| `tests/data/fixtures/collect_reconciliation/*.json` | **T06 (w3)** | T13 (w5) | T06 owns the raw capture: 3 real position lifecycles (mint → swaps → burn+collect) from the pinned pool, incl. one that exits range. T13 adds only `cases.json` describing expected values. Listed in `PLAN.md` §2.1 as shared. |
| `tests/data/fixtures/blocks_gas.json` | T07 (w3) | T11, T13 | incl. one 20× base-fee spike block |
| `tests/data/fixtures/binance_klines_api.json` | T08 (w3) | T11 | REST-shaped payload |
| `tests/data/fixtures/binance_klines_bulk.zip` | T08 (w3) | T11 | one small bulk-archive CSV zip, for the dual-path test |
| `tests/data/fixtures/reference_series.json` | **T12 (w3)** | — | T12's OWN minimal series: a flat-then-jump series for the look-ahead test, a crash series, and a gap-filled series. T12 must NOT depend on T08's fixture (same wave). |
| `tests/data/fixtures/dune_expected_magnitudes.json` | T15 (w3) | T13 | may be `status: unavailable` |
| `tests/data/conftest.py` — `tiny_dataset()` | T11 (w4) | T13, T14, T16 | the shared end-to-end factory |

Fixtures are **small** (≤ a few hundred rows), hand-trimmed from real responses, and committed. Each must
contain at least one awkward case: a negative tick where applicable, a tick crossing, an out-of-range
position, an empty page, and a block with no swaps.

**Real values, synthetic envelopes.** Field *values* must come from real responses — never invent an
`amount0`. But *envelopes* may be synthesized where the shape is what is under test: T05's >1000-row
pagination test may replicate real rows across synthetic pages, and T04's tests may use entirely fake
payloads (it does not know what a swap is). Say which you did in the fixture's header comment.
