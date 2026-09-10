# CONTRACTS.md — frozen cross-task interfaces for `undertow.sim`

**Status: FROZEN.** Every signature, dataclass field, protocol method, unit, sign convention and
enumerated value below is a contract between tasks. Implement exactly this. You may *add* (a new
keyword-only argument with a default, a new helper, a new enum member with a default for existing
match statements); you may not *rename*, *retype*, *reorder* or *remove*. If something here is
genuinely wrong, write `docs/decisions/NNN-<slug>.md`, flag it in your PR, and notify the
orchestrator — do not unilaterally change it, because other agents are already coding against it.

---

## 0. Notation & global conventions

- **Numeraire**: token0 = USDC. Wealth, PnL, fees, costs are in USDC units. Gas is computed in wei and
  converted at that block's reference ETH price. Prices follow the data module's orientation:
  **token1 per token0 = USDC per WETH** (~2000–4000 across the study window). See data
  `CONTRACTS.md` §3.0.
- **Timestamps**: `datetime` with `tzinfo=timezone.utc`. Sim time is indexed by `seq` (tape step
  index) or bar index; wall-clock never enters the state.
- **Determinism**: every stochastic component takes an explicit `numpy.random.Generator`. No
  `np.random.*` module-level calls, no global seeding. Same config + same seed ⇒ bit-identical.
- **`from __future__ import annotations`** in every file.
- **Type hints** on all public functions. `numpy` for array math, `polars` or `pyarrow` for tabular
  access.
- **Two numeric worlds**: simulator is `float64` end-to-end; backtester's fee path uses exact integer
  math via the data module's `FeeGrowthTracker` (Q128 ints). S14 quantifies the drift.
- **No look-ahead**: an observation at step *t* is built only from data at blocks/bars ≤ *t*. All
  rolling features are backward-looking, closed on the right.
- **Cost honesty**: every reported PnL/equity number is net of gas and slippage. Gross numbers exist
  only inside `RewardBreakdown` and the decomposition ledger, explicitly labelled.
- **Import boundary**: `undertow.sim` imports only names exported from `undertow.data`'s top-level
  `__init__.py`. The string `from undertow.data.` (with trailing dot) must never appear in
  `src/undertow/sim/`.

---

## 1. `undertow.sim.types` (S01) — enums, aliases, exception hierarchy

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import NewType, Literal

# -- Domain aliases --
Tick = NewType("Tick", int)
TickSpacing = NewType("TickSpacing", int)
SqrtPrice = float                # sqrt(P) in human units; P = s^2
Price = float                    # token1 per token0 (USDC per WETH)
Wealth = float                   # USDC

# -- Regime labels (consumed from data module; never recomputed in sim) --
class Regime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"
    UNKNOWN = "unknown"

# -- Price process mode --
class PriceMode(str, Enum):
    REPLAY = "replay"
    CALIBRATED = "calibrated"

# -- Action space --
# The factored discrete action: {hold} ∪ {rebalance(center_offset, width)}.
# center_offset: signed tick-spacings around current tick, ∈ {−4,…,+4}
# width: tick-spacings per side, ∈ {1, 2, 5, 10, 25, 50}
# S10 defines the policy protocol; S01 owns the action types.

@dataclass(frozen=True, slots=True)
class Action:
    action_type: Literal["hold", "rebalance"]
    center_offset: int = 0       # tick-spacings from current tick; ignored for hold
    width: int = 0               # tick-spacings per side; ignored for hold

    def is_hold(self) -> bool: ...
    def lower_tick(self, current_tick: Tick, tick_spacing: TickSpacing) -> Tick: ...
    def upper_tick(self, current_tick: Tick, tick_spacing: TickSpacing) -> Tick: ...

# -- Exception hierarchy --
class UndertowSimError(Exception): ...
class SimConfigError(UndertowSimError): ...
class MarketViewError(UndertowSimError): ...
class LookAheadError(MarketViewError): ...      # raised when a consumer tries to read beyond the wall
class PositionError(UndertowSimError): ...
class EnvError(UndertowSimError): ...
class BacktestError(UndertowSimError): ...
class ParityError(UndertowSimError): ...
```

---

## 2. `undertow.sim.config` (S01) — `SimConfig` tree

```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
import numpy as np

from undertow.sim.types import PriceMode, Regime

# -- Gas units per action type (source: representative NFT-position-manager costs) --
GAS_MINT: int = 460_000
GAS_BURN: int = 215_000
GAS_COLLECT: int = 130_000
GAS_REBALANCE_SWAP: int = 150_000  # approximate; sweepable

# -- Slippage defaults --
FIXED_IMPACT_BPS: float = 5.0     # basis points

@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    duration_days: int = 30
    step_minutes: int = 10         # decision cadence
    agent_capital_usdc: float = 100_000.0
    marginal_agent: bool = True    # agent liquidity never moves the price path
    fee_tier_bps: int = 3000       # 0.30% default for WETH/USDC
    tick_spacing: int = 60

@dataclass(frozen=True, slots=True)
class ActionGridConfig:
    center_offsets: Sequence[int] = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
    widths: Sequence[int] = (1, 2, 5, 10, 25, 50)
    include_hold: bool = True

@dataclass(frozen=True, slots=True)
class GasConfig:
    mint_units: int = GAS_MINT
    burn_units: int = GAS_BURN
    collect_units: int = GAS_COLLECT
    rebalance_swap_units: int = GAS_REBALANCE_SWAP

@dataclass(frozen=True, slots=True)
class SlippageConfig:
    fixed_impact_bps: float = FIXED_IMPACT_BPS

@dataclass(frozen=True, slots=True)
class RewardConfig:
    risk_penalty_lambda: float = 0.0                # default off; ablatable
    risk_rolling_window_steps: int = 144             # 1 day at 10-min steps
    normalize_by_capital: bool = True                # reward = ΔPnL_net / W_0
    # Ablation flags (RQ1 instrument)
    fees_enabled: bool = True
    gas_enabled: bool = True
    slippage_enabled: bool = True
    il_enabled: bool = True
    risk_penalty_enabled: bool = False

@dataclass(frozen=True, slots=True)
class SplitConfig:
    train_start_utc: datetime = datetime(2022, 1, 1, tzinfo=timezone.utc)
    train_end_utc: datetime = datetime(2023, 12, 31, tzinfo=timezone.utc)
    eval_start_utc: datetime = datetime(2024, 1, 1, tzinfo=timezone.utc)
    eval_end_utc: datetime = datetime(2024, 12, 31, tzinfo=timezone.utc)

@dataclass(frozen=True, slots=True)
class TrainingConfig:
    algorithm: Literal["ppo"] = "ppo"
    seeds: Sequence[int] = (0, 1, 2, 3, 4)
    discount_gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.15
    hidden_size: int = 256
    n_hidden: int = 2
    parallel_envs: int = 4
    total_timesteps: int = 1_000_000
    checkpoint_freq_steps: int = 100_000
    log_freq_steps: int = 10_000

@dataclass(frozen=True, slots=True)
class BacktestConfig:
    compute_decomposition: bool = True
    log_decision_every_step: bool = False

@dataclass(frozen=True, slots=True)
class SimConfig:
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    action_grid: ActionGridConfig = field(default_factory=ActionGridConfig)
    gas: GasConfig = field(default_factory=GasConfig)
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    price_mode: PriceMode = PriceMode.REPLAY
    seed: int = 0
    output_dir: Path | None = None

    def rng(self) -> np.random.Generator: ...  # constructs np.random.default_rng(self.seed)

def load_sim_config(path: str | Path) -> SimConfig: ...
```

---

## 3. `undertow.data` public API — what the sim consumes (S02)

S02 extends `undertow/data/__init__.py` to export exactly the names below. The sim imports from
`undertow.data` only.

```python
# Re-exported from undertow.data.types:
from undertow.data.types import (
    Address, BlockNumber, Tick as DataTick, FeeTier, Regime as DataRegime,
    EventType, CheckResult, Severity,
)

# Re-exported from undertow.data.config:
from undertow.data.config import (
    DataConfig, PoolConfig, WindowConfig, RegimeConfig, EndpointConfig,
    load_config as load_data_config, default_pools,
)

# Re-exported from undertow.data.schemas:
from undertow.data.schemas import (
    SWAP_SCHEMA, MINT_SCHEMA, BURN_SCHEMA, COLLECT_SCHEMA, FLASH_SCHEMA,
    FEE_GROWTH_SCHEMA, GAS_SCHEMA, REFERENCE_SCHEMA, REGIME_SCHEMA, EVENT_TAPE_SCHEMA,
    SCHEMA_REGISTRY, SCHEMA_VERSION, SORT_KEYS,
    encode_uint, decode_uint, validate_table, empty_table,
)

# Re-exported from undertow.data.fixedpoint:
from undertow.data.fixedpoint import (
    Q96, Q128, TICK_BASE, MIN_TICK, MAX_TICK,
    sqrt_price_x96_to_price, price_to_sqrt_price_x96,
    price_to_tick, tick_to_price, tick_to_sqrt_price,
)

# New exports (S02 writes these adapters in loader.py or directly in __init__.py):
from undertow.data.loader import (
    load_dataset,                     # DataConfig → Dataset (with validate + streams kwargs)
    load_tape,                        # DataConfig → pyarrow.Table of EVENT_TAPE_SCHEMA, sorted
    load_reference_feed,              # DataConfig → pyarrow.Table of REFERENCE_SCHEMA, sorted
    load_gas_feed,                    # DataConfig → pyarrow.Table of GAS_SCHEMA, sorted
    load_regime_labels,               # DataConfig → pyarrow.Table of REGIME_SCHEMA, sorted
)
```

The sim **never** imports from `undertow.data.fetchers.*`, `undertow.data.transforms.*`,
`undertow.data.storage.*`, or `undertow.data.validation.*`.

`FeeGrowthTracker` is **not** exported by S02. If the backtester (S11) needs fee-growth math, it
consumes pre-computed fee-growth columns from `load_tape()` or imports the tracker via a deliberate,
ADR-documented contract extension — not an accident of a broad export.

---

## 4. `MarketView` interface (S03)

The read-only, look-ahead-safe window onto the dataset. Every consumer of historical data in the sim
goes through this.

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence
import polars as pl        # or pyarrow — S03 chooses; rest of sim follows

@dataclass(frozen=True, slots=True)
class MarketView:
    """A look-ahead-safe view of the train OR eval split. Never both."""

    # -- Splits (disjoint; exactly one is active) --
    train: pl.DataFrame | None        # train-split event tape, seq-sorted
    eval: pl.DataFrame | None         # eval-split event tape, seq-sorted
    reference: pl.DataFrame           # reference feed, sorted by open_time
    regimes: pl.DataFrame             # regime labels, sorted by timestamp
    gas: pl.DataFrame                 # gas feed, sorted by block_number

    # -- Pinned split boundaries (UTC) --
    train_start_utc: datetime
    train_end_utc: datetime
    eval_start_utc: datetime
    eval_end_utc: datetime
    is_train: bool                    # True if this view was built for training

    # -- Query methods --
    def slice(self, start_seq: int, end_seq: int) -> MarketView:
        """Return a sub-view over [start_seq, end_seq). Raises LookAheadError if
        `end_seq` extends beyond the active split's last seq."""
        ...

    def tape_at(self, seq: int) -> dict[str, object]:
        """All columns for the event at `seq` as a dict. Raises LookAheadError for
        out-of-split access."""
        ...

    def reference_close(self, at_time: datetime) -> float | None:
        """Last reference close with close_time <= at_time. None if no bar exists
        (before the feed starts)."""
        ...

    def reference_bars(self, start_time: datetime, end_time: datetime) -> pl.DataFrame:
        """Reference bars in [start_time, end_time). Look-ahead-safe: clamped to the
        active split's time bounds."""
        ...

    def regime_at(self, at_time: datetime) -> str:
        """Regime label at `at_time` (as-of join)."""
        ...

    def gas_at_block(self, block_number: int) -> dict[str, object] | None:
        """Gas row for `block_number`. None if no gas data for that block."""
        ...

    def active_split_data(self) -> pl.DataFrame:
        """The event tape for the active split (train or eval), sorted by seq."""
        ...

    def step_count(self) -> int:
        """Number of tape events in the active split."""
        ...

    def build_episode(self, seed: int, max_steps: int | None = None) -> tuple[int, int]:
        """Sample a contiguous episode window of the active split. Returns (start_seq,
        end_seq). Never crosses the split boundary. Episode length is config-driven
        (EpisodeConfig.duration_days → steps at step_minutes cadence)."""
        ...

def build_market_view(data_config: DataConfig, split_config: SplitConfig, *,
                      split: Literal["train", "eval"] = "train") -> MarketView:
    """Load data via `load_tape|load_reference_feed|load_gas_feed|load_regime_labels`,
    then partition by the split boundaries. Raises MarketViewError if the split window
    contains no data."""
    ...
```

### 4.1 MarketView invariants (enforced in `__post_init__`, tested by S03, probed by S14)

1. `train` and `eval` are never both non-None.
2. Every query method that takes a time or seq argument raises `LookAheadError` if the argument
   exceeds the active split's range.
3. `build_episode` never produces a window that straddles the train/eval boundary.
4. The `reference` feed is the *entire* feed (both train and eval windows); `reference_close` /
   `reference_bars` enforce the split wall for the calling split.

---

## 5. Position & valuation math (S04)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from undertow.sim.types import Tick, TickSpacing, SqrtPrice, Price, Wealth

@dataclass(frozen=True, slots=True)
class Position:
    """A concentrated-liquidity position on [tick_lower, tick_upper].

    Equations in comments refer to lesson plan §5.3–§5.6.
    Matches the Uniswap V3 NFT position model: liquidity L, tick bounds, fee tier.
    """
    tick_lower: Tick
    tick_upper: Tick
    liquidity: float                 # L — the invariant quantity
    tick_spacing: TickSpacing
    fee_growth_inside_last_0: float = 0.0   # snapshot for token0 fee accrual
    fee_growth_inside_last_1: float = 0.0   # snapshot for token1 fee accrual

    def __post_init__(self) -> None:
        if self.tick_lower >= self.tick_upper:
            raise PositionError("tick_lower must be < tick_upper")
        if self.liquidity <= 0:
            raise PositionError("liquidity must be > 0")

    # -- Token amounts (eqs 8–9 of §5.3) --
    def amount0(self, sqrt_price: SqrtPrice) -> float:
        """Token0 holdings at `sqrt_price`. Returns 0 if the position has no token0
        at this price (out of range on the token0 side)."""
        ...

    def amount1(self, sqrt_price: SqrtPrice) -> float:
        """Token1 holdings at `sqrt_price`."""
        ...

    # -- Value (eq 11 of §5.3.4) --
    def value(self, sqrt_price: SqrtPrice, price: Price) -> Wealth:
        """Position value in numeraire (token0 = USDC).
        V(s) = amount0 + amount1 * price."""
        ...

    # -- IL vs HODL (§5.6) --
    def hodl_value(self, entry_sqrt_price: SqrtPrice, entry_price: Price,
                   current_price: Price) -> Wealth:
        """What the initial deposit would be worth if simply held (no LPing).
        HODL = initial_token0 + initial_token1 * current_price."""
        ...

    def il_vs_hodl(self, entry_sqrt_price: SqrtPrice, entry_price: Price,
                   current_sqrt_price: SqrtPrice, current_price: Price) -> float:
        """Impermanent loss: V(current) - HODL(current). Negative = loss."""
        ...

    def il_fraction(self, entry_sqrt_price: SqrtPrice, entry_price: Price,
                    current_sqrt_price: SqrtPrice, current_price: Price) -> float:
        """IL as a fraction of HODL value: (V - HODL) / HODL."""
        ...

    # -- Fee accrual (§10.2.4 of roadmap) --
    def uncollected_fees(self, fee_growth_inside_0: float,
                         fee_growth_inside_1: float) -> tuple[float, float]:
        """Uncollected fees = L * (current_fee_growth_inside - last_snapshot).
        Returns (fees_token0, fees_token1). Token0 fees are in USDC units."""
        ...

    def snapshot_fees(self, fee_growth_inside_0: float,
                      fee_growth_inside_1: float) -> None:
        """Update fee_growth_inside_last to current (like a Collect without withdrawing)."""
        ...

    # -- Range predicates --
    def in_range(self, sqrt_price: SqrtPrice) -> bool: ...
    def below_range(self, sqrt_price: SqrtPrice) -> bool: ...
    def above_range(self, sqrt_price: SqrtPrice) -> bool: ...

def calc_sqrt_price_a(tick: Tick) -> SqrtPrice:
    """sqrt(1.0001^tick) — the sqrt price at a tick."""
    ...

def calc_sqrt_price_b(tick: Tick) -> SqrtPrice:
    """sqrt(1.0001^tick) — alias for clarity when the tick is the upper bound."""
    ...

def initial_deposit(sqrt_price: SqrtPrice, price: Price,
                    tick_lower: Tick, tick_upper: Tick,
                    capital: Wealth, tick_spacing: TickSpacing) -> Position:
    """Given a capital budget and a range, compute the Position (i.e. solve for L)
    such that V(sqrt_price_current, price_current) == capital.

    The position's L is chosen so the real token amounts at the current price exactly
    consume the capital budget. Returns the Position with fee-growth snapshots at 0."""
    ...

def position_from_amounts(tick_lower: Tick, tick_upper: Tick,
                          amount0: float, amount1: float,
                          sqrt_price: SqrtPrice,
                          tick_spacing: TickSpacing) -> Position:
    """Given raw token amounts and a range, compute L (eqs 8–9 inverted).
    Useful for reconstructing a position from on-chain data."""
    ...
```

---

## 6. Pool engine interface (S07)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from undertow.sim.types import Tick, SqrtPrice, Price

@dataclass
class TickState:
    """Per-tick state in the lattice."""
    fee_growth_outside_0: float     # Q128 as float64
    fee_growth_outside_1: float
    liquidity_gross: float          # total L locked at this tick
    liquidity_net: float            # signed: + for lower ticks, - for upper ticks
    initialized: bool

@dataclass
class PoolState:
    """The pool's mutable state at a single point in time."""
    sqrt_price: SqrtPrice
    tick: Tick
    liquidity: float                # active L at current tick
    fee_growth_global_0: float
    fee_growth_global_1: float
    fee_tier_bps: int               # e.g. 3000 for 0.30%
    tick_spacing: int

    # -- Fee-growth inside a range (eqs 4–6 of §10.2.4) --
    def fee_growth_below(self, tick_boundary: Tick,
                         fee_growth_outside: float) -> float: ...
    def fee_growth_above(self, tick_boundary: Tick,
                         fee_growth_outside: float) -> float: ...
    def fee_growth_inside(self, tick_lower: Tick, tick_upper: Tick,
                          fg_outside_lower_0: float, fg_outside_lower_1: float,
                          fg_outside_upper_0: float, fg_outside_upper_1: float
                          ) -> tuple[float, float]: ...
    def tick_cross(self, new_tick: Tick) -> None:
        """Cross ticks from current to new_tick, updating liquidity and flipping
        fee_growth_outside at each crossed tick boundary. This is the expensive
        operation — only called when the price actually crosses a tick."""
        ...

@dataclass
class PoolEngine:
    """Owns the tick lattice and provides the step interface.

    Two modes: SIM (float64, fast) and BACKTEST (exact int, via data module's
    FeeGrowthTracker). S07 implements the SIM path; S11 uses the BACKTEST path
    through the data module. S14 quantifies the difference.
    """
    state: PoolState
    ticks: dict[Tick, TickState]    # sparse — only initialized ticks

    # -- Position management --
    def open_position(self, tick_lower: Tick, tick_upper: Tick,
                      liquidity: float) -> int:
        """Register a new position. Returns a position_id. Snaps ticks to spacing."""
        ...

    def close_position(self, position_id: int) -> tuple[float, float]:
        """Remove a position, returning its accrued fees (token0, token1)."""
        ...

    def position_fees(self, position_id: int) -> tuple[float, float]:
        """Current uncollected fees for a position."""
        ...

    # -- Core step --
    def step(self, new_sqrt_price: SqrtPrice, new_tick: Tick,
             swap_volume_0: float = 0.0, swap_volume_1: float = 0.0) -> dict:
        """Advance the pool by one step.

        Updates sqrt_price → tick → liquidity (crossing ticks as needed), updates
        fee_growth_global from swap volume at the pool's fee tier, and returns a dict
        with:
          - fees_accrued: dict[position_id, (fees_0, fees_1)]
          - ticks_crossed: list[Tick]
          - active_liquidity: float (after step)
          - pool_fee_earned_0: float
          - pool_fee_earned_1: float
        """
        ...
```

---

## 7. Price process protocols (S06)

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable
import numpy as np
from undertow.sim.types import SqrtPrice, Price, Tick

@runtime_checkable
class PriceProcess(Protocol):
    """A stochastic or deterministic price source. Call `step()` for each
    simulator step to get the next price point."""

    def reset(self, rng: np.random.Generator) -> None:
        """Reset internal state. Called at episode start."""
        ...

    def step(self, rng: np.random.Generator) -> tuple[Price, SqrtPrice, Tick]:
        """Advance one step. Returns (price, sqrt_price, tick).
        All three are consistent: price = sqrt_price^2, tick = floor(log(price)/log(1.0001))."""
        ...

    @property
    def mode(self) -> str: ...

# -- Replay price process (S06) --
class ReplayPriceProcess:
    """Replays reference-feed close prices. Deterministic given the feed slice.

    Constructed from a MarketView + a (start_seq, end_seq) episode window.
    Each step advances one bar in the reference feed."""
    def __init__(self, market_view: MarketView, start_seq: int, end_seq: int) -> None: ...
    def reset(self, rng: np.random.Generator) -> None: ...
    def step(self, rng: np.random.Generator) -> tuple[Price, SqrtPrice, Tick]: ...

# -- Calibrated MRSJD price process (S06) --
@dataclass(frozen=True, slots=True)
class MRSJDParams:
    """Parameters of a Markov-regime-switching jump-diffusion.

    Fitted on the train split only (no look-ahead). 4 latent states matching
    the regime labels. Student-t jump sizes."""
    state_names: tuple[str, ...]            # e.g. ("bull", "bear", "sideways", "high_vol")
    transition_matrix: np.ndarray           # 4x4, row-stochastic
    drift: tuple[float, ...]                # per-state μ
    diffusion: tuple[float, ...]            # per-state σ
    jump_intensity: tuple[float, ...]       # per-state λ_J
    jump_loc: tuple[float, ...]             # per-state jump mean
    jump_scale: tuple[float, ...]           # per-state jump scale
    jump_dof: tuple[float, ...]             # per-state Student-t df

class CalibratedPriceProcess:
    """A calibrated MRSJD. step() advances dt and may jump or switch regime."""
    def __init__(self, params: MRSJDParams, dt: float = 1 / (6 * 365.25)) -> None: ...
    def reset(self, rng: np.random.Generator) -> None: ...
    def step(self, rng: np.random.Generator) -> tuple[Price, SqrtPrice, Tick]: ...

    @staticmethod
    def fit(market_view: MarketView) -> MRSJDParams:
        """Fit on the train split only. market_view.is_train must be True."""
        ...
```

---

## 8. Friction model interfaces (S08)

```python
from __future__ import annotations
from typing import Protocol, runtime_checkable

@runtime_checkable
class GasModel(Protocol):
    """Computes gas cost in USDC for an action at a given block/time."""

    def gas_cost_usdc(self, action_type: str, block_number: int,
                      base_fee_per_gas_wei: int,
                      priority_fee_p50_wei: int,
                      eth_usd_price: float) -> float:
        """Gas cost = (base_fee + priority_fee) * gas_units_for(action_type) * eth_usd_price / 1e18.

        action_type ∈ {'mint','burn','collect','rebalance_swap','rebalance'}.
        'rebalance' = burn + mint + swap."""
        ...

@runtime_checkable
class SlippageModel(Protocol):
    """Computes slippage cost in USDC for a rebalancing trade."""

    def slippage_cost_usdc(self, notional_usdc: float,
                           pool_fee_tier_bps: int,
                           fixed_impact_bps: float) -> float:
        """Slippage = notional * (pool_fee_tier_bps + fixed_impact_bps) / 10000."""
        ...
```

---

## 9. Reward function (S09)

```python
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """Per-step reward decomposition. Every term is in USDC (or USDC/initial_capital if
    normalized). The reward = sum of all terms.

    Ablation flags in RewardConfig determine which terms are non-zero."""
    fees: float = 0.0
    gas: float = 0.0                   # ≤ 0 (cost)
    slippage: float = 0.0              # ≤ 0 (cost)
    il_change: float = 0.0             # ΔIL; ≤ 0 most steps
    risk_penalty: float = 0.0          # −λ * σ_PnL; ≤ 0
    reward: float = 0.0                # sum of the above
    # Reference values for debugging
    pnl_net: float = 0.0               # ΔPnL after all costs
    equity: float = 0.0               # current equity

def compute_reward(
    fees_earned: float,
    gas_cost: float,
    slippage_cost: float,
    il_change: float,
    pnl_volatility: float,
    config: RewardConfig,
) -> RewardBreakdown:
    """Compute the step reward per eq (1) of §9.2 (roadmap eq (2)).

    R_t = F_t − G_t − S_t − ΔIL_t − λ·σ_PnL,t

    Each term is gated by its `*_enabled` flag in config. Disabled terms
    are zero in the returned breakdown but the raw values are still populated."""
    ...
```

---

## 10. Policy protocol + baseline ladder (S10)

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable
import numpy as np
from undertow.sim.types import Action, Tick, SqrtPrice, Price, Wealth

@runtime_checkable
class Policy(Protocol):
    """A frozen decision rule. The backtester calls only act(); there is no learning API
    on the protocol — an attempt to call update() on the backtester's policy path is
    unrepresentable."""

    def act(self, observation: Observation, rng: np.random.Generator | None = None
            ) -> Action:
        """Return an action given the current observation. rng is None for
        deterministic policies."""
        ...

    @property
    def name(self) -> str: ...

    def reset(self) -> None:
        """Reset any policy-internal state at episode start. Default: no-op."""
        ...

# -- Baseline policies (all in baselines.py) --

class HODLPolicy:
    """Buy-and-hold the initial token mix. Always returns hold."""
    ...

class PassiveNarrowPolicy:
    """Deploy at ±5% around entry price, never rebalance. Always returns hold after
    initial deployment."""
    ...

class PassiveWidePolicy:
    """Deploy at ±20% around entry price, never rebalance."""
    ...

class FullRangeV2Policy:
    """Full-range (Uniswap V2 equivalent) position: tick_lower = MIN_TICK,
    tick_upper = MAX_TICK. Always hold."""
    ...

class TauResetPolicy:
    """Recenter when price exits the current range. New range = [current_tick - τ,
    current_tick + τ] where τ = the entry half-width in tick-spacings."""
    def __init__(self, half_width_ticks: int) -> None: ...
    def reset(self) -> None: ...

class CostAwareRebalancePolicy:
    """Recenter at most once per day, and only if estimated fee gain since
    last rebalance exceeds the gas cost of a rebalance."""
    def __init__(self, half_width_ticks: int, min_interval_steps: int,
                 gas_model: GasModel) -> None: ...
    def reset(self) -> None: ...
```

---

## 11. Observation space (S12)

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from undertow.sim.types import Tick, SqrtPrice, Price, Wealth

@dataclass(frozen=True, slots=True)
class Observation:
    """The state vector the agent sees at each decision step.

    All fields are backward-looking, closed on the right. The observation builder (S12)
    reads from MarketView + PoolEngine at the current step index — never beyond."""
    # -- Market --
    price: Price                              # current reference price
    sqrt_price: SqrtPrice                     # current sqrt(reference price)
    tick: Tick                                # current tick
    price_returns_1h: float                   # log return over last 6*10min bars
    price_returns_24h: float                  # log return over last 144*10min bars
    realized_vol_24h: float                   # annualized σ over last 144 bars

    # -- Position --
    position_in_range: bool
    position_value: Wealth
    uncollected_fees_token0: float
    uncollected_fees_token1: float
    il_vs_hodl: float                         # as fraction of initial capital

    # -- Pool --
    pool_active_liquidity: float
    pool_fee_tier_bps: int

    # -- Costs --
    gas_price_recent_wei: float               # recent base_fee + priority_fee
    eth_usd_price_recent: float

    # -- Regime --
    regime: str                               # one of the Regime enum values

    # -- Time --
    step_in_episode: int
    steps_remaining: int

    def to_vector(self) -> np.ndarray:
        """Flatten to a 1-D float64 array for the policy network."""
        ...
```

---

## 12. Gymnasium environment (S12)

```python
from __future__ import annotations
import gymnasium as gym
from undertow.sim.types import Action

class LpEnvironment(gym.Env):
    """Gymnasium environment wiring prices + pool + frictions + reward.

    Observation space: Box(shape = Observation.to_vector().shape, dtype=float64).
    Action space: Discrete(n_actions), where n_actions = 1 (hold) + |center_offsets| * |widths|.
    """

    def __init__(self, market_view: MarketView, pool_engine: PoolEngine,
                 price_process: PriceProcess, gas_model: GasModel,
                 slippage_model: SlippageModel, config: SimConfig) -> None: ...

    def reset(self, *, seed: int | None = None,
              options: dict | None = None) -> tuple[np.ndarray, dict]: ...

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Returns (observation, reward, terminated, truncated, info).
        info carries the full RewardBreakdown as a dict."""
        ...
```

---

## 13. Backtester interface (S11)

```python
from __future__ import annotations
from dataclasses import dataclass
import polars as pl
from undertow.sim.types import Wealth

@dataclass(frozen=True, slots=True)
class BacktestLedger:
    """Full record of a backtest run."""
    equity_curve: pl.DataFrame            # step, equity, pnl, fees, gas, slippage, il
    decision_log: pl.DataFrame            # step, action, tick_lower, tick_upper, gas_paid, ...
    cost_ledger: pl.DataFrame             # step, cost_type, amount_usdc
    pnl_decomposition: dict[str, float]   # total fees, total gas, total slippage, total IL, net
    config_hash: str
    git_commit: str
    policy_name: str
    split: str                            # "train" or "eval"

@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Summary of a backtest run."""
    total_pnl: Wealth
    annualized_return: float
    sharpe: float
    max_drawdown: float
    fee_income: float
    gas_paid: float
    slippage_paid: float
    il_realized: float
    hodl_return: float                    # what HODL would have earned over the same window
    excess_vs_hodl: float                 # total_pnl - hodl_return

def run_backtest(policy: Policy, market_view: MarketView,
                 config: SimConfig) -> BacktestLedger:
    """Replay a frozen policy over the real event tape.

    Uses the data module's fee-growth engine for exact fee accrual (S11 wires
    FeeGrowthTracker through S02's exports). The policy is never updated inside the
    backtester."""
    ...

def summarize_backtest(ledger: BacktestLedger) -> BacktestResult: ...
```

---

## 14. Metrics signatures (S05)

```python
from __future__ import annotations
import polars as pl
from typing import Sequence

def sharpe_ratio(equity_curve: pl.Series, risk_free_rate: float = 0.0,
                 periods_per_year: int = 365 * 24 * 6) -> float:
    """Annualized Sharpe: mean(returns - rf) / std(returns) * sqrt(periods_per_year)."""
    ...

def sortino_ratio(equity_curve: pl.Series, risk_free_rate: float = 0.0,
                  periods_per_year: int = 365 * 24 * 6) -> float:
    """Annualized Sortino: uses only downside deviation in denominator."""
    ...

def max_drawdown(equity_curve: pl.Series) -> float:
    """Maximum peak-to-trough decline as a negative fraction. -0.25 = 25% drawdown."""
    ...

def value_at_risk(returns: pl.Series, confidence: float = 0.95) -> float:
    """Historical VaR at the given confidence level."""
    ...

def cvar(returns: pl.Series, confidence: float = 0.95) -> float:
    """Conditional VaR (expected shortfall)."""
    ...

def annualized_return(equity_curve: pl.Series,
                      periods_per_year: int = 365 * 24 * 6) -> float:
    """Annualized return from the equity curve."""
    ...

def pnl_decomposition(ledger: pl.DataFrame) -> dict[str, float]:
    """PnL = ΣF − ΣΔIL − Σ(G+S). Returns a dict with keys: total_fees, total_il_change,
    total_gas, total_slippage, net_pnl."""
    ...

def per_regime_metrics(equity_curve: pl.DataFrame, regime_column: str,
                       metric_fns: dict[str, callable]) -> pl.DataFrame:
    """Compute each metric fn for each regime label. Returns a table of
    regime × metric_name → value."""
    ...

def all_metrics(equity_curve: pl.Series, returns: pl.Series,
                periods_per_year: int = 365 * 24 * 6) -> dict[str, float]:
    """Convenience: returns {sharpe, sortino, maxdd, var_95, cvar_95, annualized_return}."""
    ...
```

---

## 15. Training harness (S13)

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

@dataclass(frozen=True, slots=True)
class RunManifest:
    """Metadata for a single training run."""
    run_id: str
    seed: int
    config_hash: str
    git_commit: str
    algorithm: str
    total_timesteps: int
    best_reward: float
    checkpoint_path: Path

@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Aggregated results across all seeds."""
    seeds: Sequence[int]
    manifests: Sequence[RunManifest]
    learning_curves: dict[str, list[float]]    # metric_name → per-eval-step values
    best_checkpoint: Path

def train_ppo(env_fn: callable, config: SimConfig,
              log_dir: Path | None = None) -> TrainingResult:
    """Train PPO across config.training.seeds, returning aggregated results.

    env_fn is a zero-argument callable that creates a fresh environment (needed for
    multi-seed training — each seed gets its own env with its own RNG).
    Logs to log_dir with tensorboard-compatible format. Writes checkpoints at
    config.training.checkpoint_freq_steps."""
    ...

def train_single_seed(env_fn: callable, config: SimConfig, seed: int,
                      log_dir: Path | None = None) -> RunManifest: ...
```

---

## 16. Parity & look-ahead validation (S14)

```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ParityReport:
    """Sim-vs-backtester agreement on identical inputs."""
    fee_drift_mean: float
    fee_drift_std: float
    il_drift_mean: float
    il_drift_std: float
    pnl_drift_mean: float
    pnl_drift_std: float
    tolerance: float                 # stated, justified tolerance
    passed: bool                     # all drifts within tolerance
    justification: str               # prose justification of the tolerance

@dataclass(frozen=True, slots=True)
class LookAheadReport:
    """Results of look-ahead probes on the composed system."""
    probes_run: int
    violations: int
    details: list[str]               # one line per violation

def run_parity_check(sim_env: LpEnvironment, backtest_ledger: BacktestLedger,
                     num_episodes: int = 10, seeds: Sequence[int] = (0, 1, 2, 3, 4)
                     ) -> ParityReport:
    """Run sim and backtester on identical inputs and quantify the drift."""
    ...

def run_lookahead_probes(market_view: MarketView, env: LpEnvironment,
                         backtest_fn: callable) -> LookAheadReport:
    """Run the look-ahead probe suite. Returns 0 violations on a correct system."""
    ...
```

---

## 17. Evaluation runner (S15)

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import polars as pl

@dataclass(frozen=True, slots=True)
class AblationResult:
    """RQ1: one row of the ablation table."""
    ablation_name: str               # "full", "no_gas", "static_fee", "no_il"
    flags: dict[str, bool]
    metrics: dict[str, float]        # per-metric values
    per_regime: pl.DataFrame | None  # per-regime breakdown

@dataclass(frozen=True, slots=True)
class RegimeMatrix:
    """RQ2: per-regime results for one policy."""
    policy_name: str
    regimes: list[str]
    metrics_per_regime: pl.DataFrame  # regime × metric → value (mean ± std over seeds)

@dataclass(frozen=True, slots=True)
class GapReport:
    """RQ3: simulation-to-reality gap."""
    policy_name: str
    sim_pnl: float
    onchain_pnl: float
    gap: float                        # sim_pnl - onchain_pnl
    gap_by_cost_type: dict[str, float]

def run_ablations(config: SimConfig, policy: Policy,
                  market_view: MarketView) -> list[AblationResult]:
    """Run the RQ1 ablation table: {full, no_gas, static_fee, no_il} × metrics."""
    ...

def run_regime_evaluation(config: SimConfig, policies: dict[str, Policy],
                          market_view: MarketView) -> list[RegimeMatrix]:
    """Run the RQ2 regime matrix: every policy × every regime × metrics."""
    ...

def run_gap_analysis(config: SimConfig, policy: Policy,
                     market_view: MarketView) -> GapReport:
    """Quantify sim-to-reality gap = sim_pnl − onchain_pnl."""
    ...

def write_results(ablation: list[AblationResult],
                  regime: list[RegimeMatrix],
                  gap: GapReport,
                  output_dir: Path) -> None:
    """Write markdown/CSV artifacts."""
    ...
```

---

## 18. Fixture ownership (§10 — who owns which test fixture)

| Fixture | Owned by | Reused by | Description |
|---|---|---|---|
| `rng` | S00 | all | `np.random.default_rng(42)`, seeded per-test |
| `tiny_data_config` | S03 | S04–S14 | `DataConfig` pointing at fake/trimmed data |
| `tiny_market_view` | S03 | S04–S15 | `MarketView` built from synthesized data (~1000 tape rows, 100 reference bars) |
| `entry_position` | S04 | S07–S14 | A `Position` at tick_lower=−120, tick_upper=+120, entry price = $3000 |
| `tiny_pool_engine` | S07 | S09–S14 | `PoolEngine` with ~20 ticks initialized |
| `mock_gas_model` | S08 | S09–S14 | Returns $5 flat per action type |
| `mock_slippage_model` | S08 | S09–S14 | Returns notional * 0.003 + 5bps |
| `dummy_policy` | S10 | S11–S15 | Always-hold policy |
| `tiny_env` | S12 | S13–S15 | `LpEnvironment` with all mocks, replay prices, 100-step episodes |
| `tiny_ledger` | S11 | S14–S15 | `BacktestLedger` from a 100-step replay |

S03 owns `tiny_market_view`; every other task imports it from `tests/sim/conftest.py`. No task may
define its own alternative — consistency of the shared test data is how cross-task bugs are caught
before they become integration failures.

---

## 19. Reference values for golden tests

These numbers come from the worked examples in the lesson plan. Every task that touches these
quantities must include a golden test that reproduces at least one of these values.

| Quantity | Expected value | Source | Used by |
|---|---|---|---|
| V2 IL at r=1.2 | −0.00454 (−0.45%) | §5.6.1 table | S04 |
| V2 IL at r=0.5 | −0.0572 (−5.7%) | §5.6.1 table | S04 |
| Concentrated ±10% IL at r=1.2 | −0.066 (−6.6%) | §5.6.1 table | S04 |
| Concentrated ±10% IL at r=0.5 | −0.325 (−32.5%) | §5.6.1 table | S04 |
| Tick 196242 → price | ~3004 USDC/WETH | data CONTRACTS.md §3.0 | S04, S06 |
| sqrt_price_x96 at ETH=$3000 | 1446501726624926496477173928747177 | data CONTRACTS.md §3.0 | S04, S07 |
| Q96 = 2^96 | 79228162514264337593543950336 | data CONTRACTS.md §3.0 | S04, S07 |
| Full IL/LVR reward with all terms ← vs no-LVR → | net negative vs net positive | §8.10 example | S09 |

---

## 20. Cross-reference: which contract each task reads

| Task | Reads sections |
|---|---|
| S00 | §0 (conventions) |
| S01 | §1 (types), §2 (config) |
| S02 | §3 (data API) |
| S03 | §3 (data API), §4 (MarketView), §18 (fixtures) |
| S04 | §5 (position math), §19 (golden values) |
| S05 | §14 (metrics) |
| S06 | §4 (MarketView), §7 (price processes), §19 (golden values) |
| S07 | §5 (position math), §6 (pool engine), §19 (golden values) |
| S08 | §2 (config), §8 (frictions) |
| S09 | §2 (config), §9 (reward), §19 (golden values) |
| S10 | §1 (action types), §10 (policy protocol) |
| S11 | §3 (data API), §4 (MarketView), §5 (position), §8 (frictions), §10 (policy), §13 (backtester) |
| S12 | §4 (MarketView), §5 (position), §6 (pool engine), §7 (prices), §8 (frictions), §9 (reward), §10 (policy), §11 (observation), §12 (env) |
| S13 | §2 (config), §12 (env), §10 (policy), §14 (metrics) |
| S14 | §12 (env), §13 (backtester), §16 (parity) |
| S15 | §2 (config), §4 (MarketView), §10 (policy), §13 (backtester), §14 (metrics), §17 (evaluation) |
| S16 | §2 (config), §12 (env), §13 (backtester), §15 (training), §17 (evaluation) |