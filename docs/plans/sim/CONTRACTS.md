# CONTRACTS.md — frozen cross-task interfaces for `undertow.sim`

**Status: DRAFT.** Being defined as S01–S03 are built.

`undertow.sim` does **not** import `undertow.data`. It is self-contained: it has its own math, its
own types, and consumes event data as generic arrays/tables passed in by the caller.

---

## 1. `undertow.sim.math` — Fixed-point & tick math

Self-contained duplicate of the `undertow.data.fixedpoint` functions that the sim needs.
The sim cannot import from `data` per the scaffold boundary rule.

```python
# Price / tick conversions (identical to data.fixedpoint)
def sqrt_price_x96_to_price(sqrt_price_x96: int, dec0: int, dec1: int) -> Decimal: ...
def price_to_sqrt_price_x96(price: Decimal, dec0: int, dec1: int) -> int: ...
def tick_to_sqrt_price_x96(tick: int) -> int: ...
def sqrt_price_x96_to_tick(sqrt_price_x96: int) -> int: ...
def tick_to_price(tick: int, dec0: int, dec1: int) -> Decimal: ...
def price_to_tick(price: Decimal, dec0: int, dec1: int) -> int: ...

# Wrapping arithmetic
def wrapping_sub_256(a: int, b: int) -> int: ...
def wrapping_add_256(a: int, b: int) -> int: ...
def q128_to_decimal(value_q128: int) -> Decimal: ...

# Liquidity ↔ amounts
def get_amount0_delta(sqrt_a: int, sqrt_b: int, liquidity: int, round_up: bool) -> int: ...
def get_amount1_delta(sqrt_a: int, sqrt_b: int, liquidity: int, round_up: bool) -> int: ...
def liquidity_for_amounts(sqrt_price: int, sqrt_a: int, sqrt_b: int,
                          amount0: int, amount1: int) -> int: ...
def amounts_for_liquidity(sqrt_price: int, sqrt_a: int, sqrt_b: int,
                          liquidity: int) -> tuple[int, int]: ...

# Tick alignment
def align_tick_down(tick: int, spacing: int) -> int: ...
def align_tick_up(tick: int, spacing: int) -> int: ...

# Constants
Q96 = 2**96
Q128 = 2**128
TICK_BASE = Decimal("1.0001")
MIN_TICK = -887272
MAX_TICK = 887272
```

---

## 2. `undertow.sim.config` — SimConfig

```python
@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    duration_days: int = 30
    step_minutes: int = 60
    agent_capital_usdc: float = 100_000.0
    marginal_agent: bool = True         # agent's liquidity does not move the pool price
    fee_tier_bps: int = 30              # 1, 5, 30, or 100
    tick_spacing: int = 60              # derived from fee_tier_bps; validated in __post_init__

    @property
    def steps_per_episode(self) -> int: ...

@dataclass(frozen=True, slots=True)
class ActionGrid:
    center_offsets: tuple[float, ...] = (-0.20, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.20)
    widths: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)
    include_hold: bool = True

@dataclass(frozen=True, slots=True)
class GasConfig:
    mint_units: int = 200_000
    burn_units: int = 150_000
    collect_units: int = 100_000
    rebalance_swap_units: int = 150_000
    base_fee_per_gas_gwei: float = 20.0
    priority_fee_p50_gwei: float = 1.0

@dataclass(frozen=True, slots=True)
class SlippageConfig:
    fixed_impact_bps: float = 5.0       # basis points per unit of volume

@dataclass(frozen=True, slots=True)
class RewardConfig:
    risk_penalty_lambda: float = 0.0
    risk_rolling_window_steps: int = 24
    normalize_by_capital: bool = True
    # ablation flags
    fees_enabled: bool = True
    gas_enabled: bool = True
    slippage_enabled: bool = True
    il_enabled: bool = True
    risk_penalty_enabled: bool = False

@dataclass(frozen=True, slots=True)
class TrainingConfig:
    algorithm: str = "ppo"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    discount_gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    hidden_size: int = 128
    n_hidden: int = 2
    parallel_envs: int = 4
    total_timesteps: int = 1_000_000
    checkpoint_freq_steps: int = 50_000
    log_freq_steps: int = 10_000

@dataclass(frozen=True, slots=True)
class PriceConfig:
    mode: str = "calibrated"            # "replay" | "calibrated" | "gbm"
    mu: float = 0.0                     # annual drift for GBM
    sigma: float = 0.80                 # annual vol for GBM
    jump_intensity: float = 0.0         # jumps per year
    jump_mean: float = 0.0
    jump_std: float = 0.05
    # Regime-switching
    regime_transition_scale: float = 30.0
    high_vol_sigma: float = 1.20
    sideways_sigma: float = 0.40

@dataclass(frozen=True, slots=True)
class SimConfig:
    episode: EpisodeConfig
    action_grid: ActionGrid
    gas: GasConfig
    slippage: SlippageConfig
    reward: RewardConfig
    training: TrainingConfig
    price: PriceConfig

def load_sim_config(path: str) -> SimConfig: ...
```

---

## 3. `undertow.sim.pool` — Concentrated Liquidity Pool

```python
@dataclass
class PoolState:
    """Mutable snapshot of pool state."""
    sqrt_price_x96: int
    tick: int
    liquidity: int
    fee_growth_global_0_x128: int
    fee_growth_global_1_x128: int
    # Per-tick state
    ticks: dict[int, TickState]

@dataclass
class TickState:
    fee_growth_outside_0_x128: int
    fee_growth_outside_1_x128: int
    liquidity_gross: int
    liquidity_net: int                # signed
    initialized: bool

@dataclass(frozen=True, slots=True)
class Position:
    tick_lower: int
    tick_upper: int
    liquidity: int
    fee_growth_inside_0_last: int = 0
    fee_growth_inside_1_last: int = 0
    token0_fees_uncollected: int = 0
    token1_fees_uncollected: int = 0

class ConcentratedLiquidityPool:
    """Self-contained CL pool simulator. No external dependencies except math.py."""
    def __init__(self, fee_tier_bps: int, tick_spacing: int,
                 initial_price: float, dec0: int = 6, dec1: int = 18): ...

    # Read
    price: float                           # property — human price (token1 in token0)
    current_tick: int                      # property — current pool tick

    # Actions
    def swap(self, amount0_in: int | None, amount1_in: int | None) -> tuple[int, int, int]:
        """Execute a swap. Returns (amount0_out, amount1_out, fee_amount)."""

    def mint(self, tick_lower: int, tick_upper: int,
             amount0_desired: int, amount1_desired: int) -> tuple[Position, int, int]:
        """Mint a new position. Returns (position, amount0_used, amount1_used)."""

    def burn(self, position: Position) -> tuple[int, int]:
        """Burn a position. Returns (amount0, amount1)."""

    def collect(self, position: Position) -> tuple[int, int]:
        """Collect uncollected fees. Returns (fees0, fees1)."""
```

---

## 4. `undertow.sim.price` — Price Processes

```python
class PriceProcess(Protocol):
    def step(self, rng: np.random.Generator) -> float:
        """Advance one step, return new price."""
    def reset(self, rng: np.random.Generator, initial_price: float) -> None: ...
    def current(self) -> float: ...

class GBMPrice(PriceProcess):
    def __init__(self, mu: float, sigma: float, dt: float): ...

class RegimeSwitchingPrice(PriceProcess):
    """Markov-regime-switching jump-diffusion.

    REGIMES = ("bull", "bear", "sideways", "high_vol")
    """
    def __init__(self, dt: float,
                 mu_by_regime: dict[str, float] | None = None,
                 sigma_by_regime: dict[str, float] | None = None,
                 jump_intensity: float = 0.0,
                 jump_mean: float = 0.0, jump_std: float = 0.05,
                 transition_scale: float = 30.0): ...

class ReplayPrice(PriceProcess):
    """Replay from a real price series."""
    def __init__(self, prices: np.ndarray): ...
```

---

## 5. `undertow.sim.env` — Gymnasium Environment

```python
class UndertowEnv(gym.Env):
    """Gymnasium environment for active LP in concentrated liquidity."""
    action_space: gym.spaces.Discrete    # action grid index
    observation_space: gym.spaces.Box    # [price, portfolio_value, in_range, ...]

    def __init__(self, config: SimConfig, price_process: PriceProcess | None = None): ...
    def reset(self, *, seed: int | None = None,
              options: dict | None = None) -> tuple[np.ndarray, dict]: ...
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]: ...
```

---

## 6. `undertow.sim.metrics` — Performance Metrics

```python
def sharpe_ratio(equity: np.ndarray, periods_per_year: int = 365,
                 rf: float = 0.0) -> float: ...
def sortino_ratio(equity: np.ndarray, periods_per_year: int = 365,
                  rf: float = 0.0) -> float: ...
def max_drawdown(equity: np.ndarray) -> float: ...
def var_cvar(pnl: np.ndarray, alpha: float = 0.05) -> tuple[float, float]: ...
def decomposition(pnl_history: dict) -> dict: ...
```

## 7. Package boundary

`undertow.sim` must never import `undertow.data`. The scaffold test
(`tests/test_scaffold.py::test_sim_does_not_import_data`) enforces this.