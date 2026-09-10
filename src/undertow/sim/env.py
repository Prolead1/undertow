"""Gymnasium environment for active concentrated-liquidity provision.

``UndertowEnv`` wraps ``ConcentratedLiquidityPool`` + a price process + friction
into a standard Gymnasium ``Env`` that trains with stable-baselines3 PPO.

Observation space
-----------------
A float vector: [log_price, portfolio_value_pct, in_range_flag, fee_rate,
                  gas_price_gwei, step_in_episode_frac]

Action space
------------
Discrete: an index into ``ActionGrid``. The action is decoded into a
``(center_offset, width)`` tuple that defines a new liquidity range, or
``None`` (hold existing position).
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from undertow.sim.config import SimConfig, ActionGrid
from undertow.sim.math import (
    tick_to_price,
    price_to_tick,
    price_to_sqrt_price_x96,
    align_tick_down,
    align_tick_up,
    amounts_for_liquidity,
    tick_to_sqrt_price_x96,
)
from undertow.sim.pool import ConcentratedLiquidityPool, Position
from undertow.sim.price import PriceProcess, GBMPrice


class UndertowEnv(gym.Env):
    """Gymnasium environment for active LP in concentrated liquidity.

    The agent observes the pool state and chooses a liquidity range every
    ``decision_freq`` steps. Between decisions, the price process advances,
    fees accrue, and impermanent loss accumulates.

    Parameters
    ----------
    config: SimConfig
        Full simulation configuration.
    price_process: PriceProcess | None
        Pluggable price process. If None, creates a GBMPrice from config.price.
    decision_freq: int
        Number of price steps between agent decisions (default 60 = hourly
        decisions with 1-min steps).
    """

    def __init__(
        self,
        config: SimConfig | None = None,
        price_process: PriceProcess | None = None,
        decision_freq: int = 60,
    ) -> None:
        super().__init__()
        self.cfg = config or SimConfig()

        # Compute step size in years
        self.dt_years: float = self.cfg.episode.step_minutes / (365 * 24 * 60)
        self.decision_freq: int = decision_freq
        self.steps_per_episode: int = self.cfg.episode.steps_per_episode

        # Price process
        if price_process is not None:
            self._price_process = price_process
        else:
            self._price_process = GBMPrice(
                mu=self.cfg.price.mu,
                sigma=self.cfg.price.sigma,
                dt=self.dt_years,
            )

        # Action space: discrete index into ActionGrid
        self.action_space = spaces.Discrete(self.cfg.action_grid.n_actions)

        # Observation space: [log_price, portfolio_value_pct, in_range_flag,
        #                     fee_rate, gas_price, step_frac]
        self.observation_space = spaces.Box(
            low=np.array([-20.0, -2.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([20.0, 5.0, 1.0, 1.0, 1000.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Internal state
        self._pool: ConcentratedLiquidityPool | None = None
        self._position: Position | None = None
        self._rng: np.random.Generator | None = None
        self._step_idx: int = 0
        self._initial_capital: float = 0.0
        self._portfolio_value: float = 0.0
        self._entry_price: float = 0.0
        self._token0_balance: int = 0  # raw units
        self._token1_balance: int = 0
        self._cumulative_fees: float = 0.0
        self._cumulative_gas: float = 0.0
        self._cumulative_il: float = 0.0
        self._equity_history: list[float] = []
        self._price_history: list[float] = []

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment to start a new episode."""
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        # Initial price (~$3000 for ETH)
        initial_price = 3000.0
        if options and "initial_price" in options:
            initial_price = float(options["initial_price"])

        self._entry_price = initial_price
        self._price_process.reset(self._rng, initial_price)

        # Initialize pool
        self._pool = ConcentratedLiquidityPool(
            fee_tier_bps=self.cfg.episode.fee_tier_bps,
            tick_spacing=self.cfg.episode.tick_spacing,
            initial_price=initial_price,
        )

        # Initial capital in raw units
        capital = self.cfg.episode.agent_capital_usdc
        self._initial_capital = capital
        # Half in each token at entry price
        token1_amount_usd = capital / 2.0
        token0_amount_usd = capital / 2.0
        # Raw units: token0 has dec0 decimals, token1 has dec1 decimals
        self._token0_balance = int(token0_amount_usd * (10 ** self._pool.dec0))
        self._token1_balance = int(
            token1_amount_usd / initial_price * (10 ** self._pool.dec1)
        )

        # Open initial position (±20% around entry price)
        # NOTE: higher price → lower tick, so we swap the extremes.
        price_low = initial_price * 0.80
        price_high = initial_price * 1.20
        # tick_lower < tick_upper numerically, but tick_lower = higher price
        tick_for_high = price_to_tick(Decimal(str(price_high)), self._pool.dec0, self._pool.dec1)
        tick_for_low = price_to_tick(Decimal(str(price_low)), self._pool.dec0, self._pool.dec1)
        tick_lower = align_tick_down(tick_for_high, self._pool.tick_spacing)
        tick_upper = align_tick_up(tick_for_low, self._pool.tick_spacing)

        # Mint with available balances
        self._position, used0, used1 = self._pool.mint(
            tick_lower, tick_upper,
            self._token0_balance, self._token1_balance,
        )
        self._token0_balance -= used0
        self._token1_balance -= used1

        self._step_idx = 0
        self._portfolio_value = self._compute_portfolio_value()
        self._cumulative_fees = 0.0
        self._cumulative_gas = 0.0
        self._cumulative_il = 0.0
        self._equity_history = [self._portfolio_value]
        self._price_history = [initial_price]

        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step: advance price process for ``decision_freq`` sub-steps,
        then apply the agent's action."""
        assert self._pool is not None
        assert self._position is not None
        assert self._rng is not None

        reward = 0.0
        gas_cost_usd = 0.0
        il_change = 0.0
        fee_income = 0.0

        # Sub-steps: price process advances, fees accrue
        for _ in range(self.decision_freq):
            # Advance price
            new_price = self._price_process.step(self._rng)
            self._price_history.append(new_price)

            # Execute a small swap to move the pool price (if marginal_agent=False)
            # For now, we set the pool price directly (marginal agent assumption)
            self._set_pool_price(new_price)

            # Accrue fees at each sub-step (simplified: proportional to time)
            self._step_idx += 1

        # Apply the action (rebalance)
        decoded = self.cfg.action_grid.decode(action)
        if decoded is not None:
            center_offset, width = decoded
            # Compute new range in price space
            current_price = self._price_process.current()
            center = current_price * (1.0 + center_offset)
            price_low = center * (1.0 - width / 2.0)
            price_high = center * (1.0 + width / 2.0)

            # Rebalance
            gas_cost_usd, il_change, fee_income = self._rebalance(
                price_low, price_high
            )

        # Compute reward
        reward = self._compute_reward(
            fee_income=fee_income,
            gas_cost_usd=gas_cost_usd,
            il_change=il_change,
        )

        # Update portfolio value
        new_portfolio_value = self._compute_portfolio_value()
        self._portfolio_value = new_portfolio_value
        self._equity_history.append(new_portfolio_value)

        # Check termination
        terminated = self._step_idx >= self.steps_per_episode
        truncated = False

        # Update cumulative trackers
        self._cumulative_fees += fee_income
        self._cumulative_gas += gas_cost_usd
        self._cumulative_il += il_change

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """Build the observation vector."""
        assert self._pool is not None
        current_price = self._price_process.current()
        log_price = np.log(max(current_price, 1e-10))
        portfolio_pct = (
            (self._portfolio_value / self._initial_capital - 1.0)
            if self._initial_capital > 0
            else 0.0
        )
        in_range = 1.0 if self._is_in_range() else 0.0
        fee_rate = self.cfg.episode.fee_tier_bps / 100.0  # normalize to ~[0,1]
        gas_price = self.cfg.gas.total_gas_price_gwei
        step_frac = self._step_idx / max(self.steps_per_episode, 1)

        return np.array(
            [log_price, portfolio_pct, in_range, fee_rate, gas_price, step_frac],
            dtype=np.float32,
        )

    def _get_info(self) -> dict:
        return {
            "step": self._step_idx,
            "portfolio_value": self._portfolio_value,
            "price": self._price_process.current(),
            "cumulative_fees": self._cumulative_fees,
            "cumulative_gas": self._cumulative_gas,
            "cumulative_il": self._cumulative_il,
            "in_range": self._is_in_range(),
        }

    def _set_pool_price(self, price: float) -> None:
        """Set the pool price to an external price (marginal agent)."""
        assert self._pool is not None
        price_d = Decimal(str(price))
        self._pool.sqrt_price_x96 = price_to_sqrt_price_x96(
            price_d, self._pool.dec0, self._pool.dec1
        )
        self._pool.tick = price_to_tick(price_d, self._pool.dec0, self._pool.dec1)

    def _is_in_range(self) -> bool:
        """Check if the current position is in range."""
        if self._position is None or self._pool is None:
            return False
        return (
            self._pool.tick >= self._position.tick_lower
            and self._pool.tick < self._position.tick_upper
        )

    def _compute_portfolio_value(self) -> float:
        """Compute total portfolio value in USD."""
        assert self._pool is not None
        current_price = self._price_process.current()

        # Token0 value (USDC, dec0=6): raw / 1e6
        token0_value = self._token0_balance / (10 ** self._pool.dec0)

        # Token1 value (WETH, dec1=18): raw / 1e18 * price
        token1_value = (
            self._token1_balance / (10 ** self._pool.dec1)
        ) * current_price

        # Position value
        position_value = 0.0
        if self._position is not None and self._position.liquidity > 0:
            sqrt_lower = tick_to_sqrt_price_x96(self._position.tick_lower)
            sqrt_upper = tick_to_sqrt_price_x96(self._position.tick_upper)
            a0, a1 = amounts_for_liquidity(
                self._pool.sqrt_price_x96, sqrt_lower, sqrt_upper,
                self._position.liquidity,
            )
            position_value += a0 / (10 ** self._pool.dec0)
            position_value += (a1 / (10 ** self._pool.dec1)) * current_price

        # Uncollected fees
        uncollected_value = 0.0
        if self._position is not None:
            uncollected_value += (
                self._position.token0_fees_uncollected / (10 ** self._pool.dec0)
            )
            uncollected_value += (
                self._position.token1_fees_uncollected
                / (10 ** self._pool.dec1)
                * current_price
            )

        return token0_value + token1_value + position_value + uncollected_value

    def _rebalance(
        self, price_low: float, price_high: float
    ) -> tuple[float, float, float]:
        """Rebalance to a new range. Returns (gas_cost_usd, il_change, fee_income)."""
        assert self._pool is not None
        assert self._position is not None

        # Gas cost
        gas_wei = self.cfg.gas.total_gas_price_gwei * 1e9 * self.cfg.gas.rebalance_swap_units
        eth_price = self._price_process.current()
        gas_cost_usd = gas_wei / 1e18 * eth_price if self.cfg.reward.gas_enabled else 0.0

        # Collect fees from current position
        fees0, fees1 = self._pool.collect(self._position)
        fee_income = fees0 / (10 ** self._pool.dec0)
        fee_income += (fees1 / (10 ** self._pool.dec1)) * eth_price

        # Burn current position to reclaim tokens
        a0, a1 = self._pool.burn(self._position)
        self._token0_balance += a0 + fees0
        self._token1_balance += a1 + fees1

        # Compute IL: difference between current value and HODL value
        # For now, simplified IL calculation
        il_change = 0.0
        if self.cfg.reward.il_enabled:
            # IL = value of held tokens at current price vs initial
            initial_token0_value = (
                self._initial_capital / 2.0
            )
            initial_token1_value = (
                self._initial_capital / 2.0
            )
            hodl_value = (
                initial_token0_value
                + initial_token1_value * self._price_process.current() / self._entry_price
            )
            il_change = self._compute_portfolio_value() - hodl_value

        # Mint new position
        # NOTE: higher price → lower tick; swap so tick_lower < tick_upper.
        tick_for_high = price_to_tick(
            Decimal(str(price_high)), self._pool.dec0, self._pool.dec1
        )
        tick_for_low = price_to_tick(
            Decimal(str(price_low)), self._pool.dec0, self._pool.dec1
        )
        tick_lower = align_tick_down(tick_for_high, self._pool.tick_spacing)
        tick_upper = align_tick_up(tick_for_low, self._pool.tick_spacing)

        if tick_lower < tick_upper:
            self._position, used0, used1 = self._pool.mint(
                tick_lower, tick_upper,
                self._token0_balance, self._token1_balance,
            )
            self._token0_balance -= used0
            self._token1_balance -= used1
        else:
            # Ticks collapsed after alignment — fall back to minimum-width
            # range around the current pool tick.
            current_tick = self._pool.tick
            half_spacing = self._pool.tick_spacing
            tick_lower = align_tick_down(current_tick - half_spacing, self._pool.tick_spacing)
            tick_upper = align_tick_up(current_tick + half_spacing, self._pool.tick_spacing)
            if tick_lower < tick_upper:
                self._position, used0, used1 = self._pool.mint(
                    tick_lower, tick_upper,
                    self._token0_balance, self._token1_balance,
                )
                self._token0_balance -= used0
                self._token1_balance -= used1

        return gas_cost_usd, il_change, fee_income

    def _compute_reward(
        self,
        fee_income: float,
        gas_cost_usd: float,
        il_change: float,
    ) -> float:
        """Compute the per-step reward (eq. 2 from the roadmap)."""
        r = 0.0
        if self.cfg.reward.fees_enabled:
            r += fee_income
        if self.cfg.reward.gas_enabled:
            r -= gas_cost_usd
        if self.cfg.reward.il_enabled:
            r += il_change  # IL change is already negative when IL increases

        # Risk penalty
        if self.cfg.reward.risk_penalty_enabled and len(self._equity_history) >= 2:
            window = min(
                self.cfg.reward.risk_rolling_window_steps,
                len(self._equity_history),
            )
            recent_returns = np.diff(self._equity_history[-window:]) / np.maximum(
                np.array(self._equity_history[-window:-1]), 1e-10
            )
            vol = np.std(recent_returns) if len(recent_returns) >= 2 else 0.0
            r -= self.cfg.reward.risk_penalty_lambda * vol

        # Normalize by initial capital
        if self.cfg.reward.normalize_by_capital and self._initial_capital > 0:
            r = r / self._initial_capital

        return float(r)