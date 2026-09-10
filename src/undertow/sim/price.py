"""Price processes for the ``undertow.sim`` simulator.

Three modes:
- ``GBMPrice`` — geometric Brownian motion (simple baseline)
- ``RegimeSwitchingPrice`` — Markov-regime-switching jump-diffusion (calibrated)
- ``ReplayPrice`` — replay from a real price series (from reference feed)
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class PriceProcess(Protocol):
    """Protocol for pluggable price processes."""

    def step(self, rng: np.random.Generator) -> float:
        """Advance one step, return new price."""
        ...

    def reset(self, rng: np.random.Generator, initial_price: float) -> None:
        """Reset to initial conditions."""
        ...

    def current(self) -> float:
        """Return current price."""
        ...


class GBMPrice:
    """Geometric Brownian motion.

    dS/S = μ dt + σ dW

    Parameters
    ----------
    mu: float
        Annual drift.
    sigma: float
        Annual volatility.
    dt: float
        Time step in years (e.g., 1/525600 for 1-minute steps).
    """

    def __init__(self, mu: float, sigma: float, dt: float) -> None:
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self._price: float = 1.0

    def step(self, rng: np.random.Generator) -> float:
        z = rng.normal()
        self._price *= np.exp(
            (self.mu - 0.5 * self.sigma**2) * self.dt
            + self.sigma * np.sqrt(self.dt) * z
        )
        return self._price

    def reset(self, rng: np.random.Generator, initial_price: float) -> None:
        self._price = initial_price

    def current(self) -> float:
        return self._price


class RegimeSwitchingPrice:
    """Markov-regime-switching jump-diffusion.

    d log S = μ(r_t) dt + σ(r_t) dW + J dN

    where r_t ∈ {bull, bear, sideways, high_vol} is a latent Markov state,
    μ and σ vary by regime, and J is a jump component with Poisson arrival.

    Parameters
    ----------
    dt: float
        Time step in years.
    mu_by_regime: dict[str, float]
        Annual drift per regime.
    sigma_by_regime: dict[str, float]
        Annual volatility per regime.
    jump_intensity: float
        Expected jumps per year (λ).
    jump_mean: float
        Mean log jump size.
    jump_std: float
        Std of log jump size.
    transition_scale: float
        Mean holding time per regime in days.
    """

    REGIMES: tuple[str, ...] = ("bull", "bear", "sideways", "high_vol")

    def __init__(
        self,
        dt: float,
        mu_by_regime: dict[str, float] | None = None,
        sigma_by_regime: dict[str, float] | None = None,
        jump_intensity: float = 0.0,
        jump_mean: float = 0.0,
        jump_std: float = 0.05,
        transition_scale: float = 30.0,
    ) -> None:
        self.dt = dt
        self.mu_by_regime = mu_by_regime or {
            "bull": 1.0,
            "bear": -1.0,
            "sideways": 0.0,
            "high_vol": 0.0,
        }
        self.sigma_by_regime = sigma_by_regime or {
            "bull": 0.60,
            "bear": 0.70,
            "sideways": 0.40,
            "high_vol": 1.20,
        }
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.transition_scale = transition_scale

        # Build transition matrix
        n = len(self.REGIMES)
        p_stay = np.exp(-1.0 / transition_scale)
        p_switch = (1.0 - p_stay) / (n - 1)
        self._transition = np.full((n, n), p_switch)
        np.fill_diagonal(self._transition, p_stay)

        self._price: float = 1.0
        self._regime_idx: int = 2  # start in sideways

    @property
    def regime(self) -> str:
        return self.REGIMES[self._regime_idx]

    def step(self, rng: np.random.Generator) -> float:
        # Possibly switch regime
        probs = self._transition[self._regime_idx]
        self._regime_idx = int(rng.choice(len(self.REGIMES), p=probs))

        regime = self.REGIMES[self._regime_idx]
        mu = self.mu_by_regime[regime]
        sigma = self.sigma_by_regime[regime]

        # Diffusion component
        z = rng.normal()
        log_return = (
            (mu - 0.5 * sigma**2) * self.dt
            + sigma * np.sqrt(self.dt) * z
        )

        # Jump component (compound Poisson)
        n_jumps = rng.poisson(self.jump_intensity * self.dt)
        if n_jumps > 0:
            jumps = rng.normal(self.jump_mean, self.jump_std, size=n_jumps)
            log_return += jumps.sum()

        self._price *= np.exp(log_return)
        return self._price

    def reset(self, rng: np.random.Generator, initial_price: float) -> None:
        self._price = initial_price
        self._regime_idx = int(rng.choice(len(self.REGIMES)))

    def current(self) -> float:
        return self._price


class ReplayPrice:
    """Replay from a pre-loaded price series.

    Parameters
    ----------
    prices: np.ndarray
        1-D array of prices. step() returns successive elements.
    """

    def __init__(self, prices: np.ndarray) -> None:
        if prices.ndim != 1 or len(prices) < 2:
            raise ValueError("prices must be a 1-D array with at least 2 elements")
        self._prices = prices
        self._idx: int = 0

    def step(self, rng: np.random.Generator) -> float:
        self._idx += 1
        if self._idx >= len(self._prices):
            raise StopIteration("End of price series reached")
        return float(self._prices[self._idx])

    def reset(self, rng: np.random.Generator, initial_price: float) -> None:
        self._idx = 0
        # Note: ignores initial_price — uses the series' first value

    def current(self) -> float:
        return float(self._prices[self._idx])