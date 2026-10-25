"""Pinned configs for the synthetic test fixtures.

The shared ``tiny_market_view`` / ``tiny_env`` fixtures and their golden numbers
model a **0.30 % USDC/WETH pool with 60 tick spacing** (the fee-tier contrast
pool).  The library defaults (``EpisodeConfig`` / ``configs/sim_default.toml``)
target the primary **0.05 %** study pool, so every test that builds a config
against the synthetic fixtures states the pool explicitly through these helpers
instead of relying on the dataclass defaults.
"""

from __future__ import annotations

from undertow.sim.config import EpisodeConfig

__all__ = [
    "TINY_POOL_FEE_TIER_BPS",
    "TINY_POOL_TICK_SPACING",
    "tiny_episode",
]

#: Uniswap fee pips of the synthetic fixture pool (0.30 %).
TINY_POOL_FEE_TIER_BPS: int = 3000
#: Tick spacing of the synthetic fixture pool.
TINY_POOL_TICK_SPACING: int = 60


def tiny_episode(**overrides: object) -> EpisodeConfig:
    """``EpisodeConfig`` for the synthetic 0.30 %/spacing-60 fixture pool.

    ``overrides`` replace the pinned pool fields like any normal dataclass
    argument, so callers can still vary duration, cadence or capital.
    """
    kwargs: dict[str, object] = {
        "fee_tier_bps": TINY_POOL_FEE_TIER_BPS,
        "tick_spacing": TINY_POOL_TICK_SPACING,
    }
    kwargs.update(overrides)
    return EpisodeConfig(**kwargs)  # type: ignore[arg-type]
