"""Slippage-cost model for ``undertow.sim`` (S08, ``CONTRACTS.md`` §8).

A rebalancing trade pays the pool fee plus a small additional impact
approximation. Both terms are basis points so the cost is linear in the traded
notional::

    slippage = notional_usdc * (pool_fee_tier_bps + fixed_impact_bps) / 10_000

The cost is returned positive (the caller subtracts it from PnL) and is zero for
zero notional.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from undertow.sim.config import FIXED_IMPACT_BPS

__all__ = ["DEFAULT_FIXED_IMPACT_BPS", "ProportionalSlippageModel", "SlippageModel"]

#: Basis points per unit fraction (1 bps = 0.0001).
BPS_DENOMINATOR: float = 10_000.0

#: Default extra impact over the pool fee (``CONTRACTS.md`` §2).
DEFAULT_FIXED_IMPACT_BPS: float = FIXED_IMPACT_BPS


@runtime_checkable
class SlippageModel(Protocol):
    """Computes slippage cost in USDC for a rebalancing trade.

    The returned cost is always ``>= 0`` and is meant to be subtracted from PnL.
    """

    def slippage_cost_usdc(
        self,
        notional_usdc: float,
        pool_fee_tier_bps: int,
        fixed_impact_bps: float,
    ) -> float:
        """Slippage cost in USDC for a trade of ``notional_usdc``."""
        ...


class ProportionalSlippageModel:
    """Linear (proportional-to-notional) slippage model.

    ``pool_fee_tier_bps`` is the Uniswap fee on the swap (e.g. ``3000`` for
    0.30%) and ``fixed_impact_bps`` is an additional impact approximation. Both
    are basis points, so the per-unit cost is
    ``(pool_fee_tier_bps + fixed_impact_bps) / 10_000``.
    """

    def slippage_cost_usdc(
        self,
        notional_usdc: float,
        pool_fee_tier_bps: int,
        fixed_impact_bps: float,
    ) -> float:
        if notional_usdc < 0:
            raise ValueError(f"notional_usdc must be >= 0, got {notional_usdc!r}")
        if pool_fee_tier_bps < 0:
            raise ValueError(
                f"pool_fee_tier_bps must be >= 0, got {pool_fee_tier_bps!r}"
            )
        if fixed_impact_bps < 0:
            raise ValueError(
                f"fixed_impact_bps must be >= 0, got {fixed_impact_bps!r}"
            )
        return (
            notional_usdc * (pool_fee_tier_bps + fixed_impact_bps) / BPS_DENOMINATOR
        )
