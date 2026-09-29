"""Slippage-cost model for ``undertow.sim`` (S08, ``CONTRACTS.md`` §8).

A rebalancing trade pays the pool fee plus a small additional impact
approximation. The two terms use different unit conventions (ADR-012): the pool
``fee`` is in Uniswap pips (hundredths of a bip, ``1e-6``), while the impact is
in true basis points (``1e-4``). The cost is linear in the traded notional::

    slippage = notional_usdc * (pool_fee_tier_pips / 1_000_000
                                + fixed_impact_bps / 10_000)

The cost is returned positive (the caller subtracts it from PnL) and is zero for
zero notional.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from undertow.sim.config import FIXED_IMPACT_BPS

__all__ = ["DEFAULT_FIXED_IMPACT_BPS", "ProportionalSlippageModel", "SlippageModel"]

#: Uniswap pip denominator: the ``fee`` field is in hundredths of a bip (ADR-012).
PIPS_DENOMINATOR: float = 1_000_000.0

#: Basis points per unit fraction (1 bps = 0.0001) — the impact term.
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

    ``pool_fee_tier_bps`` holds the Uniswap ``fee`` value in **pips**
    (hundredths of a bip; e.g. ``3000`` for the 0.30% tier) and
    ``fixed_impact_bps`` is an additional impact in **basis points** (``5`` =
    0.0005). The per-unit cost is ``pips/1e6 + impact_bps/1e4`` (ADR-012).
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
        fee_fraction = pool_fee_tier_bps / PIPS_DENOMINATOR
        impact_fraction = fixed_impact_bps / BPS_DENOMINATOR
        return notional_usdc * (fee_fraction + impact_fraction)
