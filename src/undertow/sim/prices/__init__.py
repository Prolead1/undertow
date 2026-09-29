"""Price processes for ``undertow.sim`` (S06, ``CONTRACTS.md`` §7).

Two ``PriceProcess`` implementations are exposed:

* :class:`ReplayPriceProcess` — deterministic replay of the reference feed's
  human USDC/WETH closes (mode ``"replay"``).
* :class:`CalibratedPriceProcess` — a calibrated Markov-regime-switching
  jump-diffusion (mode ``"calibrated"``), parameterised by :class:`MRSJDParams`.

Both return ``(price_human, sqrt_price_raw, tick)`` following ADR-009.
"""

from __future__ import annotations

from undertow.sim.prices.base import PriceProcess
from undertow.sim.prices.regime_jump import CalibratedPriceProcess, MRSJDParams
from undertow.sim.prices.replay import ReplayPriceProcess

__all__ = [
    "CalibratedPriceProcess",
    "MRSJDParams",
    "PriceProcess",
    "ReplayPriceProcess",
]
