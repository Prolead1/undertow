"""``PriceProcess`` protocol and the ADR-009 price-point conversion (S06).

``CONTRACTS.md`` §7 pins the protocol. The conversion helper below is shared by
:mod:`undertow.sim.prices.replay` and :mod:`undertow.sim.prices.regime_jump` so
that every process emits a mutually consistent ``(price, sqrt_price, tick)``
triplet (ADR-009):

    sqrt_price_raw = 10 ** ((dec1 - dec0) / 2) / sqrt(price_human)
    tick           = undertow.data.price_to_tick(price_human, dec0, dec1)  # floor

``tick`` is **never** ``log(price) / log(1.0001)``.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Protocol, runtime_checkable

import numpy as np

from undertow.data import price_to_tick
from undertow.sim.types import Price, SqrtPrice, Tick

__all__ = ["PriceProcess", "human_price_to_triplet"]


@runtime_checkable
class PriceProcess(Protocol):
    """A stochastic or deterministic price source.

    Call :meth:`reset` at episode start, then :meth:`step` for each simulator
    step to get the next price point. Every stochastic draw consumes the
    explicitly supplied :class:`numpy.random.Generator`; no global RNG state is
    ever touched (``CONTRACTS.md`` §0 determinism).
    """

    def reset(self, rng: np.random.Generator) -> None:
        """Reset internal state. Called at episode start."""
        ...

    def step(self, rng: np.random.Generator) -> tuple[Price, SqrtPrice, Tick]:
        """Advance one step and return ``(price_human, sqrt_price_raw, tick)``.

        All three are consistent per ADR-009 (see module docstring). Raises
        :class:`StopIteration` when the process is exhausted.
        """
        ...

    @property
    def mode(self) -> str:
        """``"replay"`` or ``"calibrated"``."""
        ...


def human_price_to_triplet(
    price: float,
    dec0: int = 6,
    dec1: int = 18,
) -> tuple[Price, SqrtPrice, Tick]:
    """Convert a human USDC/WETH price to the consistent ADR-009 triplet.

    ``sqrt_price_raw = 10 ** ((dec1 - dec0) / 2) / sqrt(price)`` and
    ``tick = undertow.data.price_to_tick(price, dec0, dec1)`` (floor semantics).
    The pinned pools use ``dec0=6`` (USDC), ``dec1=18`` (WETH).

    Raises ``ValueError`` for a non-positive price or a price outside the
    protocol tick domain.
    """
    if not math.isfinite(price) or price <= 0.0:
        raise ValueError(f"price must be finite and positive, got {price!r}")
    scale = 10.0 ** ((dec1 - dec0) / 2.0)
    sqrt_price = scale / math.sqrt(price)
    tick = Tick(price_to_tick(Decimal(str(price)), dec0, dec1))
    return float(price), float(sqrt_price), tick
