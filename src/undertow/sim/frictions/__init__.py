"""Friction models — gas and slippage (S08, ``CONTRACTS.md`` §8).

Public surface:

* :class:`~undertow.sim.frictions.gas.GasModel` — protocol for gas cost in USDC.
* :class:`~undertow.sim.frictions.gas.ReplayGasModel` — tape-driven base fee.
* :class:`~undertow.sim.frictions.gas.FlatGasModel` — fixed cost per action.
* :class:`~undertow.sim.frictions.gas.SpikeGasModel` — flat cost with spikes.
* :class:`~undertow.sim.frictions.slippage.SlippageModel` — protocol for slippage.
* :class:`~undertow.sim.frictions.slippage.ProportionalSlippageModel` — linear model.

Gas follows ADRs 005/006: only the base fee is used, uplifted by a flat
:data:`~undertow.sim.frictions.gas.DEFAULT_TIP_SURCHARGE_PCT` surcharge; the
``priority_fee_p50_wei`` argument is kept for contract compatibility and ignored.
"""

from __future__ import annotations

from undertow.sim.frictions.gas import (
    DEFAULT_TIP_SURCHARGE_PCT,
    FlatGasModel,
    GasCostCalculator,
    GasModel,
    ReplayGasModel,
    SpikeGasModel,
)
from undertow.sim.frictions.slippage import (
    DEFAULT_FIXED_IMPACT_BPS,
    ProportionalSlippageModel,
    SlippageModel,
)

__all__ = [
    "DEFAULT_FIXED_IMPACT_BPS",
    "DEFAULT_TIP_SURCHARGE_PCT",
    "FlatGasModel",
    "GasCostCalculator",
    "GasModel",
    "ProportionalSlippageModel",
    "ReplayGasModel",
    "SlippageModel",
    "SpikeGasModel",
]
