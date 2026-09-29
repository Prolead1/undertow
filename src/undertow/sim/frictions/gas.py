"""Gas-cost models for ``undertow.sim`` (S08, ``CONTRACTS.md`` §8).

Every model returns a **positive** USDC cost that the caller subtracts from PnL.
The shared arithmetic lives in :class:`GasCostCalculator`; the three concrete
models differ only in where the per-block base fee comes from.

Raw gas price to USD
--------------------
The contract signature (:class:`GasModel`) carries both ``base_fee_per_gas_wei``
and ``priority_fee_p50_wei`` and its docstring describes
``(base_fee + priority_fee) * gas_units * eth_usd_price / 1e18``. ADRs 005/006
supersede that formula: the gas stream stores **base fee only** (priority-fee
percentiles are written as ``"0"``), and priority fees are approximated by a flat
:data:`DEFAULT_TIP_SURCHARGE_PCT` uplift on the base fee::

    (base_fee_per_gas * (1 + tip_surcharge_pct / 100)) * gas_units * eth_usd_price / 1e18

The ``priority_fee_p50_wei`` argument is kept for contract compatibility and is
**ignored** (see the module-level note and the S08 PR for the flagged mismatch).
The tip surcharge is a documented constructor/config parameter so S15 can sweep
it without touching the tape.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from undertow.sim.config import GasConfig
from undertow.sim.marketview import MarketView

__all__ = [
    "DEFAULT_TIP_SURCHARGE_PCT",
    "FlatGasModel",
    "GasCostCalculator",
    "GasModel",
    "ReplayGasModel",
    "SpikeGasModel",
]

# ADR-005: average post-EIP-1559 priority-fee : base-fee ratio on USDC/WETH.
DEFAULT_TIP_SURCHARGE_PCT: int = 3

_WEI_PER_ETH: float = 1e18


@runtime_checkable
class GasModel(Protocol):
    """Computes gas cost in USDC for an action at a given block/time.

    ``action_type`` is one of ``{"mint", "burn", "collect", "rebalance_swap",
    "rebalance", "hold"}``; unknown action types raise ``ValueError``. The
    returned cost is always ``>= 0`` and is meant to be subtracted from PnL.

    ``priority_fee_p50_wei`` is accepted for ``CONTRACTS.md`` §8 compatibility
    but ignored (ADR-005/006: base fee only plus a flat tip surcharge).
    """

    def gas_cost_usdc(
        self,
        action_type: str,
        block_number: int,
        base_fee_per_gas_wei: int,
        priority_fee_p50_wei: int,
        eth_usd_price: float,
    ) -> float:
        """Gas cost in USDC for ``action_type`` at ``block_number``."""
        ...


class GasCostCalculator:
    """Shared gas arithmetic: action -> gas units -> USDC (ADR-005).

    Not a :class:`GasModel` itself; the concrete models delegate here. The
    ``priority_fee_p50_wei`` argument of :meth:`gas_cost_usdc` is accepted for
    signature compatibility and ignored — the effective gas price is
    ``base_fee_per_gas_wei * (1 + tip_surcharge_pct / 100)``.
    """

    def __init__(
        self,
        gas_config: GasConfig,
        tip_surcharge_pct: int = DEFAULT_TIP_SURCHARGE_PCT,
    ) -> None:
        if tip_surcharge_pct < 0:
            raise ValueError(
                f"tip_surcharge_pct must be >= 0, got {tip_surcharge_pct!r}"
            )
        self._gas_config = gas_config
        self._tip_surcharge_pct = tip_surcharge_pct
        # `rebalance` is a full round trip: burn the old position, mint the new
        # one, and swap the inventory to the new range (S08 task brief).
        self._units: dict[str, int] = {
            "mint": gas_config.mint_units,
            "burn": gas_config.burn_units,
            "collect": gas_config.collect_units,
            "rebalance_swap": gas_config.rebalance_swap_units,
            "rebalance": (
                gas_config.burn_units
                + gas_config.mint_units
                + gas_config.rebalance_swap_units
            ),
            "hold": 0,
        }

    @property
    def tip_surcharge_pct(self) -> int:
        """The configured flat priority-fee uplift, in percent."""
        return self._tip_surcharge_pct

    def gas_units_for(self, action_type: str) -> int:
        """Gas units consumed by ``action_type``.

        Raises ``ValueError`` for an action type outside the documented map
        (``hold`` is a recognised zero-cost action).
        """
        try:
            return self._units[action_type]
        except KeyError:
            known = ", ".join(sorted(self._units))
            raise ValueError(
                f"unknown action_type {action_type!r}; expected one of: {known}"
            ) from None

    def gas_cost_usdc(
        self,
        action_type: str,
        block_number: int,
        base_fee_per_gas_wei: int,
        priority_fee_p50_wei: int,
        eth_usd_price: float,
    ) -> float:
        """``base_fee * (1 + tip/100) * units * eth_usd_price / 1e18``.

        ``block_number`` and ``priority_fee_p50_wei`` are ignored here (the
        models supply the base fee; ADR-005 keeps the priority fee out of the
        formula).
        """
        del block_number, priority_fee_p50_wei  # contract-compat, ADR-005/006
        if base_fee_per_gas_wei < 0:
            raise ValueError(
                f"base_fee_per_gas_wei must be >= 0, got {base_fee_per_gas_wei!r}"
            )
        if eth_usd_price < 0:
            raise ValueError(f"eth_usd_price must be >= 0, got {eth_usd_price!r}")

        units = self.gas_units_for(action_type)
        effective_base_fee = base_fee_per_gas_wei * (
            1.0 + self._tip_surcharge_pct / 100.0
        )
        return effective_base_fee * units * eth_usd_price / _WEI_PER_ETH


class ReplayGasModel:
    """Prices gas from the tape's actual per-block base fee.

    Wraps a :class:`~undertow.sim.marketview.MarketView`; the per-block
    base-fee argument is ignored in favour of ``market_view.gas_at_block``. A
    block with no gas row costs ``0.0`` rather than raising.
    """

    def __init__(
        self,
        market_view: MarketView,
        gas_config: GasConfig | None = None,
        tip_surcharge_pct: int = DEFAULT_TIP_SURCHARGE_PCT,
    ) -> None:
        self._market_view = market_view
        self._calculator = GasCostCalculator(
            gas_config if gas_config is not None else GasConfig(),
            tip_surcharge_pct,
        )

    @property
    def calculator(self) -> GasCostCalculator:
        return self._calculator

    def gas_cost_usdc(
        self,
        action_type: str,
        block_number: int,
        base_fee_per_gas_wei: int,
        priority_fee_p50_wei: int,
        eth_usd_price: float,
    ) -> float:
        row = self._market_view.gas_at_block(block_number)
        if row is None:
            return 0.0
        base_fee = int(row["base_fee_per_gas"])
        # Retained for the contract; `0` in the tape per ADR-005 and ignored by
        # the calculator anyway.
        priority_fee = int(row.get("priority_fee_p50_wei") or 0)
        return self._calculator.gas_cost_usdc(
            action_type,
            block_number,
            base_fee,
            priority_fee,
            eth_usd_price,
        )


class FlatGasModel:
    """Fixed gas cost per action type, independent of block.

    Params mirror the S08 brief. ``priority_fee_gwei`` is accepted for
    compatibility but ignored under ADR-005; the effective gas price is
    ``base_fee_gwei * (1 + tip_surcharge_pct / 100)``.
    """

    def __init__(
        self,
        gas_config: GasConfig,
        eth_usd_price: float = 3000.0,
        base_fee_gwei: float = 30.0,
        priority_fee_gwei: float = 1.0,
        tip_surcharge_pct: int = DEFAULT_TIP_SURCHARGE_PCT,
    ) -> None:
        if base_fee_gwei < 0:
            raise ValueError(f"base_fee_gwei must be >= 0, got {base_fee_gwei!r}")
        if priority_fee_gwei < 0:
            raise ValueError(
                f"priority_fee_gwei must be >= 0, got {priority_fee_gwei!r}"
            )
        self._eth_usd_price = eth_usd_price
        self._base_fee_wei = int(base_fee_gwei * 1e9)
        self._calculator = GasCostCalculator(gas_config, tip_surcharge_pct)

    @property
    def base_fee_wei(self) -> int:
        return self._base_fee_wei

    def gas_cost_usdc(
        self,
        action_type: str,
        block_number: int,
        base_fee_per_gas_wei: int,
        priority_fee_p50_wei: int,
        eth_usd_price: float,
    ) -> float:
        del block_number, base_fee_per_gas_wei, priority_fee_p50_wei, eth_usd_price
        return self._calculator.gas_cost_usdc(
            action_type, 0, self._base_fee_wei, 0, self._eth_usd_price
        )


class SpikeGasModel:
    """Flat base cost with an occasional multiplicative gas spike.

    Each call draws once from an explicit, instance-owned ``numpy`` Generator
    (determinism contract). ``spike_probability=1.0`` always spikes;
    ``spike_probability=0.0`` never does.
    """

    def __init__(
        self,
        base_model: GasModel,
        spike_multiplier: float = 10.0,
        spike_probability: float = 0.05,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        if spike_multiplier < 0:
            raise ValueError(
                f"spike_multiplier must be >= 0, got {spike_multiplier!r}"
            )
        if not 0.0 <= spike_probability <= 1.0:
            raise ValueError(
                f"spike_probability must be in [0, 1], got {spike_probability!r}"
            )
        self._base_model = base_model
        self._spike_multiplier = spike_multiplier
        self._spike_probability = spike_probability
        # An explicit Generator is mandatory for reproducible runs; the default
        # keeps the model usable standalone without leaking a module-level seed.
        self._rng = rng if rng is not None else np.random.default_rng(0)

    def gas_cost_usdc(
        self,
        action_type: str,
        block_number: int,
        base_fee_per_gas_wei: int,
        priority_fee_p50_wei: int,
        eth_usd_price: float,
    ) -> float:
        base_cost = self._base_model.gas_cost_usdc(
            action_type,
            block_number,
            base_fee_per_gas_wei,
            priority_fee_p50_wei,
            eth_usd_price,
        )
        if self._rng.random() < self._spike_probability:
            return base_cost * self._spike_multiplier
        return base_cost
