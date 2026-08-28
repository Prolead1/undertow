"""Undertow — friction-aware RL for concentrated liquidity provision.

Two packages live under ``undertow``:

- ``undertow.data`` — on-chain Uniswap V3 ingestion, verification, and caching.
- ``undertow.sim`` — the AMM / concentrated-liquidity simulator and RL environment.

The two packages communicate only through stable public interfaces and must never
import each other's internals. ``undertow.data`` imports nothing from
``undertow.sim``, and vice versa.
"""

__all__: list[str] = []