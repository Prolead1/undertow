"""``undertow.data`` — Uniswap V3 data ingestion and processing.

Package boundary: this package imports nothing from ``undertow.sim``, and nothing
in ``undertow.sim`` may import from here. The two communicate only through stable
public interfaces (``undertow.data.load_dataset`` and friends), not through
internal modules.

Stub module — the real public API surface is owned by T16.
"""

__all__: list[str] = []