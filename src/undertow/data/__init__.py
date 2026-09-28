"""``undertow.data`` — Uniswap V3 data ingestion and processing (T16 public API).

Package boundary: this package imports nothing from ``undertow.sim``, and nothing
in ``undertow.sim`` may import from here except through this file's public exports.
The two communicate only through stable public interfaces.

Primary entry point
-------------------
``load_dataset(config_or_path)`` is the single function ``undertow.sim`` should
call to obtain a validated, block-ordered event tape with attached reference
prices, regime labels, gas costs, and fee-growth snapshots. It does NOT re-fetch
data — it reads the parquet cache and manifest written by the CLI's
``undertow-data snapshot``.

For data producers (CLI, notebooks)
-------------------------------------
The ``DataConfig``, ``PoolConfig``, etc. types are re-exported so consumers can
build configurations programmatically. The fetchers, storage, and validation
modules are NOT part of the stable public API — import them at your own risk.
"""

from __future__ import annotations

__all__ = [
    # --- Primary entry point ---
    "load_dataset",
    # --- Core types a consumer needs ---
    "Dataset",
    "DataConfig",
    "PoolConfig",
    "WindowConfig",
    "DatasetManifest",
    # --- Schema types ---
    "EVENT_TAPE_SCHEMA",
    "SCHEMA_VERSION",
    # --- Enums a consumer needs to interpret data ---
    "Route",
    "FeeTier",
    "Regime",
    # --- Fixed-point utilities (sim needs these) ---
    "tick_to_price",
    "tick_to_sqrt_price_x96",
    "sqrt_price_x96_to_tick",
    "liquidity_for_amounts",
    "amounts_for_liquidity",
    # === S02 additions (sim public API, docs/plans/sim/CONTRACTS.md §3) ===
    # --- Re-exported from undertow.data.types ---
    "Address",
    "BlockNumber",
    "DataTick",
    "DataRegime",
    "EventType",
    "CheckResult",
    "Severity",
    # --- Re-exported from undertow.data.config ---
    "RegimeConfig",
    "EndpointConfig",
    "load_data_config",
    "default_pools",
    # --- Re-exported from undertow.data.schemas ---
    "SWAP_SCHEMA",
    "MINT_SCHEMA",
    "BURN_SCHEMA",
    "COLLECT_SCHEMA",
    "FLASH_SCHEMA",
    "FEE_GROWTH_SCHEMA",
    "GAS_SCHEMA",
    "REFERENCE_SCHEMA",
    "REGIME_SCHEMA",
    "SCHEMA_REGISTRY",
    "SORT_KEYS",
    "encode_uint",
    "decode_uint",
    "validate_table",
    "empty_table",
    # --- Re-exported from undertow.data.fixedpoint ---
    "Q96",
    "Q128",
    "TICK_BASE",
    "MIN_TICK",
    "MAX_TICK",
    "sqrt_price_x96_to_price",
    "price_to_sqrt_price_x96",
    "price_to_tick",
    "tick_to_sqrt_price",
    # --- Bridge adapters (undertow.data.loader) ---
    "load_tape",
    "load_reference_feed",
    "load_gas_feed",
    "load_regime_labels",
]

from undertow.data.config import (
    # TICK_BASE lives in .config, not the CONTRACTS §3-attributed .fixedpoint.
    # See docs/decisions/007-data-api-loader-module-attribution.md.
    TICK_BASE,
    DataConfig,
    EndpointConfig,
    PoolConfig,
    RegimeConfig,
    WindowConfig,
    default_pools,
)
from undertow.data.config import (
    load_config as load_data_config,
)
from undertow.data.fixedpoint import (
    MAX_TICK,
    MIN_TICK,
    Q96,
    Q128,
    amounts_for_liquidity,
    liquidity_for_amounts,
    price_to_sqrt_price_x96,
    price_to_tick,
    sqrt_price_x96_to_price,
    sqrt_price_x96_to_tick,
    tick_to_price,
    tick_to_sqrt_price_x96,
)
from undertow.data.loader import (
    load_dataset,
    load_gas_feed,
    load_reference_feed,
    load_regime_labels,
    load_tape,
    tick_to_sqrt_price,
)
from undertow.data.schemas import (
    BURN_SCHEMA,
    COLLECT_SCHEMA,
    EVENT_TAPE_SCHEMA,
    FEE_GROWTH_SCHEMA,
    FLASH_SCHEMA,
    GAS_SCHEMA,
    MINT_SCHEMA,
    REFERENCE_SCHEMA,
    REGIME_SCHEMA,
    SCHEMA_REGISTRY,
    SCHEMA_VERSION,
    SORT_KEYS,
    SWAP_SCHEMA,
    decode_uint,
    empty_table,
    encode_uint,
    validate_table,
)
from undertow.data.storage.manifest import DatasetManifest
from undertow.data.transforms.align import Dataset
from undertow.data.types import (
    Address,
    BlockNumber,
    CheckResult,
    EventType,
    FeeTier,
    Regime,
    Route,
    Severity,
)
from undertow.data.types import (
    Regime as DataRegime,
)
from undertow.data.types import (
    Tick as DataTick,
)
from undertow.data.types import (
    ValidationError as ValidationError,  # kept importable; not in __all__ (pre-existing)
)
