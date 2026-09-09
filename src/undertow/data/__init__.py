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
]

from pathlib import Path

import pyarrow as pa

from undertow.data.config import (
    DataConfig,
    PoolConfig,
    WindowConfig,
    load_config,
)
from undertow.data.fixedpoint import (
    amounts_for_liquidity,
    liquidity_for_amounts,
    sqrt_price_x96_to_tick,
    tick_to_price,
    tick_to_sqrt_price_x96,
)
from undertow.data.pipeline import build
from undertow.data.schemas import EVENT_TAPE_SCHEMA, SCHEMA_VERSION
from undertow.data.storage.manifest import DatasetManifest
from undertow.data.transforms.align import Dataset
from undertow.data.types import FeeTier, Regime, Route, ValidationError


def load_dataset(
    config_or_path: str | Path | DataConfig,
    *,
    streams: frozenset[str] | None = None,
) -> Dataset:
    """Load a pre-built dataset from parquet cache (T16 public entry point).

    This is the function ``undertow.sim`` calls. It reads a dataset that was
    previously built by ``undertow-data snapshot`` — it does NOT fetch new data.

    Args:
        config_or_path: Either a ``DataConfig``, a path to a TOML config file,
            or a path to a dataset output directory (containing ``manifest.json``).
            When a directory is given, a minimal config is inferred from the
            manifest on disk.
        streams: Optional set of stream names to load a partial dataset.
            ``None`` loads everything. ``frozenset({"tape"})`` loads only the
            event tape + its required side-streams.

    Returns:
        A validated ``Dataset`` with the event tape and requested side-streams.

    Raises:
        ValidationError: If the dataset hasn't been built yet (no manifest, or
            required parquet files missing). The message names the
            ``undertow-data snapshot`` command that would produce it.
    """
    if isinstance(config_or_path, str | Path):
        path = Path(config_or_path)
        if path.is_dir() and (path / "manifest.json").is_file():
            # Path to a dataset output directory — infer config from manifest.
            from undertow.data.config import EndpointConfig, RegimeConfig
            from undertow.data.storage.manifest import read_manifest

            manifest = read_manifest(path)
            config = DataConfig(
                pool=manifest.pool,
                window=manifest.window,
                regime=RegimeConfig(),
                endpoints=EndpointConfig(
                    graph_url="",  # not needed for reading
                    rpc_url="",  # not needed for reading
                    reference_base_url="",  # not needed for reading
                ),
                cache_dir=path,  # dummy — not used for reading
                output_dir=path,
            )
        else:
            # Path to a TOML config file.
            config = load_config(str(path))
    else:
        config = config_or_path

    dataset = build(config)

    if streams is not None:
        # Partial load: clear non-requested side-streams.
        # The tape is always included (required by build).
        requested = streams | {"tape", "event_tape"}
        field_map: dict[str, str] = {
            "gas": "gas",
            "reference": "reference",
            "regime": "regime",
            "fee_growth": "fee_growth",
        }
        for stream_name, field_name in field_map.items():
            if stream_name not in requested:
                object.__setattr__(dataset, field_name, None)

    return dataset