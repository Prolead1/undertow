"""Bridge adapters from ``undertow.data`` internals to the sim-facing public API.

Owned by S02 (``docs/plans/sim/CONTRACTS.md`` §3). These thin wrappers are the *only*
sanctioned way for ``undertow.sim`` to obtain data: the sim imports them from the
top-level :mod:`undertow.data` package and never reaches into
``undertow.data.pipeline`` / ``.storage`` / ``.transforms`` directly.

The loaders deliberately delegate to the real internal modules (``pipeline.build``
via :func:`load_dataset`) rather than re-implementing any I/O. When the data plan's
T16 ships the final public surface these adapters stay thin — they are the seam, not
the implementation.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from undertow.data.config import DataConfig, EndpointConfig, RegimeConfig, load_config
from undertow.data.pipeline import build
from undertow.data.schemas import SORT_KEYS
from undertow.data.storage.manifest import read_manifest
from undertow.data.transforms.align import Dataset


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


def _sorted(table: pa.Table, stream: str) -> pa.Table:
    """Return ``table`` sorted by its canonical sort keys (``SCHEMAS.SORT_KEYS``)."""
    return table.sort_by([(col, "ascending") for col in SORT_KEYS[stream]])


def load_tape(config: DataConfig) -> pa.Table:
    """Load the block-ordered event tape (``EVENT_TAPE_SCHEMA``), sorted by the
    canonical ``SORT_KEYS["event_tape"]`` = ``("block_number", "log_index")``.

    The tape step index (``seq``) is equivalent to this ordering: the dataset is
    block-ordered and ``log_index`` breaks ties within a block, so ``seq`` is the
    row position under these keys. The tape is the canonical record for the
    backtester (S11): one row per on-chain log, with the reference price, regime,
    gas and fee-growth columns already joined.
    """
    dataset = load_dataset(config, streams=frozenset({"tape"}))
    return _sorted(dataset.require("event_tape"), "event_tape")


def load_reference_feed(config: DataConfig) -> pa.Table:
    """Load the reference price feed (``REFERENCE_SCHEMA``), sorted by the canonical
    ``SORT_KEYS["reference"]`` = ``("symbol", "open_time")``.

    Note the key is the pair, not ``open_time`` alone: rows are grouped by
    ``symbol`` first, then ordered by ``open_time`` within each symbol.
    """
    dataset = load_dataset(config, streams=frozenset({"reference"}))
    return _sorted(dataset.require("reference"), "reference")


def load_gas_feed(config: DataConfig) -> pa.Table:
    """Load the per-block gas feed (``GAS_SCHEMA``), sorted by ``block_number``."""
    dataset = load_dataset(config, streams=frozenset({"gas"}))
    return _sorted(dataset.require("gas"), "gas")


def load_regime_labels(config: DataConfig) -> pa.Table:
    """Load the regime labels (``REGIME_SCHEMA``), sorted by ``timestamp``."""
    dataset = load_dataset(config, streams=frozenset({"regime"}))
    return _sorted(dataset.require("regime"), "regime")


# ---------------------------------------------------------------------------
# Fixed-point adapter pending T16.
# ---------------------------------------------------------------------------


def tick_to_sqrt_price(tick: int) -> float:
    """Human sqrt-price ``sqrt(1.0001 ** tick)`` at ``tick``.

    Pending T16 — data public API not yet complete. The internal
    ``undertow.data.fixedpoint`` module currently exposes only the exact-integer
    ``tick_to_sqrt_price_x96``; T16 owns the final fixedpoint surface, so this
    adapter is a placeholder rather than a re-implementation.
    """
    # T16 owns the final data public API; do not guess the intended units/return type.
    raise NotImplementedError("Pending T16 — data public API not yet complete")
