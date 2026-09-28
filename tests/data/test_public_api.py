"""T16 public-API guard — restored and reconciled by ADR-008.

The data plan's T16 brief required a set-equality test over ``undertow.data.__all__``
that was never committed. ADR-008 settles the sim-facing surface and this test pins
it: both a **missing** and an **extra** export fail.

Authoritative decision:
``docs/decisions/008-data-sim-public-api-reconciliation.md``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import undertow.data as data

# The settled sim-facing surface (ADR-008): T16's 16-name baseline plus the names
# sim ``CONTRACTS.md`` §3 requires, minus the phantom ``tick_to_sqrt_price``.
EXPECTED_SURFACE: frozenset[str] = frozenset(
    {
        # -- Entry point --
        "load_dataset",
        # -- Bridge loaders --
        "load_tape",
        "load_reference_feed",
        "load_gas_feed",
        "load_regime_labels",
        # -- Core / dataset / config --
        "Dataset",
        "DatasetManifest",
        "DataConfig",
        "PoolConfig",
        "WindowConfig",
        "RegimeConfig",
        "EndpointConfig",
        "default_pools",
        "load_data_config",
        # -- Types --
        "Address",
        "BlockNumber",
        "DataTick",
        "DataRegime",
        "EventType",
        "CheckResult",
        "Severity",
        "FeeTier",
        "Route",
        "Regime",
        # -- Schemas --
        "SWAP_SCHEMA",
        "MINT_SCHEMA",
        "BURN_SCHEMA",
        "COLLECT_SCHEMA",
        "FLASH_SCHEMA",
        "FEE_GROWTH_SCHEMA",
        "GAS_SCHEMA",
        "REFERENCE_SCHEMA",
        "REGIME_SCHEMA",
        "EVENT_TAPE_SCHEMA",
        "SCHEMA_REGISTRY",
        "SCHEMA_VERSION",
        "SORT_KEYS",
        "encode_uint",
        "decode_uint",
        "validate_table",
        "empty_table",
        # -- Fixed-point --
        "Q96",
        "Q128",
        "TICK_BASE",
        "MIN_TICK",
        "MAX_TICK",
        "sqrt_price_x96_to_price",
        "price_to_sqrt_price_x96",
        "price_to_tick",
        "tick_to_price",
        "tick_to_sqrt_price_x96",
        "sqrt_price_x96_to_tick",
        "liquidity_for_amounts",
        "amounts_for_liquidity",
    }
)

FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "BaseHttpFetcher",
        "FeeGrowthTracker",
        "GraphFetcher",
        "RpcFetcher",
        "GasFetcher",
        "ReferenceFetcher",
        "AbiFetcher",
    }
)


def test_all_matches_settled_surface_exactly() -> None:
    """Set equality: a missing OR an extra export fails (T16's intent)."""
    actual = frozenset(data.__all__)
    assert actual == EXPECTED_SURFACE, (
        f"missing: {sorted(EXPECTED_SURFACE - actual)}; "
        f"unexpected: {sorted(actual - EXPECTED_SURFACE)}"
    )


def test_all_has_no_duplicates() -> None:
    duplicates = sorted({n for n in data.__all__ if data.__all__.count(n) > 1})
    assert duplicates == [], f"duplicate __all__ entries: {duplicates}"


@pytest.mark.parametrize("name", sorted(EXPECTED_SURFACE))
def test_every_export_is_importable_and_not_none(name: str) -> None:
    assert hasattr(data, name), f"{name!r} in __all__ but absent from the namespace"


@pytest.mark.parametrize("name", sorted(FORBIDDEN_NAMES))
def test_internals_do_not_leak(name: str) -> None:
    assert name not in data.__all__, f"internal {name!r} leaked into the public surface"
    assert not hasattr(data, name), f"internal {name!r} is reachable from undertow.data"


def test_undertow_data_does_not_import_sim() -> None:
    """Architecture guard: ``undertow.data`` must never import ``undertow.sim``."""
    data_dir = Path(data.__file__).resolve().parent
    offenders: list[str] = []
    for path in sorted(data_dir.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("import undertow.sim", "from undertow.sim")):
                offenders.append(f"{path}:{lineno}: {stripped}")
    assert offenders == [], "undertow.data imports undertow.sim:\n" + "\n".join(offenders)
