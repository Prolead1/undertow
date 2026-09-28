"""S02 — ``undertow.data`` public-API surface and the sim/data import boundary.

Asserts the additive export extension mandated by
``docs/plans/sim/CONTRACTS.md`` §3 and the grep-able boundary rule of
``docs/plans/sim/PLAN.md`` §4: the sim may import only names exported from the
top-level ``undertow.data`` package, never an internal module.
"""

from __future__ import annotations

import importlib
import re
import shutil
import subprocess
from pathlib import Path

import pyarrow as pa
import pytest

import undertow.data as data

# ---------------------------------------------------------------------------
# The names CONTRACTS.md §3 requires ``undertow.data`` to export.
# ---------------------------------------------------------------------------

TYPES_NAMES = (
    "Address",
    "BlockNumber",
    "DataTick",
    "FeeTier",
    "DataRegime",
    "EventType",
    "CheckResult",
    "Severity",
)

CONFIG_NAMES = (
    "DataConfig",
    "PoolConfig",
    "WindowConfig",
    "RegimeConfig",
    "EndpointConfig",
    "load_data_config",
    "default_pools",
)

SCHEMA_NAMES = (
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
)

FIXEDPOINT_NAMES = (
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
)

LOADER_NAMES = (
    "load_dataset",
    "load_tape",
    "load_reference_feed",
    "load_gas_feed",
    "load_regime_labels",
)

CONTRACT_NAMES = (
    TYPES_NAMES + CONFIG_NAMES + SCHEMA_NAMES + FIXEDPOINT_NAMES + LOADER_NAMES
)

# The data internals that must NEVER leak into the sim-facing ``__all__``.
FETCHER_NAMES = (
    "BaseHttpFetcher",
    "FeeGrowthTracker",
    "GraphFetcher",
    "RpcFetcher",
    "GasFetcher",
    "ReferenceFetcher",
    "AbiFetcher",
)

BOUNDARY_MARKER = "from undertow.data."


def _repo_root() -> Path:
    """Repository root from this file: ``tests/sim/test_data_api.py`` -> repo root."""
    return Path(__file__).resolve().parents[2]


def test_undertow_data_imports() -> None:
    assert importlib.import_module("undertow.data") is data


@pytest.mark.parametrize("name", CONTRACT_NAMES)
def test_contract_name_is_in_all_and_importable(name: str) -> None:
    assert name in data.__all__, f"{name!r} missing from undertow.data.__all__"
    assert hasattr(data, name), f"{name!r} missing from the undertow.data namespace"


def test_no_duplicate_names_in_all() -> None:
    duplicates = sorted({name for name in data.__all__ if data.__all__.count(name) > 1})
    assert duplicates == [], f"duplicate __all__ entries: {duplicates}"


@pytest.mark.parametrize("name", FETCHER_NAMES)
def test_fetcher_names_are_not_exported(name: str) -> None:
    assert name not in data.__all__, f"internal fetcher {name!r} leaked into __all__"


def test_specific_import_works() -> None:
    from undertow.data import (
        Q96,
        Q128,
        price_to_tick,
        sqrt_price_x96_to_price,
        tick_to_price,
    )

    assert Q96 == 2**96
    assert Q128 == 2**128
    assert callable(sqrt_price_x96_to_price)
    assert callable(price_to_tick)
    assert callable(tick_to_price)


def test_tick_to_sqrt_price_absent_and_x96_present() -> None:
    """ADR-008: ``tick_to_sqrt_price`` is not a data name and is intentionally
    not exported. The exact-integer ``tick_to_sqrt_price_x96`` is."""
    assert "tick_to_sqrt_price" not in data.__all__
    assert not hasattr(data, "tick_to_sqrt_price")
    assert "tick_to_sqrt_price_x96" in data.__all__
    assert callable(data.tick_to_sqrt_price_x96)


@pytest.mark.parametrize(
    "loader_name, requested_stream, require_name, sort_cols, values",
    [
        (
            "load_tape",
            "tape",
            "event_tape",
            ("block_number", "log_index"),
            {"block_number": [3, 1, 2], "log_index": [7, 6, 5]},
        ),
        (
            "load_reference_feed",
            "reference",
            "reference",
            ("symbol", "open_time"),
            {"symbol": ["b", "a", "b"], "open_time": [3, 2, 1]},
        ),
        (
            "load_gas_feed",
            "gas",
            "gas",
            ("block_number",),
            {"block_number": [3, 1, 2]},
        ),
        (
            "load_regime_labels",
            "regime",
            "regime",
            ("timestamp",),
            {"timestamp": [3, 1, 2]},
        ),
    ],
)
def test_loaders_delegate_to_load_dataset_and_sort(
    monkeypatch: pytest.MonkeyPatch,
    loader_name: str,
    requested_stream: str,
    require_name: str,
    sort_cols: tuple[str, ...],
    values: dict[str, list[object]],
) -> None:
    """Each bridge adapter requests one stream, requires its table, and sorts it."""
    import undertow.data.loader as loader

    table = pa.table(values)
    calls: dict[str, object] = {}

    class _FakeDataset:
        def require(self, name: str) -> pa.Table:
            calls["require"] = name
            return table

    def fake_load_dataset(
        config: object, *, streams: frozenset[str] | None = None
    ) -> _FakeDataset:
        calls["streams"] = streams
        return _FakeDataset()

    monkeypatch.setattr(loader, "load_dataset", fake_load_dataset)

    result = getattr(loader, loader_name)(object())  # config is opaque to the adapter

    assert calls["streams"] == frozenset({requested_stream})
    assert calls["require"] == require_name
    assert result.equals(table.sort_by([(col, "ascending") for col in sort_cols]))


def test_sim_does_not_import_data_internals() -> None:
    """The string ``from undertow.data.`` must not appear anywhere in ``src/undertow/sim``."""
    sim_dir = _repo_root() / "src" / "undertow" / "sim"
    offenders: list[str] = []
    for path in sorted(sim_dir.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if BOUNDARY_MARKER in line:
                offenders.append(f"{path.relative_to(_repo_root())}:{lineno}: {line.strip()}")
    assert offenders == [], "sim imports an undertow.data internal module:\n" + "\n".join(offenders)


def test_sim_import_lines_target_only_top_level_data() -> None:
    """No *import line* in ``src/undertow/sim`` may name a dotted ``undertow.data.<submodule>``.

    This complements the literal string scan above: it catches ``import undertow.data.foo``
    (no ``from``) and is deliberately restricted to import statements so docstrings and
    prose mentions do not false-positive.
    """
    import_line = re.compile(r"^\s*(from|import)\s+undertow\.data\.")
    sim_dir = _repo_root() / "src" / "undertow" / "sim"
    offenders: list[str] = []
    for path in sorted(sim_dir.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if import_line.match(line):
                offenders.append(f"{path.relative_to(_repo_root())}:{lineno}: {line.strip()}")
    assert offenders == [], (
        "sim import line reaches an undertow.data submodule:\n" + "\n".join(offenders)
    )


def test_sim_boundary_grep_ripgrep() -> None:
    """Cross-check the boundary with ripgrep when it is installed."""
    rg = shutil.which("rg")
    if rg is None:
        pytest.skip("ripgrep not installed; pure-Python boundary scan covers this")
    sim_dir = _repo_root() / "src" / "undertow" / "sim"
    result = subprocess.run(
        [rg, "--no-filename", BOUNDARY_MARKER, str(sim_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    assert result.stdout.strip() == "", f"ripgrep found boundary violations:\n{result.stdout}"
