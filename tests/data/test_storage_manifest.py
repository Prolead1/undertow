"""Tests for ``undertow.data.storage.manifest`` (T09) — provenance and versioning.

Covers the manifest behaviours the T09 brief pins: the dataclass round-trip with
UTC-aware timestamps, ``dataset_id`` determinism over (pool, window, route set,
schema_version) with independence from the wall clock, secret hygiene (endpoints
recorded host-only — a key in a URL must never survive serialization), the
schema-version guard (major mismatch => ``ValidationError`` telling the user to
re-pull), the dirty-tree git marker, and the diff-friendliness of ``manifest.json``
(indent 2, sorted keys). All tests are offline (tmp dirs + monkeypatched
``subprocess``).
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest

from undertow.data.config import EndpointConfig, WindowConfig, default_pools
from undertow.data.schemas import SCHEMA_VERSION
from undertow.data.storage.manifest import (
    DatasetManifest,
    StreamManifest,
    compute_dataset_id,
    endpoint_hosts,
    git_commit,
    library_versions,
    read_manifest,
    write_manifest,
)
from undertow.data.types import BlockNumber, Route, ValidationError

POOL_A = default_pools()["USDC_WETH_3000"]
WINDOW = WindowConfig(start_block=BlockNumber(10_000_000), end_block=BlockNumber(20_000_000))

_FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _streams() -> dict[str, StreamManifest]:
    """Per-stream records covering every field, incl. block-less streams (min/max None)."""
    return {
        "swap": StreamManifest(
            row_count=3,
            min_block=BlockNumber(10_000_001),
            max_block=BlockNumber(10_000_003),
            content_hash="a" * 64,
            route=Route.THEGRAPH,
            n_requests=2,
            warnings=("one page truncated",),
        ),
        "gas": StreamManifest(row_count=0, content_hash="b" * 64, route=Route.RPC, n_requests=1),
        "reference": StreamManifest(
            row_count=10, content_hash="c" * 64, route=Route.REFERENCE, n_requests=4
        ),
    }


def _manifest(**overrides: object) -> DatasetManifest:
    kwargs: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "pool": POOL_A,
        "window": WINDOW,
        "streams": _streams(),
        "created_at_utc": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
        "git_commit": "deadbeefcafe",
        "library_versions": {
            "pyarrow": "25.0.1",
            "polars": "1.0.0",
            "numpy": "2.0.0",
            "requests": "2.32.0",
            "undertow": "0.1.0",
        },
    }
    kwargs.update(overrides)
    return DatasetManifest(**kwargs)


# ---------------------------------------------------------------------------
# 9. Manifest round-trip
# ---------------------------------------------------------------------------


def test_manifest_roundtrip_dataclass_equality(tmp_path: Path) -> None:
    offset = timezone(timedelta(hours=2))
    manifest = _manifest(created_at_utc=datetime(2024, 5, 6, 7, 8, 9, tzinfo=offset))
    path = write_manifest(manifest, tmp_path)
    assert path == tmp_path / "manifest.json"

    back = read_manifest(tmp_path)
    assert back is not manifest
    assert back == manifest  # full dataclass equality, including dataset_id
    # created_at_utc is normalized to UTC and stays aware.
    assert back.created_at_utc == datetime(2024, 5, 6, 5, 8, 9, tzinfo=UTC)
    assert back.created_at_utc.utcoffset() == timedelta(0)
    # Per-stream fields survive, including block-less (None) bounds and the enum route.
    assert back.streams["swap"].route == Route.THEGRAPH
    assert back.streams["swap"].warnings == ("one page truncated",)
    assert back.streams["gas"].max_block is None
    assert back.streams["reference"].content_hash == "c" * 64
    assert back.schema_version == SCHEMA_VERSION


def test_manifest_naive_created_at_rejected() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        _manifest(created_at_utc=datetime(2024, 1, 1, 0, 0, 0))  # no tzinfo


def test_manifest_json_is_pretty_sorted_and_readable(tmp_path: Path) -> None:
    write_manifest(_manifest(), tmp_path)
    text = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert "\n  " in text  # indent=2
    payload = json.loads(text)
    assert list(payload) == sorted(payload)  # sort_keys=True -> diff-friendly
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["dataset_id"] == _manifest().dataset_id


# ---------------------------------------------------------------------------
# 10. dataset_id determinism
# ---------------------------------------------------------------------------


def test_dataset_id_deterministic_and_input_sensitive() -> None:
    routes = [Route.RPC, Route.THEGRAPH, Route.REFERENCE]
    base = compute_dataset_id(POOL_A, WINDOW, routes, SCHEMA_VERSION)
    assert re.fullmatch(r"[0-9a-f]{64}", base)

    # Same logical dataset -> same id, regardless of route order or enum/str mixing.
    assert base == compute_dataset_id(POOL_A, WINDOW, list(reversed(routes)), SCHEMA_VERSION)
    assert base == compute_dataset_id(
        POOL_A, WINDOW, ["rpc", "thegraph", "reference"], SCHEMA_VERSION
    )
    # A different window is a different dataset.
    other_window = WindowConfig(start_block=BlockNumber(1), end_block=BlockNumber(2))
    assert base != compute_dataset_id(POOL_A, other_window, routes, SCHEMA_VERSION)
    # A different route set or schema version is a different dataset.
    assert base != compute_dataset_id(POOL_A, WINDOW, [Route.RPC], SCHEMA_VERSION)
    assert base != compute_dataset_id(POOL_A, WINDOW, routes, "1.1.0")


def test_dataset_id_ignores_wall_clock() -> None:
    m1 = _manifest(created_at_utc=datetime(2024, 1, 1, tzinfo=UTC))
    m2 = _manifest(created_at_utc=datetime(2025, 7, 7, tzinfo=UTC))
    assert m1.dataset_id == m2.dataset_id
    expected = compute_dataset_id(
        POOL_A, WINDOW, [Route.THEGRAPH, Route.REFERENCE, Route.RPC], SCHEMA_VERSION
    )
    assert m1.dataset_id == expected


# ---------------------------------------------------------------------------
# 11. Secret hygiene — endpoints stored host-only
# ---------------------------------------------------------------------------


def test_endpoint_hosts_never_include_secrets() -> None:
    endpoints = EndpointConfig(
        graph_url="https://gateway.thegraph.com/v0/subgraphs?x-api-key=abc123KEY",
        rpc_url="https://user:abc123KEY@archive.example.com/rpc",
        reference_base_url="https://data-api.binance.vision/api/v3",
    )
    hosts = endpoint_hosts(endpoints)
    assert hosts == {
        "graph": "https://gateway.thegraph.com",
        "rpc": "https://archive.example.com",
        "reference": "https://data-api.binance.vision",
    }


def test_manifest_json_contains_no_secret_and_no_full_url(tmp_path: Path) -> None:
    endpoints = EndpointConfig(
        graph_url="https://gateway.thegraph.com/subgraphs?id=abc123KEY",
        rpc_url="https://rpc.example.com/?token=abc123KEY",
        reference_base_url="https://data-api.binance.vision",
    )
    write_manifest(_manifest(endpoint_hosts=endpoint_hosts(endpoints)), tmp_path)
    text = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert "abc123KEY" not in text
    assert "gateway.thegraph.com/subgraphs" not in text  # never the full URL with path
    assert "rpc.example.com" in text  # the host itself is fine and expected
    # The manifest field contains only scheme://host values.
    payload = json.loads(text)
    assert payload["endpoint_hosts"]["graph"] == "https://gateway.thegraph.com"
    assert payload["endpoint_hosts"]["rpc"] == "https://rpc.example.com"


# ---------------------------------------------------------------------------
# 12. Schema-version guard
# ---------------------------------------------------------------------------


def _rewrite_schema_version(raw: str, tmp_path: Path) -> None:
    write_manifest(_manifest(), tmp_path)
    path = tmp_path / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = raw
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_schema_version_major_mismatch_rejected_with_repull_hint(tmp_path: Path) -> None:
    _rewrite_schema_version("0.9.0", tmp_path)
    with pytest.raises(ValidationError, match="re-pull"):
        read_manifest(tmp_path)


def test_schema_version_same_major_accepted(tmp_path: Path) -> None:
    _rewrite_schema_version("1.2.3", tmp_path)  # compatible: major component unchanged
    back = read_manifest(tmp_path)
    assert back.schema_version == "1.2.3"


def test_manifest_missing_schema_version_rejected(tmp_path: Path) -> None:
    write_manifest(_manifest(), tmp_path)
    path = tmp_path / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["schema_version"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="schema_version"):
        read_manifest(tmp_path)


def test_read_manifest_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="no manifest"):
        read_manifest(tmp_path)


def test_read_manifest_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValidationError, match="not valid JSON"):
        read_manifest(tmp_path)


def test_read_manifest_rejects_streams_as_non_object(tmp_path: Path) -> None:
    write_manifest(_manifest(), tmp_path)
    path = tmp_path / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["streams"] = ["not", "an", "object"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="'streams' must be an object"):
        read_manifest(tmp_path)


def test_read_manifest_rejects_non_list_warnings(tmp_path: Path) -> None:
    write_manifest(_manifest(), tmp_path)
    path = tmp_path / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["warnings"] = {"not": "a list"}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="'warnings' must be a list"):
        read_manifest(tmp_path)


# ---------------------------------------------------------------------------
# 13. git provenance
# ---------------------------------------------------------------------------


def _fake_git(*, dirty: bool = False) -> Callable[..., SimpleNamespace]:
    def run(cmd: list[str], **_: object) -> SimpleNamespace:
        if cmd == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=f"{_FAKE_COMMIT}\n")
        if cmd == ["git", "status", "--porcelain"]:
            return SimpleNamespace(
                stdout=" M src/undertow/data/storage/parquet.py\n" if dirty else ""
            )
        raise AssertionError(f"unexpected git command {cmd!r}")

    return run


def test_git_commit_dirty_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", _fake_git(dirty=True))
    assert git_commit() == f"{_FAKE_COMMIT}+dirty"


def test_git_commit_clean_no_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", _fake_git(dirty=False))
    assert git_commit() == _FAKE_COMMIT


def test_git_commit_unavailable_returns_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: object, **__: object) -> NoReturn:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", boom)
    assert git_commit() == "unknown"


# ---------------------------------------------------------------------------
# Environment provenance
# ---------------------------------------------------------------------------


def test_library_versions_has_expected_keys() -> None:
    versions = library_versions()
    assert set(versions) == {"pyarrow", "polars", "numpy", "requests", "undertow"}
    assert all(isinstance(v, str) for v in versions.values())