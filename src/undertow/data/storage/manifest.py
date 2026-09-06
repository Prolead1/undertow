"""Dataset manifest: provenance and versioning (T09, ``CONTRACTS.md`` §7).

The manifest answers "what dataset is this, from what, produced when". It is
human-readable JSON (indent 2, sorted keys) so it diffs cleanly in git.

Rules this module enforces:

- ``dataset_id`` is a **deterministic hash of (pool address, window, sorted route
  set, schema_version)** — never of the wall clock. The same logical dataset
  yields the same id, so a re-pull is recognizable as the same thing.
- Wall clock lives in the manifest (``created_at_utc``), never in the parquet data.
- ``git_commit`` is ``git rev-parse HEAD`` with a ``+dirty`` suffix when the
  working tree has uncommitted changes; ``"unknown"`` when git is unavailable
  (never a crash).
- **Secrets never enter the manifest.** Endpoints are recorded as host only
  (``https://gateway.thegraph.com``), never the full URL, never a key.
- ``read_manifest`` rejects a manifest whose ``schema_version`` differs in its
  major component from the current ``SCHEMA_VERSION`` (``ValidationError`` telling
  the user to re-pull) — a schema change must not silently mix old and new parquet.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

from undertow.data.config import EndpointConfig, PoolConfig, WindowConfig
from undertow.data.schemas import SCHEMA_VERSION
from undertow.data.types import Address, BlockNumber, FeeTier, Route, ValidationError

_MANIFEST_FILENAME: Final[str] = "manifest.json"
_VERSION_LIBRARIES: Final[tuple[str, ...]] = ("pyarrow", "polars", "numpy", "requests", "undertow")


@dataclass(frozen=True, slots=True)
class StreamManifest:
    """Per-stream provenance record (T09 brief §2). ``route`` is the ``Route`` enum value."""

    row_count: int
    content_hash: str
    min_block: BlockNumber | None = None  # None for streams without a block_number column
    max_block: BlockNumber | None = None
    route: Route | None = None
    n_requests: int = 0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """The complete provenance record for one dataset (``CONTRACTS.md`` §7).

    Field order follows the contract exactly; ``dataset_id`` is derived
    (``init=False``) rather than caller-supplied so it can never drift from its
    inputs. ``endpoint_hosts`` is an extension over the contract's field list:
    provenance must record *where* the data came from, but only the host.
    """

    schema_version: str
    dataset_id: str = field(init=False)  # computed in __post_init__ from the other fields
    pool: PoolConfig
    window: WindowConfig
    streams: dict[str, StreamManifest]
    created_at_utc: datetime  # wall clock lives HERE, not in the data
    git_commit: str
    library_versions: dict[str, str]
    warnings: tuple[str, ...] = ()
    endpoint_hosts: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.created_at_utc.tzinfo is None or self.created_at_utc.utcoffset() is None:
            raise ValidationError(
                "DatasetManifest: created_at_utc must be timezone-aware (naive datetimes "
                "are not allowed)"
            )
        object.__setattr__(self, "created_at_utc", self.created_at_utc.astimezone(UTC))
        routes = [sm.route for sm in self.streams.values() if sm.route is not None]
        object.__setattr__(
            self,
            "dataset_id",
            compute_dataset_id(self.pool, self.window, routes, self.schema_version),
        )


# ---------------------------------------------------------------------------
# dataset_id
# ---------------------------------------------------------------------------


def compute_dataset_id(
    pool: PoolConfig,
    window: WindowConfig,
    routes: Sequence[Route | str],
    schema_version: str,
) -> str:
    """Deterministic id of the logical dataset: sha256 of the canonical JSON of
    (pool address, window, sorted route set, schema_version). Independent of the
    wall clock, of row counts, and of stream contents — re-pulling the same logical
    dataset yields the same id.
    """
    route_set = sorted({r.value if isinstance(r, Route) else str(r) for r in routes})
    payload = {
        "pool_address": pool.address,
        "window": _window_to_json(window),
        "routes": route_set,
        "schema_version": schema_version,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Secrets: endpoint hosts only
# ---------------------------------------------------------------------------


def _host_only(url: str) -> str:
    """``scheme://host[:port]`` for ``url`` — never a path, query or credentials.

    Mirrors ``config._host_only`` (T01-owned; private there, so not importable
    across modules without breaking its encapsulation).
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<invalid-url>"
    netloc = parts.netloc
    if "@" in netloc:  # strip any userinfo the URL might carry
        netloc = netloc.rsplit("@", 1)[1]
    return f"{parts.scheme}://{netloc}"


def endpoint_hosts(endpoints: EndpointConfig) -> dict[str, str]:
    """The host-only view of ``endpoints`` — the ONLY endpoint data a manifest stores."""
    return {
        "graph": _host_only(endpoints.graph_url),
        "rpc": _host_only(endpoints.rpc_url),
        "reference": _host_only(endpoints.reference_base_url),
    }


# ---------------------------------------------------------------------------
# Environment provenance
# ---------------------------------------------------------------------------


def git_commit() -> str:
    """HEAD sha of the repo the dataset was built from, ``+dirty`` when the working
    tree has uncommitted changes, ``"unknown"`` when git is unavailable. Never raises."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = head.stdout.strip()
    if not commit:
        return "unknown"
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return commit  # cannot determine dirtiness; do not guess
    if status.stdout.strip():
        return f"{commit}+dirty"
    return commit


def library_versions() -> dict[str, str]:
    """Installed versions of the libraries that could affect the dataset's content."""
    out: dict[str, str] = {}
    for name in _VERSION_LIBRARIES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "unknown"
    return out


# ---------------------------------------------------------------------------
# JSON (de)serialization
# ---------------------------------------------------------------------------


def _pool_to_json(pool: PoolConfig) -> dict[str, Any]:
    return {
        "address": pool.address,
        "token0_symbol": pool.token0_symbol,
        "token1_symbol": pool.token1_symbol,
        "token0_decimals": pool.token0_decimals,
        "token1_decimals": pool.token1_decimals,
        "fee_tier": pool.fee_tier.value,
        "tick_spacing": pool.tick_spacing,
        "deployment_block": pool.deployment_block,
    }


def _window_to_json(window: WindowConfig) -> dict[str, Any]:
    def iso(value: datetime | None) -> str | None:
        return value.isoformat() if value is not None else None

    return {
        "start_block": window.start_block,
        "end_block": window.end_block,
        "start_utc": iso(window.start_utc),
        "end_utc": iso(window.end_utc),
        "train_end_utc": iso(window.train_end_utc),
        "eval_start_utc": iso(window.eval_start_utc),
    }


def _stream_to_json(stream: StreamManifest) -> dict[str, Any]:
    return {
        "row_count": stream.row_count,
        "min_block": stream.min_block,
        "max_block": stream.max_block,
        "content_hash": stream.content_hash,
        "route": stream.route.value if stream.route is not None else None,
        "n_requests": stream.n_requests,
        "warnings": list(stream.warnings),
    }


def _manifest_to_json(manifest: DatasetManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "dataset_id": manifest.dataset_id,
        "pool": _pool_to_json(manifest.pool),
        "window": _window_to_json(manifest.window),
        "streams": {name: _stream_to_json(sm) for name, sm in manifest.streams.items()},
        "created_at_utc": manifest.created_at_utc.isoformat(),
        "git_commit": manifest.git_commit,
        "library_versions": dict(sorted(manifest.library_versions.items())),
        "warnings": list(manifest.warnings),
        "endpoint_hosts": dict(sorted(manifest.endpoint_hosts.items())),
    }


def write_manifest(manifest: DatasetManifest, root: Path) -> Path:
    """Write ``manifest.json`` under ``root`` (pretty-printed, sorted keys, atomic
    temp-file replace so an interrupted write never leaves a half-written manifest)."""
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    target = root_path / _MANIFEST_FILENAME
    payload = _manifest_to_json(manifest)
    fd, tmp = tempfile.mkstemp(prefix=".manifest.json.", suffix=".tmp", dir=root_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, target)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
    return target


def _major_component(version: str) -> int | None:
    major, _, _ = version.partition(".")
    try:
        return int(major)
    except ValueError:
        return None


def read_manifest(root: Path) -> DatasetManifest:
    """Read and validate ``root/manifest.json``.

    Rejects a manifest whose ``schema_version`` major component differs from the
    current ``SCHEMA_VERSION`` with a ``ValidationError`` telling the caller to
    re-pull (a schema change must never silently mix old and new parquet).
    Unknown keys in the file are ignored so older readers tolerate additive fields.
    """
    root_path = Path(root)
    path = root_path / _MANIFEST_FILENAME
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ValidationError(
            f"read_manifest: no manifest at {path}; write the dataset first"
        ) from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValidationError(f"read_manifest: {path} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValidationError(
            f"read_manifest: {path} must contain a JSON object, got {type(data).__name__}"
        )

    stored_version = data.get("schema_version")
    if not isinstance(stored_version, str):
        raise ValidationError(
            f"read_manifest: {path} has no string schema_version; re-pull the dataset"
        )
    if _major_component(stored_version) != _major_component(SCHEMA_VERSION):
        raise ValidationError(
            f"read_manifest: manifest schema_version {stored_version!r} is incompatible "
            f"with the current {SCHEMA_VERSION!r}; re-pull the dataset"
        )

    pool_data = data.get("pool")
    window_data = data.get("window")
    if not isinstance(pool_data, dict) or not isinstance(window_data, dict):
        raise ValidationError(f"read_manifest: {path} is missing pool/window sections")

    streams_raw = data.get("streams", {})
    if not isinstance(streams_raw, dict):
        raise ValidationError(
            f"read_manifest: {path} 'streams' must be an object, "
            f"got {type(streams_raw).__name__}"
        )

    try:
        pool = _pool_from_json(pool_data)
        window = _window_from_json(window_data)
        streams = {
            name: _stream_from_json(name, sm) for name, sm in streams_raw.items()
        }
        created = _parse_aware_utc(data.get("created_at_utc"))
        warnings_raw = data.get("warnings", ())
        if not isinstance(warnings_raw, list):
            raise ValueError(
                f"manifest field 'warnings' must be a list, got {type(warnings_raw).__name__}"
            )
        warnings = tuple(str(w) for w in warnings_raw)
        endpoint_hosts_data = data.get("endpoint_hosts", {})
        hosts = (
            {str(k): str(v) for k, v in endpoint_hosts_data.items()}
            if isinstance(endpoint_hosts_data, dict)
            else {}
        )
    except (TypeError, ValueError) as e:
        raise ValidationError(f"read_manifest: {path} is corrupt: {e}") from e

    library_versions_data = data.get("library_versions")
    versions = (
        {str(k): str(v) for k, v in library_versions_data.items()}
        if isinstance(library_versions_data, dict)
        else {}
    )
    git = data.get("git_commit")
    git_commit_value = str(git) if isinstance(git, str) else "unknown"
    return DatasetManifest(
        schema_version=stored_version,
        pool=pool,
        window=window,
        streams=streams,
        created_at_utc=created,
        git_commit=git_commit_value,
        library_versions=versions,
        warnings=tuple(warnings),
        endpoint_hosts=hosts,
    )


def _parse_aware_utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"expected an ISO-8601 string, got {value!r}")
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{value!r} is not an ISO-8601 timestamp") from None
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{value!r} must be timezone-aware")
    return dt.astimezone(UTC)


def _pool_from_json(data: Mapping[str, object]) -> PoolConfig:
    return PoolConfig(
        address=Address(_need_str(data, "address")),
        token0_symbol=_need_str(data, "token0_symbol"),
        token1_symbol=_need_str(data, "token1_symbol"),
        token0_decimals=_need_int(data, "token0_decimals"),
        token1_decimals=_need_int(data, "token1_decimals"),
        fee_tier=FeeTier(_need_int(data, "fee_tier")),
        tick_spacing=_need_int(data, "tick_spacing"),
        deployment_block=BlockNumber(_need_int(data, "deployment_block")),
    )


def _window_from_json(data: Mapping[str, object]) -> WindowConfig:
    def opt_int(key: str) -> BlockNumber | None:
        value = data.get(key)
        if value is None:
            return None
        if not isinstance(value, int):
            raise ValueError(f"window.{key} must be an integer or null")
        return BlockNumber(value)

    def opt_utc(key: str) -> datetime | None:
        value = data.get(key)
        if value is None:
            return None
        return _parse_aware_utc(value)

    return WindowConfig(
        start_block=opt_int("start_block"),
        end_block=opt_int("end_block"),
        start_utc=opt_utc("start_utc"),
        end_utc=opt_utc("end_utc"),
        train_end_utc=opt_utc("train_end_utc"),
        eval_start_utc=opt_utc("eval_start_utc"),
    )


def _stream_from_json(name: str, data: Mapping[str, object]) -> StreamManifest:
    if not isinstance(data, dict):
        raise ValueError(f"streams.{name} must be an object")
    route_raw = data.get("route")
    route: Route | None = None
    if route_raw is not None:
        try:
            route = Route(str(route_raw))
        except ValueError as e:
            raise ValueError(f"streams.{name}.route: unknown route {route_raw!r}") from e
    warnings = tuple(str(w) for w in data.get("warnings", ()))
    return StreamManifest(
        row_count=_need_int(data, "row_count"),
        content_hash=_need_str(data, "content_hash"),
        min_block=_opt_block(data.get("min_block")),
        max_block=_opt_block(data.get("max_block")),
        route=route,
        n_requests=_opt_int(data, "n_requests"),
        warnings=warnings,
    )


def _opt_block(value: object) -> BlockNumber | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"expected an integer or null block number, got {value!r}")
    return BlockNumber(value)


def _need_str(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"manifest field {key!r} must be a string, got {value!r}")
    return value


def _need_int(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"manifest field {key!r} must be an integer, got {value!r}")
    return value


def _opt_int(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"manifest field {key!r} must be an integer, got {value!r}")
    return value