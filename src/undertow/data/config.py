"""Configuration, pinned constants and the pool registry for ``undertow.data``.

Owned by T01 (``CONTRACTS.md`` §2). This module reads TOML and expands ``${ENV_VAR}``
placeholders **in endpoint strings only** at load time. It performs no network I/O and no math
beyond integer / ``Decimal`` constants.

Secrets never appear in log lines, exception messages, ``repr`` output or the manifest — they are
read from the environment in this module only. The expanded value of an endpoint URL must never be
emitted anywhere; log/record the *host* only (see ``EndpointConfig.__repr__``).

The block/date window semantics are documented in ``docs/decisions/003-window-blocks-optional.md``
(repo) / ``.pi/plans/data-pipeline/adr/003-window-blocks-optional.md`` (vault).
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlsplit

from undertow.data.types import TICK_SPACING, Address, BlockNumber, ConfigError, FeeTier

# ---------------------------------------------------------------------------
# Pinned constants (CONTRACTS.md §2). No float anywhere: Q-numbers are ints,
# TICK_BASE is an exact Decimal.
# ---------------------------------------------------------------------------
Q96: int = 2**96
Q128: int = 2**128
TICK_BASE: Decimal = Decimal("1.0001")
MIN_TICK: int = -887272
MAX_TICK: int = 887272
GLOBAL_TICK_SENTINEL: int = -(2**31)  # int32 minimum — marks global-only fee-growth rows

# Order-of-magnitude gas estimates for the NonfungiblePositionManager path (mint/burn/collect)
# plus a ballpark figure for a swap; instead of a `reportGasUsed` path. These are deliberately
# NOT hardcoded into transforms: they are overridable per-configuration via the `[gas.units]`
# TOML section, so `undertow.sim` / T14 can calibrate them against real receipts later.
GAS_UNITS: Mapping[str, int] = MappingProxyType(
    {
        "mint": 400_000,
        "burn": 250_000,
        "collect": 150_000,
        "swap": 180_000,
    }
)


@dataclass(frozen=True, slots=True)
class PoolConfig:
    """A Uniswap V3 pool. ``address`` is lowercase, checksum-stripped; decimals are 0..18."""

    address: Address
    token0_symbol: str
    token1_symbol: str
    token0_decimals: int
    token1_decimals: int
    fee_tier: FeeTier
    tick_spacing: int  # must equal TICK_SPACING[fee_tier]; validated in __post_init__
    deployment_block: BlockNumber

    def __post_init__(self) -> None:
        addr = str(self.address)
        if (
            not addr.startswith("0x")
            or len(addr) != 42
            or any(c not in "0123456789abcdef" for c in addr[2:])
        ):
            raise ConfigError(
                "PoolConfig: address must be a lowercase, checksum-stripped "
                f"0x-prefixed 42-char hex string; got {addr!r}"
            )
        if self.fee_tier not in TICK_SPACING:
            raise ConfigError(
                f"PoolConfig: fee_tier {self.fee_tier!r} is not one of "
                f"{sorted(e.value for e in FeeTier)}"
            )
        expected = TICK_SPACING[self.fee_tier]
        if self.tick_spacing != expected:
            raise ConfigError(
                f"PoolConfig: tick_spacing {self.tick_spacing} must equal "
                f"TICK_SPACING[{self.fee_tier!r}] = {expected}"
            )
        if not 0 <= self.token0_decimals <= 18:
            raise ConfigError(
                f"PoolConfig: token0_decimals {self.token0_decimals} must be within 0..18"
            )
        if not 0 <= self.token1_decimals <= 18:
            raise ConfigError(
                f"PoolConfig: token1_decimals {self.token1_decimals} must be within 0..18"
            )
        if self.deployment_block <= 0:
            raise ConfigError(f"PoolConfig: deployment_block {self.deployment_block} must be > 0")


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """One study window: block bounds XOR date bounds (exactly one pair is set).

    Dates are resolved to blocks by T07's block index; ``end_block`` / ``end_utc`` are inclusive.
    The optional ``train_end_utc`` / ``eval_start_utc`` pair is a walk-forward split hook that
    ``undertow.data`` NEVER acts on — it is only recorded in the manifest so a chosen split is
    provenance-tracked. Semantics per ADR-003.
    """

    start_block: BlockNumber | None = None
    end_block: BlockNumber | None = None
    start_utc: datetime | None = None
    end_utc: datetime | None = None
    train_end_utc: datetime | None = None
    eval_start_utc: datetime | None = None

    def __post_init__(self) -> None:
        if (self.start_block is None) != (self.end_block is None):
            raise ConfigError(
                "WindowConfig: block bounds must be given together; "
                f"start_block={self.start_block!r}, end_block={self.end_block!r}"
            )
        if (self.start_utc is None) != (self.end_utc is None):
            raise ConfigError(
                "WindowConfig: date bounds must be given together; "
                f"start_utc={self.start_utc!r}, end_utc={self.end_utc!r}"
            )
        if self.start_block is not None and self.start_utc is not None:
            raise ConfigError(
                "WindowConfig: exactly one of (start_block/end_block) or (start_utc/end_utc) "
                "may be set; got both"
            )
        if self.start_block is None and self.start_utc is None:
            raise ConfigError(
                "WindowConfig: exactly one of (start_block/end_block) or (start_utc/end_utc) "
                "must be set; got neither"
            )

        if self.start_block is not None and self.end_block is not None:
            if self.start_block > self.end_block:
                raise ConfigError(
                    f"WindowConfig: start_block {self.start_block} must be <= end_block "
                    f"{self.end_block}"
                )
        else:
            start, end = self.start_utc, self.end_utc
            assert start is not None and end is not None  # established by the pair checks above
            if start.tzinfo is None or start.utcoffset() is None:
                raise ConfigError(
                    "WindowConfig: start_utc must be timezone-aware UTC "
                    "(naive datetimes are not allowed)"
                )
            if end.tzinfo is None or end.utcoffset() is None:
                raise ConfigError(
                    "WindowConfig: end_utc must be timezone-aware UTC "
                    "(naive datetimes are not allowed)"
                )
            start_utc = start.astimezone(UTC)
            end_utc = end.astimezone(UTC)
            if start_utc > end_utc:
                raise ConfigError(
                    f"WindowConfig: start_utc {start_utc.isoformat()} must be <= end_utc "
                    f"{end_utc.isoformat()}"
                )
            object.__setattr__(self, "start_utc", start_utc)
            object.__setattr__(self, "end_utc", end_utc)

        # Walk-forward split hook: validated only when BOTH bounds are present.
        if (self.train_end_utc is None) != (self.eval_start_utc is None):
            raise ConfigError(
                "WindowConfig: train_end_utc and eval_start_utc must be provided together, "
                "or both omitted"
            )
        if self.train_end_utc is not None and self.eval_start_utc is not None:
            train_end, eval_start = self.train_end_utc, self.eval_start_utc
            if train_end.tzinfo is None or train_end.utcoffset() is None:
                raise ConfigError(
                    "WindowConfig: train_end_utc must be timezone-aware UTC "
                    "(naive datetimes are not allowed)"
                )
            if eval_start.tzinfo is None or eval_start.utcoffset() is None:
                raise ConfigError(
                    "WindowConfig: eval_start_utc must be timezone-aware UTC "
                    "(naive datetimes are not allowed)"
                )
            train_end = train_end.astimezone(UTC)
            eval_start = eval_start.astimezone(UTC)
            if train_end > eval_start:
                raise ConfigError(
                    f"WindowConfig: train_end_utc {train_end.isoformat()} must be <= "
                    f"eval_start_utc {eval_start.isoformat()}"
                )
            object.__setattr__(self, "train_end_utc", train_end)
            object.__setattr__(self, "eval_start_utc", eval_start)
            # Containment into the window is only checkable when the window itself is given as
            # dates; a block-only window has no datetime reference to contain them against
            # (ADR-003). `end_utc` is inclusive, so the split may touch it.
            if (
                self.start_utc is not None
                and self.end_utc is not None
                and not (self.start_utc <= train_end <= eval_start <= self.end_utc)
            ):
                raise ConfigError(
                    f"WindowConfig: split ({train_end.isoformat()}.."
                    f"{eval_start.isoformat()}) must fall inside window "
                    f"({self.start_utc.isoformat()}..{self.end_utc.isoformat()})"
                )


@dataclass(frozen=True, slots=True)
class RegimeConfig:
    """Regime labeller thresholds — pre-committed per PLAN.md §7, sweepable via T12."""

    lookback_days: int = 30
    vol_threshold: float = 0.80  # annualized σ_rv above this => HIGH_VOL
    drift_threshold: float = 0.05  # |μ_30| above this => BULL / BEAR
    periods_per_year: int = 365 * 1440  # minute bars


def _host_only(url: str) -> str:
    """Return ``scheme://host[:port]`` for ``url`` — never a path, query or credentials."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<invalid-url>"
    netloc = parts.netloc
    if "@" in netloc:  # strip any userinfo the URL might carry
        netloc = netloc.rsplit("@", 1)[1]
    return f"{parts.scheme}://{netloc}"


@dataclass(frozen=True, slots=True)
class EndpointConfig:
    """Remote endpoints. ``graph_url`` / ``rpc_url`` / ``reference_base_url`` may contain
    ``${ENV_VAR}`` placeholders expanded by ``load_config``; a missing variable raises
    ``ConfigError``. The expanded values are secrets by policy — ``__repr__`` masks them to the
    host only (the plan says: log the host, never the key)."""

    graph_url: str
    rpc_url: str  # archive node required
    reference_base_url: str
    max_concurrency: int = 4
    request_timeout_s: float = 30.0
    max_retries: int = 5

    def __repr__(self) -> str:
        return (
            "EndpointConfig("
            f"graph_url={_host_only(self.graph_url)!r}, "
            f"rpc_url={_host_only(self.rpc_url)!r}, "
            f"reference_base_url={_host_only(self.reference_base_url)!r}, "
            f"max_concurrency={self.max_concurrency!r}, "
            f"request_timeout_s={self.request_timeout_s!r}, "
            f"max_retries={self.max_retries!r})"
        )


@dataclass(frozen=True, slots=True)
class DataConfig:
    """The complete resolved configuration for one dataset pull."""

    pool: PoolConfig
    window: WindowConfig
    regime: RegimeConfig
    endpoints: EndpointConfig
    cache_dir: Path
    output_dir: Path
    # Extension (defaulted, keyword-safe): lets T14/undertow.sim calibrate gas cost without
    # touching config.py; overridden from the `[gas.units]` TOML section when present.
    gas_units: Mapping[str, int] = GAS_UNITS


# ---------------------------------------------------------------------------
# TOML parsing helpers — the only I/O in this module is reading the file handed to load_config.
# ---------------------------------------------------------------------------

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

_TOP_KEYS = frozenset({"pool", "window", "regime", "endpoints", "paths", "gas"})
_POOL_KEYS = frozenset(
    {
        "address",
        "token0_symbol",
        "token1_symbol",
        "token0_decimals",
        "token1_decimals",
        "fee_tier",
        "tick_spacing",
        "deployment_block",
    }
)
_WINDOW_KEYS = frozenset(
    {"start_block", "end_block", "start_utc", "end_utc", "train_end_utc", "eval_start_utc"}
)
_REGIME_KEYS = frozenset({"lookback_days", "vol_threshold", "drift_threshold", "periods_per_year"})
_ENDPOINT_KEYS = frozenset(
    {
        "graph_url",
        "rpc_url",
        "reference_base_url",
        "max_concurrency",
        "request_timeout_s",
        "max_retries",
    }
)
_PATH_KEYS = frozenset({"cache_dir", "output_dir"})
_GAS_KEYS = frozenset({"units"})
_GAS_UNIT_KEYS = frozenset({"mint", "burn", "collect", "swap"})


def _check_unknown(data: Mapping[str, object], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"{where}: unknown configuration key(s): {unknown}")


def _table(data: Mapping[str, object], name: str, where: str) -> dict[str, object]:
    if name not in data:
        raise ConfigError(f"{where}: missing required section [{name}]")
    value = data[name]
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: section [{name}] must be a table, got {value!r}")
    return value


def _need_str(data: Mapping[str, object], key: str, where: str) -> str:
    if key not in data:
        raise ConfigError(f"{where}: missing required key {key!r}")
    value = data[key]
    if not isinstance(value, str):
        raise ConfigError(f"{where}: key {key!r} must be a string, got {value!r}")
    return value


def _need_int(data: Mapping[str, object], key: str, where: str) -> int:
    if key not in data:
        raise ConfigError(f"{where}: missing required key {key!r}")
    value = data[key]
    if not isinstance(value, int):
        raise ConfigError(f"{where}: key {key!r} must be an integer, got {value!r}")
    return value


def _opt_int(data: Mapping[str, object], key: str, where: str) -> int | None:
    if key not in data:
        return None
    value = data[key]
    if not isinstance(value, int):
        raise ConfigError(f"{where}: key {key!r} must be an integer, got {value!r}")
    return value


def _opt_float(data: Mapping[str, object], key: str, where: str) -> float | None:
    """Optional numeric key returned as float; accepts TOML int or float spellings."""
    if key not in data:
        return None
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}: key {key!r} must be numeric, got {value!r}")
    return float(value)


def _expand_env(value: str, where: str) -> str:
    """Expand ``${ENV_VAR}`` placeholders. Missing variable => ConfigError naming it, no value."""

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        expanded = os.environ.get(name)
        if expanded is None:
            raise ConfigError(
                f"{where}: environment variable {name!r} is required by an endpoint URL "
                "but is not set"
            )
        return expanded

    return _ENV_VAR_RE.sub(_sub, value)


def _parse_pool(section: Mapping[str, object], where: str) -> PoolConfig:
    _check_unknown(section, _POOL_KEYS, f"{where}[pool]")
    try:
        fee_tier = FeeTier(_need_int(section, "fee_tier", f"{where}[pool]"))
    except ValueError:
        raise ConfigError(
            f"{where}[pool]: fee_tier {section.get('fee_tier')!r} is not a supported "
            f"fee tier; expected one of {sorted(e.value for e in FeeTier)}"
        ) from None
    return PoolConfig(
        address=Address(_need_str(section, "address", f"{where}[pool]")),
        token0_symbol=_need_str(section, "token0_symbol", f"{where}[pool]"),
        token1_symbol=_need_str(section, "token1_symbol", f"{where}[pool]"),
        token0_decimals=_need_int(section, "token0_decimals", f"{where}[pool]"),
        token1_decimals=_need_int(section, "token1_decimals", f"{where}[pool]"),
        fee_tier=fee_tier,
        tick_spacing=_need_int(section, "tick_spacing", f"{where}[pool]"),
        deployment_block=BlockNumber(_need_int(section, "deployment_block", f"{where}[pool]")),
    )


def _to_aware_utc(value: str, field: str, where: str) -> datetime:
    """Parse an ISO-8601 string to a UTC-aware datetime. Naive => ConfigError."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        raise ConfigError(
            f"{where}.window: {field} {value!r} is not an ISO-8601 timestamp; expected e.g. "
            "'2022-01-01T00:00:00Z' or an explicit offset"
        ) from None
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ConfigError(
            f"{where}.window: {field} {value!r} must be timezone-aware UTC "
            "(naive datetimes are not allowed)"
        )
    return dt.astimezone(UTC)


def _parse_window(section: Mapping[str, object], where: str) -> WindowConfig:
    _check_unknown(section, _WINDOW_KEYS, f"{where}[window]")
    kwargs: dict[str, object] = {}
    for key in ("start_block", "end_block"):
        if key in section:
            kwargs[key] = BlockNumber(_need_int(section, key, f"{where}[window]"))
    for key in ("start_utc", "end_utc", "train_end_utc", "eval_start_utc"):
        if key in section:
            kwargs[key] = _to_aware_utc(
                _need_str(section, key, f"{where}[window]"), key, f"{where}"
            )
    # The dict keys/values match the dataclass fields by construction (single source of truth
    # is the _parse_window loop above); typing cannot see that, hence the ignore.
    return WindowConfig(**kwargs)  # type: ignore[arg-type]


def _parse_regime(section: Mapping[str, object], where: str) -> RegimeConfig:
    _check_unknown(section, _REGIME_KEYS, f"{where}[regime]")
    lookback = _opt_int(section, "lookback_days", f"{where}[regime]")
    vol = _opt_float(section, "vol_threshold", f"{where}[regime]")
    drift = _opt_float(section, "drift_threshold", f"{where}[regime]")
    periods = _opt_int(section, "periods_per_year", f"{where}[regime]")
    # Fallbacks mirror the RegimeConfig defaults; a TOML value always wins.
    if lookback is not None and lookback < 1:
        raise ConfigError(f"{where}[regime]: lookback_days must be >= 1, got {lookback}")
    if periods is not None and periods < 1:
        raise ConfigError(f"{where}[regime]: periods_per_year must be >= 1, got {periods}")
    if vol is not None and vol < 0:
        raise ConfigError(f"{where}[regime]: vol_threshold must be >= 0, got {vol}")
    if drift is not None and drift < 0:
        raise ConfigError(f"{where}[regime]: drift_threshold must be >= 0, got {drift}")
    return RegimeConfig(
        lookback_days=lookback if lookback is not None else 30,
        vol_threshold=vol if vol is not None else 0.80,
        drift_threshold=drift if drift is not None else 0.05,
        periods_per_year=periods if periods is not None else 365 * 1440,
    )


def _parse_endpoints(section: Mapping[str, object], where: str) -> EndpointConfig:
    _check_unknown(section, _ENDPOINT_KEYS, f"{where}[endpoints]")
    concurrent = _opt_int(section, "max_concurrency", f"{where}[endpoints]")
    timeout = _opt_float(section, "request_timeout_s", f"{where}[endpoints]")
    retries = _opt_int(section, "max_retries", f"{where}[endpoints]")
    # Fallbacks mirror the EndpointConfig defaults.
    if concurrent is not None and concurrent < 1:
        raise ConfigError(f"{where}[endpoints]: max_concurrency must be >= 1, got {concurrent}")
    if timeout is not None and timeout <= 0:
        raise ConfigError(f"{where}[endpoints]: request_timeout_s must be > 0, got {timeout}")
    if retries is not None and retries < 0:
        raise ConfigError(f"{where}[endpoints]: max_retries must be >= 0, got {retries}")
    return EndpointConfig(
        graph_url=_expand_env(
            _need_str(section, "graph_url", f"{where}[endpoints]"), f"{where}[endpoints]"
        ),
        rpc_url=_expand_env(
            _need_str(section, "rpc_url", f"{where}[endpoints]"), f"{where}[endpoints]"
        ),
        reference_base_url=_expand_env(
            _need_str(section, "reference_base_url", f"{where}[endpoints]"), f"{where}[endpoints]"
        ),
        max_concurrency=concurrent if concurrent is not None else 4,
        request_timeout_s=timeout if timeout is not None else 30.0,
        max_retries=retries if retries is not None else 5,
    )


def _parse_gas(section: Mapping[str, object], where: str) -> Mapping[str, int]:
    """Optional `[gas.units]` section overriding GAS_UNITS; missing section => defaults."""
    if not section:
        return GAS_UNITS
    _check_unknown(section, _GAS_KEYS, f"{where}[gas]")
    units = _table(section, "units", f"{where}[gas]")
    if set(units) != set(_GAS_UNIT_KEYS):
        raise ConfigError(
            f"{where}[gas.units]: must define exactly {sorted(_GAS_UNIT_KEYS)}; got {sorted(units)}"
        )
    out: dict[str, int] = {}
    for key in _GAS_UNIT_KEYS:
        value = _need_int(units, key, f"{where}[gas.units]")
        if value <= 0:
            raise ConfigError(f"{where}[gas.units]: {key} must be a positive integer, got {value}")
        out[key] = value
    return MappingProxyType(out)


def load_config(path: str | Path) -> DataConfig:
    """Load a ``DataConfig`` from a TOML file.

    Only I/O in this module. ``${ENV_VAR}`` placeholders are expanded in ``[endpoints]`` string
    fields; a missing variable raises ``ConfigError`` naming the variable (never its value).
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"cannot read config file {p}: {e}") from e
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"invalid TOML in {p}: {e}") from e
    _check_unknown(raw, _TOP_KEYS, str(p))
    where = str(p)
    pool = _parse_pool(_table(raw, "pool", where), where)
    window = _parse_window(_table(raw, "window", where), where)
    regime = _parse_regime(_table(raw, "regime", where), where)
    endpoints = _parse_endpoints(_table(raw, "endpoints", where), where)
    paths = _table(raw, "paths", where)
    _check_unknown(paths, _PATH_KEYS, f"{where}[paths]")
    cache_dir = Path(_need_str(paths, "cache_dir", f"{where}[paths]"))
    output_dir = Path(_need_str(paths, "output_dir", f"{where}[paths]"))
    gas = raw.get("gas")
    gas_units = _parse_gas(gas if isinstance(gas, dict) else {}, where)
    return DataConfig(
        pool=pool,
        window=window,
        regime=regime,
        endpoints=endpoints,
        cache_dir=cache_dir,
        output_dir=output_dir,
        gas_units=gas_units,
    )


def default_pools() -> dict[str, PoolConfig]:
    """The two pinned pools (PLAN.md §7), keyed "USDC_WETH_3000" / "USDC_WETH_500".

    Token ordering is contractual: Uniswap orders tokens by address and ``0xA0b8…eB48`` (USDC)
    sorts below ``0xC02a…6Cc2`` (WETH), so **token0 = USDC (6 decimals), token1 = WETH (18
    decimals)** for both pools. Getting this backwards silently inverts every price in the
    thesis — see CONTRACTS.md §3.0 (price orientation).
    """
    return {
        "USDC_WETH_3000": PoolConfig(
            address=Address("0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"),
            token0_symbol="USDC",
            token1_symbol="WETH",
            token0_decimals=6,
            token1_decimals=18,
            fee_tier=FeeTier.BPS_30,
            tick_spacing=60,
            deployment_block=BlockNumber(12370624),
        ),
        "USDC_WETH_500": PoolConfig(
            address=Address("0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"),
            token0_symbol="USDC",
            token1_symbol="WETH",
            token0_decimals=6,
            token1_decimals=18,
            fee_tier=FeeTier.BPS_5,
            tick_spacing=10,
            deployment_block=BlockNumber(12376729),
        ),
    }
