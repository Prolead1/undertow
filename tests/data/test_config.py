"""Tests for ``undertow.data.config`` (T01) — frozen dataclasses, validation, TOML loading.

Covers every ``__post_init__`` rejection path (exception type AND message content), the
price-orientation guard on ``default_pools``, environment-variable expansion (secrets never leak
into messages or ``repr``), the walk-forward split validation, and the pinned constants.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from undertow.data.config import (
    GAS_UNITS,
    GLOBAL_TICK_SENTINEL,
    MAX_TICK,
    MIN_TICK,
    Q96,
    Q128,
    TICK_BASE,
    DataConfig,
    EndpointConfig,
    PoolConfig,
    RegimeConfig,
    WindowConfig,
    default_pools,
    load_config,
)
from undertow.data.types import Address, BlockNumber, ConfigError, FeeTier

UTC = UTC

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

USDC_WETH_3000 = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
USDC_WETH_500 = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"

# A small, valid config used as the template for the load_config rejection tests. graph_url is
# gateway base + key only; the subgraph id lives in code and _endpoint() appends it. See thegraph.py.
GOOD_TOML = """\
[pool]
address = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
token0_symbol = "USDC"
token1_symbol = "WETH"
token0_decimals = 6
token1_decimals = 18
fee_tier = 3000
tick_spacing = 60
deployment_block = 12370624

[window]
start_utc = "2022-01-01T00:00:00Z"
end_utc = "2024-12-31T23:59:59Z"

[regime]
lookback_days = 30
vol_threshold = 0.80
drift_threshold = 0.05
periods_per_year = 525600

[endpoints]
graph_url = "https://gateway.thegraph.com/api/${GRAPH_API_KEY}"
rpc_url = "${ETH_RPC_URL}"
reference_base_url = "https://api.binance.com"

[paths]
cache_dir = "data/cache/test"
output_dir = "data/datasets/test"

[gas.units]
mint = 400000
burn = 250000
collect = 150000
swap = 180000
"""


def _write(tmp_path: Path, body: str) -> str:
    p = tmp_path / "cfg.toml"
    p.write_text(body, encoding="utf-8")
    return str(p)


def _drop_section(body: str, header: str) -> str:
    """Remove a whole TOML section (from ``header`` to the next ``[`` section header)."""
    start = body.index(header)
    nxt = body.find("\n[", start + 1)
    return body[:start] + (body[nxt:] if nxt != -1 else "")


def _pool(**overrides: object) -> PoolConfig:
    base: dict[str, object] = {
        "address": Address(USDC_WETH_3000),
        "token0_symbol": "USDC",
        "token1_symbol": "WETH",
        "token0_decimals": 6,
        "token1_decimals": 18,
        "fee_tier": FeeTier.BPS_30,
        "tick_spacing": 60,
        "deployment_block": BlockNumber(12370624),
    }
    base.update(overrides)
    # Keys/values match the dataclass fields by construction; typing cannot see that.
    return PoolConfig(**base)  # type: ignore[arg-type]


def _set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPH_API_KEY", "test-key-0123456789")
    monkeypatch.setenv("ETH_RPC_URL", "https://eth-mainnet.example.test/v2/rpc-secret")


# ---------------------------------------------------------------------------
# Pinned constants
# ---------------------------------------------------------------------------


def test_pinned_constants_exact() -> None:
    assert Q96 == 2**96
    assert Q128 == 2**128
    assert Decimal("1.0001") == TICK_BASE
    assert MIN_TICK == -887272
    assert MAX_TICK == 887272
    assert GLOBAL_TICK_SENTINEL == -(2**31)


# ---------------------------------------------------------------------------
# default_pools — the price-orientation guard
# ---------------------------------------------------------------------------


def test_default_pools_price_orientation_and_pins() -> None:
    pools = default_pools()
    assert set(pools) == {"USDC_WETH_3000", "USDC_WETH_500"}
    for key, fee, spacing, deployment, addr in [
        ("USDC_WETH_3000", FeeTier.BPS_30, 60, BlockNumber(12370624), USDC_WETH_3000),
        ("USDC_WETH_500", FeeTier.BPS_5, 10, BlockNumber(12376729), USDC_WETH_500),
    ]:
        pool = pools[key]
        assert pool.address == addr
        assert pool.token0_symbol == "USDC"
        assert pool.token0_decimals == 6
        assert pool.token1_symbol == "WETH"
        assert pool.token1_decimals == 18
        assert pool.fee_tier == fee
        assert pool.tick_spacing == spacing
        assert pool.deployment_block == deployment


# ---------------------------------------------------------------------------
# PoolConfig.__post_init__ rejection paths (type AND message content)
# ---------------------------------------------------------------------------


def test_pool_address_must_be_lowercase_checksum_stripped_42chars() -> None:
    bad_addresses = [
        "",  # empty
        "0x1234",  # too short
        "0x" + "a" * 38,  # too short (40 chars total)
        "0x" + "a" * 39 + "xyz",  # non-hex tail
        "0x" + "A" * 40,  # uppercase hex digits
        "0x" + "g" * 40,  # invalid hex digit
        "0X" + "a" * 40,  # uppercase 0X prefix
        "0x" + "a" * 41 + "B",  # right length, wrong case
    ]
    for bad in bad_addresses:
        with pytest.raises(ConfigError, match="42|lowercase|0x") as excinfo:
            _pool(address=Address(bad))
        msg = str(excinfo.value)
        assert "address" in msg


def test_pool_fee_tier_spacing_mismatch_rejected() -> None:
    # Contract: tick_spacing must equal TICK_SPACING[fee_tier] — 3000 -> 60, never 10.
    with pytest.raises(ConfigError) as excinfo:
        _pool(fee_tier=FeeTier.BPS_30, tick_spacing=10)
    msg = str(excinfo.value)
    assert "tick_spacing" in msg
    assert "60" in msg


def test_pool_decimals_out_of_range_rejected() -> None:
    with pytest.raises(ConfigError) as excinfo:
        _pool(token0_decimals=19)
    msg = str(excinfo.value)
    assert "token0_decimals" in msg
    assert "0..18" in msg

    with pytest.raises(ConfigError) as excinfo:
        _pool(token1_decimals=-1)
    msg = str(excinfo.value)
    assert "token1_decimals" in msg
    assert "0..18" in msg


def test_pool_deployment_block_must_be_positive() -> None:
    for bad in (0, -5):
        with pytest.raises(ConfigError, match="deployment_block") as excinfo:
            _pool(deployment_block=BlockNumber(bad))
        assert "must be > 0" in str(excinfo.value)


# ---------------------------------------------------------------------------
# WindowConfig validation
# ---------------------------------------------------------------------------


def test_window_blocks_only_constructs() -> None:
    w = WindowConfig(start_block=BlockNumber(10), end_block=BlockNumber(20))
    assert w.start_block == 10
    assert w.end_block == 20
    assert w.start_utc is None
    assert w.end_utc is None


def test_window_dates_only_constructs() -> None:
    w = WindowConfig(
        start_utc=datetime(2022, 1, 1, tzinfo=UTC),
        end_utc=datetime(2022, 6, 1, tzinfo=UTC),
    )
    assert w.start_block is None
    assert w.end_block is None
    assert w.start_utc == datetime(2022, 1, 1, tzinfo=UTC)
    assert w.end_utc == datetime(2022, 6, 1, tzinfo=UTC)


def test_window_exactly_one_bound_pair_required() -> None:
    start = datetime(2022, 1, 1, tzinfo=UTC)
    end = datetime(2022, 6, 1, tzinfo=UTC)

    with pytest.raises(ConfigError, match="got both"):
        WindowConfig(
            start_block=BlockNumber(10),
            end_block=BlockNumber(20),
            start_utc=start,
            end_utc=end,
        )
    with pytest.raises(ConfigError, match="got neither"):
        WindowConfig()
    with pytest.raises(ConfigError, match="block bounds must be given together"):
        WindowConfig(start_block=BlockNumber(10))
    with pytest.raises(ConfigError, match="date bounds must be given together"):
        WindowConfig(start_utc=start)


def test_window_start_must_not_exceed_end() -> None:
    with pytest.raises(ConfigError, match="start_block") as excinfo:
        WindowConfig(start_block=BlockNumber(20), end_block=BlockNumber(10))
    assert "must be <= end_block" in str(excinfo.value)

    with pytest.raises(ConfigError, match="start_utc") as excinfo:
        WindowConfig(
            start_utc=datetime(2022, 6, 1, tzinfo=UTC),
            end_utc=datetime(2022, 1, 1, tzinfo=UTC),
        )
    assert "must be <= end_utc" in str(excinfo.value)


def test_window_naive_datetimes_rejected() -> None:
    aware = datetime(2022, 6, 1, tzinfo=UTC)
    with pytest.raises(ConfigError, match="start_utc") as excinfo:
        WindowConfig(start_utc=datetime(2022, 1, 1), end_utc=aware)
    assert "timezone-aware UTC" in str(excinfo.value)
    assert "naive" in str(excinfo.value)

    with pytest.raises(ConfigError, match="end_utc") as excinfo:
        WindowConfig(start_utc=aware, end_utc=datetime(2022, 1, 1))
    assert "timezone-aware UTC" in str(excinfo.value)


def test_window_datetime_offsets_normalized_to_utc() -> None:
    w = WindowConfig(
        start_utc=datetime(2022, 1, 1, 2, 0, tzinfo=timezone(timedelta(hours=2))),
        end_utc=datetime(2022, 1, 2, tzinfo=timezone(timedelta(hours=-5))),
    )
    assert w.start_utc == datetime(2022, 1, 1, tzinfo=UTC)
    assert w.end_utc == datetime(2022, 1, 2, 5, 0, tzinfo=UTC)
    assert w.start_utc.tzinfo is UTC


def test_window_split_valid_inside_date_window() -> None:
    w = WindowConfig(
        start_utc=datetime(2022, 1, 1, tzinfo=UTC),
        end_utc=datetime(2022, 12, 31, tzinfo=UTC),
        train_end_utc=datetime(2022, 6, 30, tzinfo=UTC),
        eval_start_utc=datetime(2022, 7, 1, tzinfo=UTC),
    )
    assert w.train_end_utc == datetime(2022, 6, 30, tzinfo=UTC)
    assert w.eval_start_utc == datetime(2022, 7, 1, tzinfo=UTC)


def test_window_split_ordering_required() -> None:
    with pytest.raises(ConfigError, match="train_end_utc") as excinfo:
        WindowConfig(
            start_utc=datetime(2022, 1, 1, tzinfo=UTC),
            end_utc=datetime(2022, 12, 31, tzinfo=UTC),
            train_end_utc=datetime(2022, 7, 1, tzinfo=UTC),
            eval_start_utc=datetime(2022, 6, 30, tzinfo=UTC),
        )
    assert "must be <= eval_start_utc" in str(excinfo.value)


def test_window_split_must_be_provided_together() -> None:
    with pytest.raises(ConfigError, match="must be provided together"):
        WindowConfig(
            start_utc=datetime(2022, 1, 1, tzinfo=UTC),
            end_utc=datetime(2022, 12, 31, tzinfo=UTC),
            train_end_utc=datetime(2022, 6, 30, tzinfo=UTC),
        )


def test_window_split_inside_window_required() -> None:
    with pytest.raises(ConfigError, match="must fall inside window"):
        WindowConfig(
            start_utc=datetime(2022, 1, 1, tzinfo=UTC),
            end_utc=datetime(2022, 12, 31, tzinfo=UTC),
            train_end_utc=datetime(2022, 6, 30, tzinfo=UTC),
            eval_start_utc=datetime(2023, 1, 1, tzinfo=UTC),  # beyond end_utc
        )


def test_window_split_naive_datetime_rejected() -> None:
    with pytest.raises(ConfigError, match="train_end_utc"):
        WindowConfig(
            start_utc=datetime(2022, 1, 1, tzinfo=UTC),
            end_utc=datetime(2022, 12, 31, tzinfo=UTC),
            train_end_utc=datetime(2022, 6, 30),  # naive
            eval_start_utc=datetime(2022, 7, 1, tzinfo=UTC),
        )


def test_window_split_with_block_window_skips_containment() -> None:
    # ADR-003: a block-only window has no datetime reference to contain split dates against, so
    # only awareness + ordering are validated; this must construct.
    w = WindowConfig(
        start_block=BlockNumber(10),
        end_block=BlockNumber(100),
        train_end_utc=datetime(2022, 6, 30, tzinfo=UTC),
        eval_start_utc=datetime(2022, 7, 1, tzinfo=UTC),
    )
    assert w.train_end_utc == datetime(2022, 6, 30, tzinfo=UTC)


# ---------------------------------------------------------------------------
# EndpointConfig repr must not leak expanded secrets
# ---------------------------------------------------------------------------


def test_endpoint_config_repr_masks_secrets() -> None:
    ep = EndpointConfig(
        graph_url="https://gateway.thegraph.com/api/GRAPH_SECRET_123/subgraphs/id/abc",
        rpc_url="https://eth-mainnet.g.alchemy.com/v2/RPC_SECRET_456",
        reference_base_url="https://api.binance.com",
    )
    r = repr(ep)
    assert "GRAPH_SECRET_123" not in r
    assert "RPC_SECRET_456" not in r
    # Hosts are still visible so the repr stays useful for debugging.
    assert "gateway.thegraph.com" in r
    assert "eth-mainnet.g.alchemy.com" in r
    assert str(ep) == r  # str() must also be masked


# ---------------------------------------------------------------------------
# GAS_UNITS — frozen default + DataConfig extension
# ---------------------------------------------------------------------------


def test_gas_units_defaults_match_pinned_values() -> None:
    assert dict(GAS_UNITS) == {
        "mint": 400_000,
        "burn": 250_000,
        "collect": 150_000,
        "swap": 180_000,
    }
    with pytest.raises(TypeError):
        GAS_UNITS["mint"] = 1  # type: ignore[index]


def test_data_config_constructs_with_only_contract_fields() -> None:
    # The 6 contract fields of DataConfig (CONTRACTS.md §2) plus the defaulted gas_units extension.
    cfg = DataConfig(
        pool=default_pools()["USDC_WETH_3000"],
        window=WindowConfig(
            start_utc=datetime(2022, 1, 1, tzinfo=UTC),
            end_utc=datetime(2022, 1, 2, tzinfo=UTC),
        ),
        regime=RegimeConfig(),
        endpoints=EndpointConfig(
            graph_url="https://g.test",
            rpc_url="https://r.test",
            reference_base_url="https://b.test",
        ),
        cache_dir=Path("data/cache/x"),
        output_dir=Path("data/datasets/x"),
    )
    assert cfg.gas_units is GAS_UNITS


def test_pool_config_is_frozen() -> None:
    pool = default_pools()["USDC_WETH_3000"]
    with pytest.raises(FrozenInstanceError):
        pool.token0_symbol = "DAI"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# load_config — round trip on the real configs + rejection paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "address", "fee_tier", "spacing", "deployment", "cache_dir", "output_dir"),
    [
        (
            "eth_usdc_3000.toml",
            USDC_WETH_3000,
            FeeTier.BPS_30,
            60,
            BlockNumber(12370624),
            "data/cache/eth_usdc_3000",
            "data/datasets/eth_usdc_3000",
        ),
        (
            "eth_usdc_500.toml",
            USDC_WETH_500,
            FeeTier.BPS_5,
            10,
            BlockNumber(12376729),
            "data/cache/eth_usdc_500",
            "data/datasets/eth_usdc_500",
        ),
    ],
)
def test_load_config_round_trip_real_configs(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    address: str,
    fee_tier: FeeTier,
    spacing: int,
    deployment: BlockNumber,
    cache_dir: str,
    output_dir: str,
) -> None:
    _set_env(monkeypatch)
    cfg = load_config(CONFIG_DIR / name)
    assert isinstance(cfg, DataConfig)
    assert cfg.pool.address == address
    assert cfg.pool.token0_symbol == "USDC"
    assert cfg.pool.token0_decimals == 6
    assert cfg.pool.token1_symbol == "WETH"
    assert cfg.pool.token1_decimals == 18
    assert cfg.pool.fee_tier == fee_tier
    assert cfg.pool.tick_spacing == spacing
    assert cfg.pool.deployment_block == deployment
    # Window is date-specified (the pinned window), never block-specified.
    assert cfg.window.start_block is None
    assert cfg.window.end_block is None
    assert cfg.window.start_utc == datetime(2022, 1, 1, tzinfo=UTC)
    assert cfg.window.end_utc == datetime(2024, 12, 31, 23, 59, 59, tzinfo=UTC)
    assert cfg.window.train_end_utc is None
    assert cfg.window.eval_start_utc is None
    assert cfg.regime == RegimeConfig(
        lookback_days=30, vol_threshold=0.8, drift_threshold=0.05, periods_per_year=525600
    )
    # ${GRAPH_API_KEY}and ${ETH_RPC_URL} were expanded,in place,at load time.The graph_url is
    # the gateway base + key only: paying the subgraph id here too would double the path at
    # at fetch time, when _endpoint() appends /subgraphs/id/ + SUBGRAPH_ID, the gateway answers
    # with HTTP 404. (Regression guard for bug-config-graph-url.
    assert cfg.endpoints.graph_url == "https://gateway.thegraph.com/api/test-key-0123456789"
    assert cfg.endpoints.rpc_url == "https://eth-mainnet.example.test/v2/rpc-secret"
    assert cfg.endpoints.reference_base_url == "https://api.binance.com"
    assert cfg.endpoints.max_concurrency == 4
    assert cfg.endpoints.request_timeout_s == 30.0
    assert cfg.endpoints.max_retries == 5
    assert cfg.cache_dir == Path(cache_dir)
    assert cfg.output_dir == Path(output_dir)
    # The [gas.units] section re-declares the pinned defaults; DataConfig picks them up.
    assert dict(cfg.gas_units) == dict(GAS_UNITS)


def test_real_configs_use_pinned_env_placeholders() -> None:
    for name in ("eth_usdc_3000.toml", "eth_usdc_500.toml"):
        text = (CONFIG_DIR / name).read_text(encoding="utf-8")
        assert "${GRAPH_API_KEY}" in text
        assert "${ETH_RPC_URL}" in text


def test_load_config_missing_env_var_names_var_no_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GRAPH_API_KEY", raising=False)
    monkeypatch.setenv("ETH_RPC_URL", "https://example.test/v2/SECRETVALUE12345")
    with pytest.raises(ConfigError) as excinfo:
        load_config(CONFIG_DIR / "eth_usdc_3000.toml")
    msg = str(excinfo.value)
    assert "GRAPH_API_KEY" in msg
    assert "SECRETVALUE12345" not in msg


def test_load_config_rejects_fee_tier_spacing_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_env(monkeypatch)
    p = _write(tmp_path, GOOD_TOML.replace("tick_spacing = 60", "tick_spacing = 10"))
    with pytest.raises(ConfigError) as excinfo:
        load_config(p)
    msg = str(excinfo.value)
    assert "tick_spacing" in msg
    assert "60" in msg


def test_load_config_rejects_unknown_fee_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_env(monkeypatch)
    p = _write(tmp_path, GOOD_TOML.replace("fee_tier = 3000", "fee_tier = 5000"))
    with pytest.raises(ConfigError) as excinfo:
        load_config(p)
    msg = str(excinfo.value)
    assert "not a supported fee tier" in msg
    assert "5000" in msg


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read config file"):
        load_config(tmp_path / "does-not-exist.toml")


def test_load_config_invalid_toml_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, "pool = [unclosed")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(p)


def test_load_config_missing_section_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_env(monkeypatch)
    p = _write(tmp_path, _drop_section(GOOD_TOML, "[regime]"))
    with pytest.raises(ConfigError, match="missing required section") as excinfo:
        load_config(p)
    assert "[regime]" in str(excinfo.value)


def test_load_config_unknown_key_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch)
    p = _write(tmp_path, GOOD_TOML.replace("tick_spacing = 60", "tick_spacing = 60\ntoken2 = 6", 1))
    with pytest.raises(ConfigError, match="unknown configuration key") as excinfo:
        load_config(p)
    assert "token2" in str(excinfo.value)


def test_load_config_bad_datetime_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch)
    p = _write(tmp_path, GOOD_TOML.replace("2022-01-01T00:00:00Z", "not-a-date"))
    with pytest.raises(ConfigError, match="not an ISO-8601 timestamp"):
        load_config(p)


def test_load_config_naive_datetime_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_env(monkeypatch)
    p = _write(tmp_path, GOOD_TOML.replace("2022-01-01T00:00:00Z", "2022-01-01"))
    with pytest.raises(ConfigError, match="must be timezone-aware UTC"):
        load_config(p)


def test_load_config_gas_units_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch)
    p = _write(tmp_path, GOOD_TOML.replace("mint = 400000", "mint = 999999", 1))
    cfg = load_config(p)
    assert cfg.gas_units["mint"] == 999999
    assert cfg.gas_units["burn"] == 250_000
    assert cfg.gas_units["collect"] == 150_000
    assert cfg.gas_units["swap"] == 180_000


def test_load_config_gas_must_be_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-table [gas] (e.g. `gas = \"foo\"`) is rejected, matching every other section."""
    _set_env(monkeypatch)
    gas_block = "[gas.units]\nmint = 400000\nburn = 250000\ncollect = 150000\nswap = 180000\n"
    body = 'gas = "not-a-table"\n' + GOOD_TOML.replace(gas_block, "")
    p = _write(tmp_path, body)
    with pytest.raises(ConfigError, match=r"\[gas\].*table"):
        load_config(p)


def test_load_config_env_expansion_only_in_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """${VAR} placeholder outside [endpoints] stays literal; expansion applies to endpoints only."""
    _set_env(monkeypatch)
    body = GOOD_TOML.replace(
        'cache_dir = "data/cache/test"', 'cache_dir = "data/cache/${UNSET_DIR}"'
    )
    p = _write(tmp_path, body)
    cfg = load_config(p)
    assert str(cfg.cache_dir) == "data/cache/${UNSET_DIR}"  # literal, never expanded
    # and the endpoints fields were still expanded.
    assert cfg.endpoints.rpc_url == "https://eth-mainnet.example.test/v2/rpc-secret"


def test_load_config_gas_units_must_be_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_env(monkeypatch)
    p = _write(tmp_path, GOOD_TOML.replace("\ncollect = 150000", ""))
    with pytest.raises(ConfigError, match="must define exactly"):
        load_config(p)
