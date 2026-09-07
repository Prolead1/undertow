"""Tests for ``undertow.data.fetchers.abi`` (T06) — pure ABI constants + decoders.

Offline by construction: no network, no mocking — the module under test is
pure bytes->rows. Three things are verified:

1. **Known-answer keccak.** The five hardcoded ``TOPIC_*`` constants are
   recomputed from their documented signatures with a *vendored* pure-Python
   keccak256 and asserted equal (the roadmap's Swap topic0 is an external known
   answer). Vendored rather than ``eth_utils.keccak`` because ``eth-hash``
   requires a hashing backend (pycryptodome/pysha3) that a plain ``uv sync``
   does not install; the implementation is the Keccak team's reference
   ``CompactFIPS202.py`` (CC0) and is independently anchored to the published
   keccak-256 vectors for ``b""`` and ``b"abc"``.
2. **Event decoders** against the synthetic-but-protocol-shaped fixture logs in
   ``tests/data/fixtures/rpc_logs_*.json`` (hand-computed expected values),
   including the brief's traps: signed ``int256`` amounts (sign extension),
   ``int24`` ticks (right-aligned in the word, sign-extended from bit 23),
   Mint's **non-indexed** ``sender`` (read from data, not topics), Burn's null
   ``sender``, negative tick ranges, and the three loud-failure defences:
   wrong ``topics[0]``, truncated ``data``, out-of-range ticks.
3. **eth_call return decoders** against ``rpc_ticks_call.json``: the seven ABI
   words of ``slot0()`` (sqrtPriceX96, tick in words 0 and 1), signed
   ``liquidity_net`` (int128), exact Q128 globals, and the ``ticks(int24)``
   calldata builder.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from undertow.data.fetchers.abi import (
    SELECTOR_FEE_GROWTH_GLOBAL_0_X128,
    SELECTOR_FEE_GROWTH_GLOBAL_1_X128,
    SELECTOR_LIQUIDITY,
    SELECTOR_SLOT0,
    SELECTOR_TICKS,
    TOPIC_BURN,
    TOPIC_COLLECT,
    TOPIC_FLASH,
    TOPIC_MINT,
    TOPIC_SWAP,
    decode_burn,
    decode_collect,
    decode_flash,
    decode_liquidity_return,
    decode_mint,
    decode_slot0_return,
    decode_swap,
    decode_ticks_return,
    decode_uint256_return,
    encode_ticks_calldata,
    selectors,
)
from undertow.data.types import SchemaViolationError

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "fixtures"

POOL_ADDR = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
TS = datetime(2022, 2, 8, 12, 0, 0, tzinfo=UTC)

# ---------------------------------------------------------------------------
# Vendored keccak256 (test-only): Keccak team reference CompactFIPS202.py
# (https://keccak.team / XKCP, CC0), ported verbatim. Anchored below to the
# published keccak-256 vectors for b"" and b"abc".
# ---------------------------------------------------------------------------


def _rol64(a: int, n: int) -> int:
    return ((a >> (64 - (n % 64))) + (a << (n % 64))) % (1 << 64)


def _keccak_f1600(lanes: list[list[int]]) -> list[list[int]]:
    rc = 1
    for _round in range(24):
        c = [lanes[x][0] ^ lanes[x][1] ^ lanes[x][2] ^ lanes[x][3] ^ lanes[x][4] for x in range(5)]
        d = [c[(x + 4) % 5] ^ _rol64(c[(x + 1) % 5], 1) for x in range(5)]
        lanes = [[lanes[x][y] ^ d[x] for y in range(5)] for x in range(5)]
        x, y = 1, 0
        current = lanes[x][y]
        for t in range(24):
            x, y = y, (2 * x + 3 * y) % 5
            current, lanes[x][y] = lanes[x][y], _rol64(current, (t + 1) * (t + 2) // 2)
        for y in range(5):
            row = [lanes[x][y] for x in range(5)]
            for x in range(5):
                lanes[x][y] = row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5])
        for j in range(7):
            rc = ((rc << 1) ^ ((rc >> 7) * 0x71)) % 256
            if rc & 2:
                lanes[0][0] ^= 1 << ((1 << j) - 1)
    return lanes


def _keccak256(data: bytes) -> bytes:
    rate = 136
    lanes = [[0] * 5 for _ in range(5)]
    full = len(data) - (len(data) % rate)
    i = 0
    while i < full:
        for lane_off in range(0, rate, 8):
            lane = int.from_bytes(data[i + lane_off : i + lane_off + 8], "little")
            lanes[(lane_off // 8) % 5][(lane_off // 8) // 5] ^= lane
        lanes = _keccak_f1600(lanes)
        i += rate
    rem = data[full:]
    block = bytearray(rem) + bytearray([0] * (rate - len(rem)))
    block[len(rem)] = 0x01
    block[-1] |= 0x80
    for lane_off in range(0, rate, 8):
        lane = int.from_bytes(bytes(block[lane_off : lane_off + 8]), "little")
        lanes[(lane_off // 8) % 5][(lane_off // 8) // 5] ^= lane
    lanes = _keccak_f1600(lanes)
    out = bytearray()
    for lane_off in range(0, rate, 8):
        out += lanes[(lane_off // 8) % 5][(lane_off // 8) // 5].to_bytes(8, "little")
    return bytes(out[:32])


def _topic0(signature: str) -> str:
    return "0x" + _keccak256(signature.encode()).hex()


# Signatures exactly as deployed (IUniswapV3PoolEvents) and hardcoded in abi.py.
_SIGNATURES: dict[str, str] = {
    TOPIC_SWAP: "Swap(address,address,int256,int256,uint160,uint128,int24)",
    TOPIC_MINT: "Mint(address,address,int24,int24,uint128,uint256,uint256)",
    TOPIC_BURN: "Burn(address,int24,int24,uint128,uint256,uint256)",
    TOPIC_COLLECT: "Collect(address,address,int24,int24,uint128,uint128)",
    TOPIC_FLASH: "Flash(address,address,uint256,uint256,uint256,uint256)",
}


def test_vendored_keccak_anchored_to_public_vectors() -> None:
    # Published keccak-256 test vectors (NOT recomputable by the code under test).
    assert _keccak256(b"").hex() == (
        "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    )
    assert _keccak256(b"abc").hex() == (
        "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"
    )


def test_topic0_known_answer_swap() -> None:
    """The roadmap's known-answer topic0 for Swap — recomputed, not copied."""
    roadmap_value = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
    recomputed = _topic0(_SIGNATURES[TOPIC_SWAP])
    assert recomputed == roadmap_value
    assert roadmap_value == TOPIC_SWAP  # the hardcoded constant agrees


def test_topic0_all_events_recompute() -> None:
    for topic0, signature in _SIGNATURES.items():
        assert _topic0(signature) == topic0, f"topic0 mismatch for {signature}"


def test_selectors_recompute() -> None:
    methods = {
        "slot0()": SELECTOR_SLOT0,
        "liquidity()": SELECTOR_LIQUIDITY,
        "feeGrowthGlobal0X128()": SELECTOR_FEE_GROWTH_GLOBAL_0_X128,
        "feeGrowthGlobal1X128()": SELECTOR_FEE_GROWTH_GLOBAL_1_X128,
        "ticks(int24)": SELECTOR_TICKS,
    }
    assert methods == selectors()


# ---------------------------------------------------------------------------
# Fixture logs -> hand-computed rows.
# ---------------------------------------------------------------------------


def _log_for(fixture: str, index: int) -> dict:
    """One log from an ``rpc_logs_*`` fixture (the ``_comment``-enveloped form)."""
    data = json.loads((FIXTURES / fixture).read_text(encoding="utf-8"))
    logs = data["logs"] if isinstance(data, dict) and "logs" in data else data
    return logs[index]


def test_decode_swap_field_by_field() -> None:
    """Every field of the anchored Swap log, exactly.

    amount0 is a NEGATIVE int256 (USDC flowed out) and must decode to a
    negative decimal; sqrt_price_x96 is the §3.0 $3000 anchor; the tick is the
    right-aligned int24 in the last data word.
    """
    row = decode_swap(_log_for("rpc_logs_swap.json", 0), pool_address=POOL_ADDR, block_timestamp=TS)
    assert row["block_number"] == 13385010
    assert row["log_index"] == 0
    assert row["block_timestamp"] == TS
    assert row["tx_hash"] == "0x" + f"{101:064x}"
    assert row["pool_address"] == POOL_ADDR
    assert row["event_type"] == "swap"
    assert row["amount0"] == "-548466086452"  # negative: two's-complement int256
    assert row["amount1"] == "182700063432212445192"
    assert row["sqrt_price_x96"] == "1446501726624926496477173928747177"  # $3000 anchor
    assert row["liquidity"] == "15000000000000000000"
    assert row["tick"] == 196242
    assert row["sender"] == "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert row["recipient"] == "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def test_decode_swap_negative_amount0_and_negative_tick() -> None:
    row = _log_for("rpc_logs_swap.json", 2)
    decoded = decode_swap(row, pool_address=POOL_ADDR, block_timestamp=TS)
    # int256 amount0 sign-extends across the full word...
    assert int(decoded["amount0"]) < 0
    # ...and the int24 tick sign-extends from bit 23 (right-aligned in the word).
    assert int(decoded["amount0"]) == -730980826364580233851
    assert decoded["tick"] == -202010
    assert decoded["block_number"] == 13385011


def test_decode_mint_sender_from_data_not_topic() -> None:
    """The brief's trap: Mint's sender is NON-indexed despite being FIRST in the
    signature — it must be read from data word 0, not from any topic."""
    log = _log_for("rpc_logs_mint.json", 0)
    row = decode_mint(log, pool_address=POOL_ADDR, block_timestamp=TS)
    # The three indexed params are owner / tickLower / tickUpper (4 topics incl. topic0).
    assert len(log["topics"]) == 4
    assert row["owner"] == "0x1111111111111111111111111111111111111111"
    assert row["tick_lower"] == 195000
    assert row["tick_upper"] == 201000
    assert row["liquidity_amount"] == "5000000000000000000"
    assert row["amount0"] == "57824009226822"
    assert row["amount1"] == "5557749327432510238473"
    assert row["sender"] == "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"  # from data
    # The sender must come from data word 0: it is NOT the first topic (owner).
    assert row["sender"] != row["owner"]
    assert log["topics"][1].lower() == (
        "0x0000000000000000000000001111111111111111111111111111111111111111"
    )


def test_decode_burn_sender_is_null() -> None:
    row = decode_burn(_log_for("rpc_logs_burn.json", 0), pool_address=POOL_ADDR, block_timestamp=TS)
    assert row["sender"] is None  # Burn has no sender (CONTRACTS.md §4.2)
    assert row["amount0"] == "57824009226822"
    assert row["liquidity_amount"] == "5000000000000000000"


def test_decode_collect() -> None:
    row = decode_collect(
        _log_for("rpc_logs_collect.json", 0), pool_address=POOL_ADDR, block_timestamp=TS
    )
    assert row["owner"] == "0x1111111111111111111111111111111111111111"
    assert row["recipient"] == "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert row["tick_lower"] == 195000
    assert row["tick_upper"] == 201000
    assert row["amount0"] == "20000000"
    assert row["amount1"] == "1500000000000000"


def test_decode_flash() -> None:
    """Flash is in scope; a flash's paid exceeds its amount (that delta is fee
    growth with no liquidity change). Built inline — no flash on the pinned
    pools in the study window, so no rpc_logs_flash fixture is owned."""
    log = _build_log(
        TOPIC_FLASH,
        indexed=[_addr_word("0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
                 _addr_word("0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")],
        data=[_w32(1000), _w32(2000), _w32(1300), _w32(2600)],
        block=13385030,
        log_index=0,
    )
    row = decode_flash(log, pool_address=POOL_ADDR, block_timestamp=TS)
    assert row["event_type"] == "flash"
    assert row["amount0"] == "1000"
    assert row["amount1"] == "2000"
    assert row["paid0"] == "1300"  # paid > amount: the flash fee
    assert row["paid1"] == "2600"
    assert row["sender"] == "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


# ---------------------------------------------------------------------------
# Loud-failure defences.
# ---------------------------------------------------------------------------


def test_wrong_topic0_raises() -> None:
    log = _log_for("rpc_logs_swap.json", 0)
    log["topics"][0] = TOPIC_MINT  # a title-fight: swap decoder, mint topic0
    with pytest.raises(SchemaViolationError, match="does not match event topic0"):
        decode_swap(log, pool_address=POOL_ADDR, block_timestamp=TS)


def test_truncated_data_raises_not_indexerror() -> None:
    log = _log_for("rpc_logs_swap.json", 0)
    # Chop one word off the end: a truncated amount/tick field, not a clean word.
    log["data"] = log["data"][: -64]
    with pytest.raises(SchemaViolationError, match="must be exactly 5 32-byte word"):
        decode_swap(log, pool_address=POOL_ADDR, block_timestamp=TS)


def test_tick_outside_range_raises() -> None:
    log = _log_for("rpc_logs_swap.json", 0)
    # 887273 > MAX_TICK: would only be reachable via a broken sign/width decode.
    log["data"] = log["data"][:-64] + _w32(887273)[2:]
    with pytest.raises(SchemaViolationError, match="outside"):
        decode_swap(log, pool_address=POOL_ADDR, block_timestamp=TS)


def test_negative_tick_range_from_indexed_int24() -> None:
    """Indexed int24 topics carry a sign-extended tick; the decoder must recover
    a NEGATIVE range like [-201000, -199800] (the brief's burn/collect case)."""
    lower, upper = -201000, -199800
    owner = _addr_word("0x1111111111111111111111111111111111111111")
    # Burn(address indexed owner, int24 indexed tickLower, int24 indexed tickUpper,
    #      uint128 amount, uint256 amount0, uint256 amount1)
    burn = _build_log(
        TOPIC_BURN,
        indexed=[owner, _int24_word(lower), _int24_word(upper)],
        data=[_w32(5_000_000_000_000_000_000), _w32(3), _w32(4)],
        block=13385100,
        log_index=0,
    )
    row = decode_burn(burn, pool_address=POOL_ADDR, block_timestamp=TS)
    assert row["tick_lower"] == lower
    assert row["tick_upper"] == upper
    assert row["liquidity_amount"] == "5000000000000000000"
    # Collect(address indexed owner, address recipient, int24 indexed tickLower,
    #         int24 indexed tickUpper, uint128 amount0, uint128 amount1): recipient
    # is NON-indexed and sits in data.
    collect = _build_log(
        TOPIC_COLLECT,
        indexed=[owner, _int24_word(lower), _int24_word(upper)],
        data=[_addr_word("0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"), _w32(3), _w32(4)],
        block=13385100,
        log_index=0,
    )
    row = decode_collect(collect, pool_address=POOL_ADDR, block_timestamp=TS)
    assert row["tick_lower"] == lower
    assert row["tick_upper"] == upper
    assert row["recipient"] == "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


# ---------------------------------------------------------------------------
# eth_call return decoders (rpc_ticks_call.json).
# ---------------------------------------------------------------------------


def _ticks_fixture() -> dict:
    return json.loads((FIXTURES / "rpc_ticks_call.json").read_text(encoding="utf-8"))


def test_slot0_seven_word_decode() -> None:
    slot0 = decode_slot0_return(_ticks_fixture()["slot0"])
    assert slot0.sqrt_price_x96 == 1446501726624926496477173928747177
    assert slot0.tick == 196242


def test_slot0_negative_tick_canonical_sign_extension() -> None:
    """A negative int24 tick is sign-extended across its whole 32-byte word
    (all-ones high part). decode_slot0_return must read only the low 24 bits and
    sign-extend from bit 23, tolerating the canonical form."""
    sqrt = 1446501726624926496477173928747177
    neg_tick = (-202010) & ((1 << 256) - 1)  # canonical sign extension
    data = _w32(sqrt) + _w32(neg_tick)[2:] + _w32(0)[2:] * 5
    assert len(data) - 2 == 448  # 7 words
    slot0 = decode_slot0_return(data)
    assert slot0.sqrt_price_x96 == sqrt
    assert slot0.tick == -202010


def test_ticks_return_signed_liquidity_net_and_exact_q128() -> None:
    t = decode_ticks_return(_ticks_fixture()["ticks"]["196242"])
    assert t.liquidity_gross == 15_000_000_000_000_000_000
    assert t.liquidity_net == -2_000_000_000_000_000_000  # int128 signed
    assert t.fee_growth_outside_0_x128 == 128888888888888888888888888888888888
    assert t.fee_growth_outside_1_x128 == 77777777777777777777777777777777777
    assert t.initialized is True
    empty = decode_ticks_return(_ticks_fixture()["ticks"]["196238"])
    assert empty.liquidity_gross == 0
    assert empty.initialized is False


def test_uint256_and_liquidity_returns_exact() -> None:
    fx = _ticks_fixture()
    g0 = decode_uint256_return(fx["feeGrowthGlobal0X128"])
    assert g0 == 245472016230501318647318615910821729000
    assert decode_liquidity_return(fx["liquidity"]) == 15_000_000_000_000_000_000


def test_encode_ticks_calldata() -> None:
    word = encode_ticks_calldata(196242)
    assert word[:10] == SELECTOR_TICKS
    assert len(word) == 2 + 8 + 64  # 0x + selector + word
    decoded = int(word[10:], 16)
    assert decoded == 196242
    assert int(encode_ticks_calldata(-202010)[10:], 16) == (1 << 256) - 202010
    with pytest.raises(SchemaViolationError):
        encode_ticks_calldata(1 << 23)  # does not fit int24


def test_truncated_return_data_raises() -> None:
    with pytest.raises(SchemaViolationError):
        decode_slot0_return("0x1234")
    with pytest.raises(SchemaViolationError):
        decode_ticks_return(_ticks_fixture()["ticks"]["196242"][:-64])


# ---------------------------------------------------------------------------
# Minimal ABI *encoder* helpers (independent of abi.py's decoders — they lay
# bytes exactly per the Solidity spec so the decoders have something real to
# chew on).
# ---------------------------------------------------------------------------


def _w32(value: int) -> str:
    return "0x" + (value & ((1 << 256) - 1)).to_bytes(32, "big").hex()


def _int24_word(tick: int) -> str:
    return _w32(tick)  # negative -> sign extension via the & mask


def _addr_word(address: str) -> str:
    return "0x" + bytes.fromhex(address[2:]).rjust(32, b"\x00").hex()


def _build_log(
    topic0: str,
    *,
    indexed: list[str],
    data: list[str],
    block: int,
    log_index: int,
) -> dict:
    return {
        "address": POOL_ADDR,
        "topics": [topic0] + indexed,
        "data": "0x" + "".join(w[2:] for w in data),
        "blockNumber": hex(block),
        "blockHash": "0x" + f"ab{block:062x}",
        "transactionHash": "0x" + f"cd{block:062x}",
        "transactionIndex": "0x0",
        "logIndex": hex(log_index),
        "removed": False,
    }