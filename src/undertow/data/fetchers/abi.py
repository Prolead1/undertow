"""Pure ABI constants and decoders for CONTRACTS Route C (T06).

This module is deliberately **pure**: no I/O, no ``requests``, nothing that touches
the network — the decoders are unit-testable without any HTTP. ``rpc.py`` does
transport; this module does bytes->rows. The reviewer's god-module rule is the
whole reason for the split.

What lives here:

* ``TOPIC_*`` — the ``topic0`` keccak256 constants for the five pool event logs.
  Event signatures are the deployed ones (Uniswap v3-core ``IUniswapV3PoolEvents``,
  the release shipped to the pinned pools) and are quoted verbatim in comments;
  they are **hardcoded**, not computed, and ``test_abi.py`` recomputes them via an
  independent keccak256 and asserts equality (incl. the roadmap's known-answer
  Swap topic0). Note the two traps documented in the brief: ``Swap``'s amounts
  are **signed ``int256``** (not ``uint256``), and ``Mint``'s ``sender`` is a
  **non-indexed** parameter that appears *first* in the signature but is read
  from ``data``, not from a topic.

* ``SELECTOR_*`` / ``encode_ticks_calldata`` — hand-encoded ``eth_call`` function
  selectors and calldata for ``fee_growth_at`` (slot0, liquidity,
  feeGrowthGlobal0X128/1X128, ticks(int24)).

* One pure decoder per event — ``decode_swap``, ``decode_mint``, ``decode_burn``,
  ``decode_collect``, ``decode_flash`` — each taking a *raw log dict* as returned
  by ``eth_getLogs`` (``topics: list[str]``, ``data: str``, ``blockNumber``,
  ``logIndex``, ``transactionHash``) plus the pool address and block timestamp
  (which only the transport can know), and returning a **schema-shaped row dict**.

* Return decoders for the ``eth_call`` calls above (``decode_slot0_return``,
  ``decode_ticks_return``, ``decode_uint256_return``, ``decode_liquidity_return``)
  and their packed-value helpers.

Integers are arbitrary-precision ``int`` throughout; schema rows serialise them
with ``schemas.encode_uint`` (decimal strings) exactly per CONTRACTS.md §0. No
``float`` anywhere.

Decoding traps handled (all from the T06 brief, §'what to build'):

* ``int256`` amounts are two's-complement sign-extended over the full 32-byte
  word; ``int24`` ticks in ``data`` are right-aligned in the word and must be
  sign-extended from **bit 23** — never read the word as a plain uint.
* Indexed ``int24`` topics are 32-byte words carrying a sign-extended ``int24``;
  they decode via two's complement over the full word, then get range-checked
  against ``MIN_TICK``/``MAX_TICK`` — the loud-failure defense against a silent
  decode bug.
* ``topics[0]`` must equal the event's ``topic0`` or we raise
  :class:`SchemaViolationError` rather than silently decode (a topic0 filter
  mismatch means the provider returned the wrong event family).
* ``data`` must be exactly the expected number of 32-byte words for the event;
  a truncated field raises :class:`SchemaViolationError`, never ``IndexError``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from undertow.data.config import MAX_TICK, MIN_TICK
from undertow.data.types import Address, SchemaViolationError

# ---------------------------------------------------------------------------
# topic0 constants (keccak256 of the event signature; test_abi.py recomputes).
# Signature strings are the deployed IUniswapV3PoolEvents signatures verbatim.
# ---------------------------------------------------------------------------

# Swap(address,address,int256,int256,uint160,uint128,int24)
TOPIC_SWAP: Final[str] = (
    "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
)

# Mint(address,address,int24,int24,uint128,uint256,uint256)
TOPIC_MINT: Final[str] = (
    "0x7a53080ba414158be7ec69b987b5fb7d07dee101fe85488f0853ae16239d0bde"
)

# Burn(address,int24,int24,uint128,uint256,uint256)
TOPIC_BURN: Final[str] = (
    "0x0c396cd989a39f4459b5fa1aed6a9a8dcdbc45908acfd67e028cd568da98982c"
)

# Collect(address,address,int24,int24,uint128,uint128)
TOPIC_COLLECT: Final[str] = (
    "0x70935338e69775456a85ddef226c395fb668b63fa0115f5f20610b388e6ca9c0"
)

# Flash(address,address,uint256,uint256,uint256,uint256)
TOPIC_FLASH: Final[str] = (
    "0xbdbdb71d7860376ba52b25a5028beea23581364a40522f6bcfb86bb1f2dca633"
)

# ---------------------------------------------------------------------------
# eth_call function selectors — keccak256(selectorsignature())[:4].
# ---------------------------------------------------------------------------
SELECTOR_SLOT0: Final[str] = "0x3850c7bd"  # slot0()
SELECTOR_LIQUIDITY: Final[str] = "0x1a686502"  # liquidity()
SELECTOR_FEE_GROWTH_GLOBAL_0_X128: Final[str] = "0xf3058399"  # feeGrowthGlobal0X128()
SELECTOR_FEE_GROWTH_GLOBAL_1_X128: Final[str] = "0x46141319"  # feeGrowthGlobal1X128()
SELECTOR_TICKS: Final[str] = "0xf30dba93"  # ticks(int24)

# Number of 32-byte words each event's `data` payload carries: the count of the
# event's NON-indexed parameters, in declaration order.
_SWAP_WORDS: Final[int] = 5
_MINT_WORDS: Final[int] = 4
_BURN_WORDS: Final[int] = 3
_COLLECT_WORDS: Final[int] = 3
_FLASH_WORDS: Final[int] = 4

_WORD_HEX: Final[int] = 64  # 32 bytes


# ---------------------------------------------------------------------------
# Private byte-level helpers — exactness first, float nowhere.
# ---------------------------------------------------------------------------


def _strip_0x(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.startswith("0x"):
        raise SchemaViolationError(
            f"{where}: expected a 0x-prefixed hex string, got {value!r}"
        )
    return value[2:]


def _hex_to_int(value: object, where: str) -> int:
    """``"0x.."`` hex string (or int) -> int, loudly on anything else."""
    if isinstance(value, int):
        return value
    try:
        return int(_strip_0x(value, where), 16)
    except (TypeError, ValueError):
        raise SchemaViolationError(
            f"{where}: expected a hex block/log index, got {value!r}"
        ) from None


def _words(data: object, n_words: int, where: str) -> list[str]:
    """Split a data hex string into ``n_words`` 32-byte words (loud length check)."""
    try:
        body = _strip_0x(data, where)
    except SchemaViolationError:
        raise
    if len(body) != n_words * _WORD_HEX:
        raise SchemaViolationError(
            f"{where}: data must be exactly {n_words} 32-byte word(s) "
            f"({n_words * _WORD_HEX} hex chars), got {len(body)} hex chars"
        )
    try:
        return [body[i : i + _WORD_HEX] for i in range(0, len(body), _WORD_HEX)]
    except (ValueError, TypeError):
        raise SchemaViolationError(f"{where}: data is not valid hex: {data!r}") from None


def _word_int(word: str) -> int:
    return int(word, 16)


def _int256(word: str) -> int:
    """Two's-complement int over the full 32-byte word (signed int256)."""
    value = _word_int(word)
    return value - (1 << 256) if value >= (1 << 255) else value


def _int128(word: str) -> int:
    """Signed int128, sign-extended to 256 bits (Solidity pads with the sign)."""
    value = _word_int(word)
    return value - (1 << 256) if value >= (1 << 255) else value


def _int24_from_data_word(word: str) -> int:
    """``int24`` right-aligned in a 32-byte data word: sign-extend from bit 23.

    Tolerates both an all-ones high part (canonical sign extension) and an
    all-zeros high part — only the low 24 bits are read, exactly as the brief
    demands ("sign-extend from 24 bits, right-aligned in a 32-byte word").
    """
    value = _word_int(word) & 0xFFFFFF
    return value - (1 << 24) if value >= (1 << 23) else value


def _int24_from_topic(word: str) -> int:
    """Indexed ``int24`` topic: two's complement over the full word.

    The provider encodes an indexed int24 as a sign-extended 32-byte word, so
    the full-word two's complement recovers the value; a zero-padded word for a
    negative tick would decode to a huge positive and fail the range check the
    decoder applies afterwards — the loud failure the brief wants.
    """
    value = _word_int(word)
    return value - (1 << 256) if value >= (1 << 255) else value


def _tick_set_or_raise(tick: int, where: str) -> int:
    if not MIN_TICK <= tick <= MAX_TICK:
        raise SchemaViolationError(
            f"{where}: tick {tick} outside [{MIN_TICK}, {MAX_TICK}] — the decode is "
            "suspect (silent sign/width bugs surface here)"
        )
    return tick


def _uint(word: str, n_bytes: int) -> int:
    """Unsigned integer from the LOW ``n_bytes`` of a 32-byte word."""
    return _word_int(word) & ((1 << (8 * n_bytes)) - 1)


def _address_from_word(word: str) -> Address:
    """An ABI address occupies the LAST 20 bytes of a 32-byte word."""
    tail = word[-40:]
    return Address("0x" + tail)


def _topics(
    log: Mapping[str, object], expected_topic: str, indexed_count: int, where: str
) -> list[str]:
    """Validate ``topics[0]`` and length; return topics[1:] (the indexed params)."""
    raw = log.get("topics")
    if not isinstance(raw, list) or not all(isinstance(t, str) for t in raw):
        raise SchemaViolationError(
            f"{where}: log must carry a 'topics' list of hex strings, got {raw!r}"
        )
    topics = raw
    if not topics:
        raise SchemaViolationError(
            f"{where}: log carries an empty 'topics' array; expected topic0 "
            f"{expected_topic!r}"
        )
    if topics[0].lower() != expected_topic.lower():
        raise SchemaViolationError(
            f"{where}: topics[0] {topics[0]!r} does not match event topic0 "
            f"{expected_topic!r}; refusing to decode a mismatched event family"
        )
    if len(topics) < indexed_count + 1:
        raise SchemaViolationError(
            f"{where}: expected {indexed_count} indexed parameter topic(s) "
            f"after topic0, got {len(topics) - 1}"
        )
    return topics[1 : indexed_count + 1]


def _base_row(
    log: Mapping[str, object], event_type: str, pool_address: Address,
    block_timestamp: datetime, where: str,
) -> dict[str, object]:
    """The six key columns (CONTRACTS.md §4.0.1), decoded from the log envelope."""
    return {
        "block_number": _hex_to_int(log.get("blockNumber"), f"{where}.blockNumber"),
        "log_index": _hex_to_int(log.get("logIndex"), f"{where}.logIndex"),
        "block_timestamp": block_timestamp,
        "tx_hash": _normalise_tx_hash(log.get("transactionHash"), f"{where}.transactionHash"),
        "pool_address": str(pool_address).lower(),
        "event_type": event_type,
    }


def _normalise_tx_hash(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise SchemaViolationError(f"{where}: expected a hex tx hash string, got {value!r}")
    text = value[2:] if value.startswith("0x") else value
    if len(text) != 64:
        raise SchemaViolationError(
            f"{where}: expected a 64-hex-char (0x-prefixed) tx hash, got {value!r}"
        )
    try:
        int(text, 16)
    except ValueError:
        raise SchemaViolationError(f"{where}: tx hash is not hex: {value!r}") from None
    return "0x" + text.lower()


# ---------------------------------------------------------------------------
# Event decoders: raw log dict -> schema-shaped row dict.
# CLIENTS pass ``pool_address`` and ``block_timestamp`` because only the
# transport knows them. Events with a *negative tick* decode to negative ints
# here (the schema stores them as plain int32 columns).
# ---------------------------------------------------------------------------


def decode_swap(log: Mapping[str, object], *, pool_address: Address,
                block_timestamp: datetime) -> dict[str, object]:
    """Decode a ``Swap`` log. Signature: Swap(address,address,int256,int256,uint160,uint128,int24).

    ``sender``/``recipient`` are indexed -> topics[1], topics[2]; ``amount0``,
    ``amount1`` (signed int256), ``sqrtPriceX96`` (uint160), ``liquidity``
    (uint128) and ``tick`` (int24, right-aligned) are in data, in that order.
    """
    where = "decode_swap"
    indexed = _topics(log, TOPIC_SWAP, 2, where)
    w = _words(log.get("data"), _SWAP_WORDS, where)
    row = _base_row(log, "swap", pool_address, block_timestamp, where)
    _apply_uint256_fields(row, {
        "amount0": _int256(w[0]),
        "amount1": _int256(w[1]),
        "sqrt_price_x96": _uint(w[2], 20),
        "liquidity": _uint(w[3], 16),
    })
    row["tick"] = _tick_set_or_raise(_int24_from_data_word(w[4]), f"{where}.tick")
    row["sender"] = _address_from_word(indexed[0])
    row["recipient"] = _address_from_word(indexed[1])
    return row


def decode_mint(log: Mapping[str, object], *, pool_address: Address,
                block_timestamp: datetime) -> dict[str, object]:
    """Decode a ``Mint`` log. Signature: Mint(address,address,int24,int24,uint128,uint256,uint256).

    Indexed: owner, tickLower, tickUpper -> topics[1..3]. **Trap**: ``sender``
    is a NON-indexed parameter that appears FIRST in the signature but is
    carried in ``data``, not in a topic; ``data`` order is sender, amount,
    amount0, amount1.
    """
    where = "decode_mint"
    indexed = _topics(log, TOPIC_MINT, 3, where)
    w = _words(log.get("data"), _MINT_WORDS, where)
    row = _base_row(log, "mint", pool_address, block_timestamp, where)
    lower = _tick_set_or_raise(_int24_from_topic(indexed[1]), f"{where}.tickLower")
    upper = _tick_set_or_raise(_int24_from_topic(indexed[2]), f"{where}.tickUpper")
    if upper <= lower:
        raise SchemaViolationError(
            f"{where}: tick_upper {upper} must be > tick_lower {lower}"
        )
    row["owner"] = _address_from_word(indexed[0])
    row["tick_lower"] = lower
    row["tick_upper"] = upper
    row["liquidity_amount"] = _encode_uint(_uint(w[1], 16))
    _apply_uint256_fields(row, {"amount0": _word_int(w[2]), "amount1": _word_int(w[3])})
    row["sender"] = _address_from_word(w[0])  # the NON-indexed sender, from data
    return row


def decode_burn(log: Mapping[str, object], *, pool_address: Address,
                block_timestamp: datetime) -> dict[str, object]:
    """Decode a ``Burn`` log. Signature: Burn(address,int24,int24,uint128,uint256,uint256).

    Indexed: owner, tickLower, tickUpper -> topics[1..3]; data order is amount,
    amount0, amount1. ``sender`` is null on burn rows (the Burn event has no
    sender — CONTRACTS.md §4.2).
    """
    where = "decode_burn"
    indexed = _topics(log, TOPIC_BURN, 3, where)
    w = _words(log.get("data"), _BURN_WORDS, where)
    row = _base_row(log, "burn", pool_address, block_timestamp, where)
    lower = _tick_set_or_raise(_int24_from_topic(indexed[1]), f"{where}.tickLower")
    upper = _tick_set_or_raise(_int24_from_topic(indexed[2]), f"{where}.tickUpper")
    if upper <= lower:
        raise SchemaViolationError(
            f"{where}: tick_upper {upper} must be > tick_lower {lower}"
        )
    row["owner"] = _address_from_word(indexed[0])
    row["tick_lower"] = lower
    row["tick_upper"] = upper
    row["liquidity_amount"] = _encode_uint(_uint(w[0], 16))
    _apply_uint256_fields(row, {"amount0": _word_int(w[1]), "amount1": _word_int(w[2])})
    row["sender"] = None
    return row


def decode_collect(log: Mapping[str, object], *, pool_address: Address,
                   block_timestamp: datetime) -> dict[str, object]:
    """Decode a ``Collect`` log. Signature: Collect(address,address,int24,int24,uint128,uint128).

    Indexed: owner, tickLower, tickUpper -> topics[1..3]; data order is
    recipient, amount0, amount1 (amounts are uint128, the fees actually
    withdrawn — CONTRACTS.md §4.3).
    """
    where = "decode_collect"
    indexed = _topics(log, TOPIC_COLLECT, 3, where)
    w = _words(log.get("data"), _COLLECT_WORDS, where)
    row = _base_row(log, "collect", pool_address, block_timestamp, where)
    lower = _tick_set_or_raise(_int24_from_topic(indexed[1]), f"{where}.tickLower")
    upper = _tick_set_or_raise(_int24_from_topic(indexed[2]), f"{where}.tickUpper")
    if upper <= lower:
        raise SchemaViolationError(
            f"{where}: tick_upper {upper} must be > tick_lower {lower}"
        )
    row["owner"] = _address_from_word(indexed[0])
    row["recipient"] = _address_from_word(w[0])
    row["tick_lower"] = lower
    row["tick_upper"] = upper
    _apply_uint256_fields(row, {"amount0": _uint(w[1], 16), "amount1": _uint(w[2], 16)})
    return row


def decode_flash(log: Mapping[str, object], *, pool_address: Address,
                 block_timestamp: datetime) -> dict[str, object]:
    """Decode a ``Flash`` log. Signature: Flash(address,address,uint256,uint256,uint256,uint256).

    Indexed: sender, recipient -> topics[1..2]; data order is amount0, amount1,
    paid0, paid1 (unsigned; paidX - amountX is fee growth with NO liquidity
    change — CONTRACTS.md §4.3.1).
    """
    where = "decode_flash"
    indexed = _topics(log, TOPIC_FLASH, 2, where)
    w = _words(log.get("data"), _FLASH_WORDS, where)
    row = _base_row(log, "flash", pool_address, block_timestamp, where)
    _apply_uint256_fields(row, {
        "amount0": _word_int(w[0]),
        "amount1": _word_int(w[1]),
        "paid0": _word_int(w[2]),
        "paid1": _word_int(w[3]),
    })
    row["sender"] = _address_from_word(indexed[0])
    row["recipient"] = _address_from_word(indexed[1])
    return row


def _apply_uint256_fields(row: dict[str, object], fields: Mapping[str, int]) -> None:
    """Store uint256/int256 values as decimal strings (CONTRACTS.md §0)."""
    for name, value in fields.items():
        row[name] = _encode_uint(value)


def _encode_uint(value: int) -> str:
    """The only sanctioned int->string conversion (schemas.encode_uint)."""
    from undertow.data.schemas import encode_uint  # local import: keeps abi side-effect free

    return encode_uint(value)

# ---------------------------------------------------------------------------
# eth_call return decoders (for fee_growth_at). All exact integers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Slot0:
    """Decoded ``slot0()`` return."""

    sqrt_price_x96: int
    tick: int
    fee_protocol: int  # packed uint8: bits 0-3 = fp0, bits 4-7 = fp1; 0 when off


@dataclass(frozen=True, slots=True)
class TickStruct:
    """Decoded ``ticks(int24)`` return — the whitepaper Table 2 tick struct."""

    liquidity_gross: int
    liquidity_net: int  # signed int128
    fee_growth_outside_0_x128: int
    fee_growth_outside_1_x128: int
    tick_cumulative_outside: int  # signed int56
    seconds_per_liquidity_outside_x128: int  # uint160
    seconds_outside: int  # uint32
    initialized: bool


def _single_word(return_data: str, where: str) -> str:
    words = _words(return_data, 1, where)
    return words[0]


def decode_slot0_return(return_data: str) -> Slot0:
    """``slot0() -> (uint160 sqrtPriceX96, int24 tick, uint16 observationIndex,
    uint16 observationCardinality, uint16 observationCardinalityNext, uint8
    feeProtocol, bool unlocked)``.

    The ABI encodes each return value into its OWN 32-byte word (7 words, each
    right-aligned) — NOT a single bit-packed word. Returns all three fields the
    fee-growth pipeline needs: sqrtPriceX96, tick, and the packed feeProtocol
    (bits 0-3 = fp0, bits 4-7 = fp1).
    """
    where = "decode_slot0_return"
    words = _words(return_data, 7, where)
    sqrt_price_x96 = _word_int(words[0])
    if sqrt_price_x96 <= 0:
        raise SchemaViolationError(f"{where}: sqrt_price_x96 must be positive")
    tick = _int24_from_data_word(words[1])
    # feeProtocol is a uint8 in the low byte of word 5 (the 6th word)
    fee_protocol = _word_int(words[5]) & 0xFF
    return Slot0(
        sqrt_price_x96=sqrt_price_x96,
        tick=_tick_set_or_raise(tick, f"{where}.tick"),
        fee_protocol=fee_protocol,
    )


def decode_liquidity_return(return_data: str) -> int:
    """``liquidity() -> uint128`` — a single low-128-bit value in the word."""
    return _uint(_single_word(return_data, "decode_liquidity_return"), 16)


def decode_uint256_return(return_data: str) -> int:
    """``feeGrowthGlobal{0,1}X128() -> uint256`` — a full 256-bit word."""
    return _word_int(_single_word(return_data, "decode_uint256_return"))


def decode_ticks_return(return_data: str) -> TickStruct:
    """``ticks(int24) -> (uint128, int128, uint256, uint256, int56, uint160, uint32, bool)``.

    Eight static components are too big for the compact rule, so the return is
    EIGHT 32-byte head words in declaration order: liquidityGross (u128),
    liquidityNet (i128, signed), feeGrowthOutside0X128 (u256), feeGrowthOutside1X128
    (u256), tickCumulativeOutside (i56, signed), secondsPerLiquidityOutsideX128
    (u160), secondsOutside (u32), initialized (bool).
    """
    where = "decode_ticks_return"
    w = _words(return_data, 8, where)
    return TickStruct(
        liquidity_gross=_uint(w[0], 16),
        liquidity_net=_int128(w[1]),
        fee_growth_outside_0_x128=_word_int(w[2]),
        fee_growth_outside_1_x128=_word_int(w[3]),
        tick_cumulative_outside=_int56(w[4]),
        seconds_per_liquidity_outside_x128=_uint(w[5], 20),
        seconds_outside=_uint(w[6], 4),
        initialized=(_word_int(w[7]) & 0xFF) != 0,
    )


def _int56(word: str) -> int:
    """Signed int56, sign-extended to 256 bits."""
    value = _word_int(word)
    return value - (1 << 256) if value >= (1 << 255) else value


# ---------------------------------------------------------------------------
# eth_call calldata builder.
# ---------------------------------------------------------------------------


def encode_ticks_calldata(tick: int) -> str:
    """Calldata for ``ticks(int24)`` at ``tick``: selector + sign-extended int24 arg.

    The argument word is the sign-extended int24 right-aligned in 32 bytes (the
    Solidity ABI integer encoding). Addresses the fact that a coin-flip of a
    sign error here would silently query the WRONG tick.
    """
    if not -(1 << 23) <= tick < (1 << 23):
        raise SchemaViolationError(
            f"encode_ticks_calldata: tick {tick} does not fit in int24"
        )
    word = (tick & ((1 << 256) - 1)).to_bytes(32, "big").hex()
    return SELECTOR_TICKS + word


def selectors() -> dict[str, str]:
    """Map method name -> calldata selector (used by the tests and fee_growth_at)."""
    return {
        "slot0()": SELECTOR_SLOT0,
        "liquidity()": SELECTOR_LIQUIDITY,
        "feeGrowthGlobal0X128()": SELECTOR_FEE_GROWTH_GLOBAL_0_X128,
        "feeGrowthGlobal1X128()": SELECTOR_FEE_GROWTH_GLOBAL_1_X128,
        "ticks(int24)": SELECTOR_TICKS,
    }