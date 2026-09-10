"""Fixed-point, tick, and liquidity math for ``undertow.sim``.

Self-contained copy of the functions the simulator needs from
``undertow.data.fixedpoint``. The sim package must not import from ``data``
(scaffold boundary rule), so this module duplicates the math it needs.

Every integer here is arbitrary-precision ``int``; no ``float`` appears
anywhere on the tick/price/accumulator path.
"""

from __future__ import annotations

from decimal import Decimal, localcontext

# ---------------------------------------------------------------------------
# Pinned constants
# ---------------------------------------------------------------------------
Q96: int = 2**96
Q128: int = 2**128
MAX_UINT256: int = 2**256 - 1
MIN_TICK: int = -887272
MAX_TICK: int = 887272
# getSqrtRatioAtTick(MIN_TICK) = 4295128739 (from TickMath.sol)
MIN_SQRT_RATIO: int = 4295128739
# getSqrtRatioAtTick(MAX_TICK)
MAX_SQRT_RATIO: int = 1461446703485210103287273052203988822378723970342
TICK_BASE: Decimal = Decimal("1.0001")

_PRICE_PREC: int = 80  # Decimal context precision for price conversions

# ---------------------------------------------------------------------------
# TickMath decomposition constants (from Uniswap v3-core TickMath.sol)
# ---------------------------------------------------------------------------
_RATIO_INIT_TRUE: int = 0xfffcb933bd6fad37aa2d162d1a594001  # noqa: E221
_RATIO_INIT_FALSE: int = 0x100000000000000000000000000000000  # noqa: E221
_RATIO_MULS: tuple[int, ...] = (
    0xfff97272373d413259a46990580e213a,
    0xfff2e50f5f656932ef12357cf3c7fdcc,
    0xffe5caca7e10e4e61c3624eaa0941cd0,
    0xffcb9843d60f6159c9db58835c926644,
    0xff973b41fa98c081472e6896dfb254c0,
    0xff2ea16466c96a3843ec78b326b52861,
    0xfe5dee046a99a2a811c461f1969c3053,
    0xfcbe86c7900a88aedcffc83b479aa3a4,
    0xf987a7253ac413176f2b074cf7815e54,
    0xf3392b0822b70005940c7a398e4b70f3,
    0xe7159475a2c29b7443b29c7fa6e889d9,
    0xd097f3bdfd2022b8845ad8f792aa5825,
    0xa9f746462d870fdf8a65dc1f90e061e5,
    0x70d869a156d2a1b890bb3df62baf32f7,
    0x31be135f97d08fd981231505542fcfa6,
    0x9aa508b5b7a84e1c677de54f3e99bc9,
    0x5d6af8dedb81196699c329225ee604,
    0x2216e584f5fa1ea926041bedfe98,
    0x48a170391f7dc42444e8fa2,
)


# ---------------------------------------------------------------------------
# Internal division helpers (port of FullMath / UnsafeMath)
# ---------------------------------------------------------------------------
def _mul_div_floor(a: int, b: int, denominator: int) -> int:
    """``floor(a * b / denominator)``, full precision."""
    if denominator <= 0:
        raise ValueError("mulDiv denominator must be positive")
    return (a * b) // denominator


def _mul_div_ceil(a: int, b: int, denominator: int) -> int:
    """``ceil(a * b / denominator)``."""
    if denominator <= 0:
        raise ValueError("mulDivRoundingUp denominator must be positive")
    return (a * b + denominator - 1) // denominator


def _div_ceil(a: int, d: int) -> int:
    """``ceil(a / d)`` for ``d > 0``."""
    return (a + d - 1) // d


# ---------------------------------------------------------------------------
# Price conversion — Decimal, exact, never float
# ---------------------------------------------------------------------------
def sqrt_price_x96_to_price(sqrt_price_x96: int, dec0: int, dec1: int) -> Decimal:
    """Human price of token1 denominated in token0.

    ``price = 10**(dec1 - dec0) * 2**192 / sqrt_price_x96**2``
    For USDC(6)/WETH(18): USDC per WETH (~3000).
    """
    if sqrt_price_x96 <= 0:
        raise ValueError("sqrt_price_x96 must be positive")
    delta = dec1 - dec0
    if delta >= 0:
        numerator = (10**delta) * (2**192)
        denominator = sqrt_price_x96 * sqrt_price_x96
    else:
        numerator = 2**192
        denominator = sqrt_price_x96 * sqrt_price_x96 * (10 ** (-delta))
    with localcontext() as ctx:
        ctx.prec = _PRICE_PREC
        return Decimal(numerator) / Decimal(denominator)


def price_to_sqrt_price_x96(price: Decimal, dec0: int, dec1: int) -> int:
    """Inverse of ``sqrt_price_x96_to_price``."""
    if price <= 0:
        raise ValueError("price must be positive")
    with localcontext() as ctx:
        ctx.prec = _PRICE_PREC
        delta = dec1 - dec0
        if delta >= 0:
            radicand = Decimal((10**delta) * (2**192)) / price
        else:
            radicand = Decimal(2**192) / (price * Decimal(10 ** (-delta)))
        s = radicand.sqrt()
    return int(s.to_integral_value(rounding="ROUND_HALF_UP"))


def q128_to_decimal(value_q128: int) -> Decimal:
    """``value_q128 / 2**128`` as an exact Decimal."""
    if type(value_q128) is not int:
        raise TypeError("value_q128 must be an int, not a bool or subclass")
    with localcontext() as ctx:
        ctx.prec = max(60, len(str(abs(value_q128))) + 128 + 16)
        return Decimal(value_q128) / Decimal(Q128)


def tick_to_price(tick: int, dec0: int, dec1: int) -> Decimal:
    """Human price of token1 in token0 from a tick.

    ``10**(dec1-dec0) / 1.0001**tick`` — DECREASING in tick for pinned pools.
    """
    with localcontext() as ctx:
        ctx.prec = _PRICE_PREC
        delta = dec1 - dec0
        if delta >= 0:
            return Decimal(10**delta) / (TICK_BASE**tick)
        return Decimal(1) / (Decimal(10 ** (-delta)) * (TICK_BASE**tick))


def price_to_tick(price: Decimal, dec0: int, dec1: int) -> int:
    """Inverse of ``tick_to_price``, floor semantics."""
    return sqrt_price_x96_to_tick(price_to_sqrt_price_x96(price, dec0, dec1))


# ---------------------------------------------------------------------------
# Tick <-> sqrt price — exact integer ports of TickMath
# ---------------------------------------------------------------------------
def tick_to_sqrt_price_x96(tick: int) -> int:
    """Exact integer port of TickMath.getSqrtRatioAtTick."""
    if tick < MIN_TICK or tick > MAX_TICK:
        raise ValueError(f"tick {tick} outside [{MIN_TICK}, {MAX_TICK}]")
    abs_tick = -tick if tick < 0 else tick

    ratio = _RATIO_INIT_TRUE if (abs_tick & 0x1) != 0 else _RATIO_INIT_FALSE
    for bit, mul in enumerate(_RATIO_MULS, start=1):
        if (abs_tick & (1 << bit)) != 0:
            ratio = (ratio * mul) >> 128

    if tick > 0:
        ratio = MAX_UINT256 // ratio

    return (ratio >> 32) + (1 if (ratio & ((1 << 32) - 1)) else 0)


def sqrt_price_x96_to_tick(sqrt_price_x96: int) -> int:
    """Exact integer port of TickMath.getTickAtSqrtRatio — floor semantics."""
    if sqrt_price_x96 < MIN_SQRT_RATIO or sqrt_price_x96 > MAX_SQRT_RATIO:
        raise ValueError(
            f"sqrt_price_x96 {sqrt_price_x96} outside "
            f"[{MIN_SQRT_RATIO}, {MAX_SQRT_RATIO})"
        )
    ratio = sqrt_price_x96 << 32

    msb = ratio.bit_length() - 1
    r = ratio >> (msb - 127) if msb >= 128 else ratio << (127 - msb)
    log2 = (msb - 128) << 64

    for shift in range(63, 49, -1):
        r = (r * r) >> 127
        f = r >> 128
        log2 |= f << shift
        r >>= f

    log_sqrt10001 = log2 * 255738959999603826347141
    tick_low = (log_sqrt10001 - 3402992956809132418596140100660247210) >> 128
    tick_high = (log_sqrt10001 + 291339464771989622907027621153398088495) >> 128

    if tick_low == tick_high:
        return tick_low
    return tick_high if tick_to_sqrt_price_x96(tick_high) <= sqrt_price_x96 else tick_low


# ---------------------------------------------------------------------------
# Tick grid alignment
# ---------------------------------------------------------------------------
def align_tick_down(tick: int, spacing: int) -> int:
    """Floor ``tick`` to the spacing grid (correct for negatives)."""
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    return (tick // spacing) * spacing


def align_tick_up(tick: int, spacing: int) -> int:
    """Ceil ``tick`` to the spacing grid (correct for negatives)."""
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    return -((-tick) // spacing) * spacing


# ---------------------------------------------------------------------------
# Wrapping arithmetic (Solidity unchecked blocks)
# ---------------------------------------------------------------------------
def wrapping_sub_256(a: int, b: int) -> int:
    """``(a - b) mod 2**256``."""
    return (a - b) % (MAX_UINT256 + 1)


def wrapping_add_256(a: int, b: int) -> int:
    """``(a + b) mod 2**256``."""
    return (a + b) % (MAX_UINT256 + 1)


# ---------------------------------------------------------------------------
# Liquidity <-> amounts (LiquidityAmounts / SqrtPriceMath)
# ---------------------------------------------------------------------------
def get_amount0_delta(
    sqrt_price_a: int, sqrt_price_b: int, liquidity: int, round_up: bool
) -> int:
    """Amount0 delta between two prices; port of SqrtPriceMath.getAmount0Delta."""
    sa, sb = sorted((sqrt_price_a, sqrt_price_b))
    if sa <= 0:
        raise ValueError("sqrt price must be positive")
    numerator1 = liquidity << 96
    numerator2 = sb - sa
    if round_up:
        return _div_ceil(_mul_div_ceil(numerator1, numerator2, sb), sa)
    return _mul_div_floor(numerator1, numerator2, sb) // sa


def get_amount1_delta(
    sqrt_price_a: int, sqrt_price_b: int, liquidity: int, round_up: bool
) -> int:
    """Amount1 delta between two prices; port of SqrtPriceMath.getAmount1Delta."""
    sa, sb = sorted((sqrt_price_a, sqrt_price_b))
    if round_up:
        return _mul_div_ceil(liquidity, sb - sa, Q96)
    return _mul_div_floor(liquidity, sb - sa, Q96)


def liquidity_for_amounts(
    sqrt_price: int,
    sqrt_price_a: int,
    sqrt_price_b: int,
    amount0: int,
    amount1: int,
) -> int:
    """Maximum liquidity for given amounts and range."""
    sa, sb = sorted((sqrt_price_a, sqrt_price_b))
    if sqrt_price <= sa:
        return _liquidity_for_amount0(sa, sb, amount0)
    if sqrt_price < sb:
        l0 = _liquidity_for_amount0(sqrt_price, sb, amount0)
        l1 = _liquidity_for_amount1(sa, sqrt_price, amount1)
        return l0 if l0 < l1 else l1
    return _liquidity_for_amount1(sa, sb, amount1)


def amounts_for_liquidity(
    sqrt_price: int,
    sqrt_price_a: int,
    sqrt_price_b: int,
    liquidity: int,
) -> tuple[int, int]:
    """``(amount0, amount1)`` for a liquidity within a range."""
    sa, sb = sorted((sqrt_price_a, sqrt_price_b))
    if sqrt_price <= sa:
        return (_amount0_for_liquidity(sa, sb, liquidity), 0)
    if sqrt_price < sb:
        a0 = _amount0_for_liquidity(sqrt_price, sb, liquidity)
        a1 = _amount1_for_liquidity(sa, sqrt_price, liquidity)
        return (a0, a1)
    return (0, _amount1_for_liquidity(sa, sb, liquidity))


def _liquidity_for_amount0(lower_sqrt: int, upper_sqrt: int, amount0: int) -> int:
    intermediate = _mul_div_floor(lower_sqrt, upper_sqrt, Q96)
    return _mul_div_floor(amount0, intermediate, upper_sqrt - lower_sqrt)


def _liquidity_for_amount1(lower_sqrt: int, upper_sqrt: int, amount1: int) -> int:
    return _mul_div_floor(amount1, Q96, upper_sqrt - lower_sqrt)


def _amount0_for_liquidity(lower_sqrt: int, upper_sqrt: int, liquidity: int) -> int:
    return (
        _mul_div_floor(liquidity << 96, upper_sqrt - lower_sqrt, upper_sqrt) // lower_sqrt
    )


def _amount1_for_liquidity(lower_sqrt: int, upper_sqrt: int, liquidity: int) -> int:
    return _mul_div_floor(liquidity, upper_sqrt - lower_sqrt, Q96)