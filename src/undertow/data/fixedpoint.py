"""Fixed-point, tick and liquidity math for Uniswap V3 on-chain data.

Every number this module touches is a Solidity fixed-point integer from a
contract event or ``eth_call``: ``sqrtPriceX96`` in Q64.96, ``feeGrowth*X128``
in Q128, liquidity in raw base units, ticks on a ``1.0001**tick`` lattice. The
single non-negotiable rule here is **exactness**: a ``float`` anywhere on the
tick/price/accumulator path is a bug (``2**128 ~= 3.4e38`` carries ~70 bits
more than a float64 mantissa). All integer conversions use arbitrary-precision
``int``; all price conversion uses ``decimal.Decimal`` with a locally-set
context (never the process-global default).

The Solidity functions are ported faithfully, bit-for-bit. Each port names its
source (contract + function) in its docstring. ``logging`` is the only I/O.
"""

from __future__ import annotations

import logging
from decimal import Decimal, localcontext

LOGGER = logging.getLogger("undertow.data.fixedpoint")

# ---------------------------------------------------------------------------
# Pinned constants (see CONTRACTS.md §2; these mirrors live here because T02
# depends only on T00, not on T01's config.py — intentional loose coupling).
# ---------------------------------------------------------------------------
Q96 = 2**96                       # sol FixedPoint96.Q96
Q128 = 2**128                     # price / accumulator raw scale
MAX_UINT256 = 2**256 - 1          # type(uint256).max
MIN_TICK = -887272                # sol TickMath.MIN_TICK
MAX_TICK = 887272                 # sol TickMath.MAX_TICK
# The boundary sqrt ratios appear verbatim in TickMath.sol (CONTRACTS §3.0, T02
# fixture route 2): MAX_SQRT_RATIO == getSqrtRatioAtTick(MAX_TICK) and
# MIN_SQRT_RATIO == getSqrtRatioAtTick(MIN_TICK). The lower bound is the
# well-known published 10-digit constant, independently pinned: the fixture
# asserts tick_to_sqrt_price_x96(MIN_TICK) == MIN_SQRT_RATIO exactly.
MIN_SQRT_RATIO = 4295128739      # sol TickMath.MIN_SQRT_RATIO == getSqrtRatioAtTick(MIN_TICK)
MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342  # sol TickMath.MAX_SQRT_RATIO

# ---------------------------------------------------------------------------
# Internal arbitrary-precision division helpers.
# ---------------------------------------------------------------------------
def _mul_div_floor(a: int, b: int, denominator: int) -> int:
    """``floor(a*b/denominator)``, full precision.

    Port of ``FullMath.mulDiv`` (Uniswap v3-core). Python integers do not
    overflow, so the 512-bit decomposition the contract needs is unnecessary.
    """
    if denominator <= 0:
        raise ValueError("mulDiv denominator must be positive")
    return (a * b) // denominator


def _mul_div_ceil(a: int, b: int, denominator: int) -> int:
    """``ceil(a*b/denominator)``. Port of ``FullMath.mulDivRoundingUp``."""
    if denominator <= 0:
        raise ValueError("mulDivRoundingUp denominator must be positive")
    return (a * b + denominator - 1) // denominator


def _div_ceil(a: int, d: int) -> int:
    """``ceil(a/d)`` for ``d > 0``. Port of ``UnsafeMath.divRoundingUp``."""
    return (a + d - 1) // d


# ---------------------------------------------------------------------------
# Price conversion — decimal, exact, never float.
# ---------------------------------------------------------------------------
_PRICE_PREC = 80  # local Decimal context precision used by price conversions


def sqrt_price_x96_to_price(sqrt_price_x96: int, dec0: int, dec1: int) -> Decimal:
    """Human price of token1 denominated in token0 — USDC per WETH for the pinned pools (~3000).

    ``price = 10**(dec1 - dec0) * 2**192 / sqrt_price_x96**2``, as a Decimal
    built from an exact integer numerator/denominator — no float anywhere. See
    CONTRACTS.md §3.0. Raises ``ValueError`` when ``sqrt_price_x96 == 0``.
    """
    if sqrt_price_x96 <= 0:
        raise ValueError("sqrt_price_x96 must be positive")
    # Build price = 10**(dec1-dec0) * 2**192 / sqrt^2 entirely with integers. A
    # negative decimal exponent must go in the DENOMINATOR — ``10 ** -k`` in
    # Python is a float, and this module must never produce one.
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
    """Inverse of ``sqrt_price_x96_to_price``; round-trips to within 1 unit.

    ``s = sqrt(10**(dec1-dec0) * 2**192 / price)`` as a Decimal, returned as the
    nearest integer. Raises ``ValueError`` for non-positive ``price``.
    """
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
    """``value_q128 / 2**128`` as an exact Decimal (never float)."""
    if type(value_q128) is not int:
        raise TypeError("value_q128 must be an int, not a bool or subclass")
    with localcontext() as ctx:
        # Enough significant digits for the terminating binary fraction plus the
        # integer part, so the result is exact for any uint256 input.
        ctx.prec = max(60, len(str(abs(value_q128))) + 128 + 16)
        return Decimal(value_q128) / Decimal(Q128)


def tick_to_price(tick: int, dec0: int, dec1: int) -> Decimal:
    """Human price of token1 in token0; same orientation as ``sqrt_price_x96_to_price``.

    Raw tick price ``1.0001**tick`` is raw-token1-per-raw-token0, so the human
    price ``10**(dec1-dec0) / 1.0001**tick`` is **decreasing** in tick for the
    pinned pools (a higher tick means a higher raw token1/token0 ratio, i.e. a
    LOWER USDC-per-WETH price). Asserted in ``test_tick_to_price_direction``.
    """
    with localcontext() as ctx:
        ctx.prec = _PRICE_PREC
        delta = dec1 - dec0
        if delta >= 0:
            return (Decimal(10**delta)) / (Decimal("1.0001") ** tick)
        return Decimal(1) / ((Decimal(10 ** (-delta))) * (Decimal("1.0001") ** tick))


def price_to_tick(price: Decimal, dec0: int, dec1: int) -> int:
    """Inverse of ``tick_to_price``, floor semantics (PLAN.md §7), NOT round.

    The human price is threaded through ``price_to_sqrt_price_x96`` and
    ``sqrt_price_x96_to_tick`` so the floor direction is exactly the protocol's.
    """
    return sqrt_price_x96_to_tick(price_to_sqrt_price_x96(price, dec0, dec1))


# ---------------------------------------------------------------------------
# Tick <-> sqrt price — exact integer ports of TickMath.
# ---------------------------------------------------------------------------
# getSqrtRatioAtTick's magic decomposition constants, extracted from Uniswap
# v3-core contracts/libraries/TickMath.sol and verified bit-identical to the
# same tuple in @uniswap/v3-sdk utils/tickMath.ts (see T02 PR body).
_RATIO_INIT_TRUE = 0xfffcb933bd6fad37aa2d162d1a594001
_RATIO_INIT_FALSE = 0x100000000000000000000000000000000
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


def tick_to_sqrt_price_x96(tick: int) -> int:
    """Exact integer port of TickMath.getSqrtRatioAtTick (Uniswap v3-core).

    Computes ``sqrt(1.0001**tick) * 2**96`` bit-for-bit, using the contract's
    magic-constant decomposition: multiply-and-shift on Q128.128, the
    ``type(uint256).max / ratio`` reciprocal for positive ticks, and the
    round-up downcast from Q128.128 to Q64.96. Raises ``ValueError`` when
    ``|tick| > MAX_TICK`` (the Solidity ``require(absTick <= MAX_TICK, 'T')``).
    """
    if tick < MIN_TICK or tick > MAX_TICK:
        raise ValueError(f"tick {tick} outside [{MIN_TICK}, {MAX_TICK}]")
    abs_tick = -tick if tick < 0 else tick

    ratio = _RATIO_INIT_TRUE if (abs_tick & 0x1) != 0 else _RATIO_INIT_FALSE
    for bit, mul in enumerate(_RATIO_MULS, start=1):  # bits 0x2 .. 0x80000
        if (abs_tick & (1 << bit)) != 0:
            ratio = (ratio * mul) >> 128

    if tick > 0:
        ratio = MAX_UINT256 // ratio

    # Q128.128 -> Q64.96 rounding up, so that getTickAtSqrtRatio of the result
    # is always the input tick; identical to the Solidity ceil-downcast tail.
    return (ratio >> 32) + (1 if (ratio & ((1 << 32) - 1)) else 0)


def sqrt_price_x96_to_tick(sqrt_price_x96: int) -> int:
    """Exact integer port of TickMath.getTickAtSqrtRatio — **floor** semantics.

    Returns the greatest tick with ``tick_to_sqrt_price_x96(t) <= sqrt_price_x96``
    (PLAN.md §7 pins floor, not round). Bit-length + log2 approximation: the
    assembly bit scan is ``bit_length() - 1``, the fraction refinement runs the
    identical 14 squared-ratio iterations (shifts 63..50), and the final
    ``tickLow``/``tickHigh`` disambiguation calls back into the getSqrtRatio port.

    Guarded to the protocol domain ``MIN_SQRT_RATIO <= x < MAX_SQRT_RATIO``
    (matches ``require(sqrtPriceX96 >= MIN_SQRT_RATIO && sqrtPriceX96 <
    MAX_SQRT_RATIO, 'R')``); raises ``ValueError`` outside it.
    """
    if sqrt_price_x96 < MIN_SQRT_RATIO or sqrt_price_x96 >= MAX_SQRT_RATIO:
        raise ValueError(
            f"sqrt_price_x96 {sqrt_price_x96} outside [{MIN_SQRT_RATIO}, {MAX_SQRT_RATIO})"
        )
    ratio = sqrt_price_x96 << 32

    # The assembly unrolled bit scan: msb = bit_length()-1 is bit-identical.
    msb = ratio.bit_length() - 1

    r = ratio >> (msb - 127) if msb >= 128 else ratio << (127 - msb)
    log2 = (msb - 128) << 64

    for shift in range(63, 49, -1):  # shifts 63..50, 14 iterations
        r = (r * r) >> 127
        f = r >> 128
        log2 |= f << shift
        r >>= f

    log_sqrt10001 = log2 * 255738959999603826347141  # 128.128 number

    tick_low = (log_sqrt10001 - 3402992956809132418596140100660247210) >> 128
    tick_high = (log_sqrt10001 + 291339464771989622907027621153398088495) >> 128

    if tick_low == tick_high:
        return tick_low
    return tick_high if tick_to_sqrt_price_x96(tick_high) <= sqrt_price_x96 else tick_low


# ---------------------------------------------------------------------------
# Grid alignment.
# ---------------------------------------------------------------------------
def align_tick_down(tick: int, spacing: int) -> int:
    """Floor ``tick`` to the spacing grid (e.g. 60 for the 0.30% pool).

    Python ``//`` floors toward -inf, exactly what the protocol wants for
    ``tickLower``; C-style truncation is wrong for negative ticks. Correct for
    ``tick=-61, spacing=60 -> -120``; idempotent on aligned ticks.
    """
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    return (tick // spacing) * spacing


def align_tick_up(tick: int, spacing: int) -> int:
    """Ceil ``tick`` to the spacing grid, correct for negative ticks.

    ``tick=-61, spacing=60 -> -60``; idempotent on aligned ticks.
    """
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    return -((-tick) // spacing) * spacing


# ---------------------------------------------------------------------------
# Wrapping arithmetic (Solidity unchecked blocks).
# ---------------------------------------------------------------------------
def wrapping_sub_256(a: int, b: int) -> int:
    """``(a - b) mod 2**256`` — Uniswap V3 computes fee-growth deltas in
    ``unchecked`` blocks, so accumulators legitimately wrap. Required for all
    feeGrowth deltas; never assert ``inside_now >= inside_last``."""
    return (a - b) % (MAX_UINT256 + 1)


def wrapping_add_256(a: int, b: int) -> int:
    """``(a + b) mod 2**256`` — unchecked addition, the wrap companion."""
    return (a + b) % (MAX_UINT256 + 1)


# ---------------------------------------------------------------------------
# Liquidity <-> amounts (LiquidityAmounts / SqrtPriceMath), integer floor/ceil.
# ---------------------------------------------------------------------------
def get_amount0_delta(sqrt_price_a: int, sqrt_price_b: int, liquidity: int, round_up: bool) -> int:
    """Amount0 delta between two prices; port of ``SqrtPriceMath.getAmount0Delta``.

    Follows the contract rounding direction exactly: ``round_up`` computes
    ``divRoundingUp(mulDivRoundingUp(numerator1, numerator2, sqrtRatioBX),
    sqrtRatioAX)``; otherwise the double floor.
    """
    sa, sb = sorted((sqrt_price_a, sqrt_price_b))
    if sa <= 0:
        raise ValueError("sqrt price must be positive")
    numerator1 = liquidity << 96
    numerator2 = sb - sa
    if round_up:
        return _div_ceil(_mul_div_ceil(numerator1, numerator2, sb), sa)
    return _mul_div_floor(numerator1, numerator2, sb) // sa


def get_amount1_delta(sqrt_price_a: int, sqrt_price_b: int, liquidity: int, round_up: bool) -> int:
    """Amount1 delta between two prices; port of ``SqrtPriceMath.getAmount1Delta``."""
    sa, sb = sorted((sqrt_price_a, sqrt_price_b))
    if round_up:
        return _mul_div_ceil(liquidity, sb - sa, Q96)
    return _mul_div_floor(liquidity, sb - sa, Q96)


def liquidity_for_amounts(
    sqrt_price: int, sqrt_price_a: int, sqrt_price_b: int, amount0: int, amount1: int
) -> int:
    """Maximum liquidity representable for given amounts and range.

    Port of ``LiquidityAmounts.getLiquidityForAmounts`` — three branches
    (price at/below the lower bound, inside the range, at/above the upper). The
    inside branch returns min(liquidity0, liquidity1); all sub-computations
    floor (the contract's ``mulDiv``), so the returned L is protocol-exact.
    """
    sa, sb = sorted((sqrt_price_a, sqrt_price_b))
    if sqrt_price <= sa:
        return _liquidity_for_amount0(sa, sb, amount0)
    if sqrt_price < sb:
        l0 = _liquidity_for_amount0(sqrt_price, sb, amount0)
        l1 = _liquidity_for_amount1(sa, sqrt_price, amount1)
        return l0 if l0 < l1 else l1
    return _liquidity_for_amount1(sa, sb, amount1)


def amounts_for_liquidity(
    sqrt_price: int, sqrt_price_a: int, sqrt_price_b: int, liquidity: int
) -> tuple[int, int]:
    """``(amount0, amount1)`` for a liquidity within a range.

    Port of ``LiquidityAmounts.getAmountsForLiquidity``: price below the lower
    bound uses only amount0; above the upper bound only amount1; inside uses
    both. All sub-computations floor (the contract's ``mulDiv``).
    """
    sa, sb = sorted((sqrt_price_a, sqrt_price_b))
    if sqrt_price <= sa:
        return (_amount0_for_liquidity(sa, sb, liquidity), 0)
    if sqrt_price < sb:
        a0 = _amount0_for_liquidity(sqrt_price, sb, liquidity)
        a1 = _amount1_for_liquidity(sa, sqrt_price, liquidity)
        return (a0, a1)
    return (0, _amount1_for_liquidity(sa, sb, liquidity))


def _liquidity_for_amount0(lower_sqrt: int, upper_sqrt: int, amount0: int) -> int:
    """Port of ``LiquidityAmounts.getLiquidityForAmount0`` (a <= b)."""
    intermediate = _mul_div_floor(lower_sqrt, upper_sqrt, Q96)
    return _mul_div_floor(amount0, intermediate, upper_sqrt - lower_sqrt)


def _liquidity_for_amount1(lower_sqrt: int, upper_sqrt: int, amount1: int) -> int:
    """Port of ``LiquidityAmounts.getLiquidityForAmount1`` (a <= b)."""
    return _mul_div_floor(amount1, Q96, upper_sqrt - lower_sqrt)


def _amount0_for_liquidity(lower_sqrt: int, upper_sqrt: int, liquidity: int) -> int:
    """Port of ``LiquidityAmounts.getAmount0ForLiquidity`` (a <= b)."""
    return _mul_div_floor(liquidity << 96, upper_sqrt - lower_sqrt, upper_sqrt) // lower_sqrt


def _amount1_for_liquidity(lower_sqrt: int, upper_sqrt: int, liquidity: int) -> int:
    """Port of ``LiquidityAmounts.getAmount1ForLiquidity`` (a <= b)."""
    return _mul_div_floor(liquidity, upper_sqrt - lower_sqrt, Q96)