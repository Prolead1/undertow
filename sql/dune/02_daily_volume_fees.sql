-- 02_daily_volume_fees.sql
-- ============================================================================
-- WHAT IT ANSWERS
--   Daily traded volume (USD + per token) AND daily fee income (per token + USD),
--   resting on the *input* side of each swap. This is the roadmap §10.1.2 query,
--   with its own documented bug FIXED (see the big comment below).
-- STREAM (CONTRACTS.md §4)
--   swap — aggregate volume + fee-flow sanity check.
-- EXPECTED ORDER OF MAGNITUDE (primary pool, 2022-01-01..2024-12-31)
--   ~1-5k swaps/day typical, ~$1-15M traded volume/day, fee income ~0.30% of the
--   input-side volume (a few hundred to ~$30k/day). Both swap directions are
--   common, so the naive WETH-only fee version (below) under-reports by ~2x.
--
-- ============================================================================
-- WHY THIS IS THE *CORRECTED* QUERY (roadmap §10.1.2 — read this comment)
-- ----------------------------------------------------------------------------
-- The naive version sums only abs(amount1). That quietly drops every swap whose
-- fee was paid in USDC: a swap's fee is charged on the INPUT token, and roughly
-- half of the pool's swaps go the other way (USDC in, WETH out). Summing
-- abs(amount1) alone therefore misses ~all USDC-side fees AND, worse, never even
-- knows which side is the input — it just adds both signed amounts' magnitudes.
--
-- The protocol convention (CONTRACTS §4.1): amount0 and amount1 always have
-- OPPOSITE signs for a real swap, and POSITIVE = flowed INTO the pool. So:
--     amount0 > 0  => token0 (USDC) is the input  => fee  = amount0 * fee_tier/1e6  [USDC]
--     amount1 > 0  => token1 (WETH) is the input  => fee  = amount1 * fee_tier/1e6  [WETH]
-- fee_tier/1e6 is 3000/1e6 = 0.003 for this pool, applied to the INPUT amount in
-- the INPUT token. We therefore split by sign(amount0) and never assume the input
-- token — that fixable 2x blind spot is a thesis-worthy detail.
--
-- SECOND CAVEAT, carried forward from §10.1.2: this is *volume/activity* truth,
-- NOT position-level fee truth. Real per-position fee accrual is only given by
-- the feeGrowth accumulators (CONTRACTS §4.4, T10) — Dune derives a per-swap fee
-- from the flow, which is NOT the same as what a position actually accrued. Use
-- this query to sanity-check magnitudes; never as a fee data source.
-- ============================================================================
-- ENGINE NOTES
--   * Per-row conversion to human units happens BEFORE any SUM: raw amounts are
--     uint256-scale (up to ~1e21 raw units) and exceed 2^53, so a double SUM of
--     raw units would silently swallow the low-order (whole-token) digits. Dune
--     decoded amounts are doubles already, so this also keeps us magnitudes-safe.
--   * address comparisons use 0x… literals: contract_address is a VARBINARY, not
--     a string — never quote it as '0x…'.
-- RE-RUN ON THE OTHER POOL
--   Edit the single 0x… literal in `params`; fee tier and decimals are derived
--   from the chain in `pool_meta` (both pinned pools are USDC 6dp / WETH 18dp).
-- ============================================================================

WITH params AS (
    SELECT 0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8 AS pool  -- USDC/WETH, 0.30%
),
pool_meta AS (
    SELECT
        c.fee           AS fee_ppm,       -- 3000 => fee = input_amount * 3000 / 1e6
        t0.decimals     AS dec0,          -- USDC: 6
        t1.decimals     AS dec1           -- WETH: 18
    FROM uniswap_v3_ethereum.PoolFactory_evt_PoolCreated AS c
    LEFT JOIN tokens.erc20 AS t0
        ON t0.blockchain = 'ethereum' AND t0.contract_address = c.token0
    LEFT JOIN tokens.erc20 AS t1
        ON t1.blockchain = 'ethereum' AND t1.contract_address = c.token1
    WHERE c.pool = (SELECT pool FROM params)
),
swaps AS (
    SELECT
        evt_block_time AS ts,
        amount0,
        amount1
    FROM uniswap_v3_ethereum.Pair_evt_Swap
    WHERE contract_address = (SELECT pool FROM params)
      AND evt_block_time >= TIMESTAMP '2022-01-01 00:00:00'
      AND evt_block_time <  TIMESTAMP '2025-01-01 00:00:00'
),
-- Split by swap direction (sign of amount0); the fee is charged on the INPUT
-- (positive) side, in the INPUT token — see the header comment for the why.
directional AS (
    SELECT
        date_trunc('day', s.ts) AS day,
        CASE WHEN s.amount0 > 0 THEN ABS(s.amount0) / POW(10, m.dec0) ELSE 0 END AS in_usdc,
        CASE WHEN s.amount1 > 0 THEN ABS(s.amount1) / POW(10, m.dec1) ELSE 0 END AS in_weth,
        ABS(s.amount0) / POW(10, m.dec0)                                        AS out_usdc,
        ABS(s.amount1) / POW(10, m.dec1)                                        AS out_weth,
        m.fee_ppm
    FROM swaps AS s
    CROSS JOIN pool_meta AS m
),
daily AS (
    SELECT
        day,
        COUNT(*)                            AS n_swaps,
        SUM(in_usdc)                        AS in_usdc_volume,   -- input-side USDC volume (human units)
        SUM(in_weth)                        AS in_weth_volume,   -- input-side WETH volume (human units)
        SUM(in_usdc + out_usdc)             AS volume_usdc,
        SUM(in_weth + out_weth)             AS volume_weth,
        SUM(in_usdc * fee_ppm / 1e6)        AS fee_usdc,   -- 0.30% of input USDC
        SUM(in_weth * fee_ppm / 1e6)        AS fee_weth    -- 0.30% of input WETH
    FROM directional
    GROUP BY day
),
-- Daily-average WETH/USD only; USDC is a $1-stable so fee_usdc needs no oracle.
weth_price AS (
    SELECT
        date_trunc('day', minute) AS day,
        AVG(price)                AS weth_usd
    FROM prices.usd
    WHERE blockchain = 'ethereum'
      AND symbol     = 'WETH'
      AND minute >= TIMESTAMP '2022-01-01 00:00:00'
      AND minute <  TIMESTAMP '2025-01-01 00:00:00'
    GROUP BY 1
)
SELECT
    d.day,
    d.n_swaps,
    d.in_usdc_volume,                                         -- direction split sanity: both >> 0
    d.in_weth_volume,
    d.volume_usdc,
    d.volume_weth,
    (d.volume_usdc + d.volume_weth * COALESCE(w.weth_usd, 0)) AS volume_usd,
    d.fee_usdc,                                               -- fee income, in the fee's token
    d.fee_weth,
    (d.fee_usdc + d.fee_weth * COALESCE(w.weth_usd, 0))       AS fee_usd
FROM daily AS d
LEFT JOIN weth_price AS w ON w.day = d.day
ORDER BY d.day