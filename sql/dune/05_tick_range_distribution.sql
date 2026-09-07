-- 05_tick_range_distribution.sql
-- ============================================================================
-- WHAT IT ANSWERS
--   1. Histogram of mint range widths (tickUpper - tickLower) in ticks — how
--      wide are the concentrated-liquidity ranges people (via the NFT manager)
--      actually mint?
--   2. OFF-GRID share: the fraction of mints whose tickLower / tickUpper are NOT
--      on the pool's fee-tier spacing grid. EXPECTED: exactly 0.
-- STREAM (CONTRACTS.md §4)
--   mint — validates the grid assumption T13's check_ticks_on_spacing_grid uses.
-- EXPECTED ORDER OF MAGNITUDE (primary pool, spacing = 60 ticks)
--   widths of a few hundred to a few thousand ticks (a 1.0001**tick step is a
--   ~1bp price step, so a ±5-10% band is ~500-2000 ticks). OFF-GRID mints MUST
--   be 0: the position manager only mints on aligned ticks (CONTRACTS
--   TICK_SPACING: spacing 60 for the 0.30% tier, 10 for the 0.05% tier).
--   If the off-grid share is ever > 0, our grid assumption is WRONG and the
--   tick→price math in the whole pipeline needs a fresh look — flag it loudly,
--   do not paper over it.
-- ENGINE NOTE
--   Grid check uses `%` (modulo-by-remainder), not a floor/mod function: for an
--   aligned tick, remainder is 0 under BOTH sign conventions (Trino mod is
--   floor-based, DuckDB % keeps the dividend's sign), so `x % spacing = 0` is
--   sign-robust for negative ticks. tickLower/tickUpper are CAST to BIGINT so
--   the operator works whichever way Dune's decoder types the columns.
-- RE-RUN ON THE OTHER POOL: edit the single 0x… literal in `params` (tick
--   spacing is derived from the chain in `pool_meta`, so no other edit needed).
-- ============================================================================

WITH params AS (
    SELECT 0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8 AS pool  -- USDC/WETH, 0.30%
),
pool_meta AS (
    SELECT c.tick_spacing AS tick_spacing
    FROM uniswap_v3_ethereum.PoolFactory_evt_PoolCreated AS c
    WHERE c.pool = (SELECT pool FROM params)
),
mints AS (
    SELECT
        CAST(tickLower AS BIGINT) AS tick_lower,
        CAST(tickUpper AS BIGINT) AS tick_upper
    FROM uniswap_v3_ethereum.Pair_evt_Mint
    WHERE contract_address = (SELECT pool FROM params)
),
ranges AS (
    SELECT
        m.tick_upper - m.tick_lower                  AS width_ticks,
        CASE WHEN MOD(m.tick_lower, pm.tick_spacing) = 0
              AND MOD(m.tick_upper, pm.tick_spacing) = 0
             THEN 1 ELSE 0 END                        AS on_grid
    FROM mints AS m
    CROSS JOIN pool_meta AS pm
),
histogram AS (
    SELECT
        CASE
            WHEN width_ticks <  120    THEN '0-120    (<= 2 spacing steps)'
            WHEN width_ticks <  600    THEN '120-600'
            WHEN width_ticks <  1500   THEN '600-1500'
            WHEN width_ticks <  6000   THEN '1500-6000'
            ELSE                             '6000+'
        END                             AS width_bucket,
        COUNT(*)                        AS n_mints
    FROM ranges
    GROUP BY 1
)
-- One statement only (SQL CTEs are scoped to the statement that defines them):
-- the histogram rows UNION ALL a summary row carrying the off-grid ledger —
-- the number that must be zero.
SELECT
    width_bucket      AS bucket,
    n_mints           AS n_mints,
    CAST(NULL AS BIGINT)   AS off_grid_mints,
    CAST(NULL AS DOUBLE)   AS off_grid_share
FROM histogram
UNION ALL
SELECT
    'OFF-GRID (must be 0)' AS bucket,
    CAST(NULL AS BIGINT)   AS n_mints,
    COUNT(*)               AS off_grid_mints,
    COUNT(*) * 1.0 / NULLIF((SELECT COUNT(*) FROM ranges), 0) AS off_grid_share
FROM ranges
WHERE on_grid = 0
ORDER BY bucket;