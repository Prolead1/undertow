-- 01_pool_overview.sql
-- ============================================================================
-- WHAT IT ANSWERS
--   Is our config right? This pulls the pool's identity straight off the chain:
--     * token0 / token1 addresses AND decimals  (configs/eth_usdc_3000.toml pins
--       USDC 6dp / WETH 18dp — if this query shows anything else, fix the config
--       BEFORE touching any data, because the token ordering is contractual and
--       inverting it silently inverts every price in the thesis, CONTRACTS §3.0)
--     * fee tier (parts-per-million of the input amount) and tick spacing
--     * creation block + date, and lifetime swap/mint/burn/collect counts — the
--       first magnitude handshake for T13's row counts.
-- STREAM (CONTRACTS.md §4)
--   swap / mint / burn / collect — lifetime event counts. Everything else in this
--   query sanity-checks the *config*, not a stream.
-- EXPECTED ORDER OF MAGNITUDE (primary pool, lifetime through 2024-12-31)
--   swaps     ~1e6-5e6 (≈ 1-5k swaps/day lifetime average; busiest USDC/WETH v3 pool)
--   mints     ~1e4     (a few hundred positions minted per month in peak regimes)
--   burns     ~1e4     (positions churn in days–weeks)
--   collects  ~1e4     (fee withdrawals; typically Burn+Collect in one tx)
-- NOTE ON TOKEN ORDERING
--   Uniswap numbers tokens by address, and 0xa0b8…eb48 (USDC) < 0xc02a…6cc2 (WETH),
--   so token0 MUST be USDC and token1 MUST be WETH. This query exists to show that
--   both are true on-chain before anyone trusts a decoded amount.
-- RE-RUN ON THE OTHER POOL
--   Edit the single 0x… literal in `params` below (the pinned secondary pool is
--   0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640 = USDC/WETH 0.05%, spacing 10).
--   Fee tier, tick spacing and decimals are all DERIVED from the chain in
--   `pool_meta`, so nothing else changes.
-- ============================================================================

WITH params AS (
    SELECT 0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8 AS pool  -- USDC/WETH, 0.30%
),
pool_meta AS (
    SELECT
        c.pool,
        c.token0,
        c.token1,
        c.fee               AS fee_ppm,       -- parts per million of input (1e6 => 100%)
        c.tick_spacing      AS tick_spacing,
        c.evt_block_time    AS creation_time,
        c.evt_block_number  AS creation_block,
        t0.symbol           AS token0_symbol,
        t0.decimals         AS token0_decimals,
        t1.symbol           AS token1_symbol,
        t1.decimals         AS token1_decimals
    FROM uniswap_v3_ethereum.PoolFactory_evt_PoolCreated AS c
    LEFT JOIN tokens.erc20 AS t0
        ON t0.blockchain = 'ethereum' AND t0.contract_address = c.token0
    LEFT JOIN tokens.erc20 AS t1
        ON t1.blockchain = 'ethereum' AND t1.contract_address = c.token1
    WHERE c.pool = (SELECT pool FROM params)
)
SELECT
    m.pool,
    m.token0,
    m.token1,
    m.token0_symbol,
    m.token1_symbol,
    m.token0_decimals,
    m.token1_decimals,
    m.fee_ppm,
    m.fee_ppm / 1e6 AS fee_fraction,           -- 0.003 for the 0.30% tier
    m.tick_spacing,
    m.creation_block,
    m.creation_time,
    (
        SELECT COUNT(*) FROM uniswap_v3_ethereum.Pair_evt_Swap
        WHERE contract_address = m.pool
    )   AS lifetime_swaps,
    (
        SELECT COUNT(*) FROM uniswap_v3_ethereum.Pair_evt_Mint
        WHERE contract_address = m.pool
    )   AS lifetime_mints,
    (
        SELECT COUNT(*) FROM uniswap_v3_ethereum.Pair_evt_Burn
        WHERE contract_address = m.pool
    )   AS lifetime_burns,
    (
        SELECT COUNT(*) FROM uniswap_v3_ethereum.Pair_evt_Collect
        WHERE contract_address = m.pool
    )   AS lifetime_collects
FROM pool_meta AS m