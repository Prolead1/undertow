-- 04_active_positions.sql
-- ============================================================================
-- WHAT IT ANSWERS
--   Monthly count of ACTIVE positions: distinct (owner, tickLower, tickUpper)
--   whose net position liquidity at month end — Σ mint(liquidityAmount) −
--   Σ burn(liquidityAmount), as-of month end — is still positive.
--   This is the "how many positions are live" number the roadmap keeps asking
--   about (§10.1.6), straight from the on-chain log.
-- STREAM (CONTRACTS.md §4)
--   mint + burn — the position lifecycle. Collect is intentionally ignored: it
--   withdraws fees, it never changes position liquidity.
-- EXPECTED ORDER OF MAGNITUDE (primary pool)
--   hundreds of concurrent positions (peak ~1e3; a few hundred minted per month
--   and most churn within weeks). If this ever reads 1e5+, suspect a decode bug
--   (e.g. liquidityAmount units) before anything else.
-- CAVEAT — POSITIONS, NOT USERS (same record as PLAN.md §7)
--   owner is overwhelmingly the NFT position manager 0xC364…FE88, so this count
--   is positions, not humans. Real per-user attribution needs a separate
--   NFT-event join and is out of scope for this pipeline.
-- CAVEAT — WINDOW FLOOR IN EARLY 2022
--   A position minted before 2022-01-01 (the pool launched Dec 2021) is not in
--   `liq`, so it never appears in the cumulative sums — the first months' counts
--   are a FLOOR. Same blindness T13's check_liquidity_conservation has for
--   pre-window positions; the pool was a few weeks old at window start, so the
--   effect is small and bounded.
-- METHOD (why cumulative sums are safe here)
--   mint/burn liquidityAmount are non-negative (CONTRACTS §4.2, unsigned), and
--   on-chain a burn can never take a position below zero, so "net > 0 at month
--   end" ⟺ "some liquidity still live". Sum is exact; no assumption about
--   partial-burn semantics is needed.
-- RE-RUN ON THE OTHER POOL: edit the single 0x… literal in `params` below.
-- ============================================================================

WITH params AS (
    SELECT 0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8 AS pool  -- USDC/WETH, 0.30%
),
mints AS (
    SELECT owner, tickLower, tickUpper, evt_block_time, liquidityAmount AS d
    FROM uniswap_v3_ethereum.Pair_evt_Mint
    WHERE contract_address = (SELECT pool FROM params)
      AND evt_block_time >= TIMESTAMP '2022-01-01 00:00:00'
      AND evt_block_time <  TIMESTAMP '2025-01-01 00:00:00'
),
burns AS (
    SELECT owner, tickLower, tickUpper, evt_block_time, -1 * liquidityAmount AS d
    FROM uniswap_v3_ethereum.Pair_evt_Burn
    WHERE contract_address = (SELECT pool FROM params)
      AND evt_block_time >= TIMESTAMP '2022-01-01 00:00:00'
      AND evt_block_time <  TIMESTAMP '2025-01-01 00:00:00'
),
liq AS (           -- signed liquidity deltas, one row per lifecycle event
    SELECT * FROM mints
    UNION ALL
    SELECT * FROM burns
),
keys AS (
    SELECT DISTINCT owner, tickLower, tickUpper FROM liq
),
months AS (
    SELECT DISTINCT date_trunc('month', evt_block_time) AS month FROM liq
),
monthly_delta AS (
    SELECT
        date_trunc('month', evt_block_time) AS month,
        owner,
        tickLower,
        tickUpper,
        SUM(d) AS d_month
    FROM liq
    GROUP BY 1, 2, 3, 4
),
-- every (key x month) pair, zero-filled, so a position with no activity for
-- several months still keeps its accumulated net through those months.
filled AS (
    SELECT
        m.month,
        k.owner,
        k.tickLower,
        k.tickUpper,
        COALESCE(md.d_month, 0) AS d_month
    FROM months AS m
    CROSS JOIN keys AS k
    LEFT JOIN monthly_delta AS md
        ON  md.month     = m.month
        AND md.owner     = k.owner
        AND md.tickLower = k.tickLower
        AND md.tickUpper = k.tickUpper
),
cum AS (
    SELECT
        month,
        owner,
        tickLower,
        tickUpper,
        SUM(d_month) OVER (
            PARTITION BY owner, tickLower, tickUpper
            ORDER BY month
        ) AS net_liquidity
    FROM filled
)
SELECT
    month,
    COUNT(*) AS active_positions       -- (owner, tickLower, tickUpper) is unique here
FROM cum
WHERE net_liquidity > 0
GROUP BY month
ORDER BY month