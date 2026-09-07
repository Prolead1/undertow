-- 06_gas_price_context.sql
-- ============================================================================
-- WHAT IT ANSWERS
--   Daily base-fee context over the pinned window: p50/p90/p99 + mean base fee
--   (gwei) per block, plus average gas used/limit. This is T07's independent
--   reference for the `gas` stream (CONTRACTS §4.5) and makes the §10.2.3 fee
--   spikes (May 2022, 2023-04 merge, …) visible in one glance. Every tx pays at
--   least baseFeePerGas, so the RL friction G_t lives on this curve.
-- STREAM (CONTRACTS.md §4)
--   gas — chain-wide, NOT pool-scoped (the `params` pool CTE below is therefore a
--   no-op kept for uniformity across this directory; both pinned pools share one
--   gas table).
-- EXPECTED ORDER OF MAGNITUDE (2022-01-01..2024-12-31)
--   base fee ~0.02-0.2 gwei in calm traffic, spiking >1 gwei (often several gwei,
--   p90 >> p50) in congestion events. If this query ever shows an all-flat
--   ~0 line, the join/source is wrong — base fees moved by orders of magnitude
--   across the window.
-- ============================================================================

WITH params AS (
    SELECT 0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8 AS pool  -- USDC/WETH 0.30%; no-op for gas
),
blocks AS (
    SELECT
        number,
        block_time,
        base_fee_per_gas,     -- wei, EIP-1559; DOUBLE in ethereum.blocks
        gas_used,
        gas_limit
    FROM ethereum.blocks
    WHERE block_time >= TIMESTAMP '2022-01-01 00:00:00'
      AND block_time <  TIMESTAMP '2025-01-01 00:00:00'
)
SELECT
    date_trunc('day', b.block_time) AS day,
    COUNT(*)                        AS n_blocks,
    approx_percentile(b.base_fee_per_gas / 1e9, 0.50) AS base_fee_p50_gwei,
    approx_percentile(b.base_fee_per_gas / 1e9, 0.90) AS base_fee_p90_gwei,
    approx_percentile(b.base_fee_per_gas / 1e9, 0.99) AS base_fee_p99_gwei,
    AVG(b.base_fee_per_gas / 1e9)                     AS base_fee_avg_gwei,
    MAX(b.base_fee_per_gas / 1e9)                     AS base_fee_max_gwei,
    AVG(b.gas_used)                                   AS gas_used_avg,
    AVG(b.gas_limit)                                  AS gas_limit_avg
FROM blocks AS b
CROSS JOIN params AS p
GROUP BY 1
ORDER BY day