-- 03_swap_counts_by_month.sql
-- ============================================================================
-- WHAT IT ANSWERS
--   Monthly event counts per stream across the pinned window — T13's row-count
--   cross-reference. Its `expected_counts(month, stream)` lookup compares our
--   pipeline's monthly counts against these, so any drift (double-fetch, dropped
--   page, off-by-one block bucket) shows up as a >tolerance discrepancy.
-- STREAM (CONTRACTS.md §4)
--   swap / mint / burn / collect (+ flash, low volume but fetched by T06).
-- EXPECTED ORDER OF MAGNITUDE (primary pool per month, 2022-01..2024-12)
--   swaps    ~3e4-1.5e5 per month (≈ 1-5k swaps/day; 1e6-5e6 over the whole window)
--   mints    ~1e2-1e3 per month
--   burns    ~1e2-1e3 per month
--   collects ~1e2-1e3 per month
--   flash    too low to move the above (a handful overall; schema exists regardless,
--            CONTRACTS §4.3.1)
-- WHY tolerance, not exactness, on these counts (this is why T13 uses a
-- tolerance_pct): Dune's decoded tables bucket by event TIMESTAMP, while our
-- fetchers bucket by (block_number, log_index). Around a month boundary a few
-- events — those mined in the last blocks of one month but timestamped in the
-- next — sort into different months. 1-2% is a defendable tolerance.
-- RE-RUN ON THE OTHER POOL: edit the single 0x… literal in `params` below.
-- ============================================================================

WITH params AS (
    SELECT 0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8 AS pool  -- USDC/WETH, 0.30%
),
swaps AS (
    SELECT date_trunc('month', evt_block_time) AS month, COUNT(*) AS n_events
    FROM uniswap_v3_ethereum.Pair_evt_Swap
    WHERE contract_address = (SELECT pool FROM params)
      AND evt_block_time >= TIMESTAMP '2022-01-01 00:00:00'
      AND evt_block_time <  TIMESTAMP '2025-01-01 00:00:00'
    GROUP BY 1
),
mints AS (
    SELECT date_trunc('month', evt_block_time) AS month, COUNT(*) AS n_events
    FROM uniswap_v3_ethereum.Pair_evt_Mint
    WHERE contract_address = (SELECT pool FROM params)
      AND evt_block_time >= TIMESTAMP '2022-01-01 00:00:00'
      AND evt_block_time <  TIMESTAMP '2025-01-01 00:00:00'
    GROUP BY 1
),
burns AS (
    SELECT date_trunc('month', evt_block_time) AS month, COUNT(*) AS n_events
    FROM uniswap_v3_ethereum.Pair_evt_Burn
    WHERE contract_address = (SELECT pool FROM params)
      AND evt_block_time >= TIMESTAMP '2022-01-01 00:00:00'
      AND evt_block_time <  TIMESTAMP '2025-01-01 00:00:00'
    GROUP BY 1
),
collects AS (
    SELECT date_trunc('month', evt_block_time) AS month, COUNT(*) AS n_events
    FROM uniswap_v3_ethereum.Pair_evt_Collect
    WHERE contract_address = (SELECT pool FROM params)
      AND evt_block_time >= TIMESTAMP '2022-01-01 00:00:00'
      AND evt_block_time <  TIMESTAMP '2025-01-01 00:00:00'
    GROUP BY 1
),
flashes AS (
    SELECT date_trunc('month', evt_block_time) AS month, COUNT(*) AS n_events
    FROM uniswap_v3_ethereum.Pair_evt_Flash
    WHERE contract_address = (SELECT pool FROM params)
      AND evt_block_time >= TIMESTAMP '2022-01-01 00:00:00'
      AND evt_block_time <  TIMESTAMP '2025-01-01 00:00:00'
    GROUP BY 1
)
SELECT 'swap'    AS event_type, month, n_events FROM swaps
UNION ALL SELECT 'mint',    month, n_events FROM mints
UNION ALL SELECT 'burn',    month, n_events FROM burns
UNION ALL SELECT 'collect', month, n_events FROM collects
UNION ALL SELECT 'flash',   month, n_events FROM flashes
ORDER BY month, event_type