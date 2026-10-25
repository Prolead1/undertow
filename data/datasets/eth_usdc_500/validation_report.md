# Validation report — undertow.data

Generated: 2026-10-25T03:40:17.812672+00:00
- **dataset id**: `d5dbfefc03b7463f3c4077f7371471f30c81b1127b69499b591c8e60ccf56586`
- **pool**: `0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640` (USDC/WETH, 500 bps, spacing 10)
- **window**: 2022-01-01 00:00:00+00:00 → 2024-12-31 23:59:59+00:00 (UTC)
- **schema version**: 1.0.0
- **git commit**: `1ccd63331b116f2852dd8f37bd8fdccc5d44570c`

## Summary — 14 check(s), 2 failed, 8 passed, 4 skipped

| Check | Severity | Result | Key metric |
|---|---|---|---|
| check_block_ordering | critical | ok | rows=7112899 |
| check_key_uniqueness | critical | ok | rows=7112899 |
| check_monotonic_timestamps | critical | ok | rows=7112899 |
| check_no_future_data | warning | ok | rows_beyond=0 |
| check_no_block_gaps | critical | ok | first=13916165, last=21525890 |
| check_swap_sign_convention | warning | FAIL | violations=128 |
| check_tick_price_consistency | critical | ok | dev-1=3801, dev0=6858801 |
| check_ticks_on_spacing_grid | critical | ok | mints=119781, burns=130516, tick_spacing=10 |
| check_liquidity_conservation | info | skip | swap_liquidity=partial, null_rows=6859945 |
| check_reference_coverage | warning | FAIL | first_close=1640995259.999, last_close=1735689599.999, symbol=ETHUSDT |
| check_regime_labels_complete | warning | ok | symbol=ETHUSDT |
| reconcile_fees_against_collect | info | skip | lifecycles=0, ignored_external=703 |
| check_dune_counts | info | skip | skipped_reason=dune_unavailable |
| crosscheck_routes | info | skip | skipped=no route samples in the dataset (T14's route policy populates them) |

## Failures

### check_swap_sign_convention — FAIL (warning)

128 swap(s) where amount0/amount1 are same-signed or zero; example tx_hash: ['0x07587d593a33b5a131989a8ba7b00c1914d89a7771b87eadcdb9fb4e90c90f71', '0x0806cd512dccd1603aaa4832849d55493f7f9f2b495c189ca561f1af66aee1de', '0x09ac22cf4117def34487ffb25535a296a4f13924326d46890389e58a9f84a79d', '0x0a33de7d917bf7ba82c48984ea364ed91773e8095633f85fbc66d0201d4586af']
- `violations` = `128`

### check_reference_coverage — FAIL (warning)

reference bars span [1640995259.999, 1735689599.999] but the window is [1640995200.0, 1735689599.0] — the as-of join starves at the edges
- `first_close` = `1640995259.999`
- `last_close` = `1735689599.999`
- `symbol` = `ETHUSDT`

## Skipped / informational

- **check_liquidity_conservation** (info): 6859945 of 6862602 swap row(s) have null liquidity (the subgraph Swap entity has no liquidity field; only the RPC route supplies it) — comparing only the non-null rows could hide drift; liquidity conservation cannot be compared; skipped
- **reconcile_fees_against_collect** (info): no completed position lifecycles in the tape (703 burn/collect row(s) refer to positions minted before the window); nothing to reconcile
- **check_dune_counts** (info): dune magnitudes fixture is status 'unavailable' (no Dune account at authoring time); comparison skipped
- **crosscheck_routes** (info): crosscheck_routes: skipped — no route samples in the dataset (T14's route policy populates them)

---
_Rendered by `undertow.data.validation.render_report` (T13)._
