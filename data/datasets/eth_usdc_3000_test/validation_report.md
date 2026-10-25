# Validation report — undertow.data

Generated: 2026-10-25T03:40:33.917984+00:00
- **dataset id**: `13eccf7a1bd5387151dbd247761b3f4cff3b8788b2a6d60ece7f711c6960133e`
- **pool**: `0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8` (USDC/WETH, 3000 bps, spacing 60)
- **window**: 2024-01-01 00:00:00+00:00 → 2024-01-31 23:59:59+00:00 (UTC)
- **schema version**: 1.0.0
- **git commit**: `1ccd63331b116f2852dd8f37bd8fdccc5d44570c`

## Summary — 14 check(s), 2 failed, 8 passed, 4 skipped

| Check | Severity | Result | Key metric |
|---|---|---|---|
| check_block_ordering | critical | ok | rows=11190 |
| check_key_uniqueness | critical | ok | rows=11190 |
| check_monotonic_timestamps | critical | ok | rows=11190 |
| check_no_future_data | warning | ok | rows_beyond=0 |
| check_no_block_gaps | critical | ok | first=18908894, last=19129888 |
| check_swap_sign_convention | warning | ok | rows=10410 |
| check_tick_price_consistency | critical | ok | dev0=10410 |
| check_ticks_on_spacing_grid | critical | ok | mints=268, burns=512, tick_spacing=60 |
| check_liquidity_conservation | info | skip | swap_liquidity=unavailable |
| check_reference_coverage | warning | FAIL | first_close=1704067259.999, last_close=1706745599.999, symbol=ETHUSDT |
| check_regime_labels_complete | warning | FAIL | symbol=ETHUSDT |
| reconcile_fees_against_collect | info | skip | lifecycles=0, ignored_external=302 |
| check_dune_counts | info | skip | skipped_reason=dune_unavailable |
| crosscheck_routes | info | skip | skipped=no route samples in the dataset (T14's route policy populates them) |

## Failures

### check_reference_coverage — FAIL (warning)

reference bars span [1704067259.999, 1706745599.999] but the window is [1704067200.0, 1706745599.0] — the as-of join starves at the edges
- `first_close` = `1704067259.999`
- `last_close` = `1706745599.999`
- `symbol` = `ETHUSDT`

### check_regime_labels_complete — FAIL (warning)

handoff finding: never observed over the window: bull, bear, high_vol
- `rows` = `44640`
- `symbol` = `ETHUSDT`

## Skipped / informational

- **check_liquidity_conservation** (info): swap liquidity is not recorded on this route (the subgraph Swap entity has no liquidity field; only the RPC route supplies it) — liquidity conservation cannot be compared; skipped
- **reconcile_fees_against_collect** (info): no completed position lifecycles in the tape (302 burn/collect row(s) refer to positions minted before the window); nothing to reconcile
- **check_dune_counts** (info): dune magnitudes fixture is status 'unavailable' (no Dune account at authoring time); comparison skipped
- **crosscheck_routes** (info): crosscheck_routes: skipped — no route samples in the dataset (T14's route policy populates them)

---
_Rendered by `undertow.data.validation.render_report` (T13)._
