# Validation report — undertow.data

Generated: 2026-10-24T03:55:43.988356+00:00
- **dataset id**: `a5cf5d9a79d1651e29ae686c43280c320a93a959022372dba34becf8890e5a50`
- **pool**: `0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8` (USDC/WETH, 3000 bps, spacing 60)
- **window**: 2022-01-01 00:00:00+00:00 → 2024-12-31 23:59:59+00:00 (UTC)
- **schema version**: 1.0.0
- **git commit**: `745135f3b7fb9c5495bafeb5514030987b4159ee+dirty`

## Summary — 14 check(s), 2 failed, 8 passed, 4 skipped

| Check | Severity | Result | Key metric |
|---|---|---|---|
| check_block_ordering | critical | ok | rows=554854 |
| check_key_uniqueness | critical | ok | rows=554854 |
| check_monotonic_timestamps | critical | ok | rows=554854 |
| check_no_future_data | warning | ok | rows_beyond=0 |
| check_no_block_gaps | critical | ok | first=13916165, last=21525890 |
| check_swap_sign_convention | warning | FAIL | violations=26 |
| check_tick_price_consistency | critical | ok | dev-1=158, dev0=447556 |
| check_ticks_on_spacing_grid | critical | ok | mints=53510, burns=53630, tick_spacing=60 |
| check_liquidity_conservation | info | skip | swap_liquidity=unavailable |
| check_reference_coverage | warning | FAIL | first_close=1640995259.999, last_close=1735689599.999, symbol=ETHUSDT |
| check_regime_labels_complete | warning | ok | symbol=ETHUSDT |
| reconcile_fees_against_collect | info | skip | lifecycles=0, ignored_external=1861 |
| check_dune_counts | info | skip | skipped_reason=dune_unavailable |
| crosscheck_routes | info | skip | skipped=no route samples in the dataset (T14's route policy populates them) |

## Failures

### check_swap_sign_convention — FAIL (warning)

26 swap(s) where amount0/amount1 are same-signed or zero; example tx_hash: ['0x0e109f2e474d6b0a9c432cf92d444a808430bbfd26b9c12e0cc9a9204f4549ec', '0x0e8f21c1b3cd4210e3416fb76cf6c42a43071cabe58e04c95496a0f25e7c5243', '0x1cc3099ade6d283818eee104577c2b9d2606a960242dd0f72a129fd2355bea5e', '0x1e9e2099552bacbaa3ae5e904fd1a01abf4aed3446d3b5464d216d03287a28c9']
- `violations` = `26`

### check_reference_coverage — FAIL (warning)

reference bars span [1640995259.999, 1735689599.999] but the window is [1640995200.0, 1735689599.0] — the as-of join starves at the edges
- `first_close` = `1640995259.999`
- `last_close` = `1735689599.999`
- `symbol` = `ETHUSDT`

## Skipped / informational

- **check_liquidity_conservation** (info): swap liquidity is not recorded on this route (the subgraph Swap entity has no liquidity field; only the RPC route supplies it) — liquidity conservation cannot be compared; skipped
- **reconcile_fees_against_collect** (info): no completed position lifecycles in the tape (1861 burn/collect row(s) refer to positions minted before the window); nothing to reconcile
- **check_dune_counts** (info): dune magnitudes fixture is status 'unavailable' (no Dune account at authoring time); comparison skipped
- **crosscheck_routes** (info): crosscheck_routes: skipped — no route samples in the dataset (T14's route policy populates them)

---
_Rendered by `undertow.data.validation.render_report` (T13)._
