# STATUS — sim-v1 handoff board

Update this file when you **start** and when you **finish**. Keep it terse. This is how the next agent
knows whether to branch off `main` or off your branch (see `PLAN.md` §3.2).

Legend: `todo` · `in-progress` · `in-review` (code-reviewer running / findings open) · `pr-open`
(approved + PR raised, awaiting human merge) · `merged` · `blocked`

| Task | State | Branch | PR | pytest | Reviewer | Notes |
|---|---|---|---|---|---|---|
| S00 scaffold + RL-library ADR | merged | `feature-sim-scaffold` | [#26](https://github.com/Prolead1/undertow/pull/26) | 574 passed | approved | Scaffold only: deps, test markers, config skeleton, RL ADR |
| S01 types/config | merged | `feature-sim-types-config` | [#27](https://github.com/Prolead1/undertow/pull/27) | 38 passed | approved | Implements CONTRACTS §§1-2 |
| S02 data API extension | merged | `feature-sim-data-api` | [#36](https://github.com/Prolead1/undertow/pull/36) | 129 passed (API+guard); full 745 passed, 5 skipped | approved | Surface widened + phantom `tick_to_sqrt_price` dropped; ADR-007 + ADR-008 |
| S03 marketview + split | merged | `feature-sim-marketview` | [#39](https://github.com/Prolead1/undertow/pull/39) | 39 passed; full 853 passed, 6 skipped | approved | Look-ahead wall enforced; pyarrow→polars; relaxed stale scaffold guard to allow top-level data API (ADR-008) |
| S04 position math | merged | `feature-sim-position-math` | [#40](https://github.com/Prolead1/undertow/pull/40) | 35 passed; full 888 passed, 6 skipped | approved | ADR-009 orientation: raw sqrt/liquidity, human amounts/price/value |
| S05 metrics | merged | `feature-sim-metrics` | [#35](https://github.com/Prolead1/undertow/pull/35) | 39 passed; suite 655 passed, 5 skipped | approved | Sortino as RMS downside deviation; PnL IL column accepts `il` (CONTRACTS §13) or `il_change` |
| S06 price processes | merged | `feature-sim-price-processes` | [#44](https://github.com/Prolead1/undertow/pull/44) | 41 passed; full 929 passed, 6 skipped | approved | ADR-010 (dt cadence = 10 min) |
| S07 pool engine | pr-open | `feature-sim-pool-engine` | [#43](https://github.com/Prolead1/undertow/pull/43) | 36 passed; full 924 passed, 6 skipped | approved | ADR-012 fee/pip units |
| S08 frictions | todo | | | | | |
| S09 reward | pr-open | `feature-sim-reward` | [#41](https://github.com/Prolead1/undertow/pull/41) | 34 passed; full 922 passed, 6 skipped | approved | normalize_reward at env boundary; ablation flags gate but pnl_net ungated |
| S10 baselines | todo | | | | | |
| S11 backtester | todo | | | | | |
| S12 gym env | todo | | | | | |
| S13 training harness | todo | | | | | |
| S14 parity/look-ahead | todo | | | | | |
| S15 evaluation runner | todo | | | | | |
| S16 CLI + public API | todo | | | | | |

## ADRs raised

- **ADR-004** — RL library selection (`stable-baselines3 >= 2.9`), in `docs/decisions/004-rl-library.md`
- **ADR-007** — `undertow.data` sim-API loader module attribution (`TICK_BASE` imported from `.config`; `load_*` adapters in `.loader`), in `docs/decisions/007-data-api-loader-module-attribution.md`
- **ADR-008** — data ⇄ sim public-API surface reconciliation (superset of T16's baseline; drops phantom `tick_to_sqrt_price`; restores the public-API guard), in `docs/decisions/008-data-sim-public-api-reconciliation.md`
- **ADR-009** — sim price orientation: raw sqrt price, human amounts/price (USDC numeraire); V2 IL r=1.2 typo corrected to −0.004141, in `docs/decisions/009-sim-price-orientation.md`
- **ADR-010** — MRSJD step size: `dt` is the 10-minute cadence (`1/(6·24·365.25)`), correcting the literal `1/(6·365.25)` (4h) in `CONTRACTS.md` §7; σ = 0 allowed as pure-drift limit, in `docs/decisions/010-sim-mrsjd-dt-cadence.md`
- **ADR-012** — fee tier units: `fee_tier_bps` holds Uniswap pips (3000 = 0.30%), pool-fee denominator `1_000_000`, slippage = notional × (pips/1e6 + impact bps/1e4), in `docs/decisions/012-fee-and-pip-units.md`
