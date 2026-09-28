# STATUS — sim-v1 handoff board

Update this file when you **start** and when you **finish**. Keep it terse. This is how the next agent
knows whether to branch off `main` or off your branch (see `PLAN.md` §3.2).

Legend: `todo` · `in-progress` · `in-review` (code-reviewer running / findings open) · `pr-open`
(approved + PR raised, awaiting human merge) · `merged` · `blocked`

| Task | State | Branch | PR | pytest | Reviewer | Notes |
|---|---|---|---|---|---|---|
| S00 scaffold + RL-library ADR | merged | `feature-sim-scaffold` | [#26](https://github.com/Prolead1/undertow/pull/26) | 574 passed | approved | Went beyond brief — all modules built as one scaffold |
| S01 types/config | merged | `feature-sim-types-config` | [#27](https://github.com/Prolead1/undertow/pull/27) | 38 passed | approved | Implements CONTRACTS §§1-2
| S02 data API extension | pr-open | `feature-sim-data-api` | [#36](https://github.com/Prolead1/undertow/pull/36) | 129 passed (API+guard); full 745 passed, 5 skipped | approved | Surface widened + phantom `tick_to_sqrt_price` dropped; ADR-007 + ADR-008 |
| S03 marketview + split | todo | | | | | |
| S04 position math | todo | | | | | |
| S05 metrics | todo | | | | | |
| S06 price processes | todo | | | | | |
| S07 pool engine | todo | | | | | |
| S08 frictions | todo | | | | | |
| S09 reward | todo | | | | | |
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