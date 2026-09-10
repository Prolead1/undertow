# STATUS — sim-v1 handoff board

Update this file when you **start** and when you **finish**. Keep it terse. This is how the next agent
knows whether to branch off `main` or off your branch (see `PLAN.md` §3.2).

Legend: `todo` · `in-progress` · `in-review` (code-reviewer running / findings open) · `pr-open`
(approved + PR raised, awaiting human merge) · `merged` · `blocked`

| Task | State | Branch | PR | pytest | Reviewer | Notes |
|---|---|---|---|---|---|---|
| S00 scaffold + RL-library ADR | pr-open | `feature-sim-scaffold` | [#26](https://github.com/Prolead1/undertow/pull/26) | 574 passed | approved | Went beyond brief — all modules built as one scaffold |
| S01 types/config | in-progress | | | | | |
| S02 data API extension | todo | | | | | |
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