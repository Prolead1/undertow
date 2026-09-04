# STATUS — data-pipeline-v1 handoff board

Update this file when you **start** and when you **finish**. Keep it terse. This is how the next agent
knows whether to branch off `main` or off your branch (see `PLAN.md` §3.2).

Legend: `todo` · `in-progress` · `in-review` (code-reviewer running / findings open) · `pr-open`
(approved + PR raised, awaiting human merge) · `merged` · `blocked`

| Task | State | Branch | PR | pytest | Reviewer | Notes |
|---|---|---|---|---|---|---|
| T00 scaffold | merged | chore-data-scaffold | #2 | 4 passed, 1 skipped | APPROVE (merged by human) | |
| T01 config/types | pr-open | feature-data-config | #3 | 53 passed, 1 skipped | APPROVE (x2) | rescued from crashed agent; ADR-003 raised |
| T02 fixedpoint | pr-open | feature-data-fixedpoint | #4 | 27 passed, 1 skipped | APPROVE (x2) | exact integer TickMath/price/liquidity ports; fixture anchored via routes 1/2/3 |
| T03 schemas | todo | — | — | — | — | |
| T04 fetcher base | todo | — | — | — | — | |
| T05 thegraph | todo | — | — | — | — | |
| T06 rpc | todo | — | — | — | — | |
| T07 gas | todo | — | — | — | — | |
| T08 reference | todo | — | — | — | — | |
| T09 storage | todo | — | — | — | — | |
| T10 feegrowth | todo | — | — | — | — | |
| T11 align | todo | — | — | — | — | |
| T12 regimes | todo | — | — | — | — | |
| T13 validation | todo | — | — | — | — | |
| T14 cli | todo | — | — | — | — | |
| T15 dune | todo | — | — | — | — | |
| T16 public api | todo | — | — | — | — | |

## ADRs raised

_(list `adr/NNN-slug.md` files as they are created; copy `adr/TEMPLATE.md` to start one)_

- `adr/001-regime-drift-definition.md` — **accepted**, pre-written by the planner: `μ` is the total
  window log return, not the per-step mean. See `CONTRACTS.md` §6.3. Affects T12.
- `adr/002-swap-segment-apportionment.md` — **not yet written; conditional.** T10 writes it *only if* it
  cannot apportion fee growth exactly across multi-tick swaps and takes the approximation route. If T10
  achieves exact apportionment, this ADR never exists and T13 should call
  `reconcile_fees_against_collect(..., rel_tolerance=0.0)`. Note this is a **call-site argument**, not a
  change to the frozen default in `CONTRACTS.md` §8 — the signature stays as specified. T10 must state
  which case happened in its PR body, because T13's acceptance tolerance depends on the answer.
- `adr/003-window-blocks-optional.md` — **accepted, raised by T01.** `CONTRACTS.md` §2 declared
  `WindowConfig.start_block/end_block` non-optional while requiring date-only windows; widened to
  `BlockNumber | None` with XOR validation. Repo copy committed: `docs/decisions/003-window-blocks-optional.md`.
  Consumers (T03, T07, T14) must handle `start_block is None` (date-mode).

## Blockers / open questions

- **mypy tooling drift (T00-level, pre-existing):** project-wide `uv run mypy src tests` fails inside
  numpy's shipped stubs (`Type statement is only supported in Python 3.12`) under Python 3.14.7;
  reproduces on clean `main`. T01's modules pass in isolation. Needs a T00 fix (pin numpy or bump
  `python_version` in `pyproject.toml`). Not a blocker for wave-1/2 tasks that run mypy on their own
  package only.
