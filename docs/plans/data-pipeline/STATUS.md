# STATUS — data-pipeline-v1 handoff board

Update this file when you **start** and when you **finish**. Keep it terse. This is how the next agent
knows whether to branch off `main` or off your branch (see `PLAN.md` §3.2).

Legend: `todo` · `in-progress` · `in-review` (code-reviewer running / findings open) · `pr-open`
(approved + PR raised, awaiting human merge) · `merged` · `blocked`

| Task | State | Branch | PR | pytest | Reviewer | Notes |
|---|---|---|---|---|---|---|
| T00 scaffold | merged | chore-data-scaffold | #2 | 4 passed, 1 skipped | APPROVE (merged by human) | |
| T01 config/types | merged | feature-data-config | #3 | 53 passed, 1 skipped | APPROVE (x2) | rescued from crashed agent; ADR-003 raised |
| T02 fixedpoint | merged | feature-data-fixedpoint | #4 | 27 passed, 1 skipped | APPROVE (x2) | exact integer TickMath/price/liquidity ports; fixture anchored via routes 1/2/3 |
| T03 schemas | merged | feature-data-schemas | #8 | 237 passed, 1 skipped | APPROVE | handles ADR-003's `start_block: None` date-mode |
| T04 fetcher base | merged | feature-data-fetcher-base | #9 | 237 passed, 1 skipped | APPROVE | retry/chunk/cache; schema-free by the §5.1 layering rule |
| T05 thegraph | in-progress | feature-data-thegraph | — | — | — | wave 3, launched in parallel |
| T06 rpc | in-progress | feature-data-rpc | — | — | — | wave 3, launched in parallel |
| T07 gas | in-progress | feature-data-gas | — | — | — | wave 3, launched in parallel |
| T08 reference | in-progress | feature-data-reference | — | — | — | wave 3, launched in parallel |
| T09 storage | in-progress | feature-data-storage | — | — | — | wave 3, launched in parallel |
| T10 feegrowth | merged | feature-data-feegrowth | #10 | 237 passed, 1 skipped | APPROVE | **exact** multi-tick apportionment; ADR-002 never exists (see below) |
| T11 align | in-progress | feature-data-event-tape | — | — | — | tiny_dataset() factory delivered in conftest.py; tape built on polars join_asof backward |
| T12 regimes | in-progress | feature-data-regimes | — | — | — | wave 3, launched in parallel |
| T13 validation | todo | — | — | — | — | |
| T14 cli | todo | — | — | — | — | |
| T15 dune | in-progress | feature-data-dune-queries | — | — | — | wave 3, launched in parallel |
| T16 public api | todo | — | — | — | — | |

## ADRs raised

_(list `adr/NNN-slug.md` files as they are created; copy `adr/TEMPLATE.md` to start one)_

- `adr/001-regime-drift-definition.md` — **accepted**, pre-written by the planner: `μ` is the total
  window log return, not the per-step mean. See `CONTRACTS.md` §6.3. Owned by T12 (writes the standalone
  version); affects T11.
- `adr/002-swap-segment-apportionment.md` — **resolved: never exists.** T10 took the exact
  re-simulation route, so the conditional ADR is not written. Consequence for T13: use a small
  tolerance (raw-unit level), **not** `rel_tolerance=0.0` — T10's PR documents two bounded residuals
  (fee-formula second-order distribution, and a first-crossing approximation only when no
  `current_sqrt_price_x96` is available). See T10 PR #10.
- `adr/003-window-blocks-optional.md` — **accepted, raised by T01.** `CONTRACTS.md` §2 declared
  `WindowConfig.start_block/end_block` non-optional while requiring date-only windows; widened to
  `BlockNumber | None` with XOR validation. Repo copy committed: `docs/decisions/003-window-blocks-optional.md`.
  Consumers (T03 ✓, T07, T14) must handle `start_block is None` (date-mode).

## Blockers / open questions

- **mypy tooling drift (T00-level, pre-existing):** project-wide `uv run mypy src tests` fails inside
  numpy's shipped stubs (`Type statement is only supported in Python 3.12`) under Python 3.14.7;
  reproduces on clean `main`. Per-module mypy passes. Needs a T00 fix (pin numpy or bump
  `python_version` in `pyproject.toml`). Wave-3 tasks run mypy on their own package only.
- **Dune access (T15):** magnitudes fixture may be `status: "unavailable"` with a skip-test if the
  agent has no Dune account — never fabricated numbers.