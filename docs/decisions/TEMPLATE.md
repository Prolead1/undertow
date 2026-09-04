# ADR NNN — <one-line decision>

**Status:** proposed | accepted | superseded by ADR-NNN
**Affects:** <task ids and file paths>
**Raised by:** <task id>

## Context

What the plan / `CONTRACTS.md` / the roadmap chapter says, and what you were trying to do. Quote the
source requirement so the next reader does not have to go hunting.

## Problem

Why the specified approach does not work, or is wrong, or is underspecified. Be concrete — a number, a
failing case, a contradiction. "It felt cleaner" is not a problem statement.

## Decision

What you are doing instead, precisely enough that another agent could implement it identically.

## Consequences

- What changes in the code, and where.
- What other tasks must know (name them — they may already be running).
- Whether `CONTRACTS.md` needs an edit, and whether the thesis text in
  `~/Documents/fyp/lesson_plan/` needs correcting. **Vault edits are flagged to the human, never made
  from a code task.**
- What you rejected, and why — the alternative is the thing the human will ask about.

---

**Process:** write the ADR, list it in `STATUS.md` under "ADRs raised", and link it in your PR body. Do
not silently deviate from `CONTRACTS.md`: another agent is coding against it as written, and an
undocumented change surfaces as their test failing for no visible reason.
