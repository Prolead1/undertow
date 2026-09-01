# ADR 003 — `WindowConfig` block bounds are optional

**Status:** accepted
**Date:** 2026-08-28 (raised during T01)
**Affects:** `undertow.data.config.WindowConfig` and every consumer of it

## Context

The data config declares the study window in one of two mutually exclusive modes: a block range or a
date range. The original contract declaration for `WindowConfig` made `start_block` / `end_block`
non-optional:

```
start_block: BlockNumber
end_block: BlockNumber     # inclusive
# exactly one of block or date bounds is given; dates resolved to blocks by the block index
start_utc: datetime | None = None
end_utc: datetime | None = None
```

## Problem

The signature was internally inconsistent: the block fields were non-optional, but the accompanying
comment required a date-only window to be constructible ("exactly one of block or date bounds is
given"). The reference configs use the **date** mode (`2022-01-01` → `2024-12-31` UTC), so a
date-only `WindowConfig` must be expressible. With non-optional block fields, constructing a
date-mode window was impossible without also supplying block bounds — contradicting the intended
exclusivity. Any consumer that typed `window.start_block` as `int` would break on a date-mode config.

## Decision

Widen the block bounds to optional and enforce the exclusivity invariant in `__post_init__`:

```
start_block: BlockNumber | None = None
end_block: BlockNumber | None = None
start_utc: datetime | None = None
end_utc: datetime | None = None
```

Validation rules:

- block pair given together XOR date pair given together;
- exactly one pair must be set;
- `start <= end` within the chosen pair;
- dates must be timezone-aware UTC.

Date-mode windows are resolved to blocks later by the block index (T07); `undertow.data` treats the
date bounds as authoritative until then.

## Consequences

- Consumers must handle `start_block is None` / `end_block is None` (date-mode windows) rather than
  assuming an integer block number.
- No other field or method signature of `WindowConfig` changes; the `train_end_utc` /
  `eval_start_utc` walk-forward split hook is untouched.
- Rejected alternative: a separate `date_mode: bool` flag. It would add a second way to express the
  same exclusivity and push the invariant into every call site; optional-with-XOR keeps the invariant
  at the boundary where it can be validated once.
