# ADR 026 — Frontend state management for U07: React state + typed client now, Zustand/TanStack Query deferred to U08

**Status:** accepted
**Affects:** `web/src/App.tsx` · `web/src/api/client.ts` · `web/src/api/useLiveStream.ts` · `web/src/__tests__/smoke.test.tsx` · `web/package.json` · `docs/plans/ui/CONTRACTS.md` §8 · U08–U10
**Raised by:** U07 (web-scaffold)

## Context

`CONTRACTS.md` §8 pins the intended frontend data layer:

> **State:** TanStack Query for REST caches (keyed by run/query), a small Zustand store for the live
> buffer, pinned comparison set, and the shared time cursor. No prop-drilling of raw series.

The U07 brief (`docs/plans/ui/tasks/U07_web_scaffold.md`) owns the shell, the typed API client and the
live-stream hook, but not the three views that consume shared state. The U07 scaffold therefore ships
`web/package.json` with React, React Router and Plotly only: **neither `@tanstack/react-query` nor
`zustand` is installed**, and the shell keeps its state in `useState`/`useReducer` inside
`App.tsx`/`useLiveStream.ts`.

§8 also does not say where the OpenAPI document comes from for `gen:api`. U07's brief says the
`gen:api` script "runs `openapi-typescript http://127.0.0.1:8050/openapi.json` (document that the
server must be running; commit the generated file)". The implemented `gen:api` instead reads the
committed `web/openapi.json`, with a separate `gen:api:live` script for a running server.

Both are deliberate deviations from a literal reading of the contract, so this ADR records them
rather than leaving the next agent to rediscover them.

## Problem

Implementing §8 literally in U07 has no consumer and creates real costs:

1. **There is no shared state to manage yet.** The pinned comparison set, the shared time cursor and
   the multi-run live buffer only exist once the views (U08 Live Cockpit, U09 Policy Replay, U10
   Leaderboard/Compare) are composed. U07 has one run selection (`?run_id=`) and one stream hook;
   adding Redux-style stores and a query cache for a single consumer would be speculative
   infrastructure, untested by any behavioral requirement in the U07 definition of done.
2. **A query client needs a cache-key policy.** `CONTRACTS.md` §7 permits a `run_id` alias of the
   form `<experiment>/seed_<n>` and ADR-025 notes duplicate `run_id`s resolve per directory. U07 has
   not yet pinned the cache-key rule (raw id vs. resolved directory vs. query tuple); baking one into
   U07 would either be wrong for U08 or churn when U08 settles it.
3. **A live URL makes `gen:api` unreproducible.** Requiring a bound `127.0.0.1:8050` to regenerate
   types means CI, a fresh clone, and a code review can all produce a different `types.ts` (or none
   at all) depending on whether a server happens to be running. The generated file is committed, so
   regeneration should be deterministic.
4. **The Python gate must not depend on Node.** `uv run pytest` must pass with `web/node_modules`
   absent (ADR-023, U07 definition of done). Adding runtime JS dependencies in U07 does not by itself
   break that, but it does widen the surface U11's CI job must install and pin.

## Decision

1. **U07 keeps React state + a typed client; no Zustand, no TanStack Query.** The shell's run
   selection lives in the URL (`?run_id=&seed=&episode=`), the run list + live buffer + last status
   live in `useLiveStream`'s `useReducer`, and REST functions are wrapped by the plain
   `web/src/api/client.ts`. The live merge/dedupe rule (`(run_id, timestep)`) is implemented directly
   in `useLiveStream.ts` and unit-tested.
2. **Zustand and TanStack Query are deferred to U08**, which introduces the first genuinely shared
   state (live buffer shared across panels, pinned comparison set, shared time cursor) and the first
   REST cache with more than one query. U08 must revisit §8's cache-key rule at that point, taking
   ADR-025's resolved-directory keying into account.
3. **`gen:api` reads the committed `web/openapi.json`** (offline and reproducible); `gen:api:live`
   fetches `http://127.0.0.1:8050/openapi.json` for the developer's convenience. `openapi.json` and
   `src/api/types.ts` are both committed; `web/README.md` documents the one-off dump command and the
   regenerate-and-commit rule whenever U05's wire contract changes.
4. **No contract edit is proposed here beyond recording the deviation.** This ADR is the record that
   §8's TanStack/Zustand sentence is **intent for U08**, not a U07 deliverable, and that
   `gen:api`'s input is the committed snapshot rather than a live URL.

## Consequences

- **U07 code.** `App.tsx` uses `useState` for pins/dismissed toast; `useLiveStream.ts` owns the
  reducer and now also performs a one-shot `getRuns()` REST load plus a `/api/runs` refresh while the
  socket is down (see the companion review fix), so the run rail no longer depends solely on the WS
  `hello` frame.
- **U08 must know.** Adding TanStack Query/Zustand later is additive; `client.ts` already returns
  typed promises with an optional `AbortSignal`, so it can back query functions without change. U08
  owns the cache-key decision and any §8 clarification.
- **U09/U10 must know.** They are expected to consume whatever shared store U08 introduces rather
  than re-fetching through local state, per §8's "no prop-drilling of raw series".
- **`CONTRACTS.md` needs no immediate edit.** §8 remains the target for U08; this ADR only timestamps
  the deferral and the `gen:api` input. If the human wants §8 to say "deferred to U08 explicitly",
  that is a one-line contract edit U11 can own.
- **Vault/lesson-plan.** No `~/Documents/fyp/lesson_plan/` text needs correcting; the requirement is
  unchanged, only its task ownership is made explicit.
- **Rejected alternatives.** (a) Install TanStack Query + Zustand in U07 with no consumers — adds
  untested runtime dependencies and a cache-key guess; (b) keep only the live-URL `gen:api` — makes
  the committed generated types non-reproducible and blocks offline/CI regeneration; (c) hand-write
  `types.ts` — loses the single-source-of-truth OpenAPI generation that §8/ADR-023 call "the biggest
  iteration win".
