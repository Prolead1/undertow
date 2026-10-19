# U03 — Telemetry reader + live streaming hub

**Wave 1b · size M · depends on: U00, U01, U02 · blocks: U04, U05, U08**
**Branch:** `feature-ui-streaming`

> U02 creates the frozen `schemas.py` (`TelemetryPoint`, `RunStatusModel`) this task returns. Branch
> off U02's branch and mark this PR `Stacked on #<U02 PR> (branch feature-ui-registry). Merge that first.`

## Why this task exists

The Live Cockpit updates while training writes. This task owns reading `telemetry.jsonl` /
`status.json` safely (including torn trailing writes) and turning new lines into push events for the
WebSocket layer (U05 wires the transport).

## Files you own

```
src/undertow/ui/telemetry.py       (new — read/tail sidecars → TelemetryPoint / RunStatus)
src/undertow/ui/streaming.py       (new — async file-tail → in-process PubSub hub)
tests/ui/test_telemetry_reader.py  (new)
tests/ui/test_streaming.py         (new)
```

## What to build

Read `CONTRACTS.md` §1 (schemas) and §5 (tail + WS envelope) first.

1. **`telemetry.py`.**
   - `read_status(path) -> RunStatus | None` — parse `status.json`; a missing file returns `None`; a
     malformed file logs a warning and returns `None`. `RunStatus` comes from U00's
     `undertow.sim.train.telemetry`.
   - `read_points(path, *, since=0, limit=1000) -> tuple[list[TelemetryPoint], int]` — `since` is a
     **line offset**; returns parsed points and the next offset. A single unparseable line (the
     trailing partial write) is skipped and does **not** advance the offset past it. A line that
     parses but fails validation is logged and skipped (offset advances — it is complete but wrong).
   - `read_curve_points(path) -> list[TelemetryPoint]` convenience (all points) with the same skew
     protection.
   - Implement `count_lines(path)` cheaply for the registry (U02 may import it later; keep it public).
   - All reads are synchronous and use a streaming line iterator; never load the whole file twice.

2. **`streaming.py`.**
   - `class TelemetryHub` — an async pub/sub over one run directory:
     - `subscribe() -> AsyncIterator[str]` yields JSON frames (the §5.2 envelope) for connected
       clients; `publish(frame)` fan-out to all subscribers via bounded `asyncio.Queue`s (drop-oldest
       on backpressure, log the drop).
     - `start_tail(run_id)` spawns a task that polls `telemetry.jsonl` every
       `telemetry_poll_seconds`, publishing new points and status changes; it stops when
       `status.state in {"done","failed"}` **after** publishing the final status.
     - `close()` cancels tasks and drains subscribers cleanly (no "Task was destroyed" warnings).
   - No `time.sleep`; use `asyncio.sleep`. File I/O via `asyncio.to_thread`.
   - Make the tail logic independently testable: a `tail_once(offset) -> (frames, new_offset)` pure
     helper that does one poll, so tests need not drive the event loop timing.

## Tests

`test_telemetry_reader.py`:
1. Empty/missing file → `([], 0)` and `None` status.
2. Three valid lines → three points, offsets 0..3; `since=1` returns the last two.
3. **Torn trailing line**: write two complete lines + half a third; `read_points` returns two and
   `next_offset==2`; after completing the write, the next call returns the third.
4. A malformed-but-complete line (e.g. wrong type) is skipped, offset still advances, warning logged.
5. `read_status` on malformed JSON returns `None` without raising.
6. Unknown extra metric keys survive round-trip (forward compatibility).

`test_streaming.py`:
7. `tail_once` publishes only new points and advances the offset.
8. Two subscribers receive the same frames; a slow subscriber drops oldest rather than blocking.
9. Tailing a run whose status is `done` publishes the final status then stops (assert the task
   completes within a timeout).
10. `close()` leaves no pending tasks.

Use `pytest-asyncio` (add to dev deps if missing) or drive the loop with `asyncio.run` — either is
fine, but be consistent and mark async tests clearly.

## Definition of done

- `uv run pytest tests/ui -q` green; full suite green at ≥90%.
- `ruff` + `mypy src/undertow` clean.
- Torn-write and completion behaviour match `CONTRACTS.md` §5.
- `code-reviewer` APPROVE; PR from `feature-ui-streaming`; `STATUS.md` updated.
