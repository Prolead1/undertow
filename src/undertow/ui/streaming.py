"""Async file-tail and in-process pub/sub for live telemetry (U03, CONTRACTS §5).

:class:`TelemetryHub` watches one run directory.  A background task polls
``telemetry.jsonl`` and ``status.json`` through :func:`asyncio.to_thread`, turns
new records into CONTRACTS §5.2 envelopes, and fans them out to connected
WebSocket clients.  The tail stops after publishing a terminal ``done`` or
``failed`` status.

The polling iterator itself is exposed as :meth:`TelemetryHub.tail_once` so tests
can advance the file synchronously without depending on timer scheduling.  All
public frames use the frozen envelope::

    {"type": ..., "run_id": ..., "data": ..., "ts": ISO-8601-UTC}

No ``time.sleep`` is used and no ``undertow.sim``/``undertow.data`` import is
made: the UI is a pure artifact consumer (`docs/plans/ui/PLAN.md` §0.6).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from undertow.ui.paths import STATUS_FILENAME, TELEMETRY_FILENAME
from undertow.ui.schemas import RunStatusModel
from undertow.ui.telemetry import read_points, read_status

__all__ = ["TERMINAL_STATES", "TelemetryHub", "build_frame"]

logger = logging.getLogger("undertow.ui.streaming")

#: Status states that end a live tail (CONTRACTS §1.2/§5.2).
TERMINAL_STATES = frozenset({"done", "failed"})

#: JSON-safe data accepted by :func:`build_frame`.
FrameData = BaseModel | dict[str, Any] | list[Any] | str | int | float | bool | None


def _iso_z(value: datetime) -> str:
    """Format a datetime as ISO-8601 UTC with the contract's trailing ``Z``."""
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _serialise(data: FrameData) -> object:
    """Turn pydantic models into JSON-native data, leaving plain values alone."""
    if isinstance(data, BaseModel):
        return data.model_dump(mode="json")
    return data


def build_frame(
    event_type: str,
    run_id: str | None,
    data: FrameData,
    *,
    timestamp: datetime | None = None,
) -> str:
    """Serialise one CONTRACTS §5.2 WebSocket envelope as a JSON string.

    ``timestamp`` is injectable for deterministic tests; production callers let
    it default to "now" in UTC.
    """
    envelope = {
        "type": event_type,
        "run_id": run_id,
        "data": _serialise(data),
        "ts": _iso_z(timestamp or datetime.now(UTC)),
    }
    return json.dumps(envelope, separators=(",", ":"), ensure_ascii=False)


class TelemetryHub:
    """Bounded pub/sub fan-out plus a file-tail task for one run directory."""

    def __init__(
        self,
        run_dir: Path | str,
        *,
        run_id: str | None = None,
        telemetry_path: Path | str | None = None,
        status_path: Path | str | None = None,
        telemetry_poll_seconds: float = 1.0,
        max_queue: int = 256,
        start_offset: int = 0,
    ) -> None:
        if max_queue < 1:
            raise ValueError(f"max_queue must be positive, got {max_queue}")
        if telemetry_poll_seconds < 0:
            raise ValueError(
                f"telemetry_poll_seconds must be non-negative, got {telemetry_poll_seconds}"
            )
        if start_offset < 0:
            raise ValueError(f"start_offset must be non-negative, got {start_offset}")

        base = Path(run_dir)
        self.run_dir = base
        self.run_id = run_id
        self.telemetry_path = (
            Path(telemetry_path) if telemetry_path is not None else base / TELEMETRY_FILENAME
        )
        self.status_path = (
            Path(status_path) if status_path is not None else base / STATUS_FILENAME
        )
        self.telemetry_poll_seconds = float(telemetry_poll_seconds)
        self.max_queue = int(max_queue)

        self._offset = int(start_offset)
        self._subscribers: set[asyncio.Queue[str | None]] = set()
        self._tasks: set[asyncio.Task[None]] = set()
        self._last_status_key: tuple[object, ...] | None = None
        self._last_status: RunStatusModel | None = None
        self._closed = False

    # -- pub/sub ----------------------------------------------------------
    @property
    def subscriber_count(self) -> int:
        """Number of currently registered subscriber queues."""
        return len(self._subscribers)

    @property
    def pending_tasks(self) -> int:
        """Number of tail tasks still owned by the hub (for shutdown checks)."""
        return len(self._tasks)

    async def subscribe(
        self, *, on_subscribe: Callable[[], None] | None = None
    ) -> AsyncIterator[str]:
        """Yield JSON frames until :meth:`close` sends the end-of-stream marker.

        Registering is lazy (the generator body runs on first iteration), which
        matches how a WebSocket route consumes it.  The queue is removed in
        ``finally`` so a disconnected client cannot leak memory.  ``on_subscribe``
        runs after the queue is registered but before the first ``await``; the
        WebSocket route uses it to start the tail only once it can receive the
        initial status frame (avoiding a lost-status race).
        """
        if self._closed:
            return
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=self.max_queue)
        self._subscribers.add(queue)
        try:
            if on_subscribe is not None:
                on_subscribe()
            while True:
                frame = await queue.get()
                if frame is None:
                    break
                yield frame
        finally:
            self._subscribers.discard(queue)

    def publish(self, frame: str) -> None:
        """Fan a frame out to every subscriber, dropping oldest on backpressure."""
        if self._closed:
            return
        for queue in tuple(self._subscribers):
            if queue.full():
                try:
                    queue.get_nowait()  # drop the oldest queued frame
                except asyncio.QueueEmpty:  # pragma: no cover - same-loop invariant
                    pass
                else:
                    logger.warning(
                        "slow telemetry subscriber: dropping oldest frame (queue=%d)",
                        self.max_queue,
                    )
            try:
                queue.put_nowait(frame)
            except asyncio.QueueFull:  # pragma: no cover - defensive race guard
                logger.warning("slow telemetry subscriber: dropping new frame")

    # -- tail -------------------------------------------------------------
    def tail_once(self, offset: int) -> tuple[list[str], int]:
        """Do one synchronous poll: return ``(frames, new_offset)``.

        The caller supplies the line offset explicitly, so the file cursor is
        never implicit.  Status frames are emitted only when the status changes;
        the first observation always emits one.  That dedup cache is the only
        state this helper touches (``start_tail`` resets it), while all file I/O
        is blocking and therefore wrapped in :func:`asyncio.to_thread` by
        :meth:`start_tail`.
        """
        frames: list[str] = []
        points, new_offset = read_points(self.telemetry_path, since=offset, limit=None)

        # Emit new points before the status that may terminate the tail, so a
        # client sees the final telemetry lines before it stops on ``done``.
        for point in points:
            frames.append(build_frame("telemetry", self.run_id or point.run_id, point))

        status = read_status(self.status_path)
        if status is not None:
            if self.run_id is None:
                self.run_id = status.run_id
            status_key = (
                status.state,
                status.timestep,
                status.updated_at,
                status.error,
            )
            if status_key != self._last_status_key:
                self._last_status_key = status_key
                self._last_status = status
                frames.append(build_frame("status", status.run_id, status))

        return frames, new_offset

    def start_tail(self, run_id: str) -> asyncio.Task[None]:
        """Spawn the polling task for ``run_id`` and return it.

        The task publishes new telemetry/status frames until it has published a
        terminal status; :meth:`close` cancels it cleanly on shutdown.
        """
        if self._closed:
            raise RuntimeError("cannot start a tail on a closed TelemetryHub")
        self.run_id = run_id
        # A fresh tail must always announce the current status, even if a test
        # or caller polled the hub manually before starting it.
        self._last_status_key = None
        self._last_status = None
        task = asyncio.create_task(
            self._tail(run_id),
            name=f"undertow-telemetry-tail-{run_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._on_tail_done)
        return task

    async def _tail(self, run_id: str) -> None:
        offset = self._offset
        while not self._closed:
            try:
                frames, offset = await asyncio.to_thread(self.tail_once, offset)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive poll guard
                logger.warning("telemetry poll failed for %s: %s", run_id, exc)
                frames = []

            self._offset = offset
            for frame in frames:
                self.publish(frame)

            status = self._last_status
            if status is not None and status.state in TERMINAL_STATES:
                return

            await asyncio.sleep(self.telemetry_poll_seconds)

    def _on_tail_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:  # pragma: no cover - guarded by _tail's except
            logger.error("telemetry tail for %s failed: %s", self.run_id, error)

    # -- shutdown ---------------------------------------------------------
    async def close(self) -> None:
        """Cancel tail tasks and wake every subscriber with an end marker.

        Idempotent, and safe to call from an application shutdown hook.  Waiting
        for the cancelled tasks here is what prevents "Task was destroyed but it
        is pending" warnings when the server exits mid-poll.
        """
        self._closed = True
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

        for queue in tuple(self._subscribers):
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:  # pragma: no cover - same-loop invariant
                    break
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:  # pragma: no cover - defensive
                try:
                    queue.get_nowait()
                    queue.put_nowait(None)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass
        self._subscribers.clear()
