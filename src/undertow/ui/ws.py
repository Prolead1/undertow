"""WebSocket plumbing for ``undertow.ui`` (U05, CONTRACTS §5.2).

This module owns the async-only part of the API surface:

* :class:`TelemetryHubs` lazily creates one :class:`TelemetryHub` per run and
  closes/evicts it when its last WebSocket subscriber disconnects.
* :func:`pump_frames` forwards hub frames while emitting heartbeat pings and
  honouring client disconnects.  Its hand-off queue is bounded and drops the
  oldest frame under backpressure (the hub queue alone cannot protect the
  intermediate queue).
* :func:`heartbeat_loop` is the status-only mode: periodic pings plus a registry
  refresh that emits ``run_added`` for runs discovered after ``hello``.

Nothing here imports ``undertow.sim``/``undertow.data``; hubs read artifacts
through the U03 readers only.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from datetime import UTC, datetime

from fastapi import WebSocket, WebSocketDisconnect

from undertow.ui.paths import RunDir
from undertow.ui.schemas import RunRecord
from undertow.ui.streaming import TelemetryHub, build_frame
from undertow.ui.telemetry import count_lines

__all__ = [
    "HEARTBEAT_SECONDS",
    "PUMP_QUEUE_MAXSIZE",
    "TelemetryHubs",
    "heartbeat_loop",
    "now_iso",
    "pump_frames",
    "watch_disconnect",
]

logger = logging.getLogger("undertow.ui.ws")

#: Requested WebSocket heartbeat / status-only ping cadence (CONTRACTS §5.2).
HEARTBEAT_SECONDS: float = 20.0

#: Bound on the pump hand-off queue; the hub's own queue is bounded too.
PUMP_QUEUE_MAXSIZE: int = 256


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string ending in ``Z`` (CONTRACTS §0)."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class TelemetryHubs:
    """Lazily-created, leased :class:`TelemetryHub` cache per run directory.

    Hubs are keyed by ``str(RunDir.path)``, **not** ``run_id``: a duplicate
    manifest ``run_id`` must never make one directory's stream serve another
    directory's telemetry.  The key is the same resolved directory the REST
    routes use, so ``WS /api/stream?run_id=<alias>`` reads the alias's own
    artifact tree.
    """

    def __init__(self, poll_seconds: float) -> None:
        self._poll_seconds = float(poll_seconds)
        self._hubs: dict[str, TelemetryHub] = {}
        self._leases: dict[str, int] = {}
        self._lock = threading.Lock()

    @staticmethod
    def hub_key(run_dir: RunDir) -> str:
        """Return the cache key for ``run_dir`` (its resolved directory path)."""
        return str(run_dir.path)

    # -- lease management -------------------------------------------------
    def acquire(self, record: RunRecord, run_dir: RunDir) -> TelemetryHub:
        """Return the cached hub for ``run_dir`` and take a subscriber lease.

        The first acquisition creates the hub with ``start_offset`` at the
        current telemetry line count, so a connecting client receives deltas
        only (CONTRACTS §5.2); the frontend backfills history through
        ``GET .../telemetry``.  This method does blocking file I/O
        (``count_lines``) and is therefore awaited via ``asyncio.to_thread`` by
        the route.

        The blocking line count runs **outside** ``self._lock``: the lock is
        held only for the in-memory hub/lease maps, so a slow disk read can
        never stall a concurrent subscriber lookup or acquisition.  A
        double-check after the read keeps hub creation single-writer.
        """
        key = self.hub_key(run_dir)
        with self._lock:
            hub = self._hubs.get(key)
            if hub is not None:
                self._leases[key] = self._leases.get(key, 0) + 1
                return hub
        # Hub is absent: count the telemetry lines (disk I/O) without the lock,
        # then re-check in case another thread created it while we read.
        start_offset = (
            count_lines(run_dir.telemetry_path) if run_dir.telemetry_path else 0
        )
        with self._lock:
            hub = self._hubs.get(key)
            if hub is None:
                hub = TelemetryHub(
                    run_dir.path,
                    run_id=record.run_id,
                    telemetry_path=run_dir.telemetry_path,
                    status_path=run_dir.status_path,
                    telemetry_poll_seconds=self._poll_seconds,
                    start_offset=start_offset,
                )
                self._hubs[key] = hub
            self._leases[key] = self._leases.get(key, 0) + 1
            return hub

    @staticmethod
    def start_tail_if_idle(hub: TelemetryHub, run_id: str) -> None:
        """Start exactly one tail task per cached hub while it has subscribers."""
        if hub.pending_tasks == 0:
            hub.start_tail(run_id)

    async def release(self, key: str) -> None:
        """Drop one lease for directory ``key``; evict the hub at zero leases.

        Closing on the last disconnect stops the file-tail task (rather than
        letting it poll forever for a run nobody watches) and wakes any
        straggling subscriber queue.  A later subscriber gets a fresh hub.
        """
        hub: TelemetryHub | None = None
        with self._lock:
            leases = self._leases.get(key, 0) - 1
            if leases > 0:
                self._leases[key] = leases
                return
            self._leases.pop(key, None)
            hub = self._hubs.pop(key, None)
        if hub is not None:
            await hub.close()

    def subscriber_count(self, key: str) -> int:
        """Current subscriber count for directory ``key`` (0 when uncached)."""
        with self._lock:
            hub = self._hubs.get(key)
            return hub.subscriber_count if hub is not None else 0

    @property
    def cached_count(self) -> int:
        """Number of hubs currently cached (used by shutdown/leak tests)."""
        with self._lock:
            return len(self._hubs)

    async def close(self) -> None:
        """Close every cached hub, used by the application lifespan shutdown."""
        with self._lock:
            hubs = list(self._hubs.values())
            self._hubs.clear()
            self._leases.clear()
        for hub in hubs:
            await hub.close()


async def watch_disconnect(websocket: WebSocket) -> None:
    """Return once the client sends a disconnect (or the socket errors)."""
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
    except (WebSocketDisconnect, RuntimeError):
        return


async def pump_frames(
    websocket: WebSocket, frames: AsyncIterator[str], *, run_id: str | None
) -> None:
    """Forward hub frames to the socket, emitting a ``ping`` on idle.

    The hub iterator is drained by a dedicated task so the heartbeat timeout
    never cancels (and therefore never closes) the underlying subscription, and
    a watcher ends the loop promptly on client disconnect so the subscription is
    discarded instead of leaking.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=PUMP_QUEUE_MAXSIZE)

    async def pump() -> None:
        try:
            async for frame in frames:
                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:  # pragma: no cover - same-loop invariant
                        pass
                    else:
                        logger.warning(
                            "slow websocket subscriber: dropping oldest frame "
                            "(queue=%d)",
                            PUMP_QUEUE_MAXSIZE,
                        )
                try:
                    queue.put_nowait(frame)
                except asyncio.QueueFull:  # pragma: no cover - defensive race guard
                    logger.warning("slow websocket subscriber: dropping new frame")
        finally:
            if queue.full():
                with suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with suppress(asyncio.QueueFull):
                queue.put_nowait(None)

    reader = asyncio.create_task(pump())
    watcher = asyncio.create_task(watch_disconnect(websocket))
    try:
        while not watcher.done():
            getter = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait(
                {getter, watcher},
                timeout=HEARTBEAT_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if watcher in done:
                getter.cancel()
                with suppress(asyncio.CancelledError):
                    await getter
                return
            if getter in done:
                frame = getter.result()
                if frame is None:
                    return
                await websocket.send_text(frame)
            else:
                getter.cancel()
                with suppress(asyncio.CancelledError):
                    await getter
                await websocket.send_text(build_frame("ping", run_id, {}))
    except (WebSocketDisconnect, RuntimeError):
        return
    finally:
        reader.cancel()
        watcher.cancel()
        with suppress(asyncio.CancelledError):
            await reader
        with suppress(asyncio.CancelledError):
            await watcher


async def heartbeat_loop(
    websocket: WebSocket,
    *,
    run_id: str | None,
    refresh: Callable[[], list[RunRecord]] | None = None,
    known_run_ids: set[str] | None = None,
) -> None:
    """Status-only mode: periodic pings, plus ``run_added`` for new runs.

    ``refresh`` is a blocking registry refresh (rglob + parquet stats) and is
    therefore run through :func:`asyncio.to_thread`.  It is only used in
    status-only mode (``run_id is None``), matching CONTRACTS §5.2's run-list
    semantics; a run-specific stream only tails that run.
    """
    watcher = asyncio.create_task(watch_disconnect(websocket))
    known = known_run_ids if known_run_ids is not None else set()
    try:
        while not watcher.done():
            done, _ = await asyncio.wait({watcher}, timeout=HEARTBEAT_SECONDS)
            if watcher in done:
                return
            await websocket.send_text(build_frame("ping", run_id, {}))
            if refresh is None:
                continue
            records = await asyncio.to_thread(refresh)
            for record in records:
                if record.run_id in known:
                    continue
                known.add(record.run_id)
                await websocket.send_text(
                    build_frame("run_added", record.run_id, record)
                )
    except (WebSocketDisconnect, RuntimeError):
        return
    finally:
        watcher.cancel()
        with suppress(asyncio.CancelledError):
            await watcher
