"""U05 — WebSocket pump backpressure and heartbeat (``undertow.ui.ws``).

:func:`pump_frames` owns a bounded hand-off queue between the hub iterator and
the socket.  These tests drive it directly with a fake socket so the
drop-oldest backpressure path and the idle ``ping`` heartbeat are deterministic
(``TestClient`` cannot easily stall ``send_text``).
"""

from __future__ import annotations

import asyncio
import json
import logging

from undertow.ui.ws import pump_frames


class _FakeSocket:
    """Minimal ASGI websocket double: records sends, blocks/receives on command."""

    def __init__(self, *, block_first_send: bool = False) -> None:
        self.sent: list[str] = []
        self.disconnected = asyncio.Event()
        self.sending = asyncio.Event()
        self.release = asyncio.Event()
        self._block_first_send = block_first_send
        self._first_send = True

    async def send_text(self, text: str) -> None:
        self.sent.append(text)
        if self._block_first_send and self._first_send:
            self._first_send = False
            self.sending.set()
            await self.release.wait()

    async def receive(self) -> dict[str, str]:
        await self.disconnected.wait()
        return {"type": "websocket.disconnect"}


def test_pump_frames_drops_oldest_and_keeps_newest(
    monkeypatch, caplog: logging.LogRecord
) -> None:
    monkeypatch.setattr("undertow.ui.ws.PUMP_QUEUE_MAXSIZE", 2)
    frames = [f"frame-{index}" for index in range(5)]
    socket = _FakeSocket(block_first_send=True)

    async def source():
        for frame in frames:
            yield frame

    async def scenario() -> None:
        task = asyncio.create_task(pump_frames(socket, source(), run_id="dup"))
        # The socket stalls on its first send, so the producer fills the
        # bounded queue and must drop the oldest frames.
        await asyncio.wait_for(socket.sending.wait(), timeout=1)
        socket.release.set()
        await asyncio.wait_for(task, timeout=1)

    with caplog.at_level(logging.WARNING, logger="undertow.ui.ws"):
        asyncio.run(scenario())

    # Only the newest frame survives the drops (the end marker follows it).
    assert socket.sent == [frames[-1]]
    assert "dropping oldest frame" in caplog.text


def test_pump_frames_emits_heartbeat_ping(monkeypatch) -> None:
    monkeypatch.setattr("undertow.ui.ws.HEARTBEAT_SECONDS", 0.01)
    socket = _FakeSocket()
    stop = asyncio.Event()

    async def source():
        await stop.wait()
        if False:  # pragma: no cover - makes this an async generator
            yield ""

    async def scenario() -> None:
        task = asyncio.create_task(pump_frames(socket, source(), run_id="run-1"))
        for _ in range(200):
            if socket.sent:
                break
            await asyncio.sleep(0.005)
        assert socket.sent, "expected an idle heartbeat ping"
        socket.disconnected.set()
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(scenario())

    frame = json.loads(socket.sent[-1])
    assert frame["type"] == "ping"
    assert frame["run_id"] == "run-1"


def test_hubs_acquire_counts_lines_outside_the_lock(
    run_dir, tmp_path, monkeypatch
) -> None:
    """The blocking line count must not run while holding the hub lock.

    Holding ``TelemetryHubs._lock`` across a disk read would stall every other
    subscriber lookup/acquisition; the lock may cover only the in-memory maps.
    """
    import undertow.ui.ws as ws
    from undertow.ui.registry import RunRegistry

    run_dir(run_id="dup", telemetry_timesteps=[1000])
    resolved = RunRegistry(tmp_path).resolve("dup")
    assert resolved is not None
    record, run = resolved
    hubs = ws.TelemetryHubs(0.01)
    lock_states: list[bool] = []
    real_count_lines = ws.count_lines

    def spy(path):
        lock_states.append(hubs._lock.locked())
        return real_count_lines(path)

    monkeypatch.setattr("undertow.ui.ws.count_lines", spy)

    first = hubs.acquire(record, run)
    second = hubs.acquire(record, run)

    assert first is second
    # Only the first (hub-creating) acquisition reads the file, and it does so
    # with the lock released.
    assert lock_states == [False]


def test_hubs_acquire_reuses_a_hub_created_during_the_line_count(
    run_dir, tmp_path, monkeypatch
) -> None:
    """A hub created while ``count_lines`` runs must not be replaced.

    This exercises the double-checked second lock acquisition: if another
    subscriber creates the hub between the first lookup and the re-check, the
    cached hub wins and both leases are accounted for.
    """
    import undertow.ui.ws as ws
    from undertow.ui.registry import RunRegistry

    run_dir(run_id="dup", telemetry_timesteps=[1000])
    resolved = RunRegistry(tmp_path).resolve("dup")
    assert resolved is not None
    record, run = resolved
    hubs = ws.TelemetryHubs(0.01)
    real_count_lines = ws.count_lines
    created: list[ws.TelemetryHub] = []
    calls = 0

    def spy(path):
        nonlocal calls
        calls += 1
        if calls == 1:
            # Simulate a concurrent subscriber winning the creation race while
            # the outer ``acquire`` is still counting lines.
            created.append(hubs.acquire(record, run))
        return real_count_lines(path)

    monkeypatch.setattr("undertow.ui.ws.count_lines", spy)

    hub = hubs.acquire(record, run)
    key = ws.TelemetryHubs.hub_key(run)

    assert calls == 2
    assert len(created) == 1
    assert hub is created[0]
    assert hubs._leases[key] == 2
