"""U03 — file-tail pub/sub: offsets, backpressure, terminal stop and clean close."""

from __future__ import annotations

import asyncio
import json
import logging

from undertow.ui.streaming import TelemetryHub, build_frame

_RUN_ID = "ppo_s0_deadbeef"


def _line(timestep: int) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "run_id": _RUN_ID,
            "seed": 0,
            "timestep": timestep,
            "train": {"policy_loss": 0.01},
            "time": {"fps": 100.0},
        }
    )


async def _wait_for_subscribers(hub: TelemetryHub, count: int) -> None:
    for _ in range(200):
        if hub.subscriber_count == count:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"expected {count} subscribers, got {hub.subscriber_count}")


def test_tail_once_only_publishes_new_points_and_advances_offset(run_dir) -> None:
    path = run_dir(seed=0, telemetry_timesteps=[1000, 2000, 3000], status=False)
    hub = TelemetryHub(path, run_id=_RUN_ID)

    frames, offset = hub.tail_once(0)

    assert offset == 3
    decoded = [json.loads(frame) for frame in frames]
    assert [frame["type"] for frame in decoded] == ["telemetry", "telemetry", "telemetry"]
    assert [frame["data"]["timestep"] for frame in decoded] == [1000, 2000, 3000]
    assert {frame["run_id"] for frame in decoded} == {_RUN_ID}

    # Nothing new: the same offset yields no frames and does not move.
    assert hub.tail_once(offset) == ([], 3)

    with (path / "telemetry.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(_line(4000) + "\n")

    frames, offset = hub.tail_once(3)
    assert offset == 4
    assert [json.loads(frame)["data"]["timestep"] for frame in frames] == [4000]


def test_two_subscribers_and_slow_subscriber_drops_oldest(
    run_dir, caplog: logging.LogRecord
) -> None:
    path = run_dir(manifest=False, telemetry=False, status=False, rollouts=0)
    hub = TelemetryHub(path, run_id=_RUN_ID, max_queue=2)
    frames = [build_frame("telemetry", _RUN_ID, {"i": index}) for index in range(5)]

    fast_received: list[str] = []
    slow_received: list[str] = []
    slow_blocked = asyncio.Event()
    release_slow = asyncio.Event()

    async def fast_consumer() -> None:
        async for frame in hub.subscribe():
            fast_received.append(frame)
            if len(fast_received) >= len(frames):
                break

    async def slow_consumer() -> None:
        async for frame in hub.subscribe():
            slow_received.append(frame)
            if len(slow_received) == 1:
                slow_blocked.set()
                await release_slow.wait()

    async def scenario() -> None:
        fast_task = asyncio.create_task(fast_consumer())
        slow_task = asyncio.create_task(slow_consumer())
        await _wait_for_subscribers(hub, 2)

        hub.publish(frames[0])
        await asyncio.wait_for(slow_blocked.wait(), timeout=1)
        await asyncio.sleep(0)  # let the fast consumer drain its queue

        with caplog.at_level(logging.WARNING, logger="undertow.ui.streaming"):
            for frame in frames[1:]:
                hub.publish(frame)
                await asyncio.sleep(0)  # fast consumer keeps up; slow one is blocked

        await asyncio.wait_for(fast_task, timeout=1)
        assert fast_received == frames

        release_slow.set()
        for _ in range(100):
            if len(slow_received) >= 3:
                break
            await asyncio.sleep(0)

        # The slow queue held two frames, so the oldest pending frames were
        # dropped rather than blocking publish().
        assert slow_received == [frames[0], frames[3], frames[4]]

        await hub.close()
        await asyncio.wait_for(slow_task, timeout=1)

    asyncio.run(scenario())

    assert "dropping oldest frame" in caplog.text


def test_tail_stops_after_publishing_terminal_status(run_dir) -> None:
    path = run_dir(seed=0, state="done", telemetry_timesteps=[1000, 2000, 3000])
    hub = TelemetryHub(path, run_id=_RUN_ID, telemetry_poll_seconds=0.01)

    received: list[dict[str, object]] = []
    terminal_seen = asyncio.Event()

    async def collect() -> None:
        async for raw in hub.subscribe():
            frame = json.loads(raw)
            received.append(frame)
            if frame["type"] == "status" and frame["data"]["state"] == "done":
                terminal_seen.set()
                break

    async def scenario() -> None:
        collector = asyncio.create_task(collect())
        await _wait_for_subscribers(hub, 1)

        tail_task = hub.start_tail(_RUN_ID)
        await asyncio.wait_for(terminal_seen.wait(), timeout=1)
        await asyncio.wait_for(tail_task, timeout=1)

        assert tail_task.done()
        assert not tail_task.cancelled()
        assert hub.pending_tasks == 0

        types = [frame["type"] for frame in received]
        assert types.count("telemetry") == 3
        assert types[-1] == "status"
        status = received[-1]
        assert status["run_id"] == _RUN_ID
        assert status["data"]["state"] == "done"
        assert status["data"]["updated_at"].endswith("Z")

        await hub.close()
        await asyncio.wait_for(collector, timeout=1)

    asyncio.run(scenario())


def test_close_cancels_tail_and_leaves_no_pending_tasks(run_dir) -> None:
    path = run_dir(seed=0, state="running", telemetry_timesteps=[1000])
    hub = TelemetryHub(path, run_id=_RUN_ID, telemetry_poll_seconds=0.01)

    first_frame = asyncio.Event()

    async def consume() -> None:
        async for _ in hub.subscribe():
            first_frame.set()

    async def scenario() -> None:
        consumer = asyncio.create_task(consume())
        await _wait_for_subscribers(hub, 1)

        tail_task = hub.start_tail(_RUN_ID)
        await asyncio.wait_for(first_frame.wait(), timeout=1)

        await hub.close()

        assert tail_task.done()
        assert hub.pending_tasks == 0
        await asyncio.wait_for(consumer, timeout=1)

        current = asyncio.current_task()
        leftovers = [
            task for task in asyncio.all_tasks() if task is not current and not task.done()
        ]
        assert leftovers == []

    asyncio.run(scenario())
