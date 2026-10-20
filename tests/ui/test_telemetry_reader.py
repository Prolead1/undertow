"""U03 — telemetry/status readers: torn writes, offsets and forward compatibility."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from undertow.ui.schemas import RunStatusModel, SchemaVersionError, TelemetryPoint
from undertow.ui.telemetry import (
    count_lines,
    read_curve_points,
    read_points,
    read_status,
)

_RUN_ID = "ppo_s0_deadbeef"


def _point(*, timestep: int, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_id": _RUN_ID,
        "seed": 0,
        "timestep": timestep,
        "wall_time_s": float(timestep) / 1000.0,
        "ep_rew_mean": 0.5,
        "ep_len_mean": 100.0,
        "episodes": timestep // 1000,
        "train": {"policy_loss": 0.01},
        "time": {"fps": 100.0},
    }
    payload.update(overrides)
    return payload


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def _status_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": _RUN_ID,
        "seed": 0,
        "state": "running",
        "timestep": 2000,
        "total_timesteps": 100_000,
        "updated_at": "2026-10-19T18:12:00Z",
        "pid": 51234,
        "error": None,
    }


def test_empty_and_missing_files_are_empty_not_errors(tmp_path: Path) -> None:
    missing = tmp_path / "telemetry.jsonl"
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    assert read_points(missing) == ([], 0)
    assert read_points(empty) == ([], 0)
    assert read_curve_points(missing) == []
    assert count_lines(missing) == 0
    assert count_lines(empty) == 0
    assert read_status(tmp_path / "status.json") is None


def test_three_lines_and_since_line_offset(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    _write_lines(path, [json.dumps(_point(timestep=step)) for step in (1000, 2000, 3000)])

    points, next_offset = read_points(path)
    assert [point.timestep for point in points] == [1000, 2000, 3000]
    assert next_offset == 3
    assert count_lines(path) == 3

    tail, tail_offset = read_points(path, since=1)
    assert [point.timestep for point in tail] == [2000, 3000]
    assert tail_offset == 3

    limited, limited_offset = read_points(path, since=1, limit=1)
    assert [point.timestep for point in limited] == [2000]
    assert limited_offset == 2


def test_torn_trailing_line_is_not_consumed(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    first = json.dumps(_point(timestep=1000))
    second = json.dumps(_point(timestep=2000))
    third = json.dumps(_point(timestep=3000))
    # Two complete lines plus a half-written third (no trailing newline).
    path.write_text(f"{first}\n{second}\n{third[: len(third) // 2]}", encoding="utf-8")

    points, next_offset = read_points(path)
    assert [point.timestep for point in points] == [1000, 2000]
    assert next_offset == 2
    assert count_lines(path) == 2

    # The writer completes the third line; the next poll from the same offset
    # must pick it up exactly once.
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{third[len(third) // 2:]}\n")

    completed, completed_offset = read_points(path, since=next_offset)
    assert [point.timestep for point in completed] == [3000]
    assert completed_offset == 3


def test_complete_but_invalid_line_is_skipped_and_offset_advances(
    tmp_path: Path, caplog
) -> None:
    path = tmp_path / "telemetry.jsonl"
    _write_lines(
        path,
        [
            json.dumps(_point(timestep=1000)),
            json.dumps(_point(timestep=2000, seed="not-an-int")),
            json.dumps(_point(timestep=3000)),
        ],
    )

    with caplog.at_level(logging.WARNING, logger="undertow.ui.telemetry"):
        points, next_offset = read_points(path)

    assert [point.timestep for point in points] == [1000, 3000]
    assert next_offset == 3
    assert "schema-invalid telemetry line 2" in caplog.text


def test_malformed_status_returns_none_without_raising(
    tmp_path: Path, caplog
) -> None:
    path = tmp_path / "status.json"
    path.write_text("{not valid json", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="undertow.ui.telemetry"):
        status = read_status(path)

    assert status is None
    assert "unreadable JSON artifact" in caplog.text


def test_status_parses_contract_fields(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    path.write_text(json.dumps(_status_payload()), encoding="utf-8")

    status = read_status(path)

    assert isinstance(status, RunStatusModel)
    assert status.run_id == _RUN_ID
    assert status.state == "running"
    assert status.updated_at.isoformat() == "2026-10-19T18:12:00+00:00"


def test_unknown_metric_keys_survive_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    payload = _point(timestep=1000)
    payload["train"] = {"policy_loss": 0.01, "future_metric": 12.5}
    payload["time"] = {"fps": 99.0, "future_clock": 3}
    _write_lines(path, [json.dumps(payload)])

    points, next_offset = read_points(path)

    assert next_offset == 1
    [point] = points
    assert isinstance(point, TelemetryPoint)
    dumped = point.model_dump()
    assert dumped["train"]["future_metric"] == 12.5
    assert dumped["time"]["future_clock"] == 3


def test_complete_non_object_lines_are_skipped_and_offset_advances(
    tmp_path: Path, caplog
) -> None:
    path = tmp_path / "telemetry.jsonl"
    _write_lines(
        path,
        [
            json.dumps(_point(timestep=1000)),
            "42",
            '"a string"',
            "[1, 2]",
            "true",
            "null",
            json.dumps(_point(timestep=2000)),
        ],
    )

    with caplog.at_level(logging.WARNING, logger="undertow.ui.telemetry"):
        points, next_offset = read_points(path)

    assert [point.timestep for point in points] == [1000, 2000]
    assert next_offset == 7
    assert "skipping invalid telemetry line 2" in caplog.text

    # The valid trailing line still parses from a later offset (no wedged tail).
    tail, tail_offset = read_points(path, since=6)
    assert [point.timestep for point in tail] == [2000]
    assert tail_offset == 7


def test_complete_invalid_json_line_is_skipped_and_bad_status_is_rejected(
    tmp_path: Path, caplog
) -> None:
    path = tmp_path / "telemetry.jsonl"
    _write_lines(path, [json.dumps(_point(timestep=1000)), "this is not json"])

    with caplog.at_level(logging.WARNING, logger="undertow.ui.telemetry"):
        points, next_offset = read_points(path)

    assert [point.timestep for point in points] == [1000]
    assert next_offset == 2
    assert "skipping invalid telemetry line 2" in caplog.text

    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    assert read_status(status_path) is None
    assert "JSON artifact is not an object" in caplog.text

    # Schema-invalid JSON is also rejected without raising.
    status_path.write_text(json.dumps(_status_payload() | {"state": "exploded"}), encoding="utf-8")
    assert read_status(status_path) is None
    assert "invalid status sidecar" in caplog.text


def test_read_status_rejects_newer_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    path.write_text(
        json.dumps(_status_payload() | {"schema_version": 2}), encoding="utf-8"
    )

    with pytest.raises(SchemaVersionError, match="schema_version 2"):
        read_status(path)


def test_read_points_rejects_newer_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    _write_lines(path, [json.dumps(_point(timestep=1000, schema_version=2))])

    with pytest.raises(SchemaVersionError, match="schema_version 2"):
        read_points(path)
    # ``read_curve_points`` shares the same path, so it rejects too.
    with pytest.raises(SchemaVersionError, match="schema_version 2"):
        read_curve_points(path)


def test_readers_tolerate_missing_and_version_one_schema(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    missing_version = _status_payload()
    del missing_version["schema_version"]
    status_path.write_text(json.dumps(missing_version), encoding="utf-8")
    assert read_status(status_path) is not None

    telemetry_path = tmp_path / "telemetry.jsonl"
    point = _point(timestep=1000)
    del point["schema_version"]
    _write_lines(telemetry_path, [json.dumps(point)])

    points, next_offset = read_points(telemetry_path)
    assert next_offset == 1
    assert [p.timestep for p in points] == [1000]


def test_offset_beyond_end_and_zero_limit_are_stable(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    _write_lines(path, [json.dumps(_point(timestep=1000))])

    assert read_points(path, since=99) == ([], 99)
    assert read_points(path, since=0, limit=0) == ([], 0)
    assert read_curve_points(path)[0].timestep == 1000
