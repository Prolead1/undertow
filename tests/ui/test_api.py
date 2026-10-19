"""U05 — FastAPI REST + WebSocket surface (``CONTRACTS.md`` §5/§7).

Exercised end-to-end through ``fastapi.testclient.TestClient`` against synthetic
run trees (the shared ``run_dir`` factory).  The tests cover the ten groups from
the task brief plus the additive behaviours U05 owns: lazy registry refresh, CORS
opt-in, the reload config factory, and prompt WebSocket cleanup on disconnect.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from undertow.ui.app import create_app, create_app_from_env
from undertow.ui.config import UiConfig, UiConfigError

_RUN_ID = "ppo_s0_deadbeef"


def _config(tmp_path: Path, **overrides: object) -> UiConfig:
    values: dict[str, object] = {"runs_dir": tmp_path, "results_dir": tmp_path}
    values.update(overrides)
    return UiConfig(**values)  # type: ignore[arg-type]


def _telemetry_line(run_id: str, timestep: int) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "run_id": run_id,
            "seed": 0,
            "timestep": timestep,
            "train": {"policy_loss": 0.02},
            "time": {"fps": 120.0},
        }
    )


def _shift_equity(run_dir_path: Path, delta: float) -> None:
    """Offset every ``equity`` value in a run's episode-0 rollout parquet."""
    rollout = run_dir_path / "rollouts" / "episode_0.parquet"
    frame = pl.read_parquet(rollout)
    frame.with_columns((pl.col("equity") + delta).alias("equity")).write_parquet(rollout)


# 1. ----------------------------------------------------------------- health
def test_health(tmp_path: Path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "schema_version": 1}


# 2. ------------------------------------------------------------- runs / list
def test_runs_empty(tmp_path: Path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        response = client.get("/api/runs")

    assert response.status_code == 200
    assert response.json() == []


def test_runs_are_deterministically_ordered_and_filterable(
    run_dir, tmp_path: Path
) -> None:
    run_dir(experiment="exp_b", seed=0, run_id="run_b", updated_at="2026-10-19T18:12:00Z")
    run_dir(
        experiment="exp_a",
        seed=1,
        run_id="run_a",
        state="running",
        updated_at="2026-10-19T18:13:00Z",
    )

    with TestClient(create_app(_config(tmp_path))) as client:
        response = client.get("/api/runs")
        running = client.get("/api/runs", params={"state": "running"})

    assert [record["run_id"] for record in response.json()] == ["run_a", "run_b"]
    # updated_at descending: run_a (18:13) then run_b (18:12)
    assert [record["run_id"] for record in running.json()] == ["run_a"]


def test_runs_rejects_an_unknown_state_filter(tmp_path: Path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        response = client.get("/api/runs", params={"state": "runing"})

    assert response.status_code == 422


def test_refresh_discovery_does_not_rescan_every_request(
    run_dir, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("undertow.ui.app._REGISTRY_TTL_SECONDS", 3600.0)
    with TestClient(create_app(_config(tmp_path))) as client:
        assert client.get("/api/runs").json() == []
        run_dir(run_id="late_run")
        assert client.get("/api/runs").json() == []
        refreshed = client.get("/api/runs", params={"refresh": True})

    assert [record["run_id"] for record in refreshed.json()] == ["late_run"]


# 3. ------------------------------------------------------ runs / detail / 404
def test_run_detail_present_and_absent(run_dir, tmp_path: Path) -> None:
    run_dir(run_id=_RUN_ID, config_hash="cafebabe", git_commit="f00d123")

    with TestClient(create_app(_config(tmp_path))) as client:
        found = client.get(f"/api/runs/{_RUN_ID}")
        missing = client.get("/api/runs/does-not-exist")

    assert found.status_code == 200
    body = found.json()
    assert body["run_id"] == _RUN_ID
    assert body["experiment"] == "experiment_1"
    assert body["manifest"]["git_commit"] == "f00d123"
    assert missing.status_code == 404
    assert "detail" in missing.json()


def test_status_route_and_missing_sidecar(run_dir, tmp_path: Path) -> None:
    run_dir(run_id=_RUN_ID, state="running", latest_timestep=500)
    run_dir(experiment="exp_2", seed=0, run_id="no_status", status=False)

    with TestClient(create_app(_config(tmp_path))) as client:
        status = client.get(f"/api/runs/{_RUN_ID}/status")
        absent = client.get("/api/runs/no_status/status")

    assert status.status_code == 200
    assert status.json()["state"] == "running"
    assert status.json()["timestep"] == 500
    assert absent.status_code == 404


# 4. ------------------------------------------------ telemetry offsets / tail
def test_telemetry_offsets_and_done_flag(run_dir, tmp_path: Path) -> None:
    run_dir(
        run_id=_RUN_ID,
        state="done",
        telemetry_timesteps=[1000, 2000, 3000],
    )

    with TestClient(create_app(_config(tmp_path))) as client:
        first = client.get(
            f"/api/runs/{_RUN_ID}/telemetry", params={"since": 0, "limit": 2}
        )
        beyond = client.get(
            f"/api/runs/{_RUN_ID}/telemetry", params={"since": 10, "limit": 2}
        )
        negative = client.get(
            f"/api/runs/{_RUN_ID}/telemetry", params={"since": -1}
        )

    body = first.json()
    assert body["run_id"] == _RUN_ID
    assert [point["timestep"] for point in body["points"]] == [1000, 2000]
    assert body["next_offset"] == 2
    assert body["status"]["state"] == "done"
    assert body["done"] is True

    tail = beyond.json()
    assert tail["points"] == []
    assert tail["next_offset"] == 10
    assert tail["done"] is True

    assert negative.status_code == 422


def test_telemetry_without_sidecar_returns_empty(run_dir, tmp_path: Path) -> None:
    run_dir(run_id=_RUN_ID, telemetry=False, state="running")

    with TestClient(create_app(_config(tmp_path))) as client:
        response = client.get(f"/api/runs/{_RUN_ID}/telemetry")

    body = response.json()
    assert body["points"] == []
    assert body["next_offset"] == 0
    assert body["done"] is False


# 5. ---------------------------------------------------------------- figures
def test_learning_figure_and_unknown_name(run_dir, tmp_path: Path) -> None:
    run_dir(
        run_id=_RUN_ID,
        learning_curve=(0.1, 0.2, 0.3),
        telemetry_timesteps=[1000, 2000, 3000],
    )

    with TestClient(create_app(_config(tmp_path))) as client:
        response = client.get(
            f"/api/runs/{_RUN_ID}/figures/learning", params={"log_y": True}
        )
        unknown = client.get(f"/api/runs/{_RUN_ID}/figures/nope")
        unknown_run = client.get("/api/runs/nope/figures/learning")

    assert response.status_code == 200
    spec = response.json()
    assert set(spec) == {"data", "layout", "config"}
    assert spec["layout"]["template"] == "undertow_dark"
    assert spec["layout"]["yaxis"]["type"] == "log"
    assert unknown.status_code == 404
    assert unknown_run.status_code == 404


@pytest.mark.parametrize("name", ["diagnostics", "actions", "ribbon", "leaderboard"])
def test_every_registered_figure_builds(name: str, run_dir, tmp_path: Path) -> None:
    run_dir(run_id=_RUN_ID, rollouts=1)

    with TestClient(create_app(_config(tmp_path))) as client:
        response = client.get(f"/api/runs/{_RUN_ID}/figures/{name}")

    assert response.status_code == 200
    spec = response.json()
    assert set(spec) == {"data", "layout", "config"}
    assert spec["layout"]["template"] == "undertow_dark"


def test_figure_without_rollouts_renders_the_empty_state(run_dir, tmp_path: Path) -> None:
    run_dir(run_id=_RUN_ID, rollouts=0)

    with TestClient(create_app(_config(tmp_path))) as client:
        response = client.get(f"/api/runs/{_RUN_ID}/figures/ribbon")

    assert response.status_code == 200
    assert response.json()["data"] == []


def test_diagnostics_metric_is_validated_and_points_are_capped(
    run_dir, tmp_path: Path
) -> None:
    run_dir(
        run_id=_RUN_ID,
        telemetry_timesteps=[1000, 2000, 3000, 4000, 5000],
    )

    with TestClient(create_app(_config(tmp_path, max_points=2))) as client:
        unknown = client.get(
            f"/api/runs/{_RUN_ID}/figures/diagnostics",
            params={"metric": "not_a_metric"},
        )
        valid = client.get(
            f"/api/runs/{_RUN_ID}/figures/diagnostics",
            params={"metric": "policy_loss"},
        )

    assert unknown.status_code == 422
    assert "unknown diagnostic metric" in unknown.json()["detail"]
    assert valid.status_code == 200
    # ``max_points`` bounds the read: only the two most recent points are read.
    assert valid.json()["data"][0]["x"] == [4000, 5000]


def test_curve_route_and_absent_curve(run_dir, tmp_path: Path) -> None:
    run_dir(
        run_id=_RUN_ID,
        learning_curve=(0.1, 0.2, 0.3),
        telemetry_timesteps=[1000, 2000, 3000],
    )
    run_dir(experiment="exp_2", seed=0, run_id="no_curve", manifest=False)

    with TestClient(create_app(_config(tmp_path))) as client:
        curve = client.get(f"/api/runs/{_RUN_ID}/curve")
        absent = client.get("/api/runs/no_curve/curve")

    assert curve.status_code == 200
    body = curve.json()
    assert body["run_id"] == _RUN_ID
    assert body["timesteps"] == [1000, 2000, 3000]
    assert body["mean"] == [0.1, 0.2, 0.3]
    assert absent.status_code == 404


# 6. --------------------------------------------------------------- rollouts
def test_rollouts_listing_and_series(run_dir, tmp_path: Path) -> None:
    run_dir(run_id=_RUN_ID, seed=0, rollouts=2)

    with TestClient(create_app(_config(tmp_path))) as client:
        listing = client.get(f"/api/runs/{_RUN_ID}/rollouts")
        series = client.get(f"/api/rollouts/{_RUN_ID}/0/0")
        downsampled = client.get(
            f"/api/rollouts/{_RUN_ID}/0/0", params={"max_points": 2}
        )
        missing_episode = client.get(f"/api/rollouts/{_RUN_ID}/0/99")
        wrong_seed = client.get(f"/api/rollouts/{_RUN_ID}/9/0")
        unknown_run = client.get("/api/rollouts/nope/0/0")

    assert listing.status_code == 200
    summaries = listing.json()
    assert [summary["episode"] for summary in summaries] == [0, 1]
    assert summaries[0]["n_steps"] == 3
    assert summaries[0]["regime"] == "bull"

    assert series.status_code == 200
    body = series.json()
    assert body["run_id"] == _RUN_ID
    assert body["seed"] == 0
    assert body["episode"] == 0
    assert len(body["rows"]) == 3
    assert body["rows"][0]["timestamp"].endswith("Z")
    assert len(downsampled.json()["rows"]) == 2
    assert downsampled.json()["downsampled"] is True

    assert missing_episode.status_code == 404
    assert wrong_seed.status_code == 404
    assert unknown_run.status_code == 404


def test_duplicate_run_id_resolves_to_the_same_directory_everywhere(
    run_dir, tmp_path: Path
) -> None:
    run_dir(
        experiment="exp_a",
        seed=0,
        run_id="duplicate",
        latest_timestep=111,
        telemetry_timesteps=[1000],
        learning_curve=(0.1, 0.2, 0.3),
        best_reward=0.1,
        rollouts=0,
    )
    run_dir(
        experiment="exp_b",
        seed=0,
        run_id="duplicate",
        latest_timestep=222,
        telemetry_timesteps=[2000],
        learning_curve=(9.0, 9.0, 9.0),
        best_reward=0.9,
        rollouts=2,
    )

    with TestClient(create_app(_config(tmp_path))) as client:
        detail_a = client.get("/api/runs/exp_a/seed_0")
        encoded_b = client.get("/api/runs/exp_b%2Fseed_0")
        detail_b = client.get("/api/runs/exp_b/seed_0")
        status_a = client.get("/api/runs/exp_a/seed_0/status")
        status_b = client.get("/api/runs/exp_b/seed_0/status")
        tail_a = client.get("/api/runs/exp_a/seed_0/telemetry")
        tail_b = client.get("/api/runs/exp_b/seed_0/telemetry")
        rollouts_a = client.get("/api/runs/exp_a/seed_0/rollouts")
        rollouts_b = client.get("/api/runs/exp_b/seed_0/rollouts")
        rollout_a = client.get("/api/rollouts/exp_a/seed_0/0/0")
        rollout_b = client.get("/api/rollouts/exp_b/seed_0/0/0")
        ribbon_a = client.get("/api/runs/exp_a/seed_0/figures/ribbon")
        ribbon_b = client.get("/api/runs/exp_b/seed_0/figures/ribbon")
        # Curve/learning aggregate the resolved directory's experiment, so the
        # raw id and the exp_a alias return exp_a's numbers while the exp_b
        # alias returns exp_b's.
        curve_raw = client.get("/api/runs/duplicate/curve")
        curve_a = client.get("/api/runs/exp_a/seed_0/curve")
        curve_b = client.get("/api/runs/exp_b/seed_0/curve")
        learning_a = client.get("/api/runs/exp_a/seed_0/figures/learning")
        learning_b = client.get("/api/runs/exp_b/seed_0/figures/learning")
        # The leaderboard is global, but the resolved directory must be the
        # representative for its run_id (never omitted behind a collision).
        leaderboard_a = client.get(
            "/api/runs/exp_a/seed_0/figures/leaderboard",
            params={"metric": "best_reward"},
        )
        leaderboard_b = client.get(
            "/api/runs/exp_b/seed_0/figures/leaderboard",
            params={"metric": "best_reward"},
        )
        compare_b = client.post(
            "/api/compare", json={"run_ids": ["exp_b/seed_0"]}
        )

    assert detail_a.json()["experiment"] == "exp_a"
    assert detail_a.json()["latest_timestep"] == 111
    assert encoded_b.json()["experiment"] == "exp_b"
    assert detail_b.json()["experiment"] == "exp_b"
    assert detail_b.json()["latest_timestep"] == 222

    assert status_a.json()["timestep"] == 111
    assert status_b.json()["timestep"] == 222
    assert [point["timestep"] for point in tail_a.json()["points"]] == [1000]
    assert [point["timestep"] for point in tail_b.json()["points"]] == [2000]

    assert rollouts_a.json() == []
    assert [summary["episode"] for summary in rollouts_b.json()] == [0, 1]
    assert rollout_a.status_code == 404
    assert rollout_b.status_code == 200
    assert ribbon_a.json()["data"] == []
    assert ribbon_b.json()["data"] != []

    # The raw canonical id and the exp_a alias aggregate exp_a's directory.
    for curve in (curve_raw, curve_a):
        assert curve.status_code == 200
        assert curve.json()["mean"] == [0.1, 0.2, 0.3]
        assert curve.json()["timesteps"] == [1000, 2000, 3000]
    assert curve_raw.json()["run_id"] == "duplicate"
    assert curve_a.json()["run_id"] == "exp_a/seed_0"

    # The exp_b alias aggregates its own directory and echoes the alias as its
    # cache key, so it can never collide with the canonical entry.
    assert curve_b.status_code == 200
    assert curve_b.json()["run_id"] == "exp_b/seed_0"
    assert curve_b.json()["mean"] == [9.0, 9.0, 9.0]
    assert curve_b.json()["timesteps"] == [2000, 4000, 6000]

    assert learning_a.status_code == 200
    assert learning_b.status_code == 200
    assert max(learning_a.json()["data"][0]["y"]) == 0.3
    assert max(learning_b.json()["data"][0]["y"]) == 9.0

    assert leaderboard_a.json()["data"][0]["x"] == [0.1]
    assert leaderboard_b.json()["data"][0]["x"] == [0.9]

    assert compare_b.json()["records"][0]["experiment"] == "exp_b"


def test_duplicate_run_id_streams_the_aliased_directory(
    run_dir, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A colliding id must keep one telemetry hub per resolved directory."""
    monkeypatch.setattr("undertow.ui.ws.HEARTBEAT_SECONDS", 0.05)
    path_a = run_dir(
        experiment="exp_a",
        seed=0,
        run_id="duplicate",
        state="running",
        telemetry_timesteps=[1000],
    )
    path_b = run_dir(
        experiment="exp_b",
        seed=0,
        run_id="duplicate",
        state="running",
        telemetry_timesteps=[9000],
    )
    config = _config(tmp_path, telemetry_poll_seconds=0.01)

    with TestClient(create_app(config)) as client:
        # Keep exp_a's subscription open so a run_id-keyed cache would reuse its
        # hub for exp_b; the alias must still tail exp_b's directory.
        with client.websocket_connect("/api/stream?run_id=exp_a/seed_0") as websocket_a:
            assert websocket_a.receive_json()["type"] == "hello"
            _receive_until(websocket_a, "status")
            with client.websocket_connect("/api/stream?run_id=exp_b/seed_0") as websocket_b:
                assert websocket_b.receive_json()["type"] == "hello"
                _receive_until(websocket_b, "status")

                with (path_b / "telemetry.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(_telemetry_line("duplicate", 9100) + "\n")

                frame = _receive_until(websocket_b, "telemetry")
                assert frame["data"]["timestep"] == 9100

        # And a lone exp_a connection reads exp_a's lines.
        with client.websocket_connect("/api/stream?run_id=exp_a/seed_0") as websocket_a:
            assert websocket_a.receive_json()["type"] == "hello"
            _receive_until(websocket_a, "status")
            with (path_a / "telemetry.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(_telemetry_line("duplicate", 1100) + "\n")
            frame = _receive_until(websocket_a, "telemetry")
            assert frame["data"]["timestep"] == 1100


def test_compare_distinguishes_colliding_run_id_aliases(
    run_dir, tmp_path: Path
) -> None:
    run_dir(experiment="exp_a", seed=0, run_id="dup", rollouts=1)
    path_b = run_dir(experiment="exp_b", seed=0, run_id="dup", rollouts=1)
    # Distinct equity per directory, so a swapped label/data pairing is caught
    # (the shared fixture writes the same equity into every run tree).
    _shift_equity(path_b, 500.0)

    with TestClient(create_app(_config(tmp_path))) as client:
        response = client.post(
            "/api/compare", json={"run_ids": ["exp_a/seed_0", "exp_b/seed_0"]}
        )

    assert response.status_code == 200
    body = response.json()
    assert [record["experiment"] for record in body["records"]] == ["exp_a", "exp_b"]
    traces = body["figures"]["compare"]["data"]
    assert {trace["name"]: trace["y"] for trace in traces} == {
        "exp_a/seed_0": [10000.0, 10001.0, 10002.0],
        "exp_b/seed_0": [10500.0, 10501.0, 10502.0],
    }


def test_compare_dedupes_raw_id_and_alias_of_one_directory(
    run_dir, tmp_path: Path
) -> None:
    run_dir(experiment="exp_a", seed=0, run_id="dup", rollouts=1)
    run_dir(experiment="exp_b", seed=0, run_id="dup", rollouts=1)

    with TestClient(create_app(_config(tmp_path))) as client:
        response = client.post(
            "/api/compare",
            json={"run_ids": ["dup", "exp_a/seed_0", "exp_b/seed_0"]},
        )

    assert response.status_code == 200
    body = response.json()
    # ``dup`` and ``exp_a/seed_0`` resolve to the same directory, so they must
    # collapse to one record and one overlay trace; deduping on the request
    # string would leave two records behind a single (overwritten) trace.
    assert [record["experiment"] for record in body["records"]] == ["exp_a", "exp_b"]
    assert len(body["figures"]["compare"]["data"]) == 2


def test_rollouts_empty_when_none_recorded(run_dir, tmp_path: Path) -> None:
    run_dir(run_id=_RUN_ID, rollouts=0)

    with TestClient(create_app(_config(tmp_path))) as client:
        listing = client.get(f"/api/runs/{_RUN_ID}/rollouts")
        series = client.get(f"/api/rollouts/{_RUN_ID}/0/0")

    assert listing.status_code == 200
    assert listing.json() == []
    assert series.status_code == 404


# 7. ---------------------------------------------------------------- compare
def test_compare_six_ids_and_validation(run_dir, tmp_path: Path) -> None:
    ids: list[str] = []
    for index in range(7):
        run_id = f"run_{index}"
        run_dir(experiment="exp", seed=index, run_id=run_id)
        ids.append(run_id)

    with TestClient(create_app(_config(tmp_path))) as client:
        six = client.post("/api/compare", json={"run_ids": ids[:6], "metric": "sharpe"})
        none = client.post("/api/compare", json={"run_ids": []})
        seven = client.post("/api/compare", json={"run_ids": ids})
        bad_metric = client.post(
            "/api/compare", json={"run_ids": ids[:2], "metric": "nope"}
        )
        unknown = client.post("/api/compare", json={"run_ids": ["unknown"]})

    assert six.status_code == 200
    body = six.json()
    assert len(body["records"]) == 6
    assert "compare" in body["figures"]
    assert body["figures"]["compare"]["layout"]["template"] == "undertow_dark"
    assert len(body["figures"]["compare"]["data"]) == 6

    assert none.status_code == 422
    assert seven.status_code == 422
    assert bad_metric.status_code == 422
    assert unknown.status_code == 404


def test_compare_accepts_experiment_seed_reference(run_dir, tmp_path: Path) -> None:
    run_dir(experiment="experiment_1", seed=0, run_id=_RUN_ID)

    with TestClient(create_app(_config(tmp_path))) as client:
        response = client.post(
            "/api/compare", json={"run_ids": ["experiment_1/seed_0"]}
        )

    assert response.status_code == 200
    assert response.json()["records"][0]["run_id"] == _RUN_ID


# 8. ------------------------------------------------------------ WebSocket
def _receive_until(websocket, frame_type: str, *, attempts: int = 50) -> dict[str, object]:
    """Read frames until ``frame_type`` arrives, failing after ``attempts``."""
    seen: list[str] = []
    for _ in range(attempts):
        frame = websocket.receive_json()
        seen.append(str(frame["type"]))
        if frame["type"] == frame_type:
            return frame
    raise AssertionError(f"never received a {frame_type!r} frame; saw {seen}")


def test_websocket_hello_then_live_telemetry(run_dir, tmp_path: Path) -> None:
    path = run_dir(
        run_id=_RUN_ID, state="running", telemetry_timesteps=[1000]
    )
    config = _config(tmp_path, telemetry_poll_seconds=0.01)

    with TestClient(create_app(config)) as client:
        with client.websocket_connect(f"/api/stream?run_id={_RUN_ID}") as websocket:
            hello = websocket.receive_json()
            assert hello["type"] == "hello"
            assert hello["run_id"] is None
            assert hello["data"]["server_time"].endswith("Z")
            assert [run["run_id"] for run in hello["data"]["runs"]] == [_RUN_ID]

            # Wait until the tail is live (it announces the current status)
            # before appending, so the new line cannot race the start offset.
            _receive_until(websocket, "status")
            with (path / "telemetry.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(_telemetry_line(_RUN_ID, 2000) + "\n")

            frame = _receive_until(websocket, "telemetry")
            assert frame["run_id"] == _RUN_ID
            assert frame["data"]["timestep"] == 2000

        # Closing the socket must not break the server.
        assert client.get("/api/health").status_code == 200


def test_websocket_close_stops_the_tail_and_evicts_the_hub(
    run_dir, tmp_path: Path
) -> None:
    path = run_dir(run_id=_RUN_ID, state="running", telemetry_timesteps=[1000])
    app = create_app(_config(tmp_path, telemetry_poll_seconds=0.01))
    key = str(path)

    with TestClient(app) as client:
        with client.websocket_connect(f"/api/stream?run_id={_RUN_ID}") as websocket:
            assert websocket.receive_json()["type"] == "hello"
            _receive_until(websocket, "status")
            assert app.state.hubs.subscriber_count(key) == 1

        deadline = time.monotonic() + 2.0
        while app.state.hubs.cached_count and time.monotonic() < deadline:
            time.sleep(0.01)

        assert app.state.hubs.cached_count == 0
        assert app.state.hubs.subscriber_count(key) == 0
        assert client.get("/api/health").status_code == 200


def test_websocket_status_only_emits_run_added(
    run_dir, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("undertow.ui.ws.HEARTBEAT_SECONDS", 0.01)
    monkeypatch.setattr("undertow.ui.app._REGISTRY_TTL_SECONDS", 0.0)

    with (
        TestClient(create_app(_config(tmp_path))) as client,
        client.websocket_connect("/api/stream") as websocket,
    ):
        hello = websocket.receive_json()
        assert hello["data"]["runs"] == []

        run_dir(run_id=_RUN_ID)
        added = _receive_until(websocket, "run_added", attempts=50)

    assert added["run_id"] == _RUN_ID
    assert added["data"]["run_id"] == _RUN_ID


def test_websocket_unknown_run_emits_error(tmp_path: Path) -> None:
    with (
        TestClient(create_app(_config(tmp_path))) as client,
        client.websocket_connect("/api/stream?run_id=nope") as websocket,
    ):
        assert websocket.receive_json()["type"] == "hello"
        error = websocket.receive_json()

    assert error["type"] == "error"
    assert "unknown run" in error["data"]["message"]


def test_websocket_status_only_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("undertow.ui.ws.HEARTBEAT_SECONDS", 0.01)

    with (
        TestClient(create_app(_config(tmp_path))) as client,
        client.websocket_connect("/api/stream") as websocket,
    ):
        assert websocket.receive_json()["type"] == "hello"
        ping = websocket.receive_json()

    assert ping["type"] == "ping"
    assert ping["run_id"] is None


# 9. ------------------------------------------------------------ static / hint
def test_root_hint_without_web_dist(tmp_path: Path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "web_dist" in response.json()["message"]
    assert response.json()["openapi"] == "/openapi.json"


def test_spa_fallback_and_assets(tmp_path: Path) -> None:
    dist = tmp_path / "web"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>cockpit</title>", encoding="utf-8")
    (assets / "app.js").write_text("console.log('ui')", encoding="utf-8")
    config = _config(tmp_path, web_dist=dist)

    with TestClient(create_app(config)) as client:
        fallback = client.get("/some/deep/route")
        asset = client.get("/assets/app.js")
        unknown_api = client.get("/api/nope")
        health = client.get("/api/health")

    assert fallback.status_code == 200
    assert "cockpit" in fallback.text
    assert asset.status_code == 200
    assert "console.log" in asset.text
    assert unknown_api.status_code == 404
    assert health.json() == {"status": "ok", "schema_version": 1}


# 10. ----------------------------------------------------------- OpenAPI paths
def test_openapi_includes_every_contract_path(tmp_path: Path) -> None:
    expected = {
        "/api/health",
        "/api/runs",
        "/api/runs/{run_id}",
        "/api/runs/{run_id}/status",
        "/api/runs/{run_id}/telemetry",
        "/api/runs/{run_id}/curve",
        "/api/runs/{run_id}/figures/{name}",
        "/api/runs/{run_id}/rollouts",
        "/api/rollouts/{run_id}/{seed}/{episode}",
        "/api/compare",
    }

    with TestClient(create_app(_config(tmp_path))) as client:
        schema = client.get("/openapi.json").json()

    assert expected <= set(schema["paths"])
    figure_schema = schema["paths"]["/api/runs/{run_id}/figures/{name}"]["get"][
        "responses"
    ]["200"]["content"]["application/json"]["schema"]
    assert figure_schema["$ref"].endswith("/FigureSpecModel")
    assert "FigureSpecModel" in schema["components"]["schemas"]
    # WebSockets are not part of OpenAPI, but the route is still registered.
    assert "/api/stream" in {route.path for route in create_app(_config(tmp_path)).routes}


# ------------------------------------------------------- additive behaviours
def test_cors_only_when_origins_are_configured(tmp_path: Path) -> None:
    origin = "http://localhost:5173"
    with TestClient(create_app(_config(tmp_path))) as client:
        disabled = client.get("/api/health", headers={"Origin": origin})

    with TestClient(
        create_app(_config(tmp_path, cors_origins=(origin,)))
    ) as client:
        enabled = client.get("/api/health", headers={"Origin": origin})

    assert "access-control-allow-origin" not in disabled.headers
    assert enabled.headers["access-control-allow-origin"] == origin


def test_create_app_from_env_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "runs_dir": str(tmp_path),
        "results_dir": str(tmp_path),
        "host": "127.0.0.1",
        "port": 8123,
        "telemetry_poll_seconds": 0.5,
        "max_points": 123,
        "cors_origins": ["http://localhost:5173"],
        "web_dist": None,
    }
    monkeypatch.setenv("UNDERTOW_UI_CONFIG_JSON", json.dumps(payload))

    app = create_app_from_env()

    assert app.state.config == UiConfig(
        runs_dir=tmp_path,
        results_dir=tmp_path,
        port=8123,
        telemetry_poll_seconds=0.5,
        max_points=123,
        cors_origins=("http://localhost:5173",),
    )


@pytest.mark.parametrize(
    "payload",
    [
        "{not json",
        "[]",
        json.dumps({"runs_dir": "/tmp"}),
        json.dumps({"runs_dir": "/tmp", "results_dir": "/tmp", "host": "x"}),
    ],
)
def test_create_app_from_env_rejects_malformed_payload(
    monkeypatch: pytest.MonkeyPatch, payload: str
) -> None:
    monkeypatch.setenv("UNDERTOW_UI_CONFIG_JSON", payload)

    with pytest.raises(UiConfigError, match="UNDERTOW_UI_CONFIG_JSON"):
        create_app_from_env()


def test_create_app_from_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UNDERTOW_UI_CONFIG_JSON", raising=False)

    app = create_app_from_env()

    assert app.state.config == UiConfig()


def test_registry_ttl_refresh_is_bounded(
    run_dir, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("undertow.ui.app._REGISTRY_TTL_SECONDS", 0.0)
    run_dir(run_id=_RUN_ID)

    with TestClient(create_app(_config(tmp_path))) as client:
        first = client.get("/api/runs")
        run_dir(experiment="experiment_2", seed=1, run_id="late_run")
        time.sleep(0.01)
        second = client.get("/api/runs")

    assert first.status_code == 200
    assert second.status_code == 200
    assert [record["run_id"] for record in first.json()] == [_RUN_ID]
    assert {record["run_id"] for record in second.json()} == {_RUN_ID, "late_run"}
