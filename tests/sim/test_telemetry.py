"""U00 — training telemetry sidecars (``telemetry.jsonl`` + ``status.json``).

These tests exercise real behaviour on tiny SB3 runs: atomic status writes that
round-trip every contract field, exactly one JSONL point per ``log_freq_steps``
with increasing timesteps, the terminal ``done``/``failed`` transitions, the
opt-out flag, and the SB3 metric extraction rules.
"""

from __future__ import annotations

import json
import os
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3 import PPO

from undertow.sim.config import EpisodeConfig, SimConfig, TrainingConfig
from undertow.sim.train import (
    RunStatus,
    TelemetryCallback,
    ppo_runner,
    train_single_seed,
)
from undertow.sim.train.telemetry import (
    SCHEMA_VERSION,
    STATUS_FILENAME,
    TELEMETRY_FILENAME,
    _to_jsonable,
    write_status,
)


class _BanditEnv(gym.Env):
    """Deterministic one-step bandit (mirrors ``tests/sim/test_training.py``)."""

    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)
    action_space = spaces.Discrete(2)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        del seed, options
        return np.zeros(2, dtype=np.float64), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        return np.zeros(2, dtype=np.float64), 1.0, True, False, {}


def _training_config(
    *,
    total_timesteps: int = 256,
    log_freq_steps: int = 64,
    checkpoint_freq_steps: int = 10_000,
    telemetry: bool = True,
) -> SimConfig:
    """A tiny, fast CPU config that divides evenly for telemetry assertions."""
    return SimConfig(
        episode=EpisodeConfig(duration_days=1, step_minutes=10),
        training=TrainingConfig(
            seeds=(0,),
            total_timesteps=total_timesteps,
            log_freq_steps=log_freq_steps,
            checkpoint_freq_steps=checkpoint_freq_steps,
            parallel_envs=1,
            telemetry=telemetry,
        ),
    )


def _status(
    *,
    state: str = "running",
    timestep: int = 0,
    error: str | None = None,
    updated_at: datetime | None = None,
) -> RunStatus:
    return RunStatus(
        schema_version=SCHEMA_VERSION,
        run_id="ppo_s0_deadbeef",
        seed=0,
        state=state,  # type: ignore[arg-type]
        timestep=timestep,
        total_timesteps=1_000_000,
        updated_at=updated_at or datetime(2026, 10, 19, 18, 12, tzinfo=UTC),
        pid=51234,
        error=error,
    )


def _read_status(tmp_path: Path) -> dict:
    return json.loads((tmp_path / STATUS_FILENAME).read_text())


def _reject_json_constant(name: str) -> object:
    """``json.loads`` hook that rejects Python's non-standard NaN/Infinity."""
    raise ValueError(f"non-finite JSON constant: {name}")


def _strict_json_lines(path: Path) -> list[dict]:
    return [
        json.loads(line, parse_constant=_reject_json_constant)
        for line in path.read_text().splitlines()
    ]


class TestTrainingConfigAdditions:
    def test_telemetry_flags_default_additively(self) -> None:
        config = TrainingConfig()
        assert config.telemetry is True
        # Rollouts are on by default so the replay views have data (U06/U11).
        assert config.record_rollouts is True
        assert config.rollout_max_episodes == 5

    def test_telemetry_flag_is_constructible(self) -> None:
        assert TrainingConfig(telemetry=False, record_rollouts=True).telemetry is False


# ---------------------------------------------------------------------------
# write_status
# ---------------------------------------------------------------------------


class TestWriteStatus:
    def test_round_trips_every_field_and_leaves_no_temp_file(self, tmp_path: Path) -> None:
        path = tmp_path / STATUS_FILENAME
        write_status(path, _status(state="running", timestep=20_000))

        payload = json.loads(path.read_text())
        assert payload == {
            "schema_version": 1,
            "run_id": "ppo_s0_deadbeef",
            "seed": 0,
            "state": "running",
            "timestep": 20_000,
            "total_timesteps": 1_000_000,
            "updated_at": "2026-10-19T18:12:00Z",
            "pid": 51234,
            "error": None,
        }
        assert path.read_text().endswith("\n")
        # The temp file was os.replace-d away: only the final artifact remains.
        assert [p.name for p in tmp_path.iterdir()] == [STATUS_FILENAME]

    def test_replaces_existing_status_in_place(self, tmp_path: Path) -> None:
        path = tmp_path / STATUS_FILENAME
        write_status(path, _status(state="running", timestep=1))
        write_status(path, _status(state="done", timestep=2))

        payload = json.loads(path.read_text())
        assert payload["state"] == "done"
        assert payload["timestep"] == 2

    def test_naive_datetime_is_treated_as_utc(self, tmp_path: Path) -> None:
        path = tmp_path / STATUS_FILENAME
        write_status(path, _status(updated_at=datetime(2026, 1, 2, 3, 4, 5)))
        assert json.loads(path.read_text())["updated_at"] == "2026-01-02T03:04:05Z"

    def test_failed_replace_cleans_up_temp_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / STATUS_FILENAME

        def _boom(src: object, dst: object) -> None:
            raise OSError("replace failed")

        monkeypatch.setattr(os, "replace", _boom)
        with pytest.raises(OSError, match="replace failed"):
            write_status(path, _status())
        assert not path.exists()
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# full training runs
# ---------------------------------------------------------------------------


class TestTelemetryDuringTraining:
    def test_writes_one_point_per_log_frequency(self, tmp_path: Path) -> None:
        config = _training_config(total_timesteps=256, log_freq_steps=64)
        manifest = train_single_seed(_BanditEnv, config, seed=0, log_dir=tmp_path)

        # Each line is strict JSON: ``_reject_json_constant`` rejects Python's
        # non-standard NaN/Infinity while parsing.
        points = _strict_json_lines(tmp_path / TELEMETRY_FILENAME)
        assert len(points) == 256 // 64
        timesteps = [point["timestep"] for point in points]
        assert timesteps == [64, 128, 192, 256]
        assert timesteps == sorted(timesteps)

        for point in points:
            assert point["schema_version"] == SCHEMA_VERSION
            assert point["run_id"] == manifest.run_id
            assert point["seed"] == 0
            assert isinstance(point["wall_time_s"], float)
            assert isinstance(point["train"], dict)
            assert isinstance(point["time"], dict)
        # SB3 train metrics really flow through at the end of a run. ``time/*``
        # is cleared by SB3's ``Logger.dump`` before the rollout callback, so the
        # callback shadows ``logger.record`` to keep it; the shadow cache must
        # therefore carry fps by the end of the run.
        assert "policy_loss" in points[-1]["train"]
        assert "n_updates" in points[-1]["train"]
        assert "fps" in points[-1]["time"]

    def test_terminal_status_is_done_after_artifacts_exist(self, tmp_path: Path) -> None:
        config = _training_config(total_timesteps=256, log_freq_steps=64)
        manifest = train_single_seed(_BanditEnv, config, seed=0, log_dir=tmp_path)

        payload = _read_status(tmp_path)
        assert payload["state"] == "done"
        assert payload["timestep"] == 256
        assert payload["total_timesteps"] == 256
        assert payload["run_id"] == manifest.run_id
        assert payload["error"] is None
        # ``done`` implies the run's artifacts are on disk.
        assert (tmp_path / "manifest.json").exists()
        assert manifest.checkpoint_path.exists()

    def test_exception_marks_failed_and_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(self: object, *args: object, **kwargs: object) -> None:
            raise RuntimeError("training exploded")

        monkeypatch.setattr(PPO, "learn", _boom)
        config = _training_config()
        with pytest.raises(RuntimeError, match="training exploded"):
            train_single_seed(_BanditEnv, config, seed=0, log_dir=tmp_path)

        payload = _read_status(tmp_path)
        assert payload["state"] == "failed"
        assert payload["error"] == "training exploded"

    def test_telemetry_false_writes_no_sidecars(self, tmp_path: Path) -> None:
        config = _training_config(telemetry=False)
        manifest = train_single_seed(_BanditEnv, config, seed=0, log_dir=tmp_path)

        assert not (tmp_path / TELEMETRY_FILENAME).exists()
        assert not (tmp_path / STATUS_FILENAME).exists()
        # Pre-existing artifacts are unaffected.
        assert (tmp_path / "manifest.json").exists()
        assert manifest.checkpoint_path.exists()

    def test_episodes_count_keeps_rising_past_buffer_window(self, tmp_path: Path) -> None:
        # The bandit finishes an episode every step, so by 128 timesteps 128
        # episodes have completed even though SB3's ``ep_info_buffer`` caps at
        # 100: the old ``len(buffer)`` would have saturated at 100.
        config = _training_config(total_timesteps=128, log_freq_steps=64)
        train_single_seed(_BanditEnv, config, seed=0, log_dir=tmp_path)

        points = _strict_json_lines(tmp_path / TELEMETRY_FILENAME)
        episodes = [point["episodes"] for point in points]
        assert episodes == [64, 128]
        assert episodes[-1] > 100
        # The windowed means are unchanged: length 1 for the one-step bandit.
        assert points[-1]["ep_len_mean"] == pytest.approx(1.0)

    def test_status_is_running_at_zero_before_learn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, dict] = {}
        original_learn = PPO.learn

        def _capture(self: PPO, *args: object, **kwargs: object) -> object:
            captured["status"] = _read_status(tmp_path)
            return original_learn(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(PPO, "learn", _capture)
        config = _training_config(total_timesteps=256, log_freq_steps=64)
        train_single_seed(_BanditEnv, config, seed=0, log_dir=tmp_path)

        assert captured["status"]["state"] == "running"
        assert captured["status"]["timestep"] == 0
        assert captured["status"]["pid"] == os.getpid()

    def test_manifest_write_failure_marks_failed_and_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*args: object, **kwargs: object) -> None:
            raise OSError("manifest write failed")

        monkeypatch.setattr(ppo_runner, "write_run_manifest", _boom)
        config = _training_config(total_timesteps=256, log_freq_steps=64)
        with pytest.raises(OSError, match="manifest write failed"):
            train_single_seed(_BanditEnv, config, seed=0, log_dir=tmp_path)

        # A failure after ``learn`` (in the artifact bookkeeping) must not leave
        # ``done`` on disk: the invariant is ``done`` => artifacts exist.
        payload = _read_status(tmp_path)
        assert payload["state"] == "failed"
        assert "manifest write failed" in (payload["error"] or "")


# ---------------------------------------------------------------------------
# metric extraction (unit)
# ---------------------------------------------------------------------------


class TestMetricExtraction:
    def _callback(self, tmp_path: Path) -> TelemetryCallback:
        return TelemetryCallback(
            run_id="ppo_s0_abc",
            seed=0,
            total_timesteps=100,
            log_freq_steps=10,
            status_path=tmp_path / STATUS_FILENAME,
            telemetry_path=tmp_path / TELEMETRY_FILENAME,
        )

    def test_strips_prefixes_and_preserves_unknown_keys(self, tmp_path: Path) -> None:
        callback = self._callback(tmp_path)
        callback.model = SimpleNamespace(  # type: ignore[assignment]
            logger=SimpleNamespace(
                name_to_value={
                    "train/policy_loss": np.float32(0.0123),
                    "train/foo": 7,
                    "time/fps": np.float64(128.4),
                    "rollout/ep_rew_mean": 99.0,  # not train/time -> dropped
                }
            ),
            ep_info_buffer=deque(
                [{"r": 10.0, "l": 5}, {"r": 20.0, "l": 7}], maxlen=100
            ),
        )
        callback.num_timesteps = 40
        # ``episodes`` is now a cumulative count owned by the callback rather
        # than the windowed buffer length.
        callback._episode_count = 2

        point = callback._build_point()
        assert point["train"]["policy_loss"] == pytest.approx(0.0123, rel=1e-5)
        assert point["train"]["foo"] == 7
        assert point["time"]["fps"] == pytest.approx(128.4)
        assert "ep_rew_mean" not in point["train"]
        assert point["ep_rew_mean"] == pytest.approx(15.0)
        assert point["ep_len_mean"] == pytest.approx(6.0)
        assert point["episodes"] == 2
        assert point["timestep"] == 40

    def test_policy_gradient_loss_is_aliased_to_policy_loss(self, tmp_path: Path) -> None:
        # SB3 v2 emits ``train/policy_gradient_loss``; the wire schema expects
        # ``policy_loss``. The raw key is preserved and the alias added.
        callback = self._callback(tmp_path)
        callback.model = SimpleNamespace(  # type: ignore[assignment]
            logger=SimpleNamespace(
                name_to_value={"train/policy_gradient_loss": np.float32(0.5)}
            ),
            ep_info_buffer=deque(),
        )
        train = callback._build_point()["train"]
        assert train["policy_loss"] == pytest.approx(0.5)
        assert train["policy_gradient_loss"] == pytest.approx(0.5)

    def test_empty_info_buffer_is_null_not_zero(self, tmp_path: Path) -> None:
        # ``None`` means "not yet known"; ``0.0`` is a legitimate reward.
        callback = self._callback(tmp_path)
        callback.model = SimpleNamespace(  # type: ignore[assignment]
            logger=SimpleNamespace(name_to_value={}),
            ep_info_buffer=deque(maxlen=100),
        )
        point = callback._build_point()
        assert point["ep_rew_mean"] is None
        assert point["ep_len_mean"] is None
        assert point["episodes"] is None

    def test_to_jsonable_converts_numpy_scalars_and_arrays(self) -> None:
        assert _to_jsonable(np.int64(4)) == 4
        assert _to_jsonable(np.float32(1.5)) == pytest.approx(1.5)
        assert _to_jsonable(np.array([1.0, 2.0])) == [1.0, 2.0]
        assert _to_jsonable("plain") == "plain"

    def test_to_jsonable_maps_non_finite_floats_to_null(self) -> None:
        assert _to_jsonable(float("nan")) is None
        assert _to_jsonable(float("inf")) is None
        assert _to_jsonable(np.float32("nan")) is None

    def test_to_jsonable_neutralises_non_finite_array_elements(self) -> None:
        assert _to_jsonable(np.array([1.0, np.nan, np.inf, -np.inf])) == [
            1.0,
            None,
            None,
            None,
        ]
        # Nested arrays are converted element-wise too.
        assert _to_jsonable(np.array([[1.0, np.nan], [np.inf, 2.0]])) == [
            [1.0, None],
            [None, 2.0],
        ]
        # A 0-d array collapses to its scalar without iterating it.
        assert _to_jsonable(np.array(np.nan)) is None
        assert _to_jsonable(np.array(3.0)) == pytest.approx(3.0)

    def test_write_point_with_non_finite_array_is_strict_json(self, tmp_path: Path) -> None:
        callback = self._callback(tmp_path)
        callback.model = SimpleNamespace(  # type: ignore[assignment]
            logger=SimpleNamespace(
                name_to_value={"train/gradient": np.array([np.nan, np.inf, 1.0])}
            ),
            ep_info_buffer=deque(maxlen=100),
        )
        callback.num_timesteps = 10
        callback._write_point()

        # ``_strict_json_lines`` raises on a bare NaN/Infinity token.
        point = _strict_json_lines(tmp_path / TELEMETRY_FILENAME)[0]
        assert point["train"]["gradient"] == [None, None, 1.0]

    def test_episode_count_is_cumulative_past_buffer_window(self, tmp_path: Path) -> None:
        callback = self._callback(tmp_path)
        callback.model = SimpleNamespace(  # type: ignore[assignment]
            logger=SimpleNamespace(name_to_value={}),
            ep_info_buffer=deque(maxlen=100),
        )
        for step in range(1, 151):
            callback.num_timesteps = step
            callback.locals = {"infos": [{"episode": {"r": 1.0, "l": 1}}]}
            callback._on_step()

        episodes = [
            point["episodes"]
            for point in _strict_json_lines(tmp_path / TELEMETRY_FILENAME)
        ]
        assert episodes == list(range(10, 151, 10))
        assert episodes[-1] == 150 > 100

    def test_non_divisible_budget_emits_only_whole_multiples(self, tmp_path: Path) -> None:
        callback = TelemetryCallback(
            run_id="ppo_s0_abc",
            seed=0,
            total_timesteps=250,
            log_freq_steps=64,
            status_path=tmp_path / STATUS_FILENAME,
            telemetry_path=tmp_path / TELEMETRY_FILENAME,
        )
        callback.model = SimpleNamespace(  # type: ignore[assignment]
            logger=SimpleNamespace(name_to_value={}),
            ep_info_buffer=deque(maxlen=100),
        )
        for step in range(1, 251):
            callback.num_timesteps = step
            callback._on_step()

        points = _strict_json_lines(tmp_path / TELEMETRY_FILENAME)
        assert [point["timestep"] for point in points] == [64, 128, 192]

    def test_cached_record_before_init_raises(self, tmp_path: Path) -> None:
        callback = self._callback(tmp_path)
        with pytest.raises(RuntimeError, match="before _init_callback"):
            callback._cached_record("train/x", 1.0)

    def test_on_training_end_restores_logger_record(self, tmp_path: Path) -> None:
        callback = self._callback(tmp_path)

        def _original_record(key: str, value: object, exclude: object = None) -> None:
            del key, value, exclude

        logger = SimpleNamespace(name_to_value={}, record=_original_record)
        callback.model = SimpleNamespace(  # type: ignore[assignment]
            logger=logger, ep_info_buffer=deque(maxlen=100)
        )
        callback._init_callback()
        assert logger.record == callback._cached_record

        callback._on_training_end()
        assert logger.record is _original_record
        assert callback._original_record is None
        assert not hasattr(logger, "_undertow_original_record")
