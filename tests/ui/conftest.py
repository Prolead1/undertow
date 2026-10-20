"""Shared pytest fixtures for the ``undertow.ui`` test suite.

U01 seeds the config fixture; U02 appends the shared ``run_dir`` factory here.
Keep this module tiny — later tasks add fixtures, they do not edit existing ones.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from undertow.ui.config import UiConfig


@pytest.fixture
def ui_config(tmp_path: Path) -> UiConfig:
    """A :class:`UiConfig` rooted entirely inside an isolated ``tmp_path``."""
    return UiConfig(runs_dir=tmp_path, results_dir=tmp_path)


#: Rollout parquet schema (CONTRACTS.md §2.1), used by the ``run_dir`` factory.
_ROLLOUT_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.Int16,
    "run_id": pl.Utf8,
    "seed": pl.Int64,
    "episode": pl.Int32,
    "step": pl.Int32,
    "timestamp": pl.Datetime("us", "UTC"),
    "price": pl.Float64,
    "tick": pl.Int64,
    "tick_lower": pl.Int64,
    "tick_upper": pl.Int64,
    "in_range": pl.Boolean,
    "position_value": pl.Float64,
    "equity": pl.Float64,
    "action_index": pl.Int32,
    "action_type": pl.Utf8,
    "center_offset": pl.Int32,
    "width": pl.Int32,
    "fees": pl.Float64,
    "gas": pl.Float64,
    "slippage": pl.Float64,
    "il": pl.Float64,
    "risk_penalty": pl.Float64,
    "reward": pl.Float64,
    "regime": pl.Utf8,
    "gas_price_wei": pl.Float64,
}


def _write_rollout(run_path: Path, run_id: str, seed: int, episode: int) -> None:
    """Write one small but schema-complete ``rollouts/episode_<k>.parquet``."""
    rollout_dir = run_path / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for step in range(3):
        rows.append(
            {
                "schema_version": 1,
                "run_id": run_id,
                "seed": seed,
                "episode": episode,
                "step": step,
                "timestamp": start.replace(minute=step),
                "price": 3000.0 + step,
                "tick": 200000 + step,
                "tick_lower": 199000,
                "tick_upper": 201000,
                "in_range": True,
                "position_value": 10_000.0,
                "equity": 10_000.0 + step,
                "action_index": 1,
                "action_type": "rebalance" if step else "hold",
                "center_offset": 0,
                "width": 4,
                "fees": 1.5,
                "gas": -0.2,
                "slippage": -0.1,
                "il": -0.05,
                "risk_penalty": 0.0,
                "reward": 0.01 * (step + 1),
                "regime": "bull",
                "gas_price_wei": 12.0,
            }
        )
    pl.DataFrame(rows, schema=_ROLLOUT_SCHEMA).write_parquet(
        rollout_dir / f"episode_{episode}.parquet"
    )


@pytest.fixture
def run_dir(tmp_path: Path) -> Callable[..., Path]:
    """Factory producing a synthetic run tree (U02+; extended never edited).

    Returns a callable with keyword options covering the artifacts every later
    UI task needs: ``manifest.json`` (valid or corrupt), ``telemetry.jsonl``,
    ``status.json``, one-or-more rollout parquet files and both directory
    layouts (pass ``experiment=""`` for the bare ``runs/seed_<n>/`` form).

    The factory returns the run directory path; evaluation artifacts are *not*
    written here because they live in a shared ``results/`` root (write those
    directly under ``tmp_path / "results"``).
    """

    def _make(
        *,
        experiment: str = "experiment_1",
        seed: int = 0,
        run_id: str | None = None,
        state: str = "done",
        total_timesteps: int = 1_000_000,
        latest_timestep: int = 2000,
        updated_at: str = "2026-10-19T18:12:00Z",
        best_reward: float = 0.42,
        config_hash: str = "deadbeef",
        git_commit: str = "abc1234",
        learning_curve: Sequence[float] = (0.1, 0.2, 0.3),
        telemetry_timesteps: Sequence[int] | None = None,
        checkpoint_stem: str | None = None,
        manifest: bool = True,
        telemetry: bool = True,
        status: bool = True,
        rollouts: int = 1,
        corrupt_manifest: bool = False,
    ) -> Path:
        base = tmp_path / experiment if experiment else tmp_path
        path = base / f"seed_{seed}"
        path.mkdir(parents=True, exist_ok=True)
        rid = run_id or f"ppo_s{seed}_{config_hash}"

        if corrupt_manifest:
            (path / "manifest.json").write_text("{ this is not json", encoding="utf-8")
        elif manifest:
            (path / "manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": rid,
                        "seed": seed,
                        "config_hash": config_hash,
                        "git_commit": git_commit,
                        "algorithm": "ppo",
                        "total_timesteps": total_timesteps,
                        "best_reward": best_reward,
                        "checkpoint_path": f"checkpoints/{checkpoint_stem or rid}.zip",
                        "library_versions": {"stable_baselines3": "2.9"},
                        "learning_curve": [float(value) for value in learning_curve],
                    }
                ),
                encoding="utf-8",
            )

        curve = [float(value) for value in learning_curve]
        steps = (
            list(telemetry_timesteps)
            if telemetry_timesteps is not None
            else [1000 * (index + 1) for index in range(len(curve))]
        )
        if telemetry:
            lines = [
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": rid,
                        "seed": seed,
                        "timestep": timestep,
                        "wall_time_s": 1.0 * (index + 1),
                        "ep_rew_mean": 0.5,
                        "ep_len_mean": 100.0,
                        "episodes": index + 1,
                        "train": {"policy_loss": 0.01},
                        "time": {"fps": 100.0},
                    }
                )
                for index, timestep in enumerate(steps)
            ]
            (path / "telemetry.jsonl").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )

        if status:
            (path / "status.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": rid,
                        "seed": seed,
                        "state": state,
                        "timestep": latest_timestep,
                        "total_timesteps": total_timesteps,
                        "updated_at": updated_at,
                        "pid": 1234,
                        "error": None,
                    }
                ),
                encoding="utf-8",
            )

        for episode in range(rollouts):
            _write_rollout(path, rid, seed, episode)
        return path

    return _make
