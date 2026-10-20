"""U06 — UI rollout reader: summaries, corruption tolerance and downsampling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from undertow.ui import rollouts as rollouts_module
from undertow.ui.rollouts import list_rollouts, read_rollout
from undertow.ui.schemas import SCHEMA_VERSION, SchemaVersionError

_SCHEMA: dict[str, pl.DataType] = {
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

_START = datetime(2026, 1, 1, tzinfo=UTC)


def _write_episode(
    directory: Path,
    *,
    seed: int,
    episode: int,
    n_steps: int = 3,
    regimes: tuple[str, ...] = ("bull",),
    schema_version: int = 1,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in range(n_steps):
        tick_lower = 199_000
        tick_upper = 201_000
        tick = 200_000 + step
        rows.append(
            {
                "schema_version": schema_version,
                "run_id": f"ppo_s{seed}_abc",
                "seed": seed,
                "episode": episode,
                "step": step,
                "timestamp": _START + timedelta(minutes=step),
                "price": 3000.0 + step,
                "tick": tick,
                "tick_lower": tick_lower,
                "tick_upper": tick_upper,
                "in_range": tick_lower <= tick < tick_upper,
                "position_value": 10_000.0,
                "equity": 10_000.0 + step,
                "action_index": 1,
                "action_type": "hold",
                "center_offset": 0,
                "width": 4,
                "fees": 1.0,
                "gas": -0.2,
                "slippage": -0.1,
                "il": -0.05,
                "risk_penalty": 0.0,
                "reward": 0.01 * (step + 1),
                "regime": regimes[step % len(regimes)],
                "gas_price_wei": 12.0,
            }
        )
    path = directory / f"episode_{episode}.parquet"
    pl.DataFrame(rows, schema=_SCHEMA).write_parquet(path)
    return path


def test_rollouts_schema_version_is_the_single_ui_source() -> None:
    assert rollouts_module.SCHEMA_VERSION == SCHEMA_VERSION


def test_list_rollouts_missing_directory_returns_empty(tmp_path: Path) -> None:
    assert list_rollouts(tmp_path / "does-not-exist") == []


def test_list_rollouts_summarises_and_sorts(tmp_path: Path, run_dir) -> None:
    path = run_dir(seed=0, rollouts=3)
    summaries = list_rollouts(path / "rollouts")

    assert [(s.seed, s.episode) for s in summaries] == [(0, 0), (0, 1), (0, 2)]
    assert all(s.n_steps == 3 for s in summaries)
    assert all(s.regime == "bull" for s in summaries)
    assert summaries[0].start == _START
    assert summaries[0].end == _START + timedelta(minutes=2)


def test_list_rollouts_sorts_mixed_seeds_and_episodes(tmp_path: Path) -> None:
    directory = tmp_path / "rollouts"
    _write_episode(directory, seed=5, episode=2)
    _write_episode(directory, seed=1, episode=7)
    _write_episode(directory, seed=5, episode=0)

    summaries = list_rollouts(directory)
    assert [(s.seed, s.episode) for s in summaries] == [(1, 7), (5, 0), (5, 2)]


def test_read_rollout_downsamples_keeping_first_and_last(tmp_path: Path) -> None:
    directory = tmp_path / "rollouts"
    _write_episode(directory, seed=0, episode=0, n_steps=10)
    series = read_rollout(directory / "episode_0.parquet", max_points=4)

    assert series.run_id == "ppo_s0_abc"
    assert series.seed == 0
    assert series.episode == 0
    assert series.downsampled is True
    assert len(series.rows) <= 4
    steps = [row["step"] for row in series.rows]
    assert steps[0] == 0
    assert steps[-1] == 9
    assert len(steps) == len(set(steps))
    # Timestamps are ISO-8601 UTC on the wire.
    assert all(str(row["timestamp"]).endswith("Z") for row in series.rows)


def test_read_rollout_full_frame_reports_columns(tmp_path: Path) -> None:
    directory = tmp_path / "rollouts"
    _write_episode(directory, seed=3, episode=4, n_steps=3, regimes=("bear", "bull"))
    series = read_rollout(directory / "episode_4.parquet", max_points=100)

    assert series.downsampled is False
    assert series.columns == list(_SCHEMA)
    assert [row["step"] for row in series.rows] == [0, 1, 2]
    assert series.rows[0]["regime"] == "bear"


def test_list_rollouts_skips_corrupt_parquet(tmp_path: Path) -> None:
    directory = tmp_path / "rollouts"
    _write_episode(directory, seed=0, episode=0)
    corrupt = directory / "episode_1.parquet"
    corrupt.write_bytes(b"not a parquet file at all")

    summaries = list_rollouts(directory)
    assert [(s.seed, s.episode) for s in summaries] == [(0, 0)]


def test_read_rollout_rejects_newer_schema_version(tmp_path: Path) -> None:
    directory = tmp_path / "rollouts"
    path = _write_episode(directory, seed=0, episode=0, schema_version=2)
    # CONTRACTS §0: the rejection must be the *typed* error, not a bare ValueError.
    with pytest.raises(SchemaVersionError, match="schema_version 2"):
        read_rollout(path, max_points=10)


def test_read_rollout_accepts_missing_and_version_one_schema(tmp_path: Path) -> None:
    directory = tmp_path / "rollouts"
    path = _write_episode(directory, seed=0, episode=0, schema_version=1)
    assert read_rollout(path, max_points=10).seed == 0

    # Drop the ``schema_version`` column entirely: old/foreign artifacts are
    # tolerated (CONTRACTS §0).
    frame = pl.read_parquet(path).drop("schema_version")
    frame.write_parquet(path)
    assert read_rollout(path, max_points=10).seed == 0


def test_list_rollouts_skips_newer_schema_version(tmp_path: Path, caplog) -> None:
    directory = tmp_path / "rollouts"
    _write_episode(directory, seed=0, episode=0)
    _write_episode(directory, seed=0, episode=1, schema_version=2)

    with caplog.at_level("WARNING", logger="undertow.ui.rollouts"):
        summaries = list_rollouts(directory)

    assert [(s.seed, s.episode) for s in summaries] == [(0, 0)]
    assert "schema_version 2" in caplog.text


def test_read_rollout_empty_file_raises(tmp_path: Path) -> None:
    directory = tmp_path / "rollouts"
    empty = directory / "episode_0.parquet"
    empty.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(schema=_SCHEMA).write_parquet(empty)
    with pytest.raises(ValueError, match="no rows"):
        read_rollout(empty, max_points=10)
