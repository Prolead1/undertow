"""U02 — run-directory discovery, registry scan/curve/rank and corruption tolerance."""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from undertow.ui.paths import discover_run_dirs, parse_run_dir
from undertow.ui.registry import RunRegistry, downsample_indices, rank
from undertow.ui.schemas import EvalMetrics, RunManifestSummary, RunRecord

_REGIME_MD = """# RQ2 — Policy comparison by regime

Mean ± std over walk-forward episodes and seeds.

| Policy | Regime | PnL | Sharpe | MaxDD |
| --- | --- | --- | --- | --- |
| ppo | bull | 1,234.56 ± 12.3 | 1.2000 ± 0.1 | -0.0500 ± 0.01 |
| ppo | bear | -500.00 ± 5.0 | -0.3000 ± 0.1 | -0.2000 ± 0.02 |
| ppo | sideways | 200.00 ± 1.0 | 0.1000 ± 0.0 | -0.1000 ± 0.0 |
"""


def _make_record(
    run_id: str,
    *,
    sharpe: float | None = None,
    worst_regime_pnl: float | None = None,
    total_pnl: float | None = None,
    best_reward: float | None = None,
) -> RunRecord:
    has_eval = sharpe is not None or worst_regime_pnl is not None or total_pnl is not None
    return RunRecord(
        run_id=run_id,
        experiment="experiment_1",
        seed=0,
        state="done",
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        total_timesteps=100,
        latest_timestep=100,
        best_reward=best_reward,
        eval=(
            EvalMetrics(sharpe=sharpe, worst_regime_pnl=worst_regime_pnl, total_pnl=total_pnl)
            if has_eval
            else None
        ),
    )


def test_empty_root_returns_no_records(tmp_path: Path) -> None:
    assert RunRegistry(tmp_path, tmp_path / "results").scan() == []


def test_well_formed_run_populates_fields(tmp_path: Path, run_dir) -> None:
    run_dir(seed=0, best_reward=0.42)

    records = RunRegistry(tmp_path, tmp_path / "results").scan()

    assert len(records) == 1
    record = records[0]
    assert record.run_id == "ppo_s0_deadbeef"
    assert record.experiment == "experiment_1"
    assert record.seed == 0
    assert record.state == "done"
    assert record.total_timesteps == 1_000_000
    assert record.latest_timestep == 2000
    assert record.best_reward == 0.42
    assert record.config_hash == "deadbeef"
    assert record.git_commit == "abc1234"
    assert record.manifest is not None
    assert record.manifest.algorithm == "ppo"
    assert record.eval is None
    assert record.has_telemetry is True
    assert record.telemetry_points == 3
    assert record.has_rollouts is True


def test_discover_both_layouts(tmp_path: Path, run_dir) -> None:
    run_dir(experiment="experiment_1", seed=0)
    run_dir(experiment="", seed=1)

    records = RunRegistry(tmp_path).scan()

    assert {record.experiment for record in records} == {"experiment_1", ""}
    assert {record.seed for record in records} == {0, 1}
    assert len(records) == 2


def test_corrupt_manifest_is_unknown_without_raising(tmp_path: Path, run_dir) -> None:
    run_dir(seed=0, corrupt_manifest=True)

    records = RunRegistry(tmp_path).scan()

    assert len(records) == 1
    record = records[0]
    assert record.state == "unknown"
    assert record.manifest is None
    assert record.config_hash is None
    assert record.git_commit is None
    assert record.best_reward is None


def test_malformed_status_falls_back_to_unknown(tmp_path: Path, run_dir) -> None:
    path = run_dir(seed=0)
    (path / "status.json").write_text("{not json", encoding="utf-8")

    records = RunRegistry(tmp_path).scan()

    assert [record.state for record in records] == ["unknown"]


def test_schema_invalid_manifest_is_unknown_without_raising(tmp_path: Path, run_dir) -> None:
    path = run_dir(seed=0)
    # Valid JSON, but missing every required manifest field but ``run_id``.
    (path / "manifest.json").write_text(json.dumps({"run_id": "r"}), encoding="utf-8")

    records = RunRegistry(tmp_path).scan()

    assert len(records) == 1
    record = records[0]
    assert record.state == "unknown"
    assert record.manifest is None
    assert record.config_hash is None
    assert record.git_commit is None
    assert record.best_reward is None


def test_schema_invalid_status_missing_state_is_unknown_without_raising(
    tmp_path: Path, run_dir
) -> None:
    path = run_dir(seed=0)
    # Valid JSON, but ``state`` is a required status field.
    (path / "status.json").write_text(
        json.dumps({"run_id": "ppo_s0_deadbeef"}), encoding="utf-8"
    )

    records = RunRegistry(tmp_path).scan()

    assert len(records) == 1
    assert records[0].state == "unknown"
    assert records[0].config_hash == "deadbeef"


def test_scan_order_is_updated_at_desc_then_run_id(tmp_path: Path, run_dir) -> None:
    run_dir(seed=0, run_id="ppo_a", updated_at="2026-01-01T00:00:00Z")
    run_dir(seed=1, run_id="ppo_b", updated_at="2026-03-01T00:00:00Z")
    run_dir(seed=2, run_id="ppo_c", updated_at="2026-03-01T00:00:00Z")

    first = RunRegistry(tmp_path).scan()
    second = RunRegistry(tmp_path).scan()

    assert [record.run_id for record in first] == ["ppo_b", "ppo_c", "ppo_a"]
    assert [record.run_id for record in first] == [record.run_id for record in second]


def test_curve_aggregates_mean_and_std(tmp_path: Path, run_dir) -> None:
    run_dir(experiment="exp", seed=0, learning_curve=[0.0, 2.0, 4.0, 6.0, 8.0])
    run_dir(experiment="exp", seed=1, learning_curve=[2.0, 2.0, 2.0, 2.0, 2.0])

    curve = RunRegistry(tmp_path, max_points=2000).curve("ppo_s0_deadbeef")

    assert curve is not None
    assert curve.timesteps == [1000, 2000, 3000, 4000, 5000]
    assert curve.mean == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert curve.std == [1.0, 0.0, 1.0, 2.0, 3.0]
    assert curve.per_seed == {
        "0": [0.0, 2.0, 4.0, 6.0, 8.0],
        "1": [2.0, 2.0, 2.0, 2.0, 2.0],
    }
    assert curve.downsampled is False


def test_curve_downsampling_keeps_first_and_last(tmp_path: Path, run_dir) -> None:
    run_dir(experiment="exp", seed=0, learning_curve=[0.0, 2.0, 4.0, 6.0, 8.0])
    run_dir(experiment="exp", seed=1, learning_curve=[2.0, 2.0, 2.0, 2.0, 2.0])

    curve = RunRegistry(tmp_path, max_points=3).curve("ppo_s0_deadbeef")

    assert curve is not None
    assert curve.downsampled is True
    assert curve.timesteps == [1000, 3000, 5000]
    assert curve.mean == [1.0, 3.0, 5.0]
    assert curve.std == [1.0, 1.0, 3.0]
    assert curve.per_seed == {"0": [0.0, 4.0, 8.0], "1": [2.0, 2.0, 2.0]}


def test_curve_missing_run_is_none(tmp_path: Path, run_dir) -> None:
    run_dir(seed=0)
    assert RunRegistry(tmp_path).curve("does-not-exist") is None


def test_downsample_indices_respects_max_points() -> None:
    assert downsample_indices(5, 3) == [0, 2, 4]
    assert downsample_indices(10, 4) == [0, 3, 6, 9]
    assert downsample_indices(11, 4) == [0, 3, 6, 10]
    assert downsample_indices(3, 10) == [0, 1, 2]
    assert downsample_indices(0, 10) == []


def test_downsample_indices_max_points_below_two_returns_first() -> None:
    assert downsample_indices(5, 1) == [0]
    assert downsample_indices(5, 0) == [0]
    assert downsample_indices(5, -3) == [0]
    assert downsample_indices(0, 1) == []


def test_scan_attaches_eval_metrics(tmp_path: Path, run_dir) -> None:
    run_dir(seed=0)
    results = tmp_path / "results"
    results.mkdir()
    (results / "regime_comparison.md").write_text(_REGIME_MD, encoding="utf-8")
    (results / "summary.json").write_text(
        json.dumps({"result": {"sharpe": 1.75, "total_pnl": 900.0}}), encoding="utf-8"
    )

    records = RunRegistry(tmp_path, results).scan()

    assert records[0].eval is not None
    assert records[0].eval.sharpe == 1.75
    assert records[0].eval.total_pnl == 900.0
    assert records[0].eval.worst_regime_pnl == -500.0
    assert records[0].eval.source == "regime_comparison.md"


def test_scan_attaches_eval_from_nested_seed_results(tmp_path: Path, run_dir) -> None:
    run_dir(seed=0)
    results = tmp_path / "results" / "experiment_1" / "seed_0"
    results.mkdir(parents=True)
    (results / "regime_comparison.md").write_text(_REGIME_MD, encoding="utf-8")

    records = RunRegistry(tmp_path, tmp_path / "results").scan()

    assert records[0].eval is not None
    assert records[0].eval.source == "experiment_1/seed_0/regime_comparison.md"
    assert records[0].eval.worst_regime_pnl == -500.0


def test_rank_uses_eval_sharpe_then_worst_regime_then_best_reward() -> None:
    a = _make_record("a", sharpe=1.0)
    b = _make_record("b", best_reward=0.2)
    c = _make_record("c", worst_regime_pnl=0.5)
    d = _make_record("d")

    ordered = rank([b, c, d, a])

    assert [record.run_id for record in ordered] == ["a", "c", "b", "d"]


def test_rank_evaluated_run_outranks_train_only_run_with_higher_reward() -> None:
    low_sharpe = _make_record("eval_low", sharpe=-5.0)
    high_reward = _make_record("train_high", best_reward=999.0)

    assert [record.run_id for record in rank([high_reward, low_sharpe])] == [
        "eval_low",
        "train_high",
    ]


def test_rank_equal_sharpe_tie_broken_by_worst_regime_pnl() -> None:
    worse = _make_record("worse_regime", sharpe=1.0, worst_regime_pnl=-10.0)
    better = _make_record("better_regime", sharpe=1.0, worst_regime_pnl=5.0)

    assert [record.run_id for record in rank([worse, better])] == [
        "better_regime",
        "worse_regime",
    ]


def test_rank_missing_and_nan_scores_sort_last_within_their_group() -> None:
    valid = _make_record("valid", sharpe=0.0, worst_regime_pnl=-1000.0)
    missing = _make_record("missing", worst_regime_pnl=300.0)
    nan = _make_record("nan", sharpe=float("nan"), worst_regime_pnl=200.0)
    inf = _make_record("inf", sharpe=float("inf"), worst_regime_pnl=100.0)
    train = _make_record("train", best_reward=999.0)

    ordered = rank([train, missing, inf, nan, valid])

    # ``valid`` has a real Sharpe; the NaN/±inf/missing primaries sort after it,
    # ordered among themselves by the next tiebreaker (worst-regime PnL), and the
    # train-only run always trails the evaluated group.
    assert [record.run_id for record in ordered] == [
        "valid",
        "missing",
        "nan",
        "inf",
        "train",
    ]


def test_rank_is_deterministic_on_ties_and_accepts_key() -> None:
    one = _make_record("b_run", best_reward=1.0)
    two = _make_record("a_run", best_reward=1.0)
    high_pnl = _make_record("c_run", total_pnl=100.0, sharpe=0.01)
    low_pnl = _make_record("d_run", total_pnl=1.0, sharpe=0.02)

    assert [r.run_id for r in rank([one, two])] == ["a_run", "b_run"]
    assert [r.run_id for r in rank([low_pnl, high_pnl], key="total_pnl")] == [
        "c_run",
        "d_run",
    ]
    assert [r.run_id for r in rank([low_pnl, high_pnl], key=lambda r: r.eval.total_pnl)] == [
        "c_run",
        "d_run",
    ]


def test_rank_rejects_unknown_metric_key() -> None:
    with pytest.raises(ValueError, match="unknown rank key"):
        rank([], key="sharpe_ratio")


def test_telemetry_tolerates_corrupt_lines_and_uses_max_timestep(tmp_path: Path, run_dir) -> None:
    path = run_dir(seed=0, status=False, telemetry_timesteps=(900, 100))
    with (path / "telemetry.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")

    records = RunRegistry(tmp_path).scan()

    assert records[0].telemetry_points == 2
    assert records[0].latest_timestep == 900
    assert records[0].state == "unknown"


def test_run_id_and_seed_fall_back_to_status_then_path(tmp_path: Path, run_dir) -> None:
    run_dir(seed=7, run_id="from_status", manifest=False, status=True)
    run_dir(seed=9, manifest=False, status=False)

    records = RunRegistry(tmp_path).scan()
    by_run_id = {record.run_id: record for record in records}

    assert by_run_id["from_status"].seed == 7
    assert by_run_id["experiment_1/seed_9"].seed == 9


def test_get_lazily_scans_and_refresh_rescans(tmp_path: Path, run_dir) -> None:
    run_dir(seed=0)
    registry = RunRegistry(tmp_path)

    assert registry.get("ppo_s0_deadbeef") is not None
    assert registry.get("missing") is None
    assert len(registry.refresh()) == 1


def test_shared_root_results_not_applied_to_every_run(tmp_path: Path, run_dir) -> None:
    run_dir(experiment="exp", seed=0)
    run_dir(experiment="exp", seed=1)
    results = tmp_path / "results"
    results.mkdir()
    (results / "regime_comparison.md").write_text(_REGIME_MD, encoding="utf-8")

    records = RunRegistry(tmp_path, results).scan()

    assert [record.eval for record in records] == [None, None]


def test_curve_extrapolates_timesteps_when_telemetry_is_short(tmp_path: Path, run_dir) -> None:
    run_dir(
        experiment="exp",
        seed=0,
        learning_curve=[0.0, 1.0, 2.0, 3.0, 4.0],
        telemetry_timesteps=(10, 20),
    )

    curve = RunRegistry(tmp_path).curve("ppo_s0_deadbeef")

    assert curve is not None
    assert curve.timesteps == [10, 20, 30, 40, 50]
    assert curve.mean == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert curve.per_seed == {"0": [0.0, 1.0, 2.0, 3.0, 4.0]}


def test_curve_uses_evaluation_index_without_telemetry(tmp_path: Path, run_dir) -> None:
    run_dir(experiment="exp", seed=0, learning_curve=[1.0, 2.0, 3.0], telemetry=False)

    curve = RunRegistry(tmp_path).curve("ppo_s0_deadbeef")

    assert curve is not None
    assert curve.timesteps == [0, 1, 2]


def test_curve_sorts_out_of_order_timesteps(tmp_path: Path, run_dir) -> None:
    run_dir(
        experiment="exp",
        seed=0,
        learning_curve=[0.0, 1.0, 2.0],
        telemetry_timesteps=(300, 100, 200),
    )

    curve = RunRegistry(tmp_path).curve("ppo_s0_deadbeef")

    assert curve is not None
    assert curve.timesteps == [100, 200, 300]


def test_scan_tolerates_undecodable_telemetry(tmp_path: Path, run_dir) -> None:
    path = run_dir(seed=0)
    (path / "telemetry.jsonl").write_bytes(b"\xff\xfe\x00 not utf-8")

    records = RunRegistry(tmp_path).scan()

    assert len(records) == 1
    assert records[0].telemetry_points == 0
    assert records[0].latest_timestep == 2000


def test_rank_ignores_nan_fallback_scores() -> None:
    good = _make_record("a", sharpe=1.0)
    broken = _make_record(
        "b", worst_regime_pnl=float("nan"), best_reward=float("nan")
    )

    assert [record.run_id for record in rank([broken, good])] == ["a", "b"]


def test_scan_deduplicates_duplicate_run_ids(tmp_path: Path, run_dir) -> None:
    run_dir(experiment="exp_a", seed=0, run_id="dup", updated_at="2026-03-01T00:00:00Z")
    run_dir(experiment="exp_b", seed=0, run_id="dup", updated_at="2026-01-01T00:00:00Z")

    records = RunRegistry(tmp_path).scan()

    assert [record.run_id for record in records] == ["dup"]
    assert records[0].experiment == "exp_a"


def test_resolve_pairs_each_id_with_its_own_directory(tmp_path: Path, run_dir) -> None:
    path_a = run_dir(
        experiment="exp_a", seed=0, run_id="dup", updated_at="2026-03-01T00:00:00Z"
    )
    path_b = run_dir(
        experiment="exp_b", seed=0, run_id="dup", updated_at="2026-01-01T00:00:00Z"
    )
    registry = RunRegistry(tmp_path)
    registry.scan()

    primary = registry.resolve("dup")
    assert primary is not None
    assert primary[1].path == path_a

    alias_a = registry.resolve("exp_a/seed_0")
    alias_b = registry.resolve("exp_b/seed_0")
    assert alias_a is not None and alias_a[1].path == path_a
    assert alias_b is not None and alias_b[1].path == path_b
    assert alias_a[0].run_id == alias_b[0].run_id == "dup"
    assert registry.resolve("exp_c/seed_0") is None


def test_curve_resolves_each_id_to_its_own_directory_on_duplicate_run_id(
    tmp_path: Path, run_dir
) -> None:
    run_dir(
        experiment="exp_a",
        seed=0,
        run_id="dup",
        learning_curve=(0.1, 0.2, 0.3),
        telemetry_timesteps=(1000,),
        updated_at="2026-03-01T00:00:00Z",
    )
    run_dir(
        experiment="exp_b",
        seed=0,
        run_id="dup",
        learning_curve=(9.0, 9.0, 9.0),
        telemetry_timesteps=(2000,),
        updated_at="2026-01-01T00:00:00Z",
    )
    registry = RunRegistry(tmp_path)
    registry.scan()

    # The raw id resolves to the canonical exp_a directory.
    curve = registry.curve("dup")
    assert curve is not None
    assert curve.run_id == "dup"
    assert curve.mean == [0.1, 0.2, 0.3]
    assert curve.timesteps == [1000, 2000, 3000]
    assert curve.per_seed == {"0": [0.1, 0.2, 0.3]}

    # Each alias aggregates its *own* directory's experiment (and echoes the
    # alias as its run_id), so a duplicate run_id never leaks exp_a's numbers
    # into exp_b's curve.
    alias_b_curve = registry.curve("exp_b/seed_0")
    assert alias_b_curve is not None
    assert alias_b_curve.run_id == "exp_b/seed_0"
    assert alias_b_curve.mean == [9.0, 9.0, 9.0]
    assert alias_b_curve.timesteps == [2000, 4000, 6000]
    assert alias_b_curve.per_seed == {"0": [9.0, 9.0, 9.0]}

    alias_a_curve = registry.curve("exp_a/seed_0")
    assert alias_a_curve is not None
    assert alias_a_curve.run_id == "exp_a/seed_0"
    assert alias_a_curve.mean == [0.1, 0.2, 0.3]
    assert alias_a_curve.timesteps == [1000, 2000, 3000]


def test_curve_aggregates_every_seed_of_the_resolved_experiment(
    tmp_path: Path, run_dir
) -> None:
    """A duplicate run_id inside one experiment must not drop a seed.

    The raw id resolves to the canonical directory, but the curve is still the
    experiment's per-seed aggregate, so every seed directory contributes —
    including the duplicate whose record lost the canonical dedup.
    """
    run_dir(
        experiment="exp",
        seed=0,
        run_id="dup",
        learning_curve=(2.0, 2.0),
        telemetry_timesteps=(100,),
        updated_at="2026-03-01T00:00:00Z",
    )
    run_dir(
        experiment="exp",
        seed=1,
        run_id="dup",
        learning_curve=(4.0, 4.0),
        telemetry_timesteps=(100,),
        updated_at="2026-01-01T00:00:00Z",
    )
    registry = RunRegistry(tmp_path)
    registry.scan()

    curve = registry.curve("dup")

    assert curve is not None
    assert curve.run_id == "dup"
    assert curve.per_seed == {"0": [2.0, 2.0], "1": [4.0, 4.0]}
    assert curve.mean == [3.0, 3.0]
    assert curve.timesteps == [100, 200]


def test_concurrent_scan_and_resolve_never_tear_the_index(
    tmp_path: Path, run_dir
) -> None:
    """A reader must never see one scan's records teamed with another's curves."""
    path = run_dir(
        experiment="exp_a",
        seed=0,
        run_id="dup",
        learning_curve=(0.1, 0.2, 0.3),
    )
    registry = RunRegistry(tmp_path)
    registry.scan()
    errors: list[str] = []
    stop = threading.Event()
    deadline = time.monotonic() + 0.3

    def writer() -> None:
        while not stop.is_set():
            try:
                registry.refresh()
            except Exception as exc:  # pragma: no cover - defensive thread guard
                errors.append(f"refresh failed: {exc}")
                return

    def reader() -> None:
        while time.monotonic() < deadline:
            resolved = registry.resolve("dup")
            if resolved is None or resolved[1].path != path:
                errors.append(f"incoherent resolve: {resolved}")
                continue
            curve = registry.curve("dup")
            if curve is None or curve.mean != [0.1, 0.2, 0.3]:
                errors.append(f"torn curve: {None if curve is None else curve.mean}")

    writers = [threading.Thread(target=writer) for _ in range(2)]
    readers = [threading.Thread(target=reader) for _ in range(4)]
    for thread in [*writers, *readers]:
        thread.start()
    for thread in readers:
        thread.join()
    stop.set()
    for thread in writers:
        thread.join()

    assert errors == []


def test_parse_run_dir_falls_back_for_unconventional_name(tmp_path: Path) -> None:
    path = tmp_path / "exp" / "weird"
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        json.dumps({"run_id": "r", "seed": 33}), encoding="utf-8"
    )

    resolved = parse_run_dir(path, root=tmp_path)

    assert resolved.experiment == "exp"
    assert resolved.seed == 33
    assert resolved.run_id == "r"


def test_parse_run_dir_keeps_seed_zero_from_status(tmp_path: Path) -> None:
    path = tmp_path / "exp" / "not_a_seed_dir"
    path.mkdir(parents=True)
    (path / "status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "r0",
                "seed": 0,
                "state": "running",
                "timestep": 1,
                "total_timesteps": 2,
                "updated_at": "2026-01-01T00:00:00Z",
                "pid": 1,
            }
        ),
        encoding="utf-8",
    )

    resolved = parse_run_dir(path, root=tmp_path)

    assert resolved.seed == 0
    assert resolved.run_id == "r0"
    assert resolved.experiment == "exp"


def test_parse_run_dir_without_root_uses_parent_name(tmp_path: Path) -> None:
    path = tmp_path / "parent" / "seed_2"
    path.mkdir(parents=True)

    resolved = parse_run_dir(path)

    assert resolved.experiment == "parent"
    assert resolved.seed == 2
    assert resolved.run_id == "parent/seed_2"


def test_discover_run_dirs_missing_or_file_root(tmp_path: Path) -> None:
    assert discover_run_dirs(tmp_path / "nope") == []
    a_file = tmp_path / "file.txt"
    a_file.write_text("x", encoding="utf-8")
    assert discover_run_dirs(a_file) == []


def test_eval_path_containment_guard(tmp_path: Path, run_dir) -> None:
    run_dir(seed=0, run_id="../outside")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "regime_comparison.md").write_text(_REGIME_MD, encoding="utf-8")
    results = tmp_path / "results"
    results.mkdir()

    records = RunRegistry(tmp_path, results).scan()

    assert records[0].eval is None


def test_manifest_summary_drops_unknown_keys() -> None:
    summary = RunManifestSummary.model_validate(
        {
            "run_id": "r",
            "seed": 0,
            "config_hash": "abc",
            "git_commit": "def",
            "algorithm": "ppo",
            "total_timesteps": 10,
            "best_reward": 1.0,
            "checkpoint_path": "ckpt.zip",
            "learning_curve": [1.0, 2.0],
            "future_field": {"x": 1},
        }
    )

    assert summary.model_extra is None
    assert "learning_curve" not in summary.model_dump()
    assert "future_field" not in summary.model_dump()


def test_curve_without_manifest_curve_is_none(tmp_path: Path, run_dir) -> None:
    run_dir(seed=0, learning_curve=())

    assert RunRegistry(tmp_path).curve("ppo_s0_deadbeef") is None


def test_registry_rank_delegates_to_module_helper() -> None:
    records = [
        _make_record("b", best_reward=0.1),
        _make_record("a", sharpe=2.0),
        _make_record("c", total_pnl=100.0, sharpe=0.0),
    ]
    registry = RunRegistry(Path("unused"))

    for key in ("sharpe", "total_pnl"):
        delegated = [record.run_id for record in registry.rank(records, key=key)]
        direct = [record.run_id for record in rank(records, key=key)]
        assert delegated == direct


#: The non-telemetry wire models intentionally use pydantic's default
#: ``extra="ignore"`` (CONTRACTS §0: readers tolerate extra fields); the U02
#: brief's "otherwise forbid extras" wording is overridden.  Undertow.sim is
#: importable from tests — only ``src/undertow/ui`` is boundary-checked.
def test_run_status_model_mirrors_sim_dataclass() -> None:
    from dataclasses import fields as dataclass_fields

    from undertow.sim.train.telemetry import RunStatus
    from undertow.ui.schemas import RunStatusModel

    assert list(RunStatusModel.model_fields) == [
        field.name for field in dataclass_fields(RunStatus)
    ]
