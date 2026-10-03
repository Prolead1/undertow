"""S15 — evaluation runner: RQ1 ablation, RQ2 regime matrix, RQ3 gap.

Every test asserts real behavior.  The fast tests pin the frozen dataclasses,
the walk-forward splitter, the regime tagger and the artifact writer (including
malformed-table detection).  The three ``@pytest.mark.slow`` tests run the full
RQ1/RQ2/RQ3 pipelines on the shared ``tiny_market_view`` (train split, ~1000
tape rows) with deterministic baselines, so the numbers are reproducible and the
suite stays fast.

The RQ3 gap is deliberately computed on the shared tiny view, which S14 documents
as *not* parity-shaped; the test therefore asserts the identity
``gap == sim_pnl - onchain_pnl`` and the report shape, not a small gap.  A large
gap on that fixture is the honest result (PLAN.md §8 risk register).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from undertow.sim.config import EpisodeConfig, SimConfig, TrainingConfig
from undertow.sim.evaluate import (
    ABLATION_VARIANTS,
    METRIC_NAMES,
    REGIME_LABELS,
    AblationResult,
    EvaluationError,
    GapReport,
    RegimeMatrix,
    aggregate_metrics,
    dominant_regime,
    evaluate_episodes,
    evaluate_policy,
    observation_from_vector,
    regime_matrix_wide,
    run_ablations,
    run_gap_analysis,
    run_regime_evaluation,
    walk_forward_windows,
    write_results,
)
from undertow.sim.evaluate.protocol import action_to_index
from undertow.sim.marketview import MarketView
from undertow.sim.policies import HODLPolicy, HOLDAction, PassiveNarrowPolicy, TauResetPolicy

# ---------------------------------------------------------------------------
# Helpers / detectors
# ---------------------------------------------------------------------------

_FLAG_KEYS = ("fees", "gas", "slippage", "il")

#: The brief's RQ1 table: (fees, gas, slippage, il).
_EXPECTED_FLAGS: dict[str, tuple[bool, bool, bool, bool]] = {
    "full": (True, True, True, True),
    "no_gas": (True, False, True, True),
    "no_slippage": (True, True, False, True),
    "no_il": (True, True, True, False),
    "static_fee": (True, True, True, True),
}


def _config(*, duration_days: int = 4, seeds: tuple[int, ...] = (0,)) -> SimConfig:
    """A deterministic evaluation config on the tiny train split."""
    return SimConfig(
        episode=EpisodeConfig(duration_days=duration_days, step_minutes=10),
        training=TrainingConfig(seeds=seeds),
        seed=0,
    )


def _valid_pipe_tables(text: str) -> int:
    """Count GitHub pipe tables, asserting each has a header, separator and data.

    This is the test-side detector for the "markdown table is well formed"
    requirement: it rejects a header without a separator, a missing data row, and
    a column-count mismatch.
    """
    lines = text.splitlines()
    tables = 0
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.startswith("|") and index + 1 < len(lines):
            separator = lines[index + 1].strip()
            residue = (
                separator.replace("|", "").replace(":", "").replace("-", "").strip()
            )
            if separator.startswith("|") and residue == "":
                header_columns = len(line.split("|"))
                assert len(separator.split("|")) == header_columns
                data_rows = 0
                cursor = index + 2
                while cursor < len(lines) and lines[cursor].strip().startswith("|"):
                    assert len(lines[cursor].split("|")) == header_columns
                    data_rows += 1
                    cursor += 1
                assert data_rows >= 1, "pipe table has no data row"
                tables += 1
                index = cursor
                continue
        index += 1
    return tables


class _FakeRlPolicy:
    """A non-baseline policy used to exercise the RL-ablation guard."""

    @property
    def name(self) -> str:
        return "fake_rl"

    def reset(self) -> None:
        """No-op."""

    def act(self, observation: object, rng: object = None) -> object:
        return HOLDAction


class TestAblationResult:
    def test_stores_all_fields(self) -> None:
        per_regime = pl.DataFrame(
            {"regime": ["bull"], "metric": ["total_pnl"], "mean": [1.0], "std": [0.0]}
        )
        result = AblationResult(
            ablation_name="no_gas",
            flags={"fees": True, "gas": False, "slippage": True, "il": True},
            metrics={"total_pnl": 3.0},
            per_regime=per_regime,
        )
        assert result.ablation_name == "no_gas"
        assert result.flags["gas"] is False
        assert result.metrics["total_pnl"] == pytest.approx(3.0)
        assert isinstance(result.per_regime, pl.DataFrame)

    def test_is_frozen_and_slotted(self) -> None:
        result = AblationResult("full", {"gas": True}, {}, None)
        assert not hasattr(result, "__dict__")
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.ablation_name = "no_gas"  # type: ignore[misc]


class TestRegimeMatrix:
    def test_stores_all_fields(self) -> None:
        frame = pl.DataFrame(
            {
                "regime": ["bull", "bear"],
                "metric": ["total_pnl", "total_pnl"],
                "mean": [1.0, -1.0],
                "std": [0.1, 0.2],
                "n": [2, 2],
            }
        )
        matrix = RegimeMatrix("ppo_full", ["bull", "bear"], frame)
        assert matrix.policy_name == "ppo_full"
        assert matrix.regimes == ["bull", "bear"]
        assert matrix.metrics_per_regime.height == 2
        with pytest.raises(dataclasses.FrozenInstanceError):
            matrix.policy_name = "x"  # type: ignore[misc]


class TestRegimeMatrixWide:
    def test_pivots_to_one_row_per_regime_with_mean_std_columns(self) -> None:
        frame = pl.DataFrame(
            {
                "regime": ["bull", "bull", "bear", "bear"],
                "metric": ["total_pnl", "sharpe", "total_pnl", "sharpe"],
                "mean": [1.0, 2.0, 3.0, 4.0],
                "std": [0.1, 0.2, 0.3, 0.4],
                "n": [2, 2, 2, 2],
            }
        )
        matrix = RegimeMatrix("hodl", ["bull", "bear"], frame)
        wide = regime_matrix_wide(matrix)
        assert wide.height == 2
        assert set(wide.columns) >= {
            "regime",
            "total_pnl_mean",
            "total_pnl_std",
            "sharpe_mean",
            "sharpe_std",
        }
        bull = wide.filter(pl.col("regime") == "bull")
        assert bull["total_pnl_mean"][0] == pytest.approx(1.0)
        assert bull["sharpe_std"][0] == pytest.approx(0.2)


class TestGapReport:
    def test_stores_all_fields(self) -> None:
        report = GapReport(
            policy_name="hodl",
            sim_pnl=10.0,
            onchain_pnl=8.0,
            gap=2.0,
            gap_by_cost_type={"fees": 0.5, "gas": -0.5, "slippage": 0.0, "il": 1.0},
        )
        assert report.gap == pytest.approx(report.sim_pnl - report.onchain_pnl)
        assert set(report.gap_by_cost_type) == {"fees", "gas", "slippage", "il"}
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.gap = 0.0  # type: ignore[misc]


class TestWalkForwardWindows:
    def test_non_overlapping_and_cover_the_split(self, tiny_market_view: MarketView) -> None:
        config = _config(duration_days=4)
        windows = walk_forward_windows(tiny_market_view, config)
        assert len(windows) == 8
        assert windows[0][0] == 0
        assert windows[-1][1] == tiny_market_view.step_count()
        for (_, previous_end), (next_start, _) in zip(windows, windows[1:], strict=False):
            assert previous_end == next_start

    def test_duration_longer_than_split_yields_one_window(
        self, tiny_market_view: MarketView
    ) -> None:
        windows = walk_forward_windows(tiny_market_view, _config(duration_days=365))
        assert windows == [(0, tiny_market_view.step_count())]

    def test_non_positive_duration_raises(self, tiny_market_view: MarketView) -> None:
        with pytest.raises(EvaluationError, match="duration_days"):
            walk_forward_windows(tiny_market_view, _config(duration_days=0))


class TestDominantRegime:
    def test_all_four_regimes_appear(self, tiny_market_view: MarketView) -> None:
        windows = walk_forward_windows(tiny_market_view, _config(duration_days=4))
        observed = {dominant_regime(tiny_market_view, start, end) for start, end in windows}
        assert observed == set(REGIME_LABELS)

    def test_missing_regime_table_returns_unknown(self, tiny_market_view: MarketView) -> None:
        empty = tiny_market_view.regimes.clear()
        view = dataclasses.replace(tiny_market_view, regimes=empty)
        assert dominant_regime(view, 0, 100) == "unknown"


class TestEvaluatePolicy:
    def test_backtest_shape_and_metrics(self, tiny_market_view: MarketView) -> None:
        config = _config(duration_days=4)
        result = evaluate_policy(HODLPolicy(), tiny_market_view, config, num_seeds=1)
        assert result["policy"] == "hodl"
        assert result["backend"] == "backtest"
        assert result["n_episodes"] == 8
        assert result["n_seeds"] == 1
        assert set(result["metrics"]) == set(METRIC_NAMES)
        for metric, summary in result["metrics"].items():
            assert set(summary) == {"mean", "std", "n"}
            assert summary["n"] == 8, metric
            assert np.isfinite(summary["mean"])
        assert len(result["per_episode"]) == 8
        assert {"regime", "metrics", "decomposition"} <= set(result["per_episode"][0])

    def test_multi_seed_replicates_windows_and_reports_std(
        self, tiny_market_view: MarketView
    ) -> None:
        config = _config(duration_days=8, seeds=(0, 1))
        result = evaluate_policy(HODLPolicy(), tiny_market_view, config, num_seeds=2)
        assert result["n_seeds"] == 2
        assert result["n_episodes"] == 8  # 4 windows x 2 seeds
        assert result["metrics"]["total_pnl"]["std"] > 0.0

    def test_unknown_backend_raises(self, tiny_market_view: MarketView) -> None:
        with pytest.raises(EvaluationError, match="unknown backend"):
            evaluate_policy(
                HODLPolicy(), tiny_market_view, _config(), backend="teleport"
            )

    def test_sim_backend_uses_replay_env(self, tiny_market_view: MarketView) -> None:
        config = _config(duration_days=8)
        result = evaluate_policy(
            HODLPolicy(), tiny_market_view, config, num_seeds=1, backend="replay_sim"
        )
        assert result["backend"] == "replay_sim"
        assert result["n_episodes"] == 4
        assert result["metrics"]["total_pnl"]["n"] == 4

    def test_sim_backend_executes_a_rebalancing_baseline(
        self, tiny_market_view: MarketView
    ) -> None:
        # TauReset's half-width is 2 spacings, which is in the agent action grid,
        # so the simulator path can execute it and pay real gas.
        records = evaluate_episodes(
            TauResetPolicy(120),
            tiny_market_view,
            _config(duration_days=8),
            backend="replay_sim",
        )
        assert records
        assert any(record.metrics["n_rebalances"] > 0 for record in records)
        assert any(record.metrics["gas_paid"] < 0.0 for record in records)

    def test_aggregate_metrics_handles_missing_metric(self) -> None:
        record = _make_record(metrics={"total_pnl": 1.0})
        summary = aggregate_metrics([record], metric_names=("total_pnl", "sharpe"))
        assert summary["total_pnl"]["mean"] == pytest.approx(1.0)
        assert summary["sharpe"]["n"] == 0


class TestObservationAdapter:
    def test_round_trips_length_and_regime(self) -> None:
        vector = [0.0] * 18
        vector[15] = 1.0  # bear ordinal
        observation = observation_from_vector(vector)
        assert observation.regime == "bear"
        assert observation.to_vector().shape == (18,)

    def test_wrong_length_raises(self) -> None:
        with pytest.raises(EvaluationError, match="length 18"):
            observation_from_vector([0.0] * 17)


class TestActionToIndex:
    def test_hold_maps_to_zero(self, tiny_env: object) -> None:
        assert action_to_index(tiny_env, HOLDAction) == 0  # type: ignore[arg-type]

    def test_unrepresentable_action_raises(self, tiny_env: object) -> None:
        from undertow.sim.types import Action

        with pytest.raises(EvaluationError, match="not in the environment"):
            action_to_index(
                tiny_env,  # type: ignore[arg-type]
                Action(action_type="rebalance", center_offset=0, width=999),
            )


@pytest.mark.slow
class TestAblations:
    def test_returns_one_result_per_variant_with_exact_flags(
        self, tiny_market_view: MarketView
    ) -> None:
        config = _config(duration_days=8)
        results = run_ablations(config, HODLPolicy(), tiny_market_view)
        assert [r.ablation_name for r in results] == [name for name, *_ in ABLATION_VARIANTS]
        for result in results:
            expected = _EXPECTED_FLAGS[result.ablation_name]
            assert tuple(result.flags[key] for key in _FLAG_KEYS) == expected
            assert set(result.metrics) >= set(METRIC_NAMES)
            assert isinstance(result.per_regime, pl.DataFrame)
            assert result.per_regime.height == len(REGIME_LABELS) * len(METRIC_NAMES)

    def test_gating_changes_the_ledger_for_a_rebalancing_baseline(
        self, tiny_market_view: MarketView
    ) -> None:
        config = _config(duration_days=8)
        results = {
            r.ablation_name: r
            for r in run_ablations(config, TauResetPolicy(120), tiny_market_view)
        }
        full = results["full"].metrics
        no_gas = results["no_gas"].metrics
        no_slippage = results["no_slippage"].metrics
        no_il = results["no_il"].metrics
        static = results["static_fee"].metrics

        # The gated variants remove exactly the corresponding cost term.
        assert full["gas_paid"] < 0.0
        assert no_gas["gas_paid"] == pytest.approx(0.0)
        assert no_gas["total_pnl"] > full["total_pnl"]
        assert full["slippage_paid"] < 0.0
        assert no_slippage["slippage_paid"] == pytest.approx(0.0)
        assert no_slippage["total_pnl"] > full["total_pnl"]
        assert no_il["il_realized"] == pytest.approx(0.0)
        assert no_il["total_pnl"] != pytest.approx(full["total_pnl"])
        # static_fee re-prices gas flat, so its gas total differs from replay.
        assert static["gas_paid"] != pytest.approx(full["gas_paid"])

    def test_gated_full_reproduces_the_untouched_equity(
        self, tiny_market_view: MarketView
    ) -> None:
        config = _config(duration_days=8)
        results = {
            r.ablation_name: r
            for r in run_ablations(config, TauResetPolicy(120), tiny_market_view)
        }
        raw = evaluate_episodes(
            TauResetPolicy(120), tiny_market_view, config, backend="backtest"
        )
        expected = float(np.mean([record.metrics["total_pnl"] for record in raw]))
        assert results["full"].metrics["total_pnl"] == pytest.approx(expected)

    def test_rl_policy_without_train_fn_raises(self, tiny_market_view: MarketView) -> None:
        with pytest.raises(EvaluationError, match="train_fn"):
            run_ablations(_config(), _FakeRlPolicy(), tiny_market_view)

    def test_rl_policy_with_train_fn_invokes_it_per_variant(
        self, tiny_market_view: MarketView
    ) -> None:
        calls: list[str] = []

        def train_fn(
            name: str, config: SimConfig, view: MarketView, checkpoint_dir: Path | None
        ) -> HODLPolicy:
            calls.append(name)
            return HODLPolicy()

        results = run_ablations(
            _config(duration_days=8),
            _FakeRlPolicy(),
            tiny_market_view,
            train_fn=train_fn,
        )
        assert calls == [name for name, *_ in ABLATION_VARIANTS]
        assert [r.ablation_name for r in results] == [name for name, *_ in ABLATION_VARIANTS]


@pytest.mark.slow
class TestRegimeEvaluation:
    def test_returns_four_regimes_per_policy(self, tiny_market_view: MarketView) -> None:
        config = _config(duration_days=4)
        policies = {"hodl": HODLPolicy(), "tau": TauResetPolicy(120)}
        matrices = run_regime_evaluation(config, policies, tiny_market_view)
        assert [m.policy_name for m in matrices] == ["hodl", "tau"]
        for matrix in matrices:
            assert matrix.regimes == list(REGIME_LABELS)
            frame = matrix.metrics_per_regime
            assert set(frame["regime"].unique().to_list()) == set(REGIME_LABELS)
            # Every regime has at least one episode on the tiny fixture.
            counts = frame.filter(pl.col("metric") == "total_pnl")["n"].to_list()
            assert all(count > 0 for count in counts)

    def test_unknown_regime_is_reported_not_dropped(
        self, tiny_market_view: MarketView
    ) -> None:
        empty = tiny_market_view.regimes.clear()
        view = dataclasses.replace(tiny_market_view, regimes=empty)
        matrices = run_regime_evaluation(
            _config(duration_days=8), {"hodl": HODLPolicy()}, view
        )
        matrix = matrices[0]
        assert matrix.regimes[:4] == list(REGIME_LABELS)
        assert "unknown" in matrix.regimes
        unknown = matrix.metrics_per_regime.filter(
            (pl.col("regime") == "unknown") & (pl.col("metric") == "total_pnl")
        )
        assert unknown.height == 1
        assert int(unknown["n"][0]) > 0


@pytest.mark.slow
class TestGapAnalysis:
    def test_gap_identity_and_decomposition(self, tiny_market_view: MarketView) -> None:
        config = _config(duration_days=8)
        report = run_gap_analysis(config, HODLPolicy(), tiny_market_view)
        assert isinstance(report, GapReport)
        assert report.policy_name == "hodl"
        assert np.isfinite(report.sim_pnl)
        assert np.isfinite(report.onchain_pnl)
        assert report.gap == pytest.approx(report.sim_pnl - report.onchain_pnl)
        assert set(report.gap_by_cost_type) == {"fees", "gas", "slippage", "il"}

    def test_gap_matches_manual_recomputation(self, tiny_market_view: MarketView) -> None:
        config = _config(duration_days=8)
        windows = walk_forward_windows(tiny_market_view, config)
        report = run_gap_analysis(config, HODLPolicy(), tiny_market_view)
        onchain = evaluate_episodes(
            HODLPolicy(), tiny_market_view, config, backend="backtest", windows=windows
        )
        sim = evaluate_episodes(
            HODLPolicy(), tiny_market_view, config, backend="replay_sim", windows=windows
        )
        expected_onchain = float(np.mean([r.metrics["total_pnl"] for r in onchain]))
        expected_sim = float(np.mean([r.metrics["total_pnl"] for r in sim]))
        assert report.onchain_pnl == pytest.approx(expected_onchain)
        assert report.sim_pnl == pytest.approx(expected_sim)
        assert report.gap == pytest.approx(expected_sim - expected_onchain)

    def test_unrepresentable_baseline_action_raises(
        self, tiny_market_view: MarketView
    ) -> None:
        # PassiveNarrow's ±5 % band is wider than the agent action grid, so the
        # simulator path cannot execute it; that is an explicit, diagnosable error.
        with pytest.raises(EvaluationError, match="not in the environment"):
            run_gap_analysis(_config(duration_days=8), PassiveNarrowPolicy(), tiny_market_view)


class TestWriteResults:
    def _artifacts(self) -> tuple[list[AblationResult], list[RegimeMatrix], GapReport]:
        ablations = [
            AblationResult(
                ablation_name=name,
                flags=dict(flags),
                metrics={metric: float(index) for index, metric in enumerate(METRIC_NAMES)},
                per_regime=pl.DataFrame(
                    {
                        "regime": list(REGIME_LABELS),
                        "metric": ["total_pnl"] * 4,
                        "mean": [1.0, 2.0, 3.0, 4.0],
                        "std": [0.1, 0.2, 0.3, 0.4],
                        "n": [1, 1, 1, 1],
                    }
                ),
            )
            for index, (name, flags, _static_gas) in enumerate(ABLATION_VARIANTS)
        ]
        frame = pl.DataFrame(
            {
                "regime": list(REGIME_LABELS) * len(METRIC_NAMES),
                "metric": [metric for metric in METRIC_NAMES for _ in REGIME_LABELS],
                "mean": [float(i) for i in range(4 * len(METRIC_NAMES))],
                "std": [float(i) / 10 for i in range(4 * len(METRIC_NAMES))],
                "n": [2] * (4 * len(METRIC_NAMES)),
            }
        )
        matrices = [
            RegimeMatrix("hodl", list(REGIME_LABELS), frame),
            RegimeMatrix("ppo_full", list(REGIME_LABELS), frame),
        ]
        gap = GapReport(
            policy_name="ppo_full",
            sim_pnl=10.0,
            onchain_pnl=8.0,
            gap=2.0,
            gap_by_cost_type={"fees": 0.5, "gas": -0.2, "slippage": 0.1, "il": 1.6},
        )
        return ablations, matrices, gap

    def test_creates_all_expected_files(self, tmp_path: Path) -> None:
        ablations, matrices, gap = self._artifacts()
        write_results(ablations, matrices, gap, tmp_path)
        expected = {
            "ablation_table.md",
            "ablation_table.csv",
            "regime_matrix_hodl.md",
            "regime_matrix_ppo_full.md",
            "regime_comparison.md",
            "gap_report.md",
            "gap_report.csv",
            "results_summary.md",
        }
        assert {path.name for path in tmp_path.iterdir()} == expected

    def test_markdown_tables_are_well_formed(self, tmp_path: Path) -> None:
        ablations, matrices, gap = self._artifacts()
        write_results(ablations, matrices, gap, tmp_path)
        for path in sorted(tmp_path.glob("*.md")):
            assert _valid_pipe_tables(path.read_text()) >= 1, path.name

    def test_csv_files_parse_with_polars(self, tmp_path: Path) -> None:
        ablations, matrices, gap = self._artifacts()
        write_results(ablations, matrices, gap, tmp_path)
        for name in ("ablation_table.csv", "gap_report.csv"):
            frame = pl.read_csv(tmp_path / name)
            assert frame.height >= 1
            assert frame.columns
        gap_frame = pl.read_csv(tmp_path / "gap_report.csv")
        total = gap_frame.filter(pl.col("component") == "total")
        assert total["gap"][0] == pytest.approx(2.0)
        assert total["sim_pnl"][0] == pytest.approx(10.0)

    def test_empty_ablations_still_writes_csv_header(self, tmp_path: Path) -> None:
        _, matrices, gap = self._artifacts()
        write_results([], matrices, gap, tmp_path)
        frame = pl.read_csv(tmp_path / "ablation_table.csv")
        assert frame.height == 0
        assert "ablation_name" in frame.columns
        assert "total_pnl" in frame.columns

    def test_empty_inputs_still_emit_a_data_row(self, tmp_path: Path) -> None:
        gap = GapReport("none", 0.0, 0.0, 0.0, {})
        write_results([], [], gap, tmp_path)
        for name in ("ablation_table.md", "regime_comparison.md", "results_summary.md"):
            assert _valid_pipe_tables((tmp_path / name).read_text()) >= 1, name
        assert "n/a" in (tmp_path / "ablation_table.md").read_text()


def _make_record(metrics: dict[str, float]) -> object:
    """A minimal ``EpisodeRecord`` for fast aggregation tests."""
    from undertow.sim.evaluate.protocol import EpisodeRecord

    return EpisodeRecord(
        policy_name="test",
        backend="backtest",
        split="train",
        start_seq=0,
        end_seq=1,
        seed=0,
        regime="bull",
        initial_capital=100.0,
        periods_per_year=100,
        metrics=metrics,
        decomposition={},
        equity_curve=pl.DataFrame({"equity": [100.0, 101.0]}),
    )
