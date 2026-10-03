"""S16 — CLI, public API surface and user docs.

The CLI is exercised two ways:

* **End-to-end via subprocess** for everything that does not need a real data
  snapshot — ``info``, ``--help``, the invalid-policy path, the missing-config
  path, and the documented data-absent failure of the data commands.  This is the
  actual ``python -m undertow.sim.cli`` entry point a user runs.
* **In-process via** :func:`undertow.sim.cli.main` with the shared
  ``tiny_market_view`` fixture injected for the ``backtest`` happy path, so the
  full command path (policy mapping → replay → artifact writing) is covered
  without network or a real snapshot.

The public API surface is checked for importability and ``__all__`` completeness;
the docs and README sections are checked for existence and content.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from undertow.sim.cli import (
    POLICY_NAMES,
    CliError,
    _action_grid,
    _CheckpointPolicy,
    _load_checkpoint,
    _make_env_factory,
    _parity_status,
    _write_ablation_report,
    build_parser,
    build_policy,
    main,
)

#: Repository root: ``tests/sim/test_cli.py`` → parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_CONFIG = REPO_ROOT / "configs" / "sim_default.toml"

#: The names the S16 brief pins as the public ``undertow.sim`` surface.
EXPECTED_EXPORTS: frozenset[str] = frozenset(
    {
        # Config
        "SimConfig",
        "load_sim_config",
        # Types
        "Action",
        "Tick",
        "TickSpacing",
        "SqrtPrice",
        "Price",
        "Wealth",
        "Regime",
        "PriceMode",
        "UndertowSimError",
        # Core
        "Position",
        "initial_deposit",
        "PoolEngine",
        "PoolState",
        "TickState",
        # Market data
        "MarketView",
        "build_market_view",
        # Prices
        "PriceProcess",
        "ReplayPriceProcess",
        "CalibratedPriceProcess",
        "MRSJDParams",
        # Frictions
        "GasModel",
        "SlippageModel",
        # Policies
        "Policy",
        "HODLPolicy",
        "PassiveNarrowPolicy",
        "PassiveWidePolicy",
        "FullRangeV2Policy",
        "TauResetPolicy",
        "CostAwareRebalancePolicy",
        # Environment
        "LpEnvironment",
        "Observation",
        "compute_reward",
        "RewardBreakdown",
        # Backtester
        "run_backtest",
        "summarize_backtest",
        "BacktestLedger",
        "BacktestResult",
        # Metrics
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "annualized_return",
        "all_metrics",
        "pnl_decomposition",
        "per_regime_metrics",
        # Training
        "train_ppo",
        "train_single_seed",
        "TrainingResult",
        "RunManifest",
        # Validation
        "run_parity_check",
        "run_lookahead_probes",
        # Evaluation
        "run_ablations",
        "run_regime_evaluation",
        "run_gap_analysis",
        "write_results",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _StubModel:
    """A minimal SB3-model stand-in returning a fixed discrete action index."""

    def __init__(self, next_index: int = 0) -> None:
        self.next_index = int(next_index)

    def predict(
        self, vector: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        del vector, deterministic
        return np.asarray([self.next_index]), None


def _run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run the real CLI module in a subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "undertow.sim.cli", *args],
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _write_minimal_data_config(tmp_path: Path, output_dir: Path) -> Path:
    """Write a valid data TOML pointing at ``output_dir`` (usually empty).

    The endpoints are literal localhost URLs (no ``${ENV}`` placeholders) so the
    sim CLI can load the config without network or environment setup.  Because
    ``output_dir`` has no manifest, the data commands must fail with the
    documented "run ``undertow-data snapshot``" message.
    """
    from undertow.data import default_pools

    pool = default_pools()["USDC_WETH_3000"]
    config_path = tmp_path / "data_config.toml"
    config_path.write_text(
        f"""
[pool]
address = "{pool.address}"
token0_symbol = "{pool.token0_symbol}"
token1_symbol = "{pool.token1_symbol}"
token0_decimals = {pool.token0_decimals}
token1_decimals = {pool.token1_decimals}
fee_tier = {pool.fee_tier.value}
tick_spacing = {pool.tick_spacing}
deployment_block = {pool.deployment_block}

[window]
start_utc = "2022-01-01T00:00:00Z"
end_utc = "2024-12-31T23:59:59Z"

[regime]
lookback_days = 30
vol_threshold = 0.80
drift_threshold = 0.05
periods_per_year = 525600

[endpoints]
graph_url = "http://localhost:9999/graph"
rpc_url = "http://localhost:9999/rpc"
reference_base_url = "http://localhost:9999/ref"

[paths]
cache_dir = "{output_dir}/cache"
output_dir = "{output_dir}"
""",
        encoding="utf-8",
    )
    return config_path


# ---------------------------------------------------------------------------
# 1. info
# ---------------------------------------------------------------------------
class TestInfo:
    def test_info_runs_and_prints_config_summary(self, tmp_path: Path) -> None:
        result = _run_cli(
            "info", "--config", str(SIM_CONFIG), "--output", str(tmp_path)
        )
        assert result.returncode == 0, result.stderr
        assert "configuration summary" in result.stdout
        assert "action-space size:" in result.stdout
        assert "available policies:" in result.stdout
        assert "train:" in result.stdout and "eval:" in result.stdout
        assert "parity status:" in result.stdout

    def test_info_action_space_size_is_one_plus_grid(self, tmp_path: Path) -> None:
        result = _run_cli("info", "--output", str(tmp_path))
        assert result.returncode == 0, result.stderr
        # 1 hold + 9 offsets * 6 widths = 55.
        assert "action-space size: 55" in result.stdout

    def test_info_reports_parity_when_report_present(self, tmp_path: Path) -> None:
        report = {
            "passed": True,
            "tolerance": 1e-3,
            "justification": "test",
        }
        (tmp_path / "parity_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        result = _run_cli("info", "--output", str(tmp_path))
        assert result.returncode == 0, result.stderr
        assert "PASS" in result.stdout
        assert "0.001" in result.stdout

    def test_parity_status_missing_and_unreadable(self, tmp_path: Path) -> None:
        assert "not available" in _parity_status(tmp_path)
        (tmp_path / "parity_report.json").write_text("{not json", encoding="utf-8")
        assert "unreadable" in _parity_status(tmp_path)


# ---------------------------------------------------------------------------
# 2. backtest — synthetic happy path (in-process) + documented data error
# ---------------------------------------------------------------------------
class TestBacktest:
    def test_backtest_hodl_writes_artifacts(
        self, tiny_market_view: object, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        view = tiny_market_view.slice(0, 200)  # type: ignore[attr-defined]
        exit_code = main(
            [
                "backtest",
                "--policy",
                "hodl",
                "--config",
                str(SIM_CONFIG),
                "--output",
                str(tmp_path),
            ],
            market_view=view,
        )
        captured = capsys.readouterr()
        assert exit_code == 0, captured.err
        for name in (
            "equity_curve.csv",
            "decision_log.csv",
            "cost_ledger.csv",
            "summary.json",
        ):
            artifact = tmp_path / name
            assert artifact.is_file(), name
            assert artifact.stat().st_size > 0, name
        payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
        assert payload["policy_name"] == "hodl"
        assert "total_pnl" in payload["result"]

    def test_backtest_every_policy_name_builds(self) -> None:
        from undertow.sim import SimConfig

        config = SimConfig()
        names = {build_policy(name, config).name for name in POLICY_NAMES}
        assert names == set(POLICY_NAMES)

    def test_backtest_without_snapshot_reports_documented_error(
        self, tmp_path: Path
    ) -> None:
        data_config = _write_minimal_data_config(tmp_path, tmp_path / "no_snapshot")
        result = _run_cli(
            "backtest",
            "--policy",
            "hodl",
            "--data-config",
            str(data_config),
            "--output",
            str(tmp_path / "out"),
        )
        assert result.returncode != 0
        assert "Build the snapshot" in result.stderr
        assert "undertow-data snapshot" in result.stderr


# ---------------------------------------------------------------------------
# 3. invalid policy
# ---------------------------------------------------------------------------
class TestInvalidPolicy:
    def test_invalid_policy_exits_nonzero_and_lists_policies(self, tmp_path: Path) -> None:
        result = _run_cli(
            "backtest",
            "--policy",
            "not_a_policy",
            "--data-config",
            str(tmp_path / "missing.toml"),
            "--output",
            str(tmp_path / "out"),
        )
        assert result.returncode != 0
        assert "available policies" in result.stderr
        for name in POLICY_NAMES:
            assert name in result.stderr

    def test_invalid_policy_is_reported_before_data_is_touched(self, tmp_path: Path) -> None:
        # No --data-config and no snapshot in cwd; the policy error must win.
        result = _run_cli(
            "backtest", "--policy", "bogus", "--output", str(tmp_path / "out")
        )
        assert result.returncode != 0
        assert "unknown policy" in result.stderr
        assert "available policies" in result.stderr

    def test_build_policy_rejects_unknown_name(self) -> None:
        from undertow.sim import SimConfig
        from undertow.sim.cli import CliError

        with pytest.raises(CliError, match="available policies"):
            build_policy("nope", SimConfig())


# ---------------------------------------------------------------------------
# 4. missing / invalid config
# ---------------------------------------------------------------------------
class TestMissingConfig:
    def test_missing_config_exits_nonzero(self, tmp_path: Path) -> None:
        result = _run_cli(
            "info",
            "--config",
            str(tmp_path / "does_not_exist.toml"),
            "--output",
            str(tmp_path),
        )
        assert result.returncode != 0
        assert "config file not found" in result.stderr

    def test_invalid_toml_config_exits_nonzero(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.toml"
        bad.write_text("this is not = valid = toml", encoding="utf-8")
        result = _run_cli("info", "--config", str(bad), "--output", str(tmp_path))
        assert result.returncode != 0
        assert "failed to load sim config" in result.stderr

    def test_missing_data_config_exits_nonzero(self, tmp_path: Path) -> None:
        result = _run_cli(
            "ablate",
            "--data-config",
            str(tmp_path / "missing_data.toml"),
            "--output",
            str(tmp_path / "out"),
        )
        assert result.returncode != 0
        assert "data config not found" in result.stderr


# ---------------------------------------------------------------------------
# 5. help text
# ---------------------------------------------------------------------------
class TestHelp:
    def test_help_lists_all_five_subcommands(self) -> None:
        result = _run_cli("--help")
        assert result.returncode == 0
        for command in ("train", "backtest", "evaluate", "ablate", "info"):
            assert command in result.stdout

    def test_subcommand_help_runs(self) -> None:
        for command in ("train", "backtest", "evaluate", "ablate", "info"):
            result = _run_cli(command, "--help")
            assert result.returncode == 0, command
            assert "usage:" in result.stdout

    def test_parser_lists_the_five_subcommands(self) -> None:
        parser = build_parser()
        action = next(
            a for a in parser._actions if getattr(a, "choices", None) is not None
        )
        assert set(action.choices) == {"train", "backtest", "evaluate", "ablate", "info"}


# ---------------------------------------------------------------------------
# 6 & 7. public API surface
# ---------------------------------------------------------------------------
class TestPublicApi:
    def test_brief_import_line_resolves(self) -> None:
        from undertow.sim import (  # noqa: F401
            LpEnvironment,
            MarketView,
            SimConfig,
            run_backtest,
            train_ppo,
        )

        assert SimConfig is not None
        assert MarketView is not None
        assert LpEnvironment is not None
        assert callable(run_backtest)
        assert callable(train_ppo)

    def test_all_names_resolve(self) -> None:
        import undertow.sim as sim

        assert isinstance(sim.__all__, list)
        missing = [name for name in sim.__all__ if not hasattr(sim, name)]
        assert missing == []

    def test_all_is_complete_and_has_no_duplicates(self) -> None:
        import undertow.sim as sim

        assert set(sim.__all__) >= EXPECTED_EXPORTS
        assert len(sim.__all__) == len(set(sim.__all__))

    def test_no_data_internal_imports_leak_into_sim(self) -> None:
        """`undertow.sim` may import only the top-level `undertow.data` API."""
        offenders: list[str] = []
        for path in (REPO_ROOT / "src" / "undertow" / "sim").rglob("*.py"):
            if "from undertow.data." in path.read_text(encoding="utf-8"):
                offenders.append(str(path))
        assert offenders == []


# ---------------------------------------------------------------------------
# 8 & 9. docs
# ---------------------------------------------------------------------------
class TestDocs:
    def test_running_experiments_guide_exists_and_is_complete(self) -> None:
        guide = REPO_ROOT / "docs" / "running_experiments.md"
        assert guide.is_file()
        text = guide.read_text(encoding="utf-8")
        assert len(text.strip()) > 0
        for command in ("undertow-sim info", "undertow-sim backtest", "undertow-sim train"):
            assert command in text
        # The correct sync command (no non-existent `sim` group).
        assert "uv sync" in text
        assert "uv sync --group sim" not in text
        # The snapshot prerequisite is stated.
        assert "undertow-data snapshot" in text

    def test_readme_has_sim_status_section(self) -> None:
        readme = REPO_ROOT / "README.md"
        text = readme.read_text(encoding="utf-8")
        assert "`undertow.sim` — Simulator, Backtester & RL Environment" in text
        for artifact in (
            "Simulator (Gymnasium env)",
            "Backtester (on-chain replay)",
            "PPO training harness",
            "Baseline ladder",
            "Parity validation",
            "RQ1 ablation runner",
            "RQ2 regime evaluation",
            "RQ3 gap analysis",
        ):
            assert artifact in text
        assert "docs/running_experiments.md" in text


# ---------------------------------------------------------------------------
# evaluate / ablate — documented data-absent failures
# ---------------------------------------------------------------------------
class TestCliInternals:
    def test_action_grid_matches_environment_index_order(self, tiny_env: object) -> None:
        """The checkpoint adapter must use the environment's exact action order.

        ``LpEnvironment`` builds index 0 = hold then the row-major grid; a silent
        divergence would map every SB3 action to the wrong range.
        """
        expected = [tiny_env.action_for_index(i) for i in range(tiny_env.n_actions)]
        assert _action_grid(tiny_env.config) == expected

    def test_action_grid_respects_a_custom_grid_row_major_order(self) -> None:
        from undertow.sim.config import ActionGridConfig, SimConfig
        from undertow.sim.types import Action

        config = dataclasses.replace(
            SimConfig(),
            action_grid=ActionGridConfig(center_offsets=(-1, 1), widths=(2, 5)),
        )
        assert _action_grid(config) == [
            Action(action_type="hold"),
            Action(action_type="rebalance", center_offset=-1, width=2),
            Action(action_type="rebalance", center_offset=-1, width=5),
            Action(action_type="rebalance", center_offset=1, width=2),
            Action(action_type="rebalance", center_offset=1, width=5),
        ]

    def test_checkpoint_policy_maps_indices_to_environment_actions(
        self, tiny_env: object
    ) -> None:
        from undertow.sim.evaluate import observation_from_vector

        vector, _info = tiny_env.reset(seed=0)
        observation = observation_from_vector(vector)
        model = _StubModel()
        policy = _CheckpointPolicy(model, tiny_env.config, name="ppo")
        assert policy.name == "ppo"
        policy.reset()  # no-op, must not raise
        for index in range(tiny_env.n_actions):
            model.next_index = index
            assert policy.act(observation) == tiny_env.action_for_index(index)

    def test_checkpoint_policy_clamps_out_of_range_index(self, tiny_env: object) -> None:
        from undertow.sim.evaluate import observation_from_vector

        vector, _info = tiny_env.reset(seed=0)
        observation = observation_from_vector(vector)
        model = _StubModel(next_index=tiny_env.n_actions + 100)
        policy = _CheckpointPolicy(model, tiny_env.config)
        assert policy.act(observation) == tiny_env.action_for_index(
            tiny_env.n_actions - 1
        )
        model.next_index = -5
        assert policy.act(observation) == tiny_env.action_for_index(0)

    def test_calibrated_env_factory_builds_calibrated_env(
        self, tiny_market_view: object
    ) -> None:
        from undertow.sim.config import SimConfig
        from undertow.sim.prices import CalibratedPriceProcess
        from undertow.sim.types import PriceMode

        config = dataclasses.replace(
            SimConfig(), price_mode=PriceMode.CALIBRATED
        )
        env = _make_env_factory(tiny_market_view, config)()
        assert not env.is_replay
        assert isinstance(env.price_process, CalibratedPriceProcess)

    def test_calibrated_env_factory_rejects_eval_view(
        self, tiny_market_view: object
    ) -> None:
        from undertow.sim.config import SimConfig
        from undertow.sim.types import PriceMode

        eval_view = dataclasses.replace(
            tiny_market_view,
            train=None,
            eval=tiny_market_view.train,
            is_train=False,
            eval_start_utc=tiny_market_view.train_start_utc,
            eval_end_utc=tiny_market_view.train_end_utc,
        )
        config = dataclasses.replace(
            SimConfig(), price_mode=PriceMode.CALIBRATED
        )
        with pytest.raises(CliError, match="train split"):
            _make_env_factory(eval_view, config)

    def test_replay_env_factory_returns_replay_env(
        self, tiny_market_view: object
    ) -> None:
        from undertow.sim.config import SimConfig

        env = _make_env_factory(tiny_market_view, SimConfig())()
        assert env.is_replay

    def test_load_checkpoint_delegates_to_ppo_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checkpoint = tmp_path / "best.zip"
        checkpoint.write_bytes(b"placeholder")
        calls: dict[str, object] = {}

        class _FakePPO:
            @staticmethod
            def load(path: str, device: str = "cpu") -> object:
                calls["path"] = path
                calls["device"] = device
                return "loaded-model"

        monkeypatch.setattr("stable_baselines3.PPO", _FakePPO)
        assert _load_checkpoint(checkpoint) == "loaded-model"
        assert calls == {"path": str(checkpoint), "device": "cpu"}

    def test_load_checkpoint_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CliError, match="checkpoint not found"):
            _load_checkpoint(tmp_path / "nope.zip")


class TestTrainEvaluateSuccessPaths:
    def test_train_writes_summary_with_fake_trainer(
        self,
        tiny_market_view: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import undertow.sim.cli as cli
        from undertow.sim.train import RunManifest, TrainingResult

        def fake_train_ppo(
            env_fn: object, config: object, log_dir: Path | None = None
        ) -> TrainingResult:
            manifest = RunManifest(
                run_id="fake-run",
                seed=0,
                config_hash="deadbeef",
                git_commit="cafe",
                algorithm="ppo",
                total_timesteps=10,
                best_reward=1.5,
                checkpoint_path=Path(log_dir or ".") / "best.zip",
            )
            return TrainingResult(
                seeds=(0,),
                manifests=(manifest,),
                learning_curves={"mean_reward": [1.5]},
                best_checkpoint=manifest.checkpoint_path,
            )

        monkeypatch.setattr(cli, "train_ppo", fake_train_ppo)
        exit_code = cli.main(
            ["train", "--config", str(SIM_CONFIG), "--output", str(tmp_path)],
            market_view=tiny_market_view,
        )
        captured = capsys.readouterr()
        assert exit_code == 0, captured.err
        payload = json.loads(
            (tmp_path / "training_summary.json").read_text(encoding="utf-8")
        )
        assert payload["seeds"] == [0]
        assert payload["best_checkpoint"].endswith("best.zip")
        assert payload["manifests"][0]["best_reward"] == 1.5

    def test_evaluate_wires_checkpoint_and_writes_results(
        self,
        tiny_market_view: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import undertow.sim.cli as cli
        from undertow.sim.evaluate import GapReport

        checkpoint = tmp_path / "best.zip"
        checkpoint.write_bytes(b"placeholder")
        monkeypatch.setattr(cli, "_load_checkpoint", lambda path: object())
        captured: dict[str, object] = {}

        def fake_regime(
            config: object,
            policies: dict[str, object],
            view: object,
            num_seeds: int | None = None,
            **kwargs: object,
        ) -> list[object]:
            del config, view, num_seeds, kwargs
            captured["policies"] = list(policies)
            return []

        def fake_gap(config: object, policy: object, view: object, **kwargs: object) -> GapReport:
            del config, view, kwargs
            captured["gap_policy"] = policy.name
            return GapReport(
                policy_name=policy.name,
                sim_pnl=0.0,
                onchain_pnl=0.0,
                gap=0.0,
                gap_by_cost_type={},
            )

        def fake_write(
            ablation: object, regime: object, gap: object, output: Path
        ) -> None:
            del ablation, regime, gap
            captured["wrote"] = str(output)

        monkeypatch.setattr(cli, "run_regime_evaluation", fake_regime)
        monkeypatch.setattr(cli, "run_gap_analysis", fake_gap)
        monkeypatch.setattr(cli, "write_results", fake_write)

        exit_code = cli.main(
            [
                "evaluate",
                "--checkpoint",
                str(checkpoint),
                "--config",
                str(SIM_CONFIG),
                "--output",
                str(tmp_path / "out"),
            ],
            market_view=tiny_market_view,
        )
        assert exit_code == 0
        assert captured["wrote"] == str(tmp_path / "out")
        assert captured["gap_policy"] == "best"
        assert set(captured["policies"]) >= set(POLICY_NAMES)

    def test_evaluate_checkpoint_named_after_baseline_is_not_overwritten(
        self,
        tiny_market_view: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import undertow.sim.cli as cli

        checkpoint = tmp_path / "hodl.zip"
        checkpoint.write_bytes(b"placeholder")
        monkeypatch.setattr(cli, "_load_checkpoint", lambda path: object())
        captured: dict[str, object] = {}

        def fake_regime(
            config: object,
            policies: dict[str, object],
            view: object,
            num_seeds: int | None = None,
            **kwargs: object,
        ) -> list[object]:
            del config, view, num_seeds, kwargs
            captured["policies"] = list(policies)
            return []

        from undertow.sim.evaluate import GapReport

        monkeypatch.setattr(cli, "run_regime_evaluation", fake_regime)
        monkeypatch.setattr(
            cli,
            "run_gap_analysis",
            lambda config, policy, view, **kw: GapReport(
                policy_name=policy.name,
                sim_pnl=0.0,
                onchain_pnl=0.0,
                gap=0.0,
                gap_by_cost_type={},
            ),
        )
        monkeypatch.setattr(
            cli, "write_results", lambda ablation, regime, gap, output: None
        )
        exit_code = cli.main(
            [
                "evaluate",
                "--checkpoint",
                str(checkpoint),
                "--config",
                str(SIM_CONFIG),
                "--output",
                str(tmp_path / "out"),
            ],
            market_view=tiny_market_view,
        )
        assert exit_code == 0
        assert "hodl" in captured["policies"]
        assert "ppo_hodl" in captured["policies"]


class TestAblationWriter:
    def test_writes_markdown_and_csv_tables(self, tmp_path: Path) -> None:
        from undertow.sim.evaluate import AblationResult

        metrics = {
            "total_pnl": 123.456,
            "sharpe": 1.5,
            "max_drawdown": -0.2,
            "fee_income": 50.0,
            "gas_paid": -5.0,
            "slippage_paid": -2.0,
            "il_realized": -10.0,
            "n_rebalances": 3.0,
        }
        results = [
            AblationResult(
                ablation_name="full",
                flags={"fees": True, "gas": True, "slippage": True, "il": True},
                metrics=metrics,
            )
        ]
        _write_ablation_report(results, tmp_path)
        markdown = (tmp_path / "ablation_table.md").read_text(encoding="utf-8")
        assert "| Ablation |" in markdown
        assert "| full |" in markdown
        assert "123.46" in markdown
        csv_text = (tmp_path / "ablation_table.csv").read_text(encoding="utf-8")
        assert csv_text.splitlines()[0].startswith("ablation_name,fees,gas,slippage,il")
        assert "full" in csv_text

    def test_ablate_writes_ablation_only_artifacts(
        self, tiny_market_view: object, tmp_path: Path
    ) -> None:
        view = tiny_market_view.slice(0, 400)  # type: ignore[attr-defined]
        exit_code = main(
            [
                "ablate",
                "--policy",
                "hodl",
                "--config",
                str(SIM_CONFIG),
                "--output",
                str(tmp_path),
            ],
            market_view=view,
        )
        assert exit_code == 0
        markdown = (tmp_path / "ablation_table.md").read_text(encoding="utf-8")
        assert "Friction ablation table" in markdown
        for variant in ("full", "no_gas", "no_slippage", "no_il", "static_fee"):
            assert variant in markdown
        assert (tmp_path / "ablation_table.csv").is_file()
        # ablate is RQ1-only: it must not fabricate an RQ3 gap report.
        assert not (tmp_path / "gap_report.md").exists()

    def test_empty_input_writes_header_only_csv(self, tmp_path: Path) -> None:
        _write_ablation_report([], tmp_path)
        csv_text = (tmp_path / "ablation_table.csv").read_text(encoding="utf-8")
        assert csv_text.splitlines()[0].startswith("ablation_name,")
        markdown = (tmp_path / "ablation_table.md").read_text(encoding="utf-8")
        assert "Ablation" in markdown


class TestEvaluateAblateDataErrors:
    def test_ablate_without_snapshot_reports_documented_error(self, tmp_path: Path) -> None:
        data_config = _write_minimal_data_config(tmp_path, tmp_path / "no_snapshot")
        result = _run_cli(
            "ablate",
            "--data-config",
            str(data_config),
            "--output",
            str(tmp_path / "out"),
        )
        assert result.returncode != 0
        assert "Build the snapshot" in result.stderr

    def test_evaluate_without_snapshot_reports_documented_error(self, tmp_path: Path) -> None:
        data_config = _write_minimal_data_config(tmp_path, tmp_path / "no_snapshot")
        checkpoint = tmp_path / "fake.zip"
        checkpoint.write_bytes(b"not a real checkpoint")
        result = _run_cli(
            "evaluate",
            "--checkpoint",
            str(checkpoint),
            "--data-config",
            str(data_config),
            "--output",
            str(tmp_path / "out"),
        )
        assert result.returncode != 0
        assert "Build the snapshot" in result.stderr

    def test_evaluate_missing_checkpoint_exits_nonzero(self, tmp_path: Path) -> None:
        result = _run_cli(
            "evaluate",
            "--checkpoint",
            str(tmp_path / "nope.zip"),
            "--output",
            str(tmp_path / "out"),
        )
        assert result.returncode != 0
        assert "checkpoint not found" in result.stderr
