"""``undertow-sim`` — the command-line entry point for the simulator (S16).

Five subcommands (``train``, ``backtest``, ``evaluate``, ``ablate``, ``info``)
covering the whole plan's runtime surface.  The CLI is a thin composition root:

* it parses arguments (``argparse`` — stdlib, no new dependency),
* loads the frozen :class:`~undertow.sim.config.SimConfig` via
  :func:`~undertow.sim.config.load_sim_config`,
* loads historical data through the look-ahead-safe
  :func:`~undertow.sim.marketview.build_market_view` door,
* delegates the actual work to the public ``undertow.sim`` sub-packages,
* writes artifacts under ``--output`` (default ``./results/``).

Library code logs through :mod:`logging`; every user-facing line printed here is
written by this module, to stdout (results) or stderr (errors).  Exit codes:
``0`` ok, ``1`` runtime/data error, ``2`` configuration error.

The data-dependent commands read a *pre-built* dataset: build it once with
``undertow-data snapshot --config <data-config>``.  ``--data-config`` defaults to
``configs/eth_usdc_3000.toml``; the snapshot is never fetched here.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import polars as pl

from undertow.data import load_data_config
from undertow.sim.backtest import BacktestLedger, BacktestResult, run_backtest, summarize_backtest
from undertow.sim.config import SimConfig, load_sim_config
from undertow.sim.core.pool import PoolEngine, PoolState
from undertow.sim.env import LpEnvironment
from undertow.sim.evaluate import (
    METRIC_NAMES,
    AblationResult,
    build_replay_env,
    run_ablations,
    run_gap_analysis,
    run_regime_evaluation,
    write_results,
)
from undertow.sim.frictions import (
    FlatGasModel,
    ProportionalSlippageModel,
    ReplayGasModel,
)
from undertow.sim.marketview import MarketView, build_market_view
from undertow.sim.policies import (
    CostAwareRebalancePolicy,
    FullRangeV2Policy,
    HODLPolicy,
    PassiveNarrowPolicy,
    PassiveWidePolicy,
    Policy,
    TauResetPolicy,
)
from undertow.sim.prices import CalibratedPriceProcess
from undertow.sim.prices.base import human_price_to_triplet
from undertow.sim.train import train_ppo
from undertow.sim.types import Action, PriceMode

__all__ = ["main"]

PROG: str = "undertow-sim"

#: Default paths, resolved relative to the current working directory.
DEFAULT_SIM_CONFIG: Path = Path("configs/sim_default.toml")
DEFAULT_DATA_CONFIG: Path = Path("configs/eth_usdc_3000.toml")
DEFAULT_OUTPUT_DIR: Path = Path("results")

#: The six baseline policies selectable with ``--policy`` (also the order they are
#: listed when an unknown name is supplied).
POLICY_NAMES: tuple[str, ...] = (
    "hodl",
    "passive_narrow",
    "passive_wide",
    "full_range_v2",
    "tau_reset",
    "cost_aware",
)

#: Half-width, in ticks, used for the tau-reset and cost-aware baselines.  Matches
#: the backtester's default initial band (±120 ticks) so a rebalanced band is the
#: same size as the initial deployment.
DEFAULT_HALF_WIDTH_TICKS: int = 120

#: Minimum decision interval (steps) for the cost-aware policy — one day at the
#: pinned 10-minute cadence (144 steps).
DEFAULT_MIN_INTERVAL_STEPS: int = 144

#: Fallback external liquidity for a hand-built tape with no ``liquidity`` column.
_DEFAULT_BASE_LIQUIDITY: float = 1.0e18

EXIT_OK: int = 0
EXIT_ERROR: int = 1
EXIT_CONFIG: int = 2


class CliError(Exception):
    """A user-facing CLI failure carrying the process exit code."""

    def __init__(self, message: str, exit_code: int = EXIT_ERROR) -> None:
        super().__init__(message)
        self.exit_code = int(exit_code)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _add_sim_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=str(DEFAULT_SIM_CONFIG),
        help=f"path to the sim TOML config (default: {DEFAULT_SIM_CONFIG})",
    )


def _add_data_args(parser: argparse.ArgumentParser, *, split: str) -> None:
    parser.add_argument(
        "--data-config",
        default=str(DEFAULT_DATA_CONFIG),
        help=(
            "path to the data TOML config whose [paths].output_dir holds the "
            f"snapshot (default: {DEFAULT_DATA_CONFIG})"
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "eval"),
        default=split,
        help=f"walk-forward split to read (default: {split})",
    )


def _add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"directory for result artifacts (created if absent; default: {DEFAULT_OUTPUT_DIR})",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``undertow-sim`` argument parser."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Undertow AMM simulator — train PPO agents, backtest frozen policies, "
            "run the RQ1-RQ3 evaluations, and inspect the configuration."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="re-raise exceptions with full tracebacks (default: clean messages)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="train PPO across the configured seeds")
    _add_sim_config(train_p)
    _add_data_args(train_p, split="train")
    _add_output(train_p)

    backtest_p = sub.add_parser(
        "backtest", help="replay a frozen baseline policy over the event tape"
    )
    _add_sim_config(backtest_p)
    _add_data_args(backtest_p, split="eval")
    _add_output(backtest_p)
    backtest_p.add_argument(
        "--policy",
        default="passive_narrow",
        help=(
            "baseline policy to replay (default: passive_narrow; one of "
            f"{', '.join(POLICY_NAMES)})"
        ),
    )

    evaluate_p = sub.add_parser(
        "evaluate", help="walk-forward evaluation of a trained policy + baselines"
    )
    _add_sim_config(evaluate_p)
    _add_data_args(evaluate_p, split="eval")
    _add_output(evaluate_p)
    evaluate_p.add_argument(
        "--checkpoint",
        required=True,
        help="path to a trained PPO checkpoint (.zip) to evaluate",
    )
    evaluate_p.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="number of configured seeds to evaluate over (default: 5)",
    )

    ablate_p = sub.add_parser("ablate", help="run the RQ1 friction-ablation matrix")
    _add_sim_config(ablate_p)
    _add_data_args(ablate_p, split="eval")
    _add_output(ablate_p)
    ablate_p.add_argument(
        "--policy",
        default="passive_narrow",
        help=(
            "baseline policy to ablate (default: passive_narrow; one of "
            f"{', '.join(POLICY_NAMES)})"
        ),
    )

    info_p = sub.add_parser("info", help="print the configuration summary and parity status")
    _add_sim_config(info_p)
    _add_output(info_p)

    return parser


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _load_config(path_str: str | Path) -> SimConfig:
    """Load the sim config, translating any failure into a config error."""
    path = Path(path_str)
    if not path.is_file():
        raise CliError(f"config file not found: {path}", EXIT_CONFIG)
    try:
        return load_sim_config(path)
    except Exception as exc:  # noqa: BLE001 - report any config failure cleanly
        raise CliError(f"failed to load sim config {path}: {exc}", EXIT_CONFIG) from exc


def _load_market_view(
    config: SimConfig, data_config_path: str | Path, split: str
) -> MarketView:
    """Load the pre-built dataset and partition it into a :class:`MarketView`.

    Reads only through the top-level ``undertow.data`` public API.  A missing or
    invalid snapshot becomes a clear, actionable error rather than a traceback.
    """
    data_path = Path(data_config_path)
    if not data_path.is_file():
        raise CliError(
            f"data config not found: {data_path}. Build the snapshot first with "
            f"'undertow-data snapshot --config {data_path}'.",
            EXIT_CONFIG,
        )
    try:
        data_config = load_data_config(data_path)
    except Exception as exc:  # noqa: BLE001 - report any data-config failure cleanly
        raise CliError(
            f"failed to load data config {data_path}: {exc}", EXIT_CONFIG
        ) from exc
    try:
        return build_market_view(
            data_config,
            config.split,
            split=split,  # type: ignore[arg-type]
            episode_config=config.episode,
        )
    except Exception as exc:  # noqa: BLE001 - missing/invalid snapshot is expected
        raise CliError(
            f"could not load the {split} split from {data_path}: {exc}. "
            f"Build the snapshot with 'undertow-data snapshot --config {data_path}'.",
            EXIT_ERROR,
        ) from exc


def _resolve_output(output: str | Path | None) -> Path:
    """Resolve and create the output directory (default ``./results/``)."""
    path = Path(output) if output else DEFAULT_OUTPUT_DIR
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CliError(f"cannot create output directory {path}: {exc}", EXIT_ERROR) from exc
    return path


# ---------------------------------------------------------------------------
# Policy construction
# ---------------------------------------------------------------------------
def validate_policy_name(name: str) -> None:
    """Raise :class:`CliError` (with the menu) when ``name`` is not a policy."""
    if name not in POLICY_NAMES:
        raise CliError(
            f"unknown policy {name!r}; available policies: {', '.join(POLICY_NAMES)}",
            EXIT_CONFIG,
        )


def build_policy(name: str, config: SimConfig) -> Policy:
    """Map a ``--policy`` name to a constructed baseline policy.

    Raises :class:`CliError` with the list of available policies for an unknown
    name, so the CLI exits non-zero with an actionable message.
    """
    validate_policy_name(name)
    spacing = int(config.episode.tick_spacing)
    if name == "hodl":
        return HODLPolicy()
    if name == "passive_narrow":
        return PassiveNarrowPolicy(tick_spacing=spacing)
    if name == "passive_wide":
        return PassiveWidePolicy(tick_spacing=spacing)
    if name == "full_range_v2":
        return FullRangeV2Policy(tick_spacing=spacing)
    if name == "tau_reset":
        return TauResetPolicy(DEFAULT_HALF_WIDTH_TICKS, tick_spacing=spacing)
    if name == "cost_aware":
        return CostAwareRebalancePolicy(
            DEFAULT_HALF_WIDTH_TICKS,
            DEFAULT_MIN_INTERVAL_STEPS,
            FlatGasModel(config.gas),
            tick_spacing=spacing,
        )
    raise CliError(f"policy {name!r} is not implemented", EXIT_CONFIG)  # pragma: no cover


def _action_grid(config: SimConfig) -> list[Action]:
    """The discrete action list, matching ``LpEnvironment``'s index order.

    Index ``0`` is ``hold``; the remaining indices enumerate the
    ``(center_offset, width)`` grid row-major (ADR-011).
    """
    actions: list[Action] = [Action(action_type="hold")]
    for center_offset in config.action_grid.center_offsets:
        for width in config.action_grid.widths:
            actions.append(
                Action(
                    action_type="rebalance",
                    center_offset=int(center_offset),
                    width=int(width),
                )
            )
    return actions


class _CheckpointPolicy:
    """A frozen :class:`Policy` wrapping a trained SB3 PPO checkpoint.

    ``act`` runs the model deterministically on the observation vector and maps
    the returned discrete index back to the matching factored :class:`Action`
    (ADR-011).  It is the only adapter between an SB3 model and the simulator's
    policy protocol, and lives with the composition root (the CLI).
    """

    def __init__(self, model: object, config: SimConfig, name: str = "ppo") -> None:
        self._model = model
        self._name = name
        self._actions = _action_grid(config)

    @property
    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        """No episode-local state to reset."""

    def act(
        self,
        observation: object,
        rng: np.random.Generator | None = None,
    ) -> Action:
        """Predict a discrete action index and return its factored action."""
        del rng  # the checkpoint policy is deterministic
        vector = np.asarray(observation.to_vector(), dtype=np.float64)  # type: ignore[attr-defined]
        raw_action, _state = self._model.predict(vector, deterministic=True)  # type: ignore[attr-defined]
        index = int(np.asarray(raw_action).reshape(-1)[0])
        index = max(0, min(index, len(self._actions) - 1))
        return self._actions[index]


def _load_checkpoint(checkpoint: Path) -> object:
    """Load a stable-baselines3 PPO checkpoint with a clear failure message."""
    if not checkpoint.is_file():
        raise CliError(f"checkpoint not found: {checkpoint}", EXIT_CONFIG)
    try:
        from stable_baselines3 import PPO  # noqa: PLC0415 - optional runtime import

        return PPO.load(str(checkpoint), device="cpu")
    except ImportError as exc:  # pragma: no cover - dev group installs SB3
        raise CliError(
            "stable-baselines3 is not installed; install it with 'uv sync' "
            "(it is a dev-group dependency).",
            EXIT_CONFIG,
        ) from exc
    except Exception as exc:  # noqa: BLE001 - corrupt checkpoint is user input
        raise CliError(f"failed to load checkpoint {checkpoint}: {exc}", EXIT_ERROR) from exc


# ---------------------------------------------------------------------------
# Environment factory (training)
# ---------------------------------------------------------------------------
def _first_positive_liquidity(tape: pl.DataFrame) -> float:
    """First non-null, positive ``liquidity`` value in a tape frame."""
    if "liquidity" not in tape.columns:
        return _DEFAULT_BASE_LIQUIDITY
    for raw in tape["liquidity"].to_list():
        if raw is None:
            continue
        try:
            value = float(str(raw))
        except (TypeError, ValueError):  # pragma: no cover - schema guarantees strings
            continue
        if value > 0.0:
            return value
    return _DEFAULT_BASE_LIQUIDITY


def _fresh_pool_engine(market_view: MarketView, config: SimConfig) -> PoolEngine:
    """A pristine external pool (no agent positions) at the tape's entry tick."""
    tape = market_view.active_split_data().sort("seq")
    first_row = tape.row(0, named=True)
    entry_price = first_row.get("price_reference")
    if entry_price is None:
        entry_price = first_row.get("price_pool")
    if entry_price is None:
        entry_price = market_view.reference_close(first_row["block_timestamp"])
    if entry_price is None:
        raise CliError("could not resolve an entry price for the training environment")
    _, sqrt_price, tick = human_price_to_triplet(float(entry_price))
    return PoolEngine(
        state=PoolState(
            sqrt_price=sqrt_price,
            tick=tick,
            liquidity=_first_positive_liquidity(tape),
            fee_growth_global_0=0.0,
            fee_growth_global_1=0.0,
            fee_tier_bps=int(config.episode.fee_tier_bps),
            tick_spacing=int(config.episode.tick_spacing),
        ),
        ticks={},
    )


def _make_env_factory(
    market_view: MarketView, config: SimConfig
) -> Callable[[], LpEnvironment]:
    """Return a zero-argument environment factory for ``train_ppo``.

    Replay mode delegates to S15's :func:`build_replay_env`; calibrated mode fits
    the MRSJD on the train split and builds a fresh pool + process per call.  A
    fresh object per call is required by ``train_ppo`` (one env per seed/parallel
    slot).
    """
    if config.price_mode is PriceMode.CALIBRATED:
        if not market_view.is_train:
            raise CliError(
                "calibrated price mode must be trained on the train split "
                "(fitting on eval data would be look-ahead).",
                EXIT_CONFIG,
            )
        params = CalibratedPriceProcess.fit(market_view)
        tape = market_view.active_split_data().sort("seq")
        first_row = tape.row(0, named=True)
        initial_price = first_row.get("price_reference") or first_row.get("price_pool")
        if initial_price is None:
            initial_price = market_view.reference_close(first_row["block_timestamp"])
        if initial_price is None:
            raise CliError("could not resolve an initial price for the MRSJD process")

        def calibrated_factory() -> LpEnvironment:
            process = CalibratedPriceProcess(params, initial_price=float(initial_price))
            return LpEnvironment(
                market_view,
                _fresh_pool_engine(market_view, config),
                process,
                ReplayGasModel(market_view, config.gas),
                ProportionalSlippageModel(),
                config,
            )

        return calibrated_factory

    def replay_factory() -> LpEnvironment:
        return build_replay_env(market_view, config)

    return replay_factory


# ---------------------------------------------------------------------------
# Artifact writers and command implementations
# ---------------------------------------------------------------------------
#: Column headers of the RQ1 markdown table (kept in step with S15's writer).
_ABLATION_HEADERS: tuple[str, ...] = (
    "Ablation",
    "PnL (USDC)",
    "Sharpe",
    "MaxDD",
    "Fee Income",
    "Gas Paid",
    "Slippage Paid",
    "IL Realized",
)


def _write_ablation_report(ablations: list[AblationResult], output: Path) -> None:
    """Write the RQ1 ablation table (markdown + CSV) on its own.

    ``run_ablations`` is a backtester-only measurement for deterministic
    baselines, so ``ablate`` does not run the RQ3 replay-simulator gap (a
    baseline band is not representable in the agent's discrete action grid).
    The format mirrors S15's ``write_results`` ablation artifact so consumers
    see one layout.
    """
    lines = [
        "# RQ1 — Friction ablation table",
        "",
        "Backtester (ground-truth) evaluation, mean over walk-forward episodes and "
        "seeds. Costs are signed (gas/slippage/IL ≤ 0); every PnL is net of gas and "
        "slippage.",
        "",
        "| " + " | ".join(_ABLATION_HEADERS) + " |",
        "| " + " | ".join("---" for _ in _ABLATION_HEADERS) + " |",
    ]
    for result in ablations:
        metrics = result.metrics
        cells = [
            result.ablation_name,
            f"{metrics.get('total_pnl', 0.0):,.2f}",
            f"{metrics.get('sharpe', 0.0):.4f}",
            f"{metrics.get('max_drawdown', 0.0):.4f}",
            f"{metrics.get('fee_income', 0.0):,.2f}",
            f"{metrics.get('gas_paid', 0.0):,.2f}",
            f"{metrics.get('slippage_paid', 0.0):,.2f}",
            f"{metrics.get('il_realized', 0.0):,.2f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    (output / "ablation_table.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    columns = ["ablation_name", "fees", "gas", "slippage", "il", *METRIC_NAMES]
    rows: list[dict[str, object]] = []
    for result in ablations:
        row: dict[str, object] = {"ablation_name": result.ablation_name}
        for flag in ("fees", "gas", "slippage", "il"):
            row[flag] = bool(result.flags.get(flag, False))
        for metric in METRIC_NAMES:
            row[metric] = float(result.metrics.get(metric, 0.0))
        rows.append(row)
    if rows:
        frame = pl.DataFrame(rows)
        present = [column for column in columns if column in frame.columns]
        remaining = [column for column in frame.columns if column not in present]
        frame.select(present + remaining).write_csv(output / "ablation_table.csv")
    else:
        (output / "ablation_table.csv").write_text(
            ",".join(columns) + "\n", encoding="utf-8"
        )


def _write_backtest_outputs(
    ledger: BacktestLedger, result: BacktestResult, output: Path
) -> None:
    """Write the equity curve, decision log, cost ledger and JSON summary."""
    ledger.equity_curve.write_csv(output / "equity_curve.csv")
    ledger.decision_log.write_csv(output / "decision_log.csv")
    ledger.cost_ledger.write_csv(output / "cost_ledger.csv")
    payload = {
        "policy_name": ledger.policy_name,
        "split": ledger.split,
        "config_hash": ledger.config_hash,
        "git_commit": ledger.git_commit,
        "n_steps": ledger.n_steps,
        "initial_capital": ledger.initial_capital,
        "fee_growth_exact": ledger.fee_growth_exact,
        "pnl_decomposition": dict(ledger.pnl_decomposition),
        "result": {
            "total_pnl": result.total_pnl,
            "annualized_return": result.annualized_return,
            "sharpe": result.sharpe,
            "max_drawdown": result.max_drawdown,
            "fee_income": result.fee_income,
            "gas_paid": result.gas_paid,
            "slippage_paid": result.slippage_paid,
            "il_realized": result.il_realized,
            "hodl_return": result.hodl_return,
            "excess_vs_hodl": result.excess_vs_hodl,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


def _parity_status(output: Path) -> str:
    """Read ``parity_report.json`` from ``output`` if a parity run recorded one."""
    report_path = output / "parity_report.json"
    if not report_path.is_file():
        return f"not available (no {report_path})"
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"unreadable ({exc})"
    passed = payload.get("passed")
    tolerance = payload.get("tolerance")
    verdict = "PASS" if passed is True else "FAIL" if passed is False else "unknown"
    detail = f", tolerance={tolerance}" if tolerance is not None else ""
    return f"{verdict} ({report_path}{detail})"


def _cmd_info(config: SimConfig, config_path: str, output: Path) -> int:
    """Print a human-readable configuration summary."""
    # ``LpEnvironment`` always includes the hold action at index 0 (ADR-011),
    # regardless of the ``include_hold`` convenience flag.
    n_actions = 1 + len(config.action_grid.center_offsets) * len(
        config.action_grid.widths
    )
    split = config.split
    lines = [
        f"{PROG} — configuration summary",
        f"  config file:      {config_path}",
        f"  price mode:       {config.price_mode.value}",
        f"  seed:             {config.seed}",
        (
            "  episode:          "
            f"{config.episode.duration_days} days, "
            f"{config.episode.step_minutes}-min steps, "
            f"capital {config.episode.agent_capital_usdc:,.0f} USDC"
        ),
        (
            "  fee tier/spacing: "
            f"{config.episode.fee_tier_bps} pips / {config.episode.tick_spacing} ticks"
        ),
        f"  action-space size: {n_actions}",
        "  split boundaries (UTC):",
        f"    train: {split.train_start_utc.isoformat()} → {split.train_end_utc.isoformat()}",
        f"    eval:  {split.eval_start_utc.isoformat()} → {split.eval_end_utc.isoformat()}",
        f"  available policies: {', '.join(POLICY_NAMES)}",
        f"  parity status:    {_parity_status(output)}",
    ]
    print("\n".join(lines))
    return EXIT_OK


def _cmd_backtest(
    args: argparse.Namespace, config: SimConfig, view: MarketView, output: Path
) -> int:
    """Replay a frozen baseline policy and write its ledger + summary."""
    policy = build_policy(args.policy, config)
    ledger = run_backtest(policy, view, config)
    result = summarize_backtest(ledger)
    _write_backtest_outputs(ledger, result, output)
    print(f"Backtest complete: policy={ledger.policy_name} split={ledger.split}")
    print(f"  steps:            {ledger.n_steps}")
    print(f"  total PnL:        {result.total_pnl:,.2f} USDC")
    print(f"  Sharpe:           {result.sharpe:.4f}")
    print(f"  max drawdown:     {result.max_drawdown:.4f}")
    print(f"  excess vs HODL:   {result.excess_vs_hodl:,.2f} USDC")
    print(f"  artifacts:        {output}")
    return EXIT_OK


def _cmd_train(args: argparse.Namespace, config: SimConfig, view: MarketView, output: Path) -> int:
    """Train PPO across the configured seeds and write checkpoints + manifests."""
    env_factory = _make_env_factory(view, config)
    result = train_ppo(env_factory, config, log_dir=output)
    payload = {
        "seeds": [int(seed) for seed in result.seeds],
        "best_checkpoint": str(result.best_checkpoint),
        "manifests": [
            {
                "run_id": manifest.run_id,
                "seed": manifest.seed,
                "config_hash": manifest.config_hash,
                "git_commit": manifest.git_commit,
                "algorithm": manifest.algorithm,
                "total_timesteps": manifest.total_timesteps,
                "best_reward": manifest.best_reward,
                "checkpoint_path": str(manifest.checkpoint_path),
            }
            for manifest in result.manifests
        ],
    }
    (output / "training_summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(f"Training complete: {len(result.manifests)} seed(s)")
    print(f"  best checkpoint:  {result.best_checkpoint}")
    print(f"  artifacts:        {output}")
    return EXIT_OK


def _cmd_evaluate(
    args: argparse.Namespace, config: SimConfig, view: MarketView, output: Path
) -> int:
    """Walk-forward evaluate a trained checkpoint + every baseline; write results."""
    checkpoint = Path(args.checkpoint)
    model = _load_checkpoint(checkpoint)
    policies: dict[str, Policy] = {
        name: build_policy(name, config) for name in POLICY_NAMES
    }
    # Never let a checkpoint file named after a baseline silently replace it.
    policy_name = checkpoint.stem or "ppo"
    if policy_name in policies:
        policy_name = f"ppo_{policy_name}"
    checkpoint_policy = _CheckpointPolicy(model, config, name=policy_name)
    policies[checkpoint_policy.name] = checkpoint_policy
    matrices = run_regime_evaluation(
        config, policies, view, num_seeds=int(args.num_seeds)
    )
    gap = run_gap_analysis(config, checkpoint_policy, view)
    write_results([], matrices, gap, output)
    print(f"Evaluation complete: {len(policies)} policies × walk-forward episodes")
    print(f"  checkpoint:       {checkpoint}")
    print(f"  artifacts:        {output}")
    return EXIT_OK


def _cmd_ablate(
    args: argparse.Namespace, config: SimConfig, view: MarketView, output: Path
) -> int:
    """Run the RQ1 friction-ablation matrix and write its tables."""
    policy = build_policy(args.policy, config)
    ablations = run_ablations(config, policy, view)
    _write_ablation_report(ablations, output)
    print(f"Ablation complete: policy={policy.name}, {len(ablations)} variants")
    print(f"  artifacts:        {output}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# Dispatch / entry point
# ---------------------------------------------------------------------------
def _run(args: argparse.Namespace, market_view: MarketView | None) -> int:
    """Execute the parsed command; raises :class:`CliError` on user errors."""
    config = _load_config(args.config)
    output = _resolve_output(getattr(args, "output", None))

    if args.command == "info":
        return _cmd_info(config, args.config, output)

    # Validate cheap, user-facing arguments *before* touching the dataset so an
    # invalid --policy / --checkpoint is reported even when no snapshot exists.
    if args.command in ("backtest", "ablate"):
        validate_policy_name(args.policy)
    if args.command == "evaluate" and not Path(args.checkpoint).is_file():
        raise CliError(f"checkpoint not found: {args.checkpoint}", EXIT_CONFIG)

    view = market_view if market_view is not None else _load_market_view(
        config, args.data_config, args.split
    )

    if args.command == "backtest":
        return _cmd_backtest(args, config, view, output)
    if args.command == "train":
        return _cmd_train(args, config, view, output)
    if args.command == "evaluate":
        return _cmd_evaluate(args, config, view, output)
    if args.command == "ablate":
        return _cmd_ablate(args, config, view, output)
    raise CliError(f"unknown command {args.command!r}", EXIT_CONFIG)  # pragma: no cover


def main(
    argv: Sequence[str] | None = None,
    *,
    market_view: MarketView | None = None,
) -> int:
    """CLI entry point.

    ``market_view`` is an optional dependency-injection seam used by the test
    suite to exercise the full command path against a synthetic view; the console
    script calls :func:`main` with no arguments and loads data from disk.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.debug:
        return _run(args, market_view)
    try:
        return _run(args, market_view)
    except CliError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # noqa: BLE001 - a CLI must not leak tracebacks
        print(f"Error: unexpected failure: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover - exercised via ``python -m``
    raise SystemExit(main())
