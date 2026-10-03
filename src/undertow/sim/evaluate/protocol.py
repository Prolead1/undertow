"""Core walk-forward, multi-seed evaluation runner (S15, CONTRACTS.md §17).

This module is the shared engine behind the three RQ deliverables:

* :func:`evaluate_policy` runs a frozen policy over non-overlapping walk-forward
  episodes of the active split and aggregates S05 metrics as mean ± std over
  episodes and seeds.
* :func:`evaluate_episodes` is the episode-granular version used by the RQ1
  ablation table and the RQ3 gap report; it returns the per-episode records
  (metrics, decomposition and equity curve) instead of a summary dict.

Ground truth is the backtester
------------------------------
The default ``backend="backtest"`` evaluates every episode with
:func:`~undertow.sim.backtest.run_backtest` (S11) against the real event tape.
That is the measurement instrument.  ``backend="replay_sim"`` runs the same
episodes through the replay-mode :class:`~undertow.sim.env.LpEnvironment` (S12)
so the RQ3 gap report can compare the float64 simulator against the exact
backtester; S14's parity harness already established they agree within the
stated tolerance on identical inputs.

Walk-forward and seeds
----------------------
The active split is divided into non-overlapping episodes of
``config.episode.duration_days`` (half-open on the wall clock; the last episode
is the remainder).  For each seed in ``config.training.seeds`` (capped at
``num_seeds``) the same deterministic episode windows are evaluated.  Policies
are frozen — :class:`Policy` has no learning method — so a seed only changes
which window the *calibrated* simulator would sample; for the replay backtester
it changes nothing and the per-seed records are identical (std = 0).  Keeping
the seed dimension explicit lets the RQ2 matrix report the same mean ± std shape
for baselines and RL agents.

Sign conventions (cost honesty, PLAN.md §4)
-------------------------------------------
Every reported PnL is net of gas and slippage.  ``gas_paid``,
``slippage_paid`` and ``il_realized`` carry the ledger's convention (costs
``<= 0``) and ``fee_income >= 0``; ``total_pnl`` is the absolute
``equity - initial_capital``.  The decomposition keys use the CONTRACTS §13
spelling (``total_fees``/``total_gas``/``total_slippage``/``total_il``/``net``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from undertow.sim.backtest import run_backtest
from undertow.sim.config import SimConfig
from undertow.sim.core.pool import PoolEngine, PoolState
from undertow.sim.env import OBSERVATION_VECTOR_LENGTH, LpEnvironment, Observation
from undertow.sim.frictions import (
    GasModel,
    ProportionalSlippageModel,
    ReplayGasModel,
    SlippageModel,
)
from undertow.sim.marketview import MarketView
from undertow.sim.metrics import (
    DEFAULT_PERIODS_PER_YEAR,
    annualized_return,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from undertow.sim.policies import Policy
from undertow.sim.prices import ReplayPriceProcess
from undertow.sim.prices.base import human_price_to_triplet
from undertow.sim.types import Action, Tick, UndertowSimError

if TYPE_CHECKING:
    from undertow.sim.backtest.ledger import BacktestLedger

LOGGER = logging.getLogger("undertow.sim.evaluate.protocol")

__all__ = [
    "METRIC_NAMES",
    "REGIME_LABELS",
    "EpisodeRecord",
    "EvaluationError",
    "aggregate_metrics",
    "build_replay_env",
    "dominant_regime",
    "evaluate_episodes",
    "evaluate_policy",
    "episode_metrics",
    "observation_from_vector",
    "per_regime_frame",
    "walk_forward_windows",
]

#: Canonical regime labels (CONTRACTS §1 / §17).  Regime labels are consumed
#: from the data module, never recomputed in the sim.
REGIME_LABELS: tuple[str, ...] = ("bull", "bear", "sideways", "high_vol")

#: Per-episode metrics aggregated by :func:`aggregate_metrics` and reported in the
#: RQ1/RQ2 tables.  Costs keep the ledger's convention (``<= 0``).
METRIC_NAMES: tuple[str, ...] = (
    "total_pnl",
    "annualized_return",
    "sharpe",
    "sortino",
    "max_drawdown",
    "fee_income",
    "gas_paid",
    "slippage_paid",
    "il_realized",
    "n_rebalances",
)

#: Initial half-width, in tick-spacings, the replay environment deploys.  Two
#: spacings is ±120 ticks, matching the backtester's
#: ``DEFAULT_INITIAL_HALF_WIDTH_TICKS`` so a sim-vs-backtest comparison starts
#: from the same token mix.
DEFAULT_INITIAL_WIDTH: int = 2

#: Fallback external liquidity when the tape carries no ``liquidity`` column
#: value (a hand-built tape).  Large so the agent stays marginal.
DEFAULT_BASE_LIQUIDITY: float = 1.0e18

#: Hard cap on simulator steps per episode; a non-terminating env is a bug, not
#: an infinite loop.
_MAX_SIM_STEPS: int = 1_000_000


class EvaluationError(UndertowSimError):
    """Raised when an evaluation request cannot be honoured."""


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """One evaluated walk-forward episode.

    ``metrics`` holds the S05-derived scalars of the episode; ``decomposition``
    holds the cost ledger (``total_fees``/``total_gas``/``total_slippage``/
    ``total_il``/``net``) so the ablation layer can re-gate it without
    recomputing a backtest.  ``equity_curve`` is the per-step curve used to
    rebuild ablated equity paths.
    """

    policy_name: str
    backend: str
    split: str
    start_seq: int
    end_seq: int
    seed: int
    regime: str
    initial_capital: float
    periods_per_year: int
    metrics: dict[str, float]
    decomposition: dict[str, float]
    equity_curve: pl.DataFrame


def walk_forward_windows(
    market_view: MarketView, config: SimConfig
) -> list[tuple[int, int]]:
    """Split the active split into non-overlapping ``duration_days`` episodes.

    Returns a list of half-open ``(start_seq, end_seq)`` windows in tape order.
    The final window is the remaining slice up to the active split wall; no
    window crosses the train/eval boundary because the active split itself does
    not.  Raises :class:`EvaluationError` for a non-positive duration or an
    empty tape.
    """
    duration_days = int(config.episode.duration_days)
    if duration_days <= 0:
        raise EvaluationError(
            f"episode.duration_days must be > 0, got {duration_days}"
        )
    tape = market_view.active_split_data().sort("seq")
    if tape.height == 0:
        raise EvaluationError("walk_forward_windows: the active split is empty")

    times: list[Any] = tape["block_timestamp"].to_list()
    seqs: list[int] = [int(v) for v in tape["seq"].to_list()]
    duration = timedelta(days=duration_days)
    windows: list[tuple[int, int]] = []
    cursor = 0
    window_start = market_view.active_start_utc
    wall = market_view.active_end_utc
    total = len(times)
    while cursor < total and window_start < wall:
        window_end = min(window_start + duration, wall)
        first = cursor
        while cursor < total and times[cursor] < window_end:
            cursor += 1
        if cursor > first:
            windows.append((seqs[first], seqs[cursor - 1] + 1))
        window_start = window_end
    if not windows:
        raise EvaluationError(
            "walk_forward_windows produced no episodes; check the active split bounds"
        )
    return windows


def dominant_regime(
    market_view: MarketView, start_seq: int, end_seq: int
) -> str:
    """Dominant regime label of the ``[start_seq, end_seq)`` episode.

    Counts ``market_view.regime_at(block_timestamp)`` over every tape row in the
    episode and returns the most frequent label.  Ties break deterministically
    (most frequent, then alphabetical) so results are reproducible.  An episode
    with no labels returns ``"unknown"``.
    """
    tape = market_view.active_split_data()
    sub = tape.filter(
        (pl.col("seq") >= int(start_seq)) & (pl.col("seq") < int(end_seq))
    ).sort("seq")
    counts: dict[str, int] = {}
    for timestamp in sub["block_timestamp"].to_list():
        label = market_view.regime_at(timestamp)
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return "unknown"
    return min(counts, key=lambda label: (-counts[label], label))


def observation_from_vector(vector: Sequence[float]) -> Observation:
    """Rebuild the §11 :class:`Observation` from its flattened vector.

    The environment's public API returns the stable-order ``to_vector()`` array.
    :class:`Policy` expects the structured ``Observation``, so this inverts the
    documented field order.  It is the only adapter between the two S12 surfaces
    and is deliberately strict about the vector length.
    """
    if len(vector) != OBSERVATION_VECTOR_LENGTH:
        raise EvaluationError(
            "observation vector must have length "
            f"{OBSERVATION_VECTOR_LENGTH}, got {len(vector)}"
        )
    from undertow.sim.env.observations import REGIME_ORDINAL

    inverse = {value: name for name, value in REGIME_ORDINAL.items()}
    values = [float(v) for v in vector]
    return Observation(
        price=values[0],
        sqrt_price=values[1],
        tick=Tick(int(values[2])),
        price_returns_1h=values[3],
        price_returns_24h=values[4],
        realized_vol_24h=values[5],
        position_in_range=values[6] >= 0.5,
        position_value=values[7],
        uncollected_fees_token0=values[8],
        uncollected_fees_token1=values[9],
        il_vs_hodl=values[10],
        pool_active_liquidity=values[11],
        pool_fee_tier_bps=int(values[12]),
        gas_price_recent_wei=values[13],
        eth_usd_price_recent=values[14],
        regime=inverse.get(int(round(values[15])), "unknown"),
        step_in_episode=int(values[16]),
        steps_remaining=int(values[17]),
    )


def action_to_index(env: LpEnvironment, action: Action) -> int:
    """Map a policy :class:`Action` to the environment's discrete action index.

    The environment's action space is the factored ``ActionGridConfig`` grid
    (ADR-011); baseline policies return bands outside that grid and are declared
    out of scope for simulator stepping.  Raises :class:`EvaluationError` with a
    clear message when the action is not representable rather than silently
    holding.
    """
    for index in range(env.n_actions):
        if env.action_for_index(index) == action:
            return index
    raise EvaluationError(
        f"action {action!r} is not in the environment's discrete action space "
        f"({env.n_actions} entries); the RQ3 simulator path cannot execute a "
        "baseline band outside the agent action grid"
    )


def _first_positive_liquidity(tape: pl.DataFrame) -> float:
    """First non-null, positive ``liquidity`` value in the tape."""
    if "liquidity" not in tape.columns:
        return DEFAULT_BASE_LIQUIDITY
    for raw in tape["liquidity"].to_list():
        if raw is None:
            continue
        try:
            value = float(str(raw))
        except (TypeError, ValueError):  # pragma: no cover - schema guarantees numeric strings
            continue
        if value > 0.0:
            return value
    return DEFAULT_BASE_LIQUIDITY


def build_replay_env(
    market_view: MarketView,
    config: SimConfig,
    *,
    gas_model: GasModel | None = None,
    slippage_model: SlippageModel | None = None,
) -> LpEnvironment:
    """Construct a replay-mode training environment for a single episode view.

    The environment captures the external pool base at construction and deploys
    its own initial position on every :meth:`reset` (S12).  The base external
    liquidity is taken from the tape's recorded active liquidity (marginal-agent
    model); the tick lattice is built by the agent's own position, so no
    synthetic background ticks are invented.  Gas replays the tape's per-block
    base fee; slippage is the pinned proportional model.
    """
    tape = market_view.active_split_data().sort("seq")
    if tape.height == 0:
        raise EvaluationError("build_replay_env: the episode view is empty")

    first_row = tape.row(0, named=True)
    entry_price = first_row.get("price_reference")
    if entry_price is None:
        entry_price = first_row.get("price_pool")
    if entry_price is None:
        entry_price = market_view.reference_close(first_row["block_timestamp"])
    if entry_price is None:
        raise EvaluationError("build_replay_env: no usable entry price")

    _, sqrt_price, tick = human_price_to_triplet(float(entry_price))
    state = PoolState(
        sqrt_price=sqrt_price,
        tick=tick,
        liquidity=_first_positive_liquidity(tape),
        fee_growth_global_0=0.0,
        fee_growth_global_1=0.0,
        fee_tier_bps=int(config.episode.fee_tier_bps),
        tick_spacing=int(config.episode.tick_spacing),
    )
    engine = PoolEngine(state=state, ticks={})
    process = ReplayPriceProcess(
        market_view,
        int(tape["seq"][0]),
        int(tape["seq"][-1]) + 1,
    )
    gas = gas_model if gas_model is not None else ReplayGasModel(market_view, config.gas)
    slippage = (
        slippage_model if slippage_model is not None else ProportionalSlippageModel()
    )
    return LpEnvironment(
        market_view,
        engine,
        process,
        gas,
        slippage,
        config,
        initial_width=DEFAULT_INITIAL_WIDTH,
    )


def _periods_per_year_from_config(config: SimConfig) -> int:
    """Annualization basis for the simulator's decision cadence."""
    step_minutes = int(config.episode.step_minutes)
    if step_minutes <= 0:
        return DEFAULT_PERIODS_PER_YEAR
    return int(round(365 * 24 * 60 / step_minutes))


def episode_metrics(
    equity: pl.Series,
    initial_capital: float,
    decomposition: dict[str, float],
    periods_per_year: int,
    n_rebalances: int,
) -> dict[str, float]:
    """S05 metrics plus the cost ledger for one episode."""
    return {
        "total_pnl": float(equity[-1]) - initial_capital if equity.len() else 0.0,
        "annualized_return": annualized_return(equity, periods_per_year),
        "sharpe": sharpe_ratio(equity, periods_per_year=periods_per_year),
        "sortino": sortino_ratio(equity, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "fee_income": float(decomposition.get("total_fees", 0.0)),
        "gas_paid": float(decomposition.get("total_gas", 0.0)),
        "slippage_paid": float(decomposition.get("total_slippage", 0.0)),
        "il_realized": float(decomposition.get("total_il", 0.0)),
        "n_rebalances": float(n_rebalances),
    }


def _record_from_ledger(
    policy: Policy,
    ledger: BacktestLedger,
    market_view: MarketView,
    start_seq: int,
    end_seq: int,
    seed: int,
) -> EpisodeRecord:
    """Package a backtest ledger into an :class:`EpisodeRecord`."""
    equity = ledger.equity_curve["equity"]
    n_rebalances = (
        int((ledger.decision_log["action_type"] == "rebalance").sum())
        if ledger.decision_log.height
        else 0
    )
    capital = float(ledger.initial_capital)
    return EpisodeRecord(
        policy_name=policy.name,
        backend="backtest",
        split=ledger.split,
        start_seq=int(start_seq),
        end_seq=int(end_seq),
        seed=int(seed),
        regime=dominant_regime(market_view, start_seq, end_seq),
        initial_capital=capital,
        periods_per_year=int(ledger.periods_per_year),
        metrics=episode_metrics(
            equity,
            capital,
            ledger.pnl_decomposition,
            int(ledger.periods_per_year),
            n_rebalances,
        ),
        decomposition=dict(ledger.pnl_decomposition),
        equity_curve=ledger.equity_curve,
    )


def _run_sim_window(
    policy: Policy,
    window_view: MarketView,
    config: SimConfig,
    seed: int,
    env_factory: Callable[[MarketView, SimConfig], LpEnvironment],
) -> EpisodeRecord:
    """Run one frozen policy episode in the replay simulator."""
    env = env_factory(window_view, config)
    capital = float(config.episode.agent_capital_usdc)
    period_basis = _periods_per_year_from_config(config)

    policy.reset()
    rng = np.random.default_rng(seed)
    vector, _info = env.reset(seed=seed)

    equities: list[float] = [capital]
    fees = 0.0
    gas = 0.0
    slippage = 0.0
    il = 0.0
    n_rebalances = 0
    equity = capital
    done = False
    steps = 0
    while not done:
        action = policy.act(observation_from_vector(vector), rng)
        index = action_to_index(env, action)
        vector, _reward, terminated, truncated, info = env.step(index)
        raw = info["reward_breakdown_raw"]
        fees += float(raw["fees"])
        gas += float(raw["gas"])
        slippage += float(raw["slippage"])
        il += float(raw["il_change"])
        equity = float(info["equity"]) if "equity" in info else float(raw.get("equity", equity))
        equities.append(equity)
        if action.action_type == "rebalance":
            n_rebalances += 1
        steps += 1
        if steps > _MAX_SIM_STEPS:  # pragma: no cover - defensive
            raise EvaluationError(
                f"simulator episode exceeded {_MAX_SIM_STEPS} steps without terminating"
            )
        done = bool(terminated or truncated)

    net = fees + gas + slippage + il
    decomposition = {
        "total_fees": fees,
        "total_gas": gas,
        "total_slippage": slippage,
        "total_il": il,
        "net": net,
    }
    equity_series = pl.Series("equity", equities)
    start_seq = int(env.price_process.start_seq)
    end_seq = int(env.price_process.end_seq)
    return EpisodeRecord(
        policy_name=policy.name,
        backend="replay_sim",
        split="train" if window_view.is_train else "eval",
        start_seq=start_seq,
        end_seq=end_seq,
        seed=int(seed),
        regime=dominant_regime(window_view, start_seq, end_seq),
        initial_capital=capital,
        periods_per_year=period_basis,
        metrics=episode_metrics(
            equity_series, capital, decomposition, period_basis, n_rebalances
        ),
        decomposition=decomposition,
        equity_curve=pl.DataFrame({"equity": equities}),
    )


def evaluate_episodes(
    policy: Policy,
    market_view: MarketView,
    config: SimConfig,
    num_seeds: int | None = None,
    *,
    backend: str = "backtest",
    env_factory: Callable[[MarketView, SimConfig], LpEnvironment] | None = None,
    windows: Sequence[tuple[int, int]] | None = None,
) -> list[EpisodeRecord]:
    """Evaluate ``policy`` over every walk-forward episode and seed.

    ``num_seeds`` caps ``config.training.seeds`` (``None`` uses them all).  The
    backtest backend runs each deterministic window once and replicates the
    record across seeds (a frozen policy cannot depend on the seed); the replay
    simulator backend runs every (window, seed) pair with its own env.
    """
    if backend not in ("backtest", "replay_sim"):
        raise EvaluationError(
            f"unknown backend {backend!r}; expected 'backtest' or 'replay_sim'"
        )
    seeds = tuple(int(s) for s in config.training.seeds)
    if num_seeds is not None:
        cap = max(0, int(num_seeds))
        seeds = seeds[:cap]
    if not seeds:
        seeds = (int(config.seed),)

    episode_windows = (
        list(windows) if windows is not None else walk_forward_windows(market_view, config)
    )
    if not episode_windows:
        raise EvaluationError("evaluate_episodes: no walk-forward windows")

    if backend == "backtest":
        ledgers: list[tuple[int, int, BacktestLedger]] = []
        for start_seq, end_seq in episode_windows:
            ledger = run_backtest(
                policy, market_view.slice(start_seq, end_seq), config
            )
            ledgers.append((int(start_seq), int(end_seq), ledger))
        return [
            _record_from_ledger(policy, ledger, market_view, start_seq, end_seq, seed)
            for seed in seeds
            for start_seq, end_seq, ledger in ledgers
        ]

    factory = env_factory if env_factory is not None else build_replay_env
    records: list[EpisodeRecord] = []
    for seed in seeds:
        for start_seq, end_seq in episode_windows:
            window_view = market_view.slice(start_seq, end_seq)
            records.append(_run_sim_window(policy, window_view, config, seed, factory))
    return records


def aggregate_metrics(
    records: Sequence[EpisodeRecord],
    metric_names: Sequence[str] = METRIC_NAMES,
) -> dict[str, dict[str, float]]:
    """Mean and population std of each metric across episodes and seeds.

    Returns ``{metric: {"mean": float, "std": float, "n": int}}`` for every
    requested metric; a metric with no observations reports zeros and ``n=0``.
    """
    aggregated: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = [float(record.metrics[name]) for record in records if name in record.metrics]
        if not values:
            aggregated[name] = {"mean": 0.0, "std": 0.0, "n": 0}
            continue
        array = np.asarray(values, dtype=np.float64)
        aggregated[name] = {
            "mean": float(np.mean(array)),
            "std": float(np.std(array, ddof=0)),
            "n": len(values),
        }
    return aggregated


def per_regime_frame(
    records: Sequence[EpisodeRecord],
    metric_names: Sequence[str] = METRIC_NAMES,
) -> pl.DataFrame:
    """Long-format per-regime aggregation: ``regime, metric, mean, std, n``.

    Every canonical regime gets a row per metric even when no episode matched
    (``mean=std=0``, ``n=0``); any non-canonical label observed in the data is
    appended so nothing is silently dropped.
    """
    observed = [
        label
        for label in dict.fromkeys(record.regime for record in records)
        if label not in REGIME_LABELS
    ]
    regimes = list(REGIME_LABELS) + observed
    rows: list[dict[str, object]] = []
    for regime in regimes:
        subset = [record for record in records if record.regime == regime]
        for metric in metric_names:
            values = [
                float(record.metrics[metric])
                for record in subset
                if metric in record.metrics
            ]
            if values:
                array = np.asarray(values, dtype=np.float64)
                mean_value = float(np.mean(array))
                std_value = float(np.std(array, ddof=0))
            else:
                mean_value = 0.0
                std_value = 0.0
            rows.append(
                {
                    "regime": regime,
                    "metric": metric,
                    "mean": mean_value,
                    "std": std_value,
                    "n": len(values),
                }
            )
    return pl.DataFrame(rows)


def _record_to_dict(record: EpisodeRecord) -> dict[str, object]:
    """JSON-friendly view of an episode record (no DataFrame)."""
    return {
        "policy_name": record.policy_name,
        "backend": record.backend,
        "split": record.split,
        "start_seq": record.start_seq,
        "end_seq": record.end_seq,
        "seed": record.seed,
        "regime": record.regime,
        "metrics": dict(record.metrics),
        "decomposition": dict(record.decomposition),
    }


def evaluate_policy(
    policy: Policy,
    market_view: MarketView,
    config: SimConfig,
    num_seeds: int = 5,
    *,
    backend: str = "backtest",
    env_factory: Callable[[MarketView, SimConfig], LpEnvironment] | None = None,
) -> dict[str, Any]:
    """Walk-forward, multi-seed evaluation of a frozen policy (CONTRACTS §17).

    Returns a dict with ``policy``, ``backend``, ``split``, ``n_episodes``,
    ``n_seeds``, ``metrics`` (each metric's mean/std/n) and ``per_episode``
    (the episode records without their equity curves).
    """
    records = evaluate_episodes(
        policy,
        market_view,
        config,
        num_seeds=num_seeds,
        backend=backend,
        env_factory=env_factory,
    )
    return {
        "policy": policy.name,
        "backend": backend,
        "split": "train" if market_view.is_train else "eval",
        "n_episodes": len(records),
        "n_seeds": len({record.seed for record in records}),
        "metrics": aggregate_metrics(records),
        "per_episode": [_record_to_dict(record) for record in records],
    }
