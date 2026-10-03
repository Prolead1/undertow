"""Look-ahead probe suite for the composed simulator + backtester (S14).

Implements ``CONTRACTS.md`` §16's :func:`run_lookahead_probes`.  The simulator and
the backtester are separately unit-tested for look-ahead safety (S03/S11/S12);
this module attacks the **composed** system: it verifies that an observation or a
decision made at step ``t`` cannot be changed by mutating, removing or poisoning
data from after ``t``.

The probes are deliberately adversarial.  Each one either passes (records a
detail line) or increments the violation counter and records what was attempted,
what was expected and what actually happened, so a failing system yields a
diagnosable report rather than a bare boolean.

The ``backtest_fn`` argument is a callable ``(MarketView) -> BacktestLedger``
(typically ``functools.partial(run_backtest, policy, config=config)``): the suite
runs it on the original view and on a truncated/mutated view and requires the
decisions at or before ``t`` to be identical.

S12 owns ``LpEnvironment`` and this task may not edit it, so the probes that
rebuild the composed environment necessarily read its captured base pool and
initial-deployment parameters (``_base_state``, ``_base_ticks``,
``_initial_width``/``_initial_center_offset``) and the live episode state it
exposes.  Those attributes are stable across S12's merged implementation; the
probes fail loudly (``AttributeError``) rather than silently if a future S12
rename removes one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from undertow.sim.core.pool import PoolEngine
from undertow.sim.env.observations import ObservationBuilder
from undertow.sim.types import LookAheadError, MarketViewError

if TYPE_CHECKING:
    from undertow.sim.backtest import BacktestLedger
    from undertow.sim.env.lp_env import LpEnvironment
    from undertow.sim.marketview import MarketView

__all__ = ["LookAheadReport", "run_lookahead_probes"]


@dataclass(frozen=True, slots=True)
class LookAheadReport:
    """Results of the composed look-ahead probe suite (CONTRACTS §16).

    ``probes_run`` is the number of independent probes executed, ``violations``
    the number that detected a leak, and ``details`` one line per probe.
    """

    probes_run: int
    violations: int
    details: list[str]


def _clone_env(env: LpEnvironment, market_view: MarketView) -> LpEnvironment:
    """Rebuild ``env`` against ``market_view`` with a pristine external pool.

    Uses the base pool state the environment captured at construction (so the
    clone starts from the same external pool) and a fresh replay price process
    over ``market_view``.  Used only by the mutation probes, where comparing two
    otherwise-identical environments isolates the effect of the mutated rows.
    """
    from undertow.sim.env.lp_env import LpEnvironment
    from undertow.sim.prices.replay import ReplayPriceProcess

    engine = PoolEngine(
        state=replace(env._base_state),  # type: ignore[attr-defined]
        ticks={
            tick: replace(state)
            for tick, state in env._base_ticks.items()  # type: ignore[attr-defined]
        },
    )
    process = ReplayPriceProcess(market_view, 0, market_view.step_count())
    return LpEnvironment(
        market_view,
        engine,
        process,
        env.gas_model,
        env.slippage_model,
        env.config,
        initial_width=env._initial_width,  # type: ignore[attr-defined]
        initial_center_offset=env._initial_center_offset,  # type: ignore[attr-defined]
    )


def _mutate_future(
    market_view: MarketView,
    cutoff: datetime,
    *,
    sentinel_price: float | None = None,
    sentinel_regime: str | None = None,
) -> MarketView:
    """Return a view with every row after ``cutoff`` perturbed.

    When ``sentinel_price``/``sentinel_regime`` are given they are written into
    future reference closes / regime labels; otherwise future closes are scaled by
    a large factor.  Future tape ``price_reference``/``price_pool`` values are
    also perturbed so both the reference feed and the event tape carry poison.
    The raw ``sqrt_price_x96``/``tick`` columns are intentionally left untouched:
    perturbing them would make future swaps direction-inconsistent and could make
    the backtester raise while replaying rows past the probe's comparison cutoff
    (the backtester walks the *whole* tape).  Row counts and timestamps are
    unchanged, so the view's split invariants still hold.
    """
    reference = market_view.reference
    future_ref = pl.col("close_time") > cutoff
    if sentinel_price is None:
        close_expr = pl.when(future_ref).then(pl.col("close") * 1.5).otherwise(pl.col("close"))
    else:
        close_expr = (
            pl.when(future_ref).then(pl.lit(float(sentinel_price))).otherwise(pl.col("close"))
        )
    mutated_reference = reference.with_columns(close_expr.alias("close"))

    regimes = market_view.regimes
    mutated_regimes = regimes
    if sentinel_regime is not None and regimes.height:
        mutated_regimes = regimes.with_columns(
            pl.when(pl.col("timestamp") > cutoff)
            .then(pl.lit(sentinel_regime))
            .otherwise(pl.col("regime"))
            .alias("regime")
        )

    active = market_view.active_split_data()
    if active.height:
        future_tape = pl.col("block_timestamp") > cutoff
        mutated_active = active.with_columns(
            [
                pl.when(future_tape)
                .then(pl.col("price_reference") * 1.5)
                .otherwise(pl.col("price_reference"))
                .alias("price_reference"),
                pl.when(future_tape)
                .then(pl.col("price_pool") * 1.5)
                .otherwise(pl.col("price_pool"))
                .alias("price_pool"),
            ]
        )
    else:
        mutated_active = active

    return replace(
        market_view,
        reference=mutated_reference,
        regimes=mutated_regimes,
        train=mutated_active if market_view.is_train else None,
        eval=mutated_active if not market_view.is_train else None,
    )


class _ProbeSuite:
    """Stateful runner for the individual look-ahead probes."""

    def __init__(
        self,
        market_view: MarketView,
        env: LpEnvironment,
        backtest_fn: Callable[[MarketView], BacktestLedger],
    ) -> None:
        self.market_view = market_view
        self.env = env
        self.backtest_fn = backtest_fn
        self.probes_run = 0
        self.violations = 0
        self.details: list[str] = []

    # -- reporting -------------------------------------------------------
    def _record(self, name: str, ok: bool, message: str) -> None:
        self.probes_run += 1
        if ok:
            self.details.append(f"PASS {name}: {message}")
        else:
            self.violations += 1
            self.details.append(f"VIOLATION {name}: {message}")

    # -- helpers ---------------------------------------------------------
    def _active_range(self) -> tuple[int, int]:
        active = self.market_view.active_split_data()
        return int(active["seq"].min()), int(active["seq"].max())

    def _sample_time(self) -> datetime:
        active = self.market_view.active_split_data()
        midpoint = active.height // 2
        return active["block_timestamp"][midpoint]

    def _run(self) -> None:
        for name in (
            "split_wall_slice",
            "split_wall_tape_at",
            "reference_wall",
            "reference_bars_clamped",
            "episode_bounds",
            "regime_asof",
            "regime_future_mutation",
            "env_step_count",
            "observation_future_mutation",
            "observation_sentinel",
            "env_future_mutation",
            "backtester_truncation",
            "backtester_future_mutation",
        ):
            getattr(self, f"probe_{name}")()

    # -- probes ----------------------------------------------------------
    def probe_split_wall_slice(self) -> None:
        min_seq, last_seq = self._active_range()
        try:
            self.market_view.slice(min_seq, last_seq + 2)
            ok, detail = False, f"slice([{min_seq}, {last_seq + 2})) did not raise"
        except LookAheadError as exc:
            ok, detail = True, f"slice past the wall raised LookAheadError ({exc})"
        self._record("split_wall_slice", ok, detail)

    def probe_split_wall_tape_at(self) -> None:
        _min_seq, last_seq = self._active_range()
        try:
            self.market_view.tape_at(last_seq + 1)
            ok, detail = False, f"tape_at({last_seq + 1}) did not raise"
        except LookAheadError as exc:
            ok, detail = True, f"tape_at past the wall raised LookAheadError ({exc})"
        self._record("split_wall_tape_at", ok, detail)

    def probe_reference_wall(self) -> None:
        past = self.market_view.active_end_utc + timedelta(seconds=1)
        failures: list[str] = []
        try:
            self.market_view.reference_close(past)
            failures.append("reference_close past the wall returned a value")
        except LookAheadError:
            pass
        try:
            self.market_view.reference_bars(past, past + timedelta(days=1))
            failures.append("reference_bars past the wall returned a frame")
        except LookAheadError:
            pass
        ok = not failures
        detail = (
            "reference reads past the active wall are rejected"
            if ok
            else "; ".join(failures)
        )
        self._record("reference_wall", ok, detail)

    def probe_reference_bars_clamped(self) -> None:
        start = self.market_view.active_start_utc
        end = self.market_view.active_end_utc + timedelta(days=1)
        bars = self.market_view.reference_bars(start, end)
        if bars.height == 0:
            self._record("reference_bars_clamped", True, "no bars in range")
            return
        max_close = bars["close_time"].max()
        ok = max_close is None or max_close <= self.market_view.active_end_utc
        detail = (
            f"reference_bars clamped at the wall (max close {max_close})"
            if ok
            else f"reference_bars leaked a bar closing at {max_close}"
        )
        self._record("reference_bars_clamped", ok, detail)

    def probe_episode_bounds(self) -> None:
        min_seq, last_seq = self._active_range()
        start, end = self.market_view.build_episode(0)
        in_seq = min_seq <= start < end <= last_seq + 1
        active = self.market_view.active_split_data()
        times = active["block_timestamp"].to_list()
        first_time = times[0]
        last_time = times[-1]
        in_time = (
            self.market_view.active_start_utc <= first_time
            and last_time <= self.market_view.active_end_utc
        )
        ok = in_seq and in_time
        detail = f"build_episode(0) -> ({start}, {end}) within [{min_seq}, {last_seq}]"
        if not ok:
            detail = (
                f"build_episode(0) -> ({start}, {end}) escapes [{min_seq}, {last_seq}] "
                f"or the active time window"
            )
        self._record("episode_bounds", ok, detail)

    def probe_regime_asof(self) -> None:
        at_time = self._sample_time()
        label = self.market_view.regime_at(at_time)
        regimes = self.market_view.regimes
        if regimes.height:
            prior = regimes.filter(pl.col("timestamp") <= at_time).sort("timestamp")
            expected = str(prior["regime"][-1]) if prior.height else "unknown"
        else:
            expected = "unknown"
        ok = label == expected
        detail = (
            f"regime_at({at_time}) == {label!r} (as-of)"
            if ok
            else f"regime_at({at_time}) == {label!r}, expected {expected!r}"
        )
        self._record("regime_asof", ok, detail)

    def probe_regime_future_mutation(self) -> None:
        at_time = self._sample_time()
        before = self.market_view.regime_at(at_time)
        mutated = _mutate_future(
            self.market_view, at_time, sentinel_regime="__sentinel__"
        )
        after = mutated.regime_at(at_time)
        ok = before == after
        detail = (
            f"future regime mutation left regime_at({at_time}) == {before!r}"
            if ok
            else f"future regime mutation changed regime_at from {before!r} to {after!r}"
        )
        self._record("regime_future_mutation", ok, detail)

    def probe_env_step_count(self) -> None:
        self.env.reset(seed=0)
        episode_length = int(self.env.episode_length)
        n = min(3, episode_length)
        problems: list[str] = []
        if self.env.step_count != 0:
            problems.append(f"step_count after reset was {self.env.step_count}, expected 0")
        for k in range(n):
            _obs, _rew, terminated, truncated, info = self.env.step(0)
            if self.env.step_count != k + 1:
                problems.append(
                    f"step_count after {k + 1} steps was {self.env.step_count}"
                )
            if info.get("step_in_episode") != k + 1:
                problems.append(
                    f"info step_in_episode at step {k + 1} was "
                    f"{info.get('step_in_episode')}"
                )
            if terminated or truncated:
                break
        ok = not problems
        detail = (
            f"step_count tracked 0..{n} over {n} steps (episode_length={episode_length})"
            if ok
            else "; ".join(problems)
        )
        self._record("env_step_count", ok, detail)

    def _observation_at(
        self, view: MarketView, step: int, at_time: datetime
    ) -> tuple[object, object]:
        """Build the step-``step`` observation from ``view`` + the live env state."""
        position = self.env._position  # type: ignore[attr-defined]
        pool_engine = self.env.pool_engine
        builder = ObservationBuilder(view)
        observation = builder.build_at(
            at_time=at_time,
            tape_row=self.env._last_tape_row,  # type: ignore[attr-defined]
            step_in_episode=step,
            steps_remaining=max(0, int(self.env.episode_length) - 1 - step),
            position=position,
            pool_engine=pool_engine,
            gas_model=self.env.gas_model,
            entry_price=self.env._entry_price,  # type: ignore[attr-defined]
            entry_sqrt_price=self.env._entry_sqrt,  # type: ignore[attr-defined]
            initial_capital=self.env._capital,  # type: ignore[attr-defined]
        )
        return observation, observation.to_vector()

    def probe_observation_future_mutation(self) -> None:
        self.env.reset(seed=0)
        episode_length = int(self.env.episode_length)
        step = max(1, episode_length // 2)
        for _ in range(step):
            _obs, _rew, terminated, truncated, _info = self.env.step(0)
            if terminated or truncated:
                break
        at_time = self.env._last_at_time  # type: ignore[attr-defined]
        assert at_time is not None
        original, original_vector = self._observation_at(self.market_view, step, at_time)
        mutated_view = _mutate_future(self.market_view, at_time)
        mutated, mutated_vector = self._observation_at(mutated_view, step, at_time)
        ok = bool((original_vector == mutated_vector).all())
        detail = (
            f"step-{step} observation unchanged by future reference/tape mutation"
            if ok
            else (
                "future mutation changed the step-"
                f"{step} observation: price {original.price} -> {mutated.price}"
            )
        )
        self._record("observation_future_mutation", ok, detail)

    def probe_observation_sentinel(self) -> None:
        self.env.reset(seed=0)
        episode_length = int(self.env.episode_length)
        step = max(1, episode_length // 2)
        for _ in range(step):
            _obs, _rew, terminated, truncated, _info = self.env.step(0)
            if terminated or truncated:
                break
        at_time = self.env._last_at_time  # type: ignore[attr-defined]
        assert at_time is not None
        sentinel = 1.0e12
        mutated_view = _mutate_future(
            self.market_view,
            at_time,
            sentinel_price=sentinel,
            sentinel_regime="__sentinel__",
        )
        observation, vector = self._observation_at(mutated_view, step, at_time)
        leaked = sentinel in vector or observation.regime == "__sentinel__"
        ok = not leaked
        detail = (
            f"step-{step} observation ignored a future price/regime sentinel"
            if ok
            else "future price/regime sentinel leaked into the current observation"
        )
        self._record("observation_sentinel", ok, detail)

    def probe_env_future_mutation(self) -> None:
        """Composed-system probe: the env's step-t obs is future-mutation-invariant."""
        if not getattr(self.env, "is_replay", False):
            self._record(
                "env_future_mutation",
                True,
                "skipped: calibrated env has no replay tape to mutate",
            )
            return
        self.env.reset(seed=0)
        episode_length = int(self.env.episode_length)
        step = max(1, episode_length // 2)
        original_observations: list[np.ndarray] = []
        at_time: datetime | None = None
        for _ in range(step):
            obs, _rew, terminated, truncated, _info = self.env.step(0)
            original_observations.append(np.asarray(obs, dtype=float))
            at_time = self.env._last_at_time  # type: ignore[attr-defined]
            if terminated or truncated:
                break
        assert at_time is not None
        mutated_view = _mutate_future(self.market_view, at_time)
        clone = _clone_env(self.env, mutated_view)
        clone.reset(seed=0)
        mutated_observations: list[np.ndarray] = []
        for _ in range(len(original_observations)):
            obs, _rew, terminated, truncated, _info = clone.step(0)
            mutated_observations.append(np.asarray(obs, dtype=float))
            if terminated or truncated:
                break
        ok = len(original_observations) == len(mutated_observations) and all(
            np.array_equal(a, b)
            for a, b in zip(original_observations, mutated_observations, strict=True)
        )
        detail = (
            f"env observations for steps 1..{len(original_observations)} unchanged by "
            "future reference/tape mutation"
            if ok
            else "env observation sequence changed when future rows were mutated"
        )
        self._record("env_future_mutation", ok, detail)

    def _decision_cutoff(self) -> tuple[int, datetime]:
        self.env.reset(seed=0)
        episode_length = int(self.env.episode_length)
        step = max(1, episode_length // 2)
        for _ in range(step):
            _obs, _rew, terminated, truncated, _info = self.env.step(0)
            if terminated or truncated:
                break
        at_time = self.env._last_at_time  # type: ignore[attr-defined]
        assert at_time is not None
        active = self.market_view.active_split_data()
        prior = active.filter(pl.col("block_timestamp") <= at_time)
        cutoff_seq = int(prior["seq"].max())
        return cutoff_seq, at_time

    def _compare_ledgers(
        self, original: BacktestLedger, other: BacktestLedger, cutoff_seq: int
    ) -> list[str]:
        problems: list[str] = []
        cols = ["step", "action_type", "center_offset", "width", "gas_paid", "slippage_paid"]
        left = original.decision_log.filter(pl.col("step") <= cutoff_seq)
        right = other.decision_log.filter(pl.col("step") <= cutoff_seq)
        if left.height != right.height:
            problems.append(
                f"decision count up to seq {cutoff_seq} differs: "
                f"{left.height} vs {right.height}"
            )
        else:
            for column in cols:
                if not left[column].equals(right[column]):
                    problems.append(f"decision column {column!r} differs up to seq {cutoff_seq}")
        left_eq = original.equity_curve.filter(pl.col("step") <= cutoff_seq)
        right_eq = other.equity_curve.filter(pl.col("step") <= cutoff_seq)
        if left_eq.height != right_eq.height:
            problems.append(
                f"equity row count up to seq {cutoff_seq} differs: "
                f"{left_eq.height} vs {right_eq.height}"
            )
        elif not left_eq["equity"].equals(right_eq["equity"]):
            problems.append(f"equity differs up to seq {cutoff_seq}")
        return problems

    def probe_backtester_truncation(self) -> None:
        cutoff_seq, _at_time = self._decision_cutoff()
        full = self.backtest_fn(self.market_view)
        start = int(self.market_view.active_split_data()["seq"].min())
        truncated_view = self.market_view.slice(start, cutoff_seq + 1)
        truncated = self.backtest_fn(truncated_view)
        problems = self._compare_ledgers(full, truncated, cutoff_seq)
        ok = not problems
        detail = (
            f"backtester decisions up to seq {cutoff_seq} are truncation-invariant"
            if ok
            else "; ".join(problems)
        )
        self._record("backtester_truncation", ok, detail)

    def probe_backtester_future_mutation(self) -> None:
        cutoff_seq, at_time = self._decision_cutoff()
        full = self.backtest_fn(self.market_view)
        mutated = self.backtest_fn(_mutate_future(self.market_view, at_time))
        problems = self._compare_ledgers(full, mutated, cutoff_seq)
        ok = not problems
        detail = (
            f"backtester decisions up to seq {cutoff_seq} are mutation-invariant"
            if ok
            else "; ".join(problems)
        )
        self._record("backtester_future_mutation", ok, detail)


def run_lookahead_probes(
    market_view: MarketView,
    env: LpEnvironment,
    backtest_fn: Callable[[MarketView], BacktestLedger],
) -> LookAheadReport:
    """Run the composed look-ahead probe suite (CONTRACTS §16).

    ``backtest_fn`` must be a callable ``(MarketView) -> BacktestLedger``.  The
    suite returns zero violations on a correct system and a non-zero count when a
    leak is injected (see ``tests/sim/test_validation.py``).

    Raises
    ------
    TypeError
        If ``backtest_fn`` is not callable.
    MarketViewError
        If the supplied ``market_view`` has no usable active split.
    """
    if not callable(backtest_fn):
        raise TypeError("run_lookahead_probes requires a callable backtest_fn")
    try:
        if market_view.step_count() == 0:
            raise MarketViewError("run_lookahead_probes: active split has no tape rows")
    except AttributeError as exc:  # pragma: no cover - defensive
        raise MarketViewError(f"run_lookahead_probes: invalid market_view ({exc})") from exc

    suite = _ProbeSuite(market_view, env, backtest_fn)
    suite._run()
    return LookAheadReport(
        probes_run=suite.probes_run,
        violations=suite.violations,
        details=suite.details,
    )
