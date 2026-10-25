"""Look-ahead-safe window onto the historical dataset (S03).

Implements ``docs/plans/sim/CONTRACTS.md`` §4: ``MarketView`` is the *only* door
through which ``undertow.sim`` reads historical data. It owns the walk-forward
split balled in ``SplitConfig`` and makes it impossible for a consumer building a
training view to read evaluation-window rows.

The data loaders shipped by S02 return pyarrow ``Table`` objects; the frozen
contract for this module declares polars ``DataFrame`` fields, so every table is
converted with ``polars.from_arrow`` at the boundary. Historical data is reached
only through the top-level ``undertow.data`` public API
(``docs/decisions/008-data-sim-public-api-reconciliation.md``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, cast

import numpy as np
import polars as pl

from undertow.data import (
    DataConfig,
    PoolConfig,
    load_data_config,
    load_fee_growth,
    load_funding,
    load_gas_feed,
    load_reference_feed,
    load_regime_labels,
    load_tape,
)
from undertow.sim.config import EpisodeConfig, SplitConfig
from undertow.sim.types import LookAheadError, MarketViewError, Regime

__all__ = ["MarketView", "build_market_view", "load_market_view"]

LOGGER = logging.getLogger("undertow.sim.marketview")

SplitName = Literal["train", "eval"]


def _to_polars(table: object) -> pl.DataFrame:
    """Convert a pyarrow ``Table`` (or pass through a polars ``DataFrame``)."""
    if isinstance(table, pl.DataFrame):
        return table
    return cast(pl.DataFrame, pl.from_arrow(table))


def _sort(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Sort ``df`` by ``columns`` when they are present and the frame is non-empty."""
    if df.height == 0:
        return df
    present = [c for c in columns if c in df.columns]
    if not present:
        return df
    return df.sort(present)


def _between(df: pl.DataFrame, column: str, start: datetime, end: datetime) -> pl.DataFrame:
    """Rows with ``start <= df[column] < end`` (half-open, UTC).

    Half-open on the upper bound keeps the train/eval splits disjoint even when
    ``SplitConfig.train_end_utc == eval_start_utc`` (the boundary row belongs to
    the eval split only).
    """
    if df.height == 0:
        return df
    return df.filter((pl.col(column) >= start) & (pl.col(column) < end))


@dataclass(frozen=True, slots=True)
class MarketView:
    """A look-ahead-safe view of the train OR eval split. Never both.

    See ``CONTRACTS.md`` §4/§4.1. ``reference`` is always the *entire* reference
    feed; ``reference_close`` / ``reference_bars`` enforce the active split's
    upper wall so an evaluation-window row can never be read while building a
    training view.
    """

    # -- Splits (disjoint; exactly one is active) --
    train: pl.DataFrame | None  # train-split event tape, seq-sorted
    eval: pl.DataFrame | None  # eval-split event tape, seq-sorted
    reference: pl.DataFrame  # reference feed, sorted by open_time
    regimes: pl.DataFrame  # regime labels, sorted by timestamp
    gas: pl.DataFrame  # gas feed, sorted by block_number

    # -- Pinned split boundaries (UTC) --
    train_start_utc: datetime
    train_end_utc: datetime
    eval_start_utc: datetime
    eval_end_utc: datetime
    is_train: bool  # True if this view was built for training

    # -- Extension (additive, defaulted; PLAN §0.2) --
    #: Episode length driver for :meth:`build_episode`. Kept out of the frozen
    #: CONTRACTS signature, which only passes ``seed``/``max_steps``.
    episode_config: EpisodeConfig | None = None
    #: The ``PoolConfig`` the active tape was pulled from (ADR-013 additive
    #: extension). ``build_market_view`` populates it from ``DataConfig.pool``;
    #: hand-built views may leave it ``None`` and the backtester falls back to
    #: a pool derived from ``SimConfig.episode``.
    pool: PoolConfig | None = None
    #: Pool-state snapshot (``FEE_GROWTH_SCHEMA`` as polars) for the active split's
    #: first block — specifically the state **before** its first event
    #: (``first_block - 1``): one global row plus one row per initialized tick.
    #: Seeds the backtester's exact fee-growth engine when the window starts
    #: mid-lifecycle (positions minted before it are absent from the tape). ``None``
    #: for hand-built views or datasets without the boundary snapshot.
    pool_state: pl.DataFrame | None = None
    #: Tape rows *before* the active window, kept when a full lane is :meth:`slice`d
    #: so the backtester can warm its exact fee engine from the lane-start
    #: ``pool_state`` to the window start (ADR-018). ``None`` for a full lane.
    warmup: pl.DataFrame | None = None
    #: Binance USDT-perp funding feed (``FUNDING_SCHEMA`` as polars), sorted by
    #: ``funding_time`` and restricted to the configured primary symbol. ``None``
    #: (or empty) when the dataset carries no funding stream; a view without
    #: funding is valid and accrues zero (S20).  The hedge marks against the
    #: sim's own reference price, so the stream's nullable ``mark_price`` is not
    #: used (2005/3288 rows are null in 2022-23); it is returned by
    #: :meth:`funding_at` for callers that want it.
    funding: pl.DataFrame | None = None
    #: Lazily built ``(block_timestamp, seq)`` arrays for episode sampling.  A
    #: per-reset rebuild converted every active row to Python lists; the cache
    #: makes that a one-time conversion per view (ADR-033).
    _episode_index: tuple[np.ndarray, np.ndarray] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    # ------------------------------------------------------------------
    # Invariants (CONTRACTS §4.1)
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.train is not None and self.eval is not None:
            raise MarketViewError(
                "MarketView: train and eval cannot both be active (look-ahead risk)"
            )
        if self.train is None and self.eval is None:
            raise MarketViewError("MarketView: exactly one of train/eval must be active")
        if self.train is not None and not self.is_train:
            raise MarketViewError("MarketView: train data present but is_train is False")
        if self.eval is not None and self.is_train:
            raise MarketViewError("MarketView: eval data present but is_train is True")

        active = self.active_split_data()
        if active.height == 0:
            raise MarketViewError("MarketView: active split contains no tape rows")
        if "seq" not in active.columns:
            raise MarketViewError("MarketView: active split has no 'seq' column")

        self._check_reference_spans_split()

    def _check_reference_spans_split(self) -> None:
        """Invariant 3: the reference feed covers the active split's bounds."""
        if self.reference.height == 0:
            raise MarketViewError("MarketView: reference feed is empty")
        for column in ("open_time", "close_time"):
            if column not in self.reference.columns:
                raise MarketViewError(
                    f"MarketView: reference feed has no '{column}' column"
                )
        ref_start_raw = self.reference["open_time"].min()
        ref_end_raw = self.reference["close_time"].max()
        if ref_start_raw is None or ref_end_raw is None:
            raise MarketViewError("MarketView: reference feed has no timestamps")
        ref_start = cast(datetime, ref_start_raw)
        ref_end = cast(datetime, ref_end_raw)
        if ref_start > self.active_start_utc or ref_end < self.active_end_utc:
            raise MarketViewError(
                "MarketView: reference feed does not span the active split "
                f"[{self.active_start_utc}, {self.active_end_utc}] "
                f"(feed covers [{ref_start}, {ref_end}])"
            )

    # ------------------------------------------------------------------
    # Active-split helpers
    # ------------------------------------------------------------------
    @property
    def active_start_utc(self) -> datetime:
        """Lower time bound of the active split."""
        return self.train_start_utc if self.is_train else self.eval_start_utc

    @property
    def active_end_utc(self) -> datetime:
        """Upper time bound of the active split (data rows are ``<`` this bound)."""
        return self.train_end_utc if self.is_train else self.eval_end_utc

    def _active_range(self) -> tuple[int, int]:
        """Inclusive ``(min_seq, max_seq)`` of the active split."""
        active = self.active_split_data()
        return cast(int, active["seq"].min()), cast(int, active["seq"].max())

    def _check_time(self, at_time: datetime) -> None:
        """Raise ``LookAheadError`` when ``at_time`` is past the active wall."""
        if at_time > self.active_end_utc:
            raise LookAheadError(
                f"time {at_time} is past the active split wall {self.active_end_utc}"
            )

    # ------------------------------------------------------------------
    # Query methods (CONTRACTS §4)
    # ------------------------------------------------------------------
    def funding_at(self, timestamp: datetime) -> tuple[float, float] | None:
        """As-of (backward) funding settlement at or before ``timestamp``.

        Returns ``(funding_rate, mark_price)`` of the latest settlement with
        ``funding_time <= timestamp``, or ``None`` before the first settlement
        (or when the view has no funding feed).  ``mark_price`` is the stream's
        raw value and may be ``NaN`` where Binance omitted it — the hedge marks
        against the sim's own reference price and consumes only the rate (S20).
        Raises ``LookAheadError`` when ``timestamp`` is past the active wall.
        """
        self._check_time(timestamp)
        funding = self.funding
        if funding is None or funding.height == 0:
            return None
        times = funding["funding_time"]
        idx = int(times.search_sorted(timestamp, side="right")) - 1
        if idx < 0:
            return None
        rate = float(funding["funding_rate"][idx])
        raw_mark = (
            funding["mark_price"][idx] if "mark_price" in funding.columns else None
        )
        mark = float("nan") if raw_mark is None else float(raw_mark)
        return rate, mark

    def slice(self, start_seq: int, end_seq: int) -> MarketView:
        """Return a sub-view over ``[start_seq, end_seq)``.

        Raises ``LookAheadError`` if the requested range extends beyond the
        active split's seq range (or starts before it).
        """
        active = self.active_split_data()
        min_seq, last_seq = self._active_range()
        if start_seq < min_seq or end_seq > last_seq + 1:
            raise LookAheadError(
                f"slice [{start_seq}, {end_seq}) is outside the active split "
                f"seq range [{min_seq}, {last_seq}]"
            )
        if start_seq >= end_seq:
            raise MarketViewError(
                f"slice start_seq ({start_seq}) must be < end_seq ({end_seq})"
            )

        sub = active.filter((pl.col("seq") >= start_seq) & (pl.col("seq") < end_seq))
        if sub.height == 0:
            raise MarketViewError(
                f"slice [{start_seq}, {end_seq}) produced an empty view"
            )

        sub_start_raw = sub["block_timestamp"].min()
        sub_end_raw = sub["block_timestamp"].max()
        if sub_start_raw is None or sub_end_raw is None:
            raise MarketViewError("slice produced rows with no block_timestamp")
        sub_start = cast(datetime, sub_start_raw)
        sub_end = cast(datetime, sub_end_raw)

        # Keep the parent lane's rows before this window so the backtester can
        # warm its exact fee engine from the lane-start pool_state.
        prior = active.filter(pl.col("seq") < start_seq)
        warmup: pl.DataFrame | None
        if self.warmup is not None and self.warmup.height:
            warmup = pl.concat([self.warmup, prior]) if prior.height else self.warmup
        else:
            warmup = prior if prior.height else None

        if self.is_train:
            return MarketView(
                train=sub,
                eval=None,
                reference=self.reference,
                regimes=self.regimes,
                gas=self.gas,
                train_start_utc=sub_start,
                train_end_utc=sub_end,
                eval_start_utc=self.eval_start_utc,
                eval_end_utc=self.eval_end_utc,
                is_train=True,
                episode_config=self.episode_config,
                pool=self.pool,
                pool_state=self.pool_state,
                warmup=warmup,
                funding=self.funding,
            )
        return MarketView(
            train=None,
            eval=sub,
            reference=self.reference,
            regimes=self.regimes,
            gas=self.gas,
            train_start_utc=self.train_start_utc,
            train_end_utc=self.train_end_utc,
            eval_start_utc=sub_start,
            eval_end_utc=sub_end,
            is_train=False,
            episode_config=self.episode_config,
            pool=self.pool,
            pool_state=self.pool_state,
            warmup=warmup,
            funding=self.funding,
        )

    def window(self, start_seq: int, end_seq: int) -> pl.DataFrame:
        """Active-split rows with ``start_seq <= seq < end_seq`` (ADR-033).

        The replay environment only needs the window's own rows; unlike
        :meth:`slice` this returns them without materialising the lane's
        pre-window history into ``warmup`` (which a late-window reset would
        otherwise copy in full).  ``seq`` is dense and sorted, so the bounds are
        two binary searches plus one contiguous slice.  Range validation matches
        :meth:`slice` — this is still a look-ahead door.
        """
        _, seq_arr = self._episode_arrays()
        min_seq = int(seq_arr[0])
        last_seq = int(seq_arr[-1])
        if start_seq < min_seq or end_seq > last_seq + 1:
            raise LookAheadError(
                f"window [{start_seq}, {end_seq}) is outside the active split "
                f"seq range [{min_seq}, {last_seq}]"
            )
        if start_seq >= end_seq:
            raise MarketViewError(
                f"window start_seq ({start_seq}) must be < end_seq ({end_seq})"
            )
        active = self.active_split_data()
        seqs = active["seq"]
        lo = int(seqs.search_sorted(int(start_seq), side="left"))
        hi = int(seqs.search_sorted(int(end_seq) - 1, side="right"))
        if hi <= lo:
            raise MarketViewError(
                f"window [{start_seq}, {end_seq}) produced an empty view"
            )
        return active.slice(lo, hi - lo)

    def tape_at(self, seq: int) -> dict[str, object]:
        """All columns for the event at ``seq`` as a dict.

        Raises ``LookAheadError`` for a seq outside the active split's range.
        """
        min_seq, last_seq = self._active_range()
        if seq < min_seq or seq > last_seq:
            raise LookAheadError(
                f"seq {seq} is outside the active split seq range [{min_seq}, {last_seq}]"
            )
        sub = self.active_split_data().filter(pl.col("seq") == seq)
        if sub.height == 0:
            raise MarketViewError(f"no tape row at seq={seq}")
        return dict(sub.row(0, named=True))

    def reference_close(self, at_time: datetime) -> float | None:
        """Last reference close with ``close_time <= at_time``.

        Returns ``None`` if no bar precedes ``at_time`` (before the feed starts).
        Raises ``LookAheadError`` when ``at_time`` is past the active wall.
        """
        self._check_time(at_time)
        sub = self.reference.filter(pl.col("close_time") <= at_time)
        if sub.height == 0:
            return None
        return float(sub.sort("close_time")["close"][-1])

    def reference_bars(self, start_time: datetime, end_time: datetime) -> pl.DataFrame:
        """Reference bars in ``[start_time, end_time)``, clamped to the split.

        The upper bound is clamped to the active wall (it can never leak an
        evaluation bar into a training view). A ``start_time`` at or beyond the
        wall is a look-ahead read and raises ``LookAheadError``.
        """
        if start_time >= self.active_end_utc:
            raise LookAheadError(
                f"reference_bars start {start_time} is past the active split "
                f"wall {self.active_end_utc}"
            )
        clamped_start = max(start_time, self.active_start_utc)
        clamped_end = min(end_time, self.active_end_utc)
        if clamped_start >= clamped_end:
            return self.reference.clear()
        # No look-ahead: a bar may only be used once its close_time is known,
        # so exclude any bar whose close falls after the wall. Filtering on
        # open_time alone would leak a bar that straddles the boundary.
        return self.reference.filter(
            (pl.col("open_time") >= clamped_start) & (pl.col("close_time") <= clamped_end)
        ).sort("open_time")

    def regime_at(self, at_time: datetime) -> str:
        """Regime label at ``at_time`` (as-of join).

        Returns ``"unknown"`` if no label covers ``at_time``. Raises
        ``LookAheadError`` when ``at_time`` is past the active wall.  The regime
        feed is sorted by ``timestamp`` by construction, so this binary-searches
        instead of re-scanning the full feed (the evaluation's dominant-regime
        pass and the look-ahead probes call it).
        """
        self._check_time(at_time)
        if self.regimes.height == 0:
            return Regime.UNKNOWN.value
        idx = int(self.regimes["timestamp"].search_sorted(at_time, side="right")) - 1
        if idx < 0:
            return Regime.UNKNOWN.value
        return str(self.regimes["regime"][idx])

    def gas_at_block(self, block_number: int) -> dict[str, object] | None:
        """Gas row for ``block_number``. ``None`` if no gas data for that block.

        The gas feed is chain-wide (not split-scoped), so this takes no time/seq
        argument and needs no wall check.  The feed is sorted by ``block_number``
        by construction, so this binary-searches instead of scanning the
        multi-million-row feed per call (each rebalance asks for it twice).
        """
        blocks = self.gas["block_number"]
        idx = int(blocks.search_sorted(block_number))
        if idx < self.gas.height and int(blocks[idx]) == block_number:
            return dict(self.gas.row(idx, named=True))
        return None

    def active_split_data(self) -> pl.DataFrame:
        """The event tape for the active split (train or eval), sorted by seq."""
        active = self.train if self.is_train else self.eval
        if active is None:  # pragma: no cover - __post_init__ guarantees this
            raise MarketViewError("MarketView: no active split data")
        if "seq" in active.columns:
            # Every frame built by build_market_view/slice is already seq-sorted;
            # checking avoids re-allocating a full sort on each call (ADR-033).
            if active["seq"].is_sorted():
                return active
            return active.sort("seq")
        return active

    def step_count(self) -> int:
        """Number of tape events in the active split."""
        return self.active_split_data().height

    def _episode_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Cached ascending ``(block_timestamp, seq)`` arrays of the active split.

        ``block_timestamp`` is non-decreasing in ``seq`` (the data pipeline's
        monotonic-timestamp check), so both arrays are searchable with
        ``np.searchsorted``.  Built once per view; the frozen dataclass is
        updated through ``object.__setattr__`` (ADR-033).
        """
        cached = self._episode_index
        if cached is None:
            active = self.active_split_data()
            cached = (
                active["block_timestamp"].to_numpy(),
                active["seq"].to_numpy(),
            )
            object.__setattr__(self, "_episode_index", cached)
        return cached

    def build_episode(
        self,
        seed: int,
        max_steps: int | None = None,
        *,
        episode_config: EpisodeConfig | None = None,
    ) -> tuple[int, int]:
        """Sample a contiguous wall-clock episode window of the active split.

        Returns ``(start_seq, end_seq)`` where ``end_seq`` is exclusive. The
        window spans ``desired_steps * step_minutes`` of wall-clock time, where
        ``desired_steps`` defaults to ``EpisodeConfig.duration_days`` at the
        ``step_minutes`` cadence and is overridden by ``max_steps`` when supplied.
        It is sampled uniformly from the active split and never crosses the
        train/eval boundary; a window at least as long as the split is the whole
        split. ``seq`` is the dense canonical tape index (EVENT_TAPE_SCHEMA), so
        the returned range is contiguous in ``seq`` but its length is the number
        of tape *events* in the window, not the number of decision steps.
        """
        ep = episode_config or self.episode_config or EpisodeConfig()
        step_minutes = int(ep.step_minutes)
        if step_minutes <= 0:
            raise MarketViewError("build_episode: step_minutes must be > 0")
        config_steps = (ep.duration_days * 24 * 60) // step_minutes
        desired = config_steps if max_steps is None else int(max_steps)
        if desired <= 0:
            raise MarketViewError(f"build_episode: desired window must be > 0, got {desired}")
        window = timedelta(minutes=desired * step_minutes)

        times, seqs = self._episode_arrays()
        if times.size == 0:
            raise MarketViewError("build_episode: the active split is empty")
        # All stamps are UTC; compare the naive numpy grid against the wall made
        # naive, preserving the previous aware-datetime ordering exactly.
        split_end = self.active_end_utc.replace(tzinfo=None)
        split_start = times[0].item()
        if window >= split_end - split_start:
            # The window cannot fit anywhere but at the start: return the split.
            return int(seqs[0]), int(seqs[-1]) + 1

        # Candidate starts = rows with room for the whole window before the wall.
        limit = split_end - window
        n_candidates = int(
            np.searchsorted(times, np.datetime64(limit, "us"), side="right")
        )
        if n_candidates <= 0:  # pragma: no cover - guarded by the fit check above
            return int(seqs[0]), int(seqs[-1]) + 1
        rng = np.random.default_rng(seed)
        start_idx = int(rng.integers(0, n_candidates))
        window_us = int(window.total_seconds() * 1_000_000)
        end_limit = times[start_idx] + np.timedelta64(window_us, "us")
        end_idx = int(np.searchsorted(times, end_limit, side="left"))
        if end_idx <= start_idx:  # pragma: no cover - window > 0 implies a row
            end_idx = start_idx + 1
        return int(seqs[start_idx]), int(seqs[end_idx - 1]) + 1


def _select_primary_symbol(frame: pl.DataFrame, configured: str, stream: str) -> pl.DataFrame:
    """Restrict a reference/regime feed to the configured primary symbol.

    The reference and regime streams deliberately carry both the primary Binance
    symbol and its cross-check (data ``PLAN.md`` §7: ``ETHUSDT`` primary,
    ``ETHUSDC`` as cross-check). Every sim consumer needs a *single* series — the
    raw multi-symbol feed has two bars per minute, so ``reference_bars`` would
    otherwise yield duplicate ``close_time`` rows that alternate symbols (and
    diverge wildly where one symbol was gap-filled). Select
    ``DataConfig.reference_symbol`` here, at the one door to history.

    A feed that does **not** contain the configured symbol raises rather than
    silently falling back to another series — including a single-symbol feed whose
    symbol is not the configured primary, because valuing the sim against the
    cross-check while the tape's ``price_reference`` uses the primary would produce
    two reference prices in one run. Frames without a ``symbol`` column (a bare
    hand-built series) and empty frames are passed through.
    """
    if frame.height == 0 or "symbol" not in frame.columns:
        return frame
    symbols = frame["symbol"].unique().to_list()
    if configured not in symbols:
        raise MarketViewError(
            f"MarketView: {stream} feed carries symbols {sorted(symbols)} but the "
            f"configured reference_symbol {configured!r} is not among them"
        )
    return frame.filter(pl.col("symbol") == configured)


def load_market_view(
    data_config_path: str | Path,
    split_config: SplitConfig,
    *,
    split: str = "train",
    episode_config: EpisodeConfig | None = None,
) -> MarketView:
    """Load a built dataset snapshot into a :class:`MarketView`.

    The simulator never fetches, so the endpoint ``${ENV_VAR}`` placeholders are
    left unexpanded (``expand_endpoints=False``): reading a dataset must not
    require ``GRAPH_API_KEY``/``ETH_RPC_URL``.  ``split`` is validated by
    :func:`build_market_view`; this wrapper only narrows the type.
    """
    data_config = load_data_config(data_config_path, expand_endpoints=False)
    return build_market_view(
        data_config,
        split_config,
        split=cast(SplitName, split),
        episode_config=episode_config,
    )


def build_market_view(
    data_config: DataConfig,
    split_config: SplitConfig,
    *,
    split: SplitName = "train",
    episode_config: EpisodeConfig | None = None,
) -> MarketView:
    """Load the dataset and partition it into a look-ahead-safe ``MarketView``.

    Historical tables are read only through the top-level ``undertow.data``
    public API and converted from pyarrow to polars. Raises ``MarketViewError``
    if the requested split window contains no tape rows.
    """
    if split not in ("train", "eval"):
        raise MarketViewError(f"build_market_view: unknown split {split!r}")

    tape = _sort(_to_polars(load_tape(data_config)), ["block_number", "log_index"])
    reference = _sort(_to_polars(load_reference_feed(data_config)), ["open_time"])
    gas = _sort(_to_polars(load_gas_feed(data_config)), ["block_number"])
    regimes = _sort(_to_polars(load_regime_labels(data_config)), ["timestamp"])
    fee_growth = _sort(_to_polars(load_fee_growth(data_config)), ["block_number", "tick"])
    funding = _sort(
        _to_polars(load_funding(data_config)), ["symbol", "funding_time"]
    )
    # One series, always: the feeds carry the primary + cross-check symbols.
    reference = _select_primary_symbol(reference, data_config.reference_symbol, "reference")
    regimes = _select_primary_symbol(regimes, data_config.reference_symbol, "regime")
    funding = _select_primary_symbol(funding, data_config.reference_symbol, "funding")

    def _state_at(active: pl.DataFrame) -> pl.DataFrame | None:
        """The pool-state snapshot for ``active``'s first block (or ``None``).

        The seed block is ``first_block - 1``: a snapshot at block ``B`` is the
        state *after* ``B``, and the backtester replays block ``B``'s events, so
        seeding at ``B`` would double-apply them.
        """
        if fee_growth.height == 0 or "block_number" not in fee_growth.columns:
            return None
        seed_block = cast(int, active["block_number"].min()) - 1
        exact = fee_growth.filter(pl.col("block_number") == seed_block)
        if exact.height == 0:
            LOGGER.warning(
                "build_market_view: a fee_growth stream is present but has no "
                "snapshot at block %d (the active split's first block minus one); "
                "the backtester will fall back to an empty lattice. Re-run "
                "`undertow-data pool-state` after changing the split.",
                seed_block,
            )
            return None
        return exact

    if split == "train":
        start, end = split_config.train_start_utc, split_config.train_end_utc
        active = _between(tape, "block_timestamp", start, end)
        if active.height == 0:
            raise MarketViewError(
                f"train split [{start}, {end}] contains no tape rows"
            )
        return MarketView(
            train=active,
            eval=None,
            reference=reference,
            regimes=regimes,
            gas=gas,
            train_start_utc=split_config.train_start_utc,
            train_end_utc=split_config.train_end_utc,
            eval_start_utc=split_config.eval_start_utc,
            eval_end_utc=split_config.eval_end_utc,
            is_train=True,
            episode_config=episode_config,
            pool=data_config.pool,
            pool_state=_state_at(active),
            funding=funding if funding.height else None,
        )

    start, end = split_config.eval_start_utc, split_config.eval_end_utc
    active = _between(tape, "block_timestamp", start, end)
    if active.height == 0:
        raise MarketViewError(f"eval split [{start}, {end}] contains no tape rows")
    return MarketView(
        train=None,
        eval=active,
        reference=reference,
        regimes=regimes,
        gas=gas,
        train_start_utc=split_config.train_start_utc,
        train_end_utc=split_config.train_end_utc,
        eval_start_utc=split_config.eval_start_utc,
        eval_end_utc=split_config.eval_end_utc,
        is_train=False,
        episode_config=episode_config,
        pool=data_config.pool,
        pool_state=_state_at(active),
        funding=funding if funding.height else None,
    )