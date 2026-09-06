"""Backward-looking regime labels from the reference price feed (T12).

This is roadmap gap **G4** (regime robustness) made reproducible: every bar of the
reference feed gets a label — ``bull`` / ``bear`` / ``sideways`` / ``high_vol`` —
computed from the two §10.3 statistics over a rolling window of closes:

- ``sigma_rv`` — **annualized realized volatility**: sample std of the per-bar log
  returns over the window (``ddof=1``), scaled by ``sqrt(periods_per_year)``.
- ``mu`` — **total log return over the window**: ``log(close_end / close_start)``.

The ``mu`` definition deliberately deviates from the roadmap text: §10.3 writes
``mu = (1/T) sum log(S_t/S_{t-1})`` (the *mean per-minute* log return) and then
thresholds it at ±5%. A mean per-minute return of 5% is a factor of ``e^2160`` over
30 days — the threshold and the statistic are dimensionally inconsistent, and
implemented literally *every* window on earth would be labelled ``sideways``,
making the regime split vacuous. Per ADR-001 (``docs/decisions/001-regime-drift-definition.md``)
the drift is the **total** window log return, the quantity ±5% is meaningful for.

Properties every label honours (the look-ahead risk, ``PLAN.md`` §8):

- **Backward-looking, closed on the right.** The label at time ``t`` uses only bars
  with ``close_time <= t``, so ``label_series`` never peeks past its own row. This
  is guarded by an explicit truncated-vs-full-series test.
- **Rows are never dropped.** The first ``W-1`` rows (window not yet full; ``W`` is
  ``lookback_days`` of bars) get ``regime = "unknown"`` with
  ``window_complete = False``. On a minute grid a 30-day window is ``W = 43_200``
  bars, so the warmup is ``43_199`` rows: the discrete-window ±1 grid effect on
  "the first 30 days", documented because someone will ask.
- **Gap-filled bars count.** Forward-filled bars (``is_gap_filled``) enter the
  window with zero returns, which biases volatility *downward*; any window whose
  filled-bar share exceeds ``gap_fill_warning_fraction`` (default 1%) triggers a
  ``GapFilledWindowWarning``.
- **Thresholds come from ``RegimeConfig`` only.** This module contains no magic
  numbers: the 0.80 / ±0.05 cutoffs are read from ``cfg``, and the window size is
  derived from ``cfg.lookback_days * cfg.periods_per_year // 365`` (43_200 bars for
  the default minute-bar config) so the configuration stays the single source of
  truth. The only literal is 365 (days per year), the unit relation the contract
  itself is written in.

Efficiency: the 3-year default input is ~1.6M minute bars against a 43_200-bar
window — the naive O(n·w) loop is ~7e10 operations. Instead, ``sigma_rv`` uses
polars' native rolling ``std`` (a numerically stable online algorithm — the
in-repo fast-path test pins 1e-10 agreement with brute force on a 500-bar series,
and a 1.6M-row spot check agreed to ~1e-18 — see the PR body), and ``mu`` is a
single log-ratio of two endpoints, so no long cumulative sums appear anywhere.

Public surface matches ``CONTRACTS.md`` §6.3 exactly. Everything below is typed;
``from __future__ import annotations`` per repo convention.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Sequence
from typing import cast

import numpy as np
import polars as pl
import pyarrow as pa

from undertow.data.config import RegimeConfig
from undertow.data.schemas import REFERENCE_SCHEMA, REGIME_SCHEMA, validate_table
from undertow.data.types import Regime

LOGGER = logging.getLogger("undertow.data.transforms.regimes")


class GapFilledWindowWarning(UserWarning):
    """Emitted by ``label_series`` when a window's gap-filled share is too high.

    A volatility estimate over forward-filled prices is biased **downward** (the
    filled bars contribute zero returns) and can mislabel a crash as sideways —
    the exact trap §10.3's regime split exists to catch, so it is surfaced, never
    silent.
    """


# Output of sensitivity_sweep — an extension table (CONTRACTS.md §6.3 lists the
# function but no schema; the schema is T12's to define).
_SWEEP_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("vol_threshold", pa.float64(), nullable=False),
        pa.field("drift_threshold", pa.float64(), nullable=False),
        pa.field("regime", pa.string(), nullable=False),
        pa.field("share_of_time", pa.float64(), nullable=False),
        pa.field("n_windows", pa.int64(), nullable=False),
    ],
    metadata={"stream": "regime_sensitivity", "schema_version": "1.0.0"},
)

_DAYS_PER_YEAR = 365  # unit relation behind cfg.periods_per_year (default 525_600 = 365*1440)


def _window_size_bars(cfg: RegimeConfig) -> int:
    """Number of bars in one lookback window, derived from ``cfg`` only."""
    return cfg.lookback_days * cfg.periods_per_year // _DAYS_PER_YEAR


def _validate_closes(closes: np.ndarray) -> np.ndarray:
    """Coerce closes to a 1-D float64 array; reject non-finite or non-positive prices.

    Prices must be strictly positive for a log return to exist. Guarding here (rather
    than producing ``nan`` deep inside a rolling computation) keeps a corrupt feed
    loud at the boundary.
    """
    arr = np.asarray(closes, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"closes must be 1-D; got {arr.ndim}-D array of shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("closes must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("closes must be finite; got non-finite value(s)")
    if np.any(arr <= 0.0):
        raise ValueError("closes must be strictly positive; log returns are undefined otherwise")
    return arr


# ---------------------------------------------------------------------------
# The two statistics (CONTRACTS.md §6.3).
# ---------------------------------------------------------------------------


def realized_volatility(closes: np.ndarray, periods_per_year: int) -> float:
    """Annualized realized volatility over a window of close prices.

    ``rv = std(log returns, ddof=1) * sqrt(periods_per_year)`` where the log returns
    are between consecutive closes of ``closes``. The **sample** std (``ddof=1``) is
    used, not the population std — the population/sample choice changes the third
    decimal of the annualized number (``sqrt((n-1)/n)``), so it is stated here on
    purpose and pinned by the tests.

    With minute bars ``periods_per_year = 365 * 1440 = 525_600``; the caller's config
    owns the value. Requires at least 3 closes (2 log returns — a sample variance
    with ``ddof=1`` needs n >= 2).
    """
    if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, int):
        raise ValueError(f"periods_per_year must be an int; got {type(periods_per_year).__name__}")
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be >= 1; got {periods_per_year}")
    arr = _validate_closes(closes)
    if arr.size < 3:
        raise ValueError(
            f"realized_volatility needs at least 3 closes (2 log returns); got {arr.size}"
        )
    log_returns = np.diff(np.log(arr))
    return float(np.std(log_returns, ddof=1) * math.sqrt(periods_per_year))


def window_drift(closes: np.ndarray) -> float:
    """Total log return over the window: ``log(closes[-1] / closes[0])``.

    Per **ADR-001** (``docs/decisions/001-regime-drift-definition.md``): the roadmap's
    §10.3 mean per-minute log return ``mu = (1/T) sum log(S_t/S_{t-1})`` thresholded
    at ±5% is dimensionally inconsistent (a 5% *per-minute* mean is ``e^2160`` over
    30 days, so every window would label ``sideways``). ``mu`` here is the **total**
    window log return — the quantity ±5% is meaningful for — i.e. the single log
    ratio of the window's endpoints, which is also exact arithmetic on two floats
    with no long summation to lose precision.
    """
    arr = _validate_closes(closes)
    if arr.size < 2:
        raise ValueError(f"window_drift needs at least 2 closes; got {arr.size}")
    # log(a/b), exactly as ADR-001 writes it: for a/b near 1 this keeps the full
    # float64 precision that a log(a) - log(b) subtraction would cancel away.
    return float(np.log(arr[-1] / arr[0]))


def label_regime(sigma_rv: float, mu: float, cfg: RegimeConfig) -> Regime:
    """Label one (sigma_rv, mu) pair with §10.3's decision order, exactly:

    1. ``sigma_rv > cfg.vol_threshold``            -> ``HIGH_VOL``  (vol dominates)
    2. else ``mu > cfg.drift_threshold``           -> ``BULL``
    3. else ``mu < -cfg.drift_threshold``          -> ``BEAR``
    4. else                                        -> ``SIDEWAYS``

    Comparisons are strict (``>`` / ``<``), so a value exactly on a threshold falls
    to the *less extreme* label. Non-finite inputs (nan/inf) compare False on every
    branch and therefore classify ``SIDEWAYS``; ``label_series`` validates feeds so
    this path only matters for direct callers.
    """
    if sigma_rv > cfg.vol_threshold:
        return Regime.HIGH_VOL
    if mu > cfg.drift_threshold:
        return Regime.BULL
    if mu < -cfg.drift_threshold:
        return Regime.BEAR
    return Regime.SIDEWAYS


# ---------------------------------------------------------------------------
# Vectorised classifier backend — must match label_regime's branch order exactly.
# ---------------------------------------------------------------------------


def _classify(
    sigma: np.ndarray, mu: np.ndarray, vol_threshold: float, drift_threshold: float
) -> np.ndarray:
    """Numpy bulk version of ``label_regime`` (same decision order, strict edges).

    Used by ``sensitivity_sweep`` over the whole series per grid point without
    re-running the rolling statistics.
    """
    labels = np.full(sigma.shape, Regime.SIDEWAYS.value, dtype=object)
    high_vol = sigma > vol_threshold
    labels[high_vol] = Regime.HIGH_VOL.value
    rest = ~high_vol
    bull = rest & (mu > drift_threshold)
    labels[bull] = Regime.BULL.value
    labels[rest & ~bull & (mu < -drift_threshold)] = Regime.BEAR.value
    return labels


# ---------------------------------------------------------------------------
# Rolling labeller internals.
# ---------------------------------------------------------------------------


def _label_symbol(
    cuts: pl.DataFrame, cfg: RegimeConfig, gap_fill_warning_fraction: float
) -> pl.DataFrame:
    """Label one symbol's bars (already sorted by close_time).

    Returns a DataFrame with the ``REGIME_SCHEMA`` columns plus a private
    ``_gap_fraction`` column the caller drops after warning.
    """
    w = _window_size_bars(cfg)
    n_ret = w - 1  # log returns inside a w-bar window
    if w < 3:
        raise ValueError(
            f"regime lookback window of {w} bars is too small; need w >= 3 "
            "(raise lookback_days or periods_per_year)"
        )
    if cuts["close_time"].is_duplicated().any():
        raise ValueError(
            f"reference symbol {cuts['symbol'][0]!r}: duplicate close_time bars break "
            "backward-looking windows; expected one bar per close"
        )

    log_close = pl.col("close").log()
    log_ret = log_close.diff()

    # sigma at bar i: std over the n_ret log returns inside the w-bar window
    # [i-n_ret .. i] (i.e. bars [i-w+1 .. i]), ddof=1, annualized. Polars'
    # rolling_std is a numerically stable online algorithm — verified against a
    # brute-force per-window recomputation at 1.6M rows (see test_regimes).
    sigma = log_ret.rolling_std(window_size=n_ret, ddof=1) * math.sqrt(cfg.periods_per_year)
    # mu at bar i: total log return over the same window — one subtraction.
    mu = log_close - log_close.shift(n_ret)

    # Complete window ⇔ at least w bars with close_time <= this bar (1-based count).
    complete = pl.col("close_time").cum_count() > n_ret

    gap_fraction = (pl.col("is_gap_filled").cast(pl.Int64).rolling_sum(window_size=w) / w).alias(
        "_gap_fraction"
    )

    label = (
        pl.when(complete & (sigma > cfg.vol_threshold))
        .then(pl.lit(Regime.HIGH_VOL.value))
        .when(complete & (mu > cfg.drift_threshold))
        .then(pl.lit(Regime.BULL.value))
        .when(complete & (mu < -cfg.drift_threshold))
        .then(pl.lit(Regime.BEAR.value))
        .when(complete)
        .then(pl.lit(Regime.SIDEWAYS.value))
        .otherwise(pl.lit(Regime.UNKNOWN.value))
    )

    out = cuts.with_columns(
        pl.col("close_time").alias("timestamp"),
        pl.when(complete).then(sigma).otherwise(pl.lit(float("nan"))).alias("sigma_rv"),
        pl.when(complete).then(mu).otherwise(pl.lit(float("nan"))).alias("mu"),
        label.alias("regime"),
        complete.alias("window_complete"),
        gap_fraction,
    ).select(
        ["timestamp", "sigma_rv", "mu", "regime", "window_complete", "symbol", "_gap_fraction"]
    )
    _warn_gap_filled(out, str(cuts["symbol"][0]), gap_fill_warning_fraction)
    return out.drop("_gap_fraction")


def _warn_gap_filled(out: pl.DataFrame, symbol: str, threshold: float) -> None:
    """Warn once per symbol if any complete window's filled-bar share is too high.

    ``threshold`` is a fraction (default 0.01 from ``label_series``). One summary
    warning per call keeps the log readable over a whole multi-year pull.
    """
    fraction = out["_gap_fraction"].to_numpy()
    complete = out["window_complete"].to_numpy()
    bad = complete & (fraction > threshold)
    n = int(np.count_nonzero(bad))
    if n == 0:
        return
    max_frac = float(np.max(fraction[bad]))
    stamps = out["timestamp"].to_numpy()[bad]
    first = str(np.datetime_as_string(stamps.min(), unit="us"))
    last = str(np.datetime_as_string(stamps.max(), unit="us"))
    message = (
        f"regime labelling {symbol}: {n} window(s) exceed the gap-filled bar threshold "
        f"of {threshold:.1%} (worst {max_frac:.1%} of a window). Forward-filled bars "
        "contribute zero returns, biasing volatility downward: a crash hidden behind an "
        "exchange outage may be labelled sideways. First affected window ends "
        f"{first}, last {last}."
    )
    warnings.warn(message, GapFilledWindowWarning, stacklevel=3)
    LOGGER.warning(message)


# ---------------------------------------------------------------------------
# Public rolling labeller and sweep (CONTRACTS.md §6.3).
# ---------------------------------------------------------------------------


def label_series(
    reference: pa.Table,
    cfg: RegimeConfig,
    *,
    gap_fill_warning_fraction: float = 0.01,
) -> pa.Table:
    """Label every bar of a ``REFERENCE_SCHEMA`` table (``REGIME_SCHEMA`` output).

    One output row per input bar (stride 1 — dense, so T11's backward as-of join on
    the regime table can only land on the most recent label). The window is
    ``cfg.lookback_days`` of bars, **backward-looking and closed on the right**: the
    label at time ``t`` uses only bars with ``close_time <= t``, never future bars.
    Incomplete warmup rows (the first ``w-1`` bars) get ``regime = "unknown"`` with
    ``window_complete = False`` and carry ``nan`` in ``sigma_rv``/``mu``; they are
    *never* dropped, so downstream row counts keep agreeing with the reference feed.

    Multi-symbol tables are labelled per symbol (each symbol gets its own windows)
    and returned sorted by ``(timestamp, symbol)`` — deterministic bytes per the
    reproducibility rule.

    ``gap_fill_warning_fraction`` extends the frozen contract signature with a
    keyword-only default: any complete window whose share of ``is_gap_filled`` bars
    exceeds it emits a ``GapFilledWindowWarning`` (see the module docstring).
    """
    validate_table(reference, REFERENCE_SCHEMA, strict=False)
    if reference.num_rows == 0:
        return REGIME_SCHEMA.empty_table()
    if not 0.0 <= gap_fill_warning_fraction <= 1.0:
        raise ValueError(
            "gap_fill_warning_fraction must be a fraction in [0, 1]; "
            f"got {gap_fill_warning_fraction!r}"
        )

    frame: pl.DataFrame = cast(pl.DataFrame, pl.from_arrow(reference)).sort(
        ["symbol", "close_time"]
    )
    parts: list[pl.DataFrame] = []
    for symbol in sorted(frame["symbol"].unique().to_list()):
        parts.append(
            _label_symbol(frame.filter(pl.col("symbol") == symbol), cfg, gap_fill_warning_fraction)
        )
    out = pl.concat(parts).sort(["timestamp", "symbol"])
    table = out.to_arrow()
    if table.schema != REGIME_SCHEMA:
        # polars round-trips strings as large_string on the arrow boundary; the
        # contract pins `string` (utf8), so coerce to the canonical schema.
        table = table.cast(REGIME_SCHEMA)
    validate_table(table, REGIME_SCHEMA, strict=True)
    return table


def sensitivity_sweep(
    reference: pa.Table,
    cfg: RegimeConfig,
    vol_grid: Sequence[float],
    drift_grid: Sequence[float],
) -> pa.Table:
    """Regime-share table over the threshold grid — the §10.3 pre-commitment sweep.

    For every ``(vol_threshold, drift_threshold)`` combination the rolling statistics
    are identical (they do not depend on the cutoffs — they are computed **once**),
    so the sweep relabels the same ``sigma_rv``/``mu`` pair per grid point: each
    combination appears exactly once and the four regime shares within it sum to 1
    (floating-point tolerance). This is what answers *"does the headline survive
    ±10% on the cutoff?"* and gets reported in the thesis.

    Columns (T12's own extension schema, not in the frozen contract)::

        vol_threshold  f64   vol cutoff used for this combination
        drift_threshold f64  drift cutoff used for this combination
        regime         str   bull | bear | sideways | high_vol
        share_of_time  f64   share of complete windows labelled `regime`
        n_windows      i64   number of complete windows labelled `regime`

    Shares are taken over **complete** windows only (``window_complete``); the
    ``unknown`` warmup rows are excluded from both the denominator and the table.
    """
    vol_values = [float(v) for v in vol_grid]
    drift_values = [float(d) for d in drift_grid]
    _check_threshold_grid(vol_values, "vol_grid")
    _check_threshold_grid(drift_values, "drift_grid")
    validate_table(reference, REFERENCE_SCHEMA, strict=False)
    if reference.num_rows == 0:
        return _SWEEP_SCHEMA.empty_table()

    frame: pl.DataFrame = cast(pl.DataFrame, pl.from_arrow(reference)).sort(
        ["symbol", "close_time"]
    )
    w = _window_size_bars(cfg)
    n_ret = w - 1
    if w < 3:
        raise ValueError(
            f"regime lookback window of {w} bars is too small; need w >= 3 "
            "(raise lookback_days or periods_per_year)"
        )
    complete_list: list[np.ndarray] = []
    sigma_list: list[np.ndarray] = []
    mu_list: list[np.ndarray] = []
    for symbol in sorted(frame["symbol"].unique().to_list()):
        cuts = frame.filter(pl.col("symbol") == symbol)
        log_close = pl.col("close").log()
        log_ret = log_close.diff()
        sigma_expr = log_ret.rolling_std(window_size=n_ret, ddof=1) * math.sqrt(
            cfg.periods_per_year
        )
        mu_expr = log_close - log_close.shift(n_ret)
        complete_expr = pl.col("close_time").cum_count() > n_ret
        stats = cuts.with_columns(
            sigma_expr.alias("_s"), mu_expr.alias("_m"), complete_expr.alias("_c")
        )
        sigma_list.append(stats["_s"].to_numpy())
        mu_list.append(stats["_m"].to_numpy())
        complete_list.append(stats["_c"].to_numpy())
    sigma = np.concatenate(sigma_list)
    mu = np.concatenate(mu_list)
    complete = np.concatenate(complete_list)
    n_total = int(np.count_nonzero(complete))
    if n_total == 0:
        # Window larger than the whole feed: nothing complete to sweep over.
        return _SWEEP_SCHEMA.empty_table()

    rows_vol: list[float] = []
    rows_drift: list[float] = []
    rows_regime: list[str] = []
    rows_share: list[float] = []
    rows_n: list[int] = []
    for vol in vol_values:
        for drift in drift_values:
            labels = _classify(sigma, mu, vol, drift)
            for regime in (Regime.BULL, Regime.BEAR, Regime.SIDEWAYS, Regime.HIGH_VOL):
                in_regime = complete & (labels == regime.value)
                count = int(np.count_nonzero(in_regime))
                rows_vol.append(vol)
                rows_drift.append(drift)
                rows_regime.append(regime.value)
                rows_share.append(count / n_total)
                rows_n.append(count)
    return pa.table(
        {
            "vol_threshold": rows_vol,
            "drift_threshold": rows_drift,
            "regime": rows_regime,
            "share_of_time": rows_share,
            "n_windows": rows_n,
        },
        schema=_SWEEP_SCHEMA,
    )


def _check_threshold_grid(values: Sequence[float], name: str) -> None:
    """Threshold grids must be non-empty, finite and non-negative (a negative cutoff
    would make every window ``sideways``)."""
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty")
    if any(not math.isfinite(v) or v < 0.0 for v in values):
        raise ValueError(f"{name} thresholds must be finite and >= 0; got {values!r}")


__all__ = [
    "GapFilledWindowWarning",
    "realized_volatility",
    "window_drift",
    "label_regime",
    "label_series",
    "sensitivity_sweep",
]
