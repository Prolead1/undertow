"""Tests for ``undertow.data.transforms.regimes`` (T12).

Covers CONTRACTS.md §6.3: the two statistics (with the ADR-001 drift definition), the
§10.3 decision order and its strict boundaries, the backward-looking rolling
labeller (including the mandatory truncated-vs-full look-ahead test and the no-row-
dropping invariant), the fast-path-vs-brute-force equivalence, the gap-fill warning,
schema conformance of ``label_series``/``sensitivity_sweep``, and the end-to-end
crash labelling that proves the thresholds mean what §10.3 intends.

Fixture provenance (``tests/data/fixtures/reference_series.json``): fully synthetic —
wave 3 has no real Binance klines in this sandbox (T08 supplies them in the same
wave), so values are designed to shape, not trimmed from real responses. Bars are
**hourly** (the statistics are scale-free: the annualization factor and the window
size both come from ``RegimeConfig``), so tests select period scales matched to the
fixture length — a 30-day minute window would need a 43,200-bar fixture, which the
"fixtures are small" rule forbids. The same formulas are verified against the real
minute-scale ``periods_per_year = 525_600`` analytically in ``test_realized_volatility``
and at 1.6M minute rows by the benchmark noted in the PR body.
"""

from __future__ import annotations

import json
import math
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from undertow.data.config import RegimeConfig
from undertow.data.schemas import REFERENCE_SCHEMA, REGIME_SCHEMA, validate_table
from undertow.data.transforms.regimes import (
    GapFilledWindowWarning,
    label_regime,
    label_series,
    realized_volatility,
    sensitivity_sweep,
    window_drift,
)
from undertow.data.types import Regime

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "reference_series.json"

# Hourly bars -> periods_per_year = 365*24, so lookback_days=1 is a 24-bar window.
CFG_HOURLY = RegimeConfig(lookback_days=1, periods_per_year=365 * 24)
CFG_60 = RegimeConfig(lookback_days=1, periods_per_year=365 * 60)  # 60-bar window
CFG_1000 = RegimeConfig(lookback_days=1, periods_per_year=365 * 1000)  # 1000-bar window


# ---------------------------------------------------------------------------
# Fixture loader
# ---------------------------------------------------------------------------


def _build_reference_table(series: dict[str, object]) -> pa.Table:
    """Turn one fixture series dict into a REFERENCE_SCHEMA table.

    The fixture stores only `close_time`-derived data (close, is_gap_filled, symbol);
    the other columns are synthetic envelopes derived from the closes (open = previous
    close, high/low from open/close, constant volumes/trades, open_time = close_time
    minus the interval). See the fixture's header comment.
    """
    start = datetime.fromisoformat(str(series["start_utc"]).replace("Z", "+00:00")).astimezone(UTC)
    interval = timedelta(seconds=int(series["interval_seconds"]))
    closes = [float(x) for x in series["close"]]
    n = len(closes)
    closes_arr = np.asarray(closes, dtype=np.float64)
    open_arr = np.concatenate([[closes_arr[0]], closes_arr[:-1]])
    close_time = [start + i * interval for i in range(n)]
    symbol = str(series["symbol"])
    table = pa.table(
        {
            "open_time": [t - interval for t in close_time],
            "close_time": close_time,
            "open": open_arr,
            "high": np.maximum(open_arr, closes_arr),
            "low": np.minimum(open_arr, closes_arr),
            "close": closes_arr,
            "volume_base": np.full(n, 100.0),
            "quote_volume": closes_arr * 100.0,
            "trades": np.full(n, 10, dtype=np.int64),
            "symbol": [symbol] * n,
            "is_gap_filled": [bool(x) for x in series["is_gap_filled"]],
        },
        schema=REFERENCE_SCHEMA,
    )
    validate_table(table, REFERENCE_SCHEMA, strict=True)
    return table


def _fixture(name: str) -> pa.Table:
    with open(FIXTURE) as f:
        raw = json.load(f)
    entry = next(s for s in raw["series"] if s["name"] == name)
    return _build_reference_table(entry)


def _synthetic_closes(n: int, seed: int = 7, shock: int = -1) -> np.ndarray:
    """A random-walk close series with optional one-bar +200% shock (for the fast-vs-brute test)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.004, n)
    if shock >= 0:
        rets[shock] = math.log(3.0)  # one huge return
    return 2000.0 * np.exp(np.cumsum(rets))


def _table_from_closes(closes: np.ndarray, *, symbol: str = "ETHUSDT") -> pa.Table:
    start = datetime(2023, 1, 1, tzinfo=UTC)
    interval = timedelta(hours=1)
    n = len(closes)
    table = pa.table(
        {
            "open_time": [start + (i - 1) * interval for i in range(n)],
            "close_time": [start + i * interval for i in range(n)],
            "open": np.concatenate([[closes[0]], closes[:-1]]),
            "high": np.maximum(closes, np.concatenate([[closes[0]], closes[:-1]])),
            "low": np.minimum(closes, np.concatenate([[closes[0]], closes[:-1]])),
            "close": closes,
            "volume_base": np.full(n, 100.0),
            "quote_volume": closes * 100.0,
            "trades": np.full(n, 10, dtype=np.int64),
            "symbol": [symbol] * n,
            "is_gap_filled": np.zeros(n, dtype=bool),
        },
        schema=REFERENCE_SCHEMA,
    )
    validate_table(table, REFERENCE_SCHEMA, strict=True)
    return table


# ---------------------------------------------------------------------------
# 1-3. The two statistics.
# ---------------------------------------------------------------------------


def test_realized_volatility_matches_analytic_sample_std() -> None:
    """rv == std(log returns, ddof=1) * sqrt(periods_per_year), at real minute scale."""
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0002, 0.01, 250)
    closes = 2000.0 * np.exp(np.cumsum(rets))
    p = 525_600  # minute bars per year — the real pipeline scale
    got = realized_volatility(closes, p)
    expected = float(np.std(np.diff(np.log(closes)), ddof=1) * math.sqrt(p))
    assert got == pytest.approx(expected, rel=1e-12)

    # Hand-computed 5-point series: log returns [0.01, -0.02, 0.03, -0.01] have
    # sample variance sum((x-mean)^2)/3 with mean 0.0025; periods_per_year = 4.
    closes5 = 100.0 * np.exp(np.cumsum([0.0, 0.01, -0.02, 0.03, -0.01]))
    hand_std = math.sqrt(((0.0075) ** 2 + (-0.0225) ** 2 + (0.0275) ** 2 + (-0.0125) ** 2) / 3.0)
    assert realized_volatility(closes5, 4) == pytest.approx(hand_std * 2.0, abs=1e-12)


def test_realized_volatility_constant_price_is_exactly_zero() -> None:
    closes = np.full(50, 2000.0)
    assert realized_volatility(closes, 525_600) == 0.0


def test_window_drift_double_flat_halving() -> None:
    closes = np.concatenate([np.full(24, 2000.0), np.full(24, 4000.0)])
    assert window_drift(closes) == pytest.approx(math.log(2.0), abs=1e-12)
    assert window_drift(np.full(48, 2000.0)) == 0.0
    halves = np.concatenate([np.full(24, 4000.0), np.full(24, 2000.0)])
    assert window_drift(halves) == pytest.approx(-math.log(2.0), abs=1e-12)


# ---------------------------------------------------------------------------
# 4-6. label_regime: truth table, precedence, strict boundaries.
# ---------------------------------------------------------------------------


def test_label_regime_truth_table() -> None:
    cfg = RegimeConfig()
    assert label_regime(0.10, 0.10, cfg) is Regime.BULL
    assert label_regime(0.10, -0.10, cfg) is Regime.BEAR
    assert label_regime(0.10, 0.02, cfg) is Regime.SIDEWAYS
    assert label_regime(1.20, 0.30, cfg) is Regime.HIGH_VOL
    assert label_regime(1.20, -0.30, cfg) is Regime.HIGH_VOL
    assert label_regime(0.90, 0.00, cfg) is Regime.HIGH_VOL
    assert label_regime(0.50, 0.06, cfg) is Regime.BULL
    assert label_regime(0.50, -0.06, cfg) is Regime.BEAR


def test_label_regime_vol_dominates_decision_order() -> None:
    cfg = RegimeConfig()
    # sigma above the threshold wins over an extreme positive (and negative) drift.
    assert label_regime(1.2, 0.5, cfg) is Regime.HIGH_VOL
    assert label_regime(1.2, -0.5, cfg) is Regime.HIGH_VOL


def test_label_regime_strict_boundaries() -> None:
    cfg = RegimeConfig(vol_threshold=0.8, drift_threshold=0.05)
    # Exactly on a threshold falls to the LESS extreme label.
    assert label_regime(0.8, 0.5, cfg) is Regime.BULL  # not HIGH_VOL
    assert label_regime(0.1, 0.05, cfg) is Regime.SIDEWAYS  # not BULL
    assert label_regime(0.1, -0.05, cfg) is Regime.SIDEWAYS  # not BEAR
    assert label_regime(0.8, -0.05, cfg) is Regime.SIDEWAYS  # both boundaries
    # Just off the boundary does flip.
    assert label_regime(0.8 + 1e-12, 0.0, cfg) is Regime.HIGH_VOL
    assert label_regime(0.1, 0.05 + 1e-12, cfg) is Regime.BULL


# ---------------------------------------------------------------------------
# 7-8. label_series: look-ahead guard and the never-drop invariant.
# ---------------------------------------------------------------------------


def test_label_series_look_ahead_guard() -> None:
    """Flat for 40h then a +50% jump: labels before the jump are unaffected by it.

    The jump bar is at index 40; labelling the full series and the series truncated
    just before the jump must yield identical labels (and statistics) on the shared
    prefix, because every window at time t uses only bars with close_time <= t.
    """
    full = _fixture("flat_then_jump")  # 50 bars: 40 flat @ 2000, then 10 @ 3000
    assert full.num_rows == 50
    truncated = full.slice(0, 40)  # ending just before the jump

    labels_full = label_series(full, CFG_HOURLY)
    labels_trunc = label_series(truncated, CFG_HOURLY)

    for col in ("regime", "sigma_rv", "mu", "window_complete"):
        a = labels_full.column(col).to_numpy(zero_copy_only=False)[:40]
        b = labels_trunc.column(col).to_numpy(zero_copy_only=False)[:40]
        if col in ("sigma_rv", "mu"):  # NaN-bearing float stats
            assert np.array_equal(a, b, equal_nan=True), f"prefix mismatch in {col!r}"
        else:
            assert np.array_equal(a, b), f"prefix mismatch in {col!r}"

    # The jump is felt AT its own bar (window [17..40] contains the huge return)...
    assert labels_full.column("regime")[40].as_py() == Regime.HIGH_VOL.value
    assert labels_full.column("mu")[40].as_py() > math.log(1.5) * 0.5  # ~+40.5%
    # ...but not one bar earlier.
    assert (
        labels_full.column("regime")[39].as_py()
        == labels_trunc.column("regime")[39].as_py()
        == Regime.SIDEWAYS.value
    )


def test_label_series_incomplete_window_unknown_not_dropped() -> None:
    table = _fixture("crash")  # 160 hourly bars
    w = 24  # CFG_HOURLY window
    out = label_series(table, CFG_HOURLY)

    # Nothing dropped: output rows == input bars.
    assert out.num_rows == table.num_rows == 160
    # First w-1 rows are the incomplete warmup: UNKNOWN, window_complete False, nan stats.
    warmup = out.slice(0, w - 1)
    assert warmup.column("window_complete").to_pylist() == [False] * (w - 1)
    assert set(warmup.column("regime").to_pylist()) == {Regime.UNKNOWN.value}
    assert all(math.isnan(x) for x in warmup.column("sigma_rv").to_pylist()) and all(
        math.isnan(x) for x in warmup.column("mu").to_pylist()
    )
    # The w-th row is the first complete one, and it is a real label.
    row = out.slice(w - 1, 1)
    assert row.column("window_complete")[0].as_py() is True
    assert row.column("regime")[0].as_py() in {
        r.value for r in (Regime.BULL, Regime.BEAR, Regime.SIDEWAYS, Regime.HIGH_VOL)
    }
    # The invariant: regime == unknown exactly when window_complete is False, everywhere.
    regimes = out.column("regime").to_pylist()
    complete = out.column("window_complete").to_pylist()
    assert [(r == Regime.UNKNOWN.value) for r in regimes] == [not c for c in complete]


# ---------------------------------------------------------------------------
# 9. Fast path vs brute force.
# ---------------------------------------------------------------------------


def test_fast_path_matches_brute_force() -> None:
    closes = _synthetic_closes(500, seed=11, shock=250)  # 500 bars, one big shock
    table = _table_from_closes(closes)
    p = CFG_60.periods_per_year
    w = 60

    fast = label_series(table, CFG_60)
    sigma_fast = np.asarray(fast.column("sigma_rv").to_numpy(zero_copy_only=False), dtype=float)
    mu_fast = np.asarray(fast.column("mu").to_numpy(zero_copy_only=False), dtype=float)
    regime_fast = fast.column("regime").to_pylist()

    for i in range(w - 1, 500):
        win = closes[i - w + 1 : i + 1]
        rv = realized_volatility(win, p)
        mu = window_drift(win)
        assert abs(sigma_fast[i] - rv) <= 1e-10, (i, sigma_fast[i], rv)
        assert abs(mu_fast[i] - mu) <= 1e-10, (i, mu_fast[i], mu)
        assert regime_fast[i] == label_regime(rv, mu, CFG_60).value


# ---------------------------------------------------------------------------
# 10. Gap-fill warning.
# ---------------------------------------------------------------------------


def test_gap_filled_window_warning() -> None:
    table = _fixture("gap_filled")  # 1080 bars: 1000 flat (1 filled), 60-bar outage, 20 lower
    assert table.num_rows == 1080
    prefix = table.slice(0, 1000)  # max 1/1000 = 0.1% filled bars in any window

    # 6.1% filled in the outage windows: warning fires.
    with pytest.warns(GapFilledWindowWarning, match="gap-filled"):
        label_series(table, CFG_1000)
    # 0.1% filled: warning does not fire (threshold default 1%).
    with warnings.catch_warnings():
        warnings.simplefilter("error", GapFilledWindowWarning)
        label_series(prefix, CFG_1000)
    # Same fixture, but a higher configured tolerance: no warning either.
    with warnings.catch_warnings():
        warnings.simplefilter("error", GapFilledWindowWarning)
        label_series(table, CFG_1000, gap_fill_warning_fraction=0.2)

    # The trap itself: while the outage is the newest data, the label is sideways —
    # the forward-filled bars have zero returns, so the crash is invisible until real
    # bars arrive. The warning is what makes the sideways label trustworthy.
    out = label_series(table, CFG_1000, gap_fill_warning_fraction=0.2)
    assert out.column("regime")[1020].as_py() == Regime.SIDEWAYS.value
    # Once real (lower) bars arrive, the drop is seen and vol dominates.
    assert out.column("regime")[1079].as_py() == Regime.HIGH_VOL.value


# ---------------------------------------------------------------------------
# 11. Schema conformance.
# ---------------------------------------------------------------------------


def test_label_series_output_conforms_to_regime_schema() -> None:
    out = label_series(_fixture("crash"), CFG_HOURLY)
    validate_table(out, REGIME_SCHEMA, strict=True)
    ts_type = out.schema.field("timestamp").type
    assert ts_type == pa.timestamp("us", tz="UTC")
    assert out.schema.field("sigma_rv").type == pa.float64()
    n = out.num_rows
    assert n == 160
    allowed = {r.value for r in Regime}
    assert set(out.column("regime").to_pylist()) <= allowed
    assert out.column("symbol").to_pylist() == ["ETHUSDT"] * n
    # Sorted by the contract's sort key (timestamp,).
    stamps = out.column("timestamp").to_pylist()
    assert stamps == sorted(stamps)


# ---------------------------------------------------------------------------
# 12. Sensitivity sweep.
# ---------------------------------------------------------------------------


def test_sensitivity_sweep_shares_sum_to_one_and_monotone() -> None:
    table = _fixture("crash")
    vol_grid = [0.7, 0.8, 0.9, 1.5, 10.0]
    drift_grid = [0.04, 0.05, 0.06]
    sweep = sensitivity_sweep(table, CFG_HOURLY, vol_grid, drift_grid)

    assert sweep.num_rows == len(vol_grid) * len(drift_grid) * 4
    by = {c: sweep.column(c).to_pylist() for c in sweep.column_names}
    regimes = {r.value for r in (Regime.BULL, Regime.BEAR, Regime.SIDEWAYS, Regime.HIGH_VOL)}

    n_windows_total: int | None = None
    for (_vol, _drift), rows in _group_sweep(by).items():
        shares = {r: rows[r][0] for r in rows}
        assert sum(shares.values()) == pytest.approx(1.0, abs=1e-9)
        assert set(shares) == regimes
        # n_windows is the per-regime window count; they must partition a constant total.
        total = sum(rows[r][1] for r in rows)
        if n_windows_total is None:
            n_windows_total = total
            assert n_windows_total > 0  # fixture is long enough to have complete windows
        assert total == n_windows_total
        # share_of_time == n_windows / total, exactly as documented.
        for regime in regimes:
            share, n = rows[regime]
            assert share == pytest.approx(n / total, abs=1e-12)
    # Monotonicity: a higher vol threshold never increases the HIGH_VOL share.
    for drift in drift_grid:
        shares = [
            by["share_of_time"][_index_of((v, drift, Regime.HIGH_VOL.value), by)] for v in vol_grid
        ]
        assert shares == sorted(shares, reverse=True)
    # A threshold above every realised sigma_rv (10.0) empties the HIGH_VOL bin.
    assert by["share_of_time"][_index_of((10.0, 0.05, Regime.HIGH_VOL.value), by)] == 0.0


def _group_sweep(
    by: dict[str, list[object]],
) -> dict[tuple[float, float], dict[str, tuple[float, int]]]:
    out: dict[tuple[float, float], dict[str, tuple[float, int]]] = {}
    for vol, drift, regime, share, n in zip(
        by["vol_threshold"],
        by["drift_threshold"],
        by["regime"],
        by["share_of_time"],
        by["n_windows"],
        strict=True,
    ):
        assert isinstance(vol, float) and isinstance(drift, float)
        assert isinstance(regime, str) and isinstance(share, float) and isinstance(n, int)
        out.setdefault((vol, drift), {})[regime] = (share, n)
    return out


def _index_of(key: tuple[float, float, str], by: dict[str, list[object]]) -> int:
    for i, (v, d, r) in enumerate(
        zip(by["vol_threshold"], by["drift_threshold"], by["regime"], strict=True)
    ):
        if (v, d, r) == key:
            return i
    raise AssertionError(f"combination not found: {key}")


# ---------------------------------------------------------------------------
# 13. End-to-end crash labelling — the thresholds mean what §10.3 intends.
# ---------------------------------------------------------------------------


def test_crash_series_labels_high_vol_then_bear() -> None:
    table = _fixture("crash")  # 43h flat, -7.5%/-24% two-hour crash, recovery, grind
    out = label_series(table, CFG_HOURLY)
    regimes = out.column("regime").to_pylist()
    sigma = np.asarray(out.column("sigma_rv").to_numpy(zero_copy_only=False), dtype=float)
    mu = np.asarray(out.column("mu").to_numpy(zero_copy_only=False), dtype=float)

    # The bar right after the last crash bar (index 44) is HIGH_VOL, with a big
    # annualised sigma AND a large negative drift — a crash is a crash.
    assert regimes[44] == Regime.HIGH_VOL.value
    assert sigma[44] > 0.80
    assert mu[44] < -0.05
    # The crash day produced the largest annualised vol of the whole series
    # (NaN warmup rows excluded) — the peak window is right after the crash bars.
    complete = np.asarray(out.column("window_complete").to_numpy(zero_copy_only=False), dtype=bool)
    peak = 23 + int(np.argmax(sigma[complete]))  # first complete row is index 23
    assert 43 <= peak <= 67, peak
    assert sigma[43] > 0.80 and sigma[44] > 0.80
    # A few bars later the ~30% drop is still inside the window: still HIGH_VOL...
    assert regimes[60] == Regime.HIGH_VOL.value
    # ...and once the calm recovery is the entire window, BULL appears (grind later
    # turns BEAR; early flat is SIDEWAYS) — seeds for the four-bin check below.
    assert Regime.BULL.value in regimes
    assert Regime.BEAR.value in regimes
    assert Regime.SIDEWAYS.value in regimes


# ---------------------------------------------------------------------------
# 14. Four-bin coverage (synthetic data only in this sandbox).
# ---------------------------------------------------------------------------


def test_all_four_regime_bins_non_empty() -> None:
    """All four bins are reachable under one config on one real-shaped series.

    On real data this is the finding the T12 brief's handoff notes ask for; in the
    sandbox it is demonstrated synthetically (T08's klines arrive in the same wave).
    """
    out = label_series(_fixture("crash"), CFG_HOURLY)
    complete = [
        r
        for r, c in zip(
            out.column("regime").to_pylist(),
            out.column("window_complete").to_pylist(),
            strict=True,
        )
        if c
    ]
    assert set(complete) == {"bull", "bear", "sideways", "high_vol"}


# ---------------------------------------------------------------------------
# 15. Empty inputs, multi-symbol, and error paths.
# ---------------------------------------------------------------------------


def test_label_series_empty_input_returns_empty_regime_table() -> None:
    empty = REFERENCE_SCHEMA.empty_table()
    out = label_series(empty, RegimeConfig())
    assert out.num_rows == 0
    validate_table(out, REGIME_SCHEMA, strict=True)
    sweep = sensitivity_sweep(empty, RegimeConfig(), [0.8], [0.05])
    assert sweep.num_rows == 0


def test_label_series_multi_symbol_labels_per_symbol() -> None:
    single = _fixture("flat_then_jump")
    other = _table_from_closes(
        np.asarray(single.column("close").to_numpy(zero_copy_only=False), dtype=float),
        symbol="ETHUSDC",
    )
    both = pa.concat_tables([single, other])
    out = label_series(both, CFG_HOURLY)
    assert out.num_rows == 2 * single.num_rows
    ethusdt = out.filter(pa.compute.equal(out.column("symbol"), "ETHUSDT"))
    # Per-symbol windows: the ETHUSDT half must match the single-symbol run exactly.
    single_labels = label_series(single, CFG_HOURLY)
    for col in ("regime", "sigma_rv", "mu", "window_complete"):
        a = ethusdt.column(col).to_pylist()
        b = single_labels.column(col).to_pylist()
        if col in ("sigma_rv", "mu"):
            assert all(
                (x == y) or (math.isnan(x) and math.isnan(y)) for x, y in zip(a, b, strict=True)
            )
        else:
            assert a == b
    assert set(out.column("symbol").to_pylist()) == {"ETHUSDT", "ETHUSDC"}


def test_error_paths() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        realized_volatility(np.array([0.0, 5.0, 6.0]), 4)
    with pytest.raises(ValueError, match="at least 3 closes"):
        realized_volatility(np.array([1.0, 2.0]), 4)
    with pytest.raises(ValueError, match="at least 2 closes"):
        window_drift(np.array([1.0]))
    with pytest.raises(ValueError, match="too small"):
        label_series(
            _table_from_closes(np.full(50, 2000.0)),
            RegimeConfig(lookback_days=1, periods_per_year=2),
        )
    with pytest.raises(ValueError, match="fraction"):
        label_series(_fixture("crash"), CFG_HOURLY, gap_fill_warning_fraction=1.5)
    with pytest.raises(ValueError, match="vol_grid"):
        sensitivity_sweep(_fixture("crash"), CFG_HOURLY, [], [0.05])
    with pytest.raises(ValueError, match="drift_grid"):
        sensitivity_sweep(_fixture("crash"), CFG_HOURLY, [0.8], [-1.0])
