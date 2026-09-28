"""Per-regime metric slicing.

Regime labels are **consumed** from the data module's regime table — they are never
recomputed in ``undertow.sim`` (``PLAN.md`` §7 "Regime labels"; ADR-001
``docs/decisions/001-regime-drift-definition.md`` defines how the upstream labels were
produced).  This module only groups an evaluation frame by its existing label column and
applies each caller-supplied metric function to that regime's equity sub-series.
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

#: Callable applied to a regime's equity sub-series, returning one metric value.
MetricFn = Callable[[pl.Series], float]

#: Default equity column name — matches ``BacktestLedger.equity_curve``.
DEFAULT_EQUITY_COLUMN: str = "equity"


def per_regime_metrics(
    equity_curve: pl.DataFrame,
    regime_column: str,
    metric_fns: dict[str, MetricFn],
    *,
    equity_column: str = DEFAULT_EQUITY_COLUMN,
) -> pl.DataFrame:
    """Compute each metric for each regime label.

    Parameters
    ----------
    equity_curve:
        Per-step DataFrame with an equity column and a regime-label column.
    regime_column:
        Name of the string/regime-label column to group by.  Null labels are dropped.
    metric_fns:
        Mapping ``metric_name -> fn`` where ``fn`` takes the regime's ``pl.Series`` of
        equity values and returns a ``float``.
    equity_column:
        Name of the equity column (default ``"equity"``).

    Returns
    -------
    polars.DataFrame
        One row per regime (in first-appearance order), with a leading ``"regime"``
        column and one float column per metric.

    Raises
    ------
    KeyError
        If ``regime_column`` or ``equity_column`` is absent from ``equity_curve``.
    """
    for column in (regime_column, equity_column):
        if column not in equity_curve.columns:
            raise KeyError(f"equity_curve is missing required column: {column!r}")

    metric_names = list(metric_fns)
    rows: list[dict[str, object]] = []
    labels = equity_curve[regime_column].drop_nulls().unique(maintain_order=True)

    for label in labels:
        sub = equity_curve.filter(pl.col(regime_column) == label)[equity_column]
        row: dict[str, object] = {"regime": label}
        for name in metric_names:
            row[name] = float(metric_fns[name](sub))
        rows.append(row)

    if not rows:
        schema: dict[str, pl.DataType] = {"regime": pl.Utf8}
        schema.update({name: pl.Float64 for name in metric_names})
        return pl.DataFrame(schema=schema)

    return pl.DataFrame(rows)


__all__ = ["DEFAULT_EQUITY_COLUMN", "MetricFn", "per_regime_metrics"]
