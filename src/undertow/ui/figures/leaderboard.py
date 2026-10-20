"""Leaderboard bar builder (U04, ``CONTRACTS.md`` §4/§6).

``leaderboard_figure`` renders the top-N runs by a chosen metric as a horizontal
bar chart, coloured by run state, with run ids as categories.  It consumes plain
``RunRecord``-shaped mappings; the ranking rule itself (eval-first, Sharpe then
worst-regime then reward) lives in :mod:`undertow.ui.registry` and is applied by
U10 — this builder is a faithful plotter of whatever it is handed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from undertow.ui.figures.theme import (
    TOKENS,
    FigureSpec,
    build_spec,
    empty_figure,
    finite,
)

__all__ = ["MAX_LEADERBOARD_BARS", "leaderboard_figure"]

#: The brief pins "top N"; 15 keeps the chart readable without scrolling.
MAX_LEADERBOARD_BARS = 15

_STATE_COLORS: dict[str, str] = {
    "running": TOKENS["accent"],
    "done": TOKENS["bull"],
    "failed": TOKENS["bear"],
    "unknown": TOKENS["sideways"],
}


def _metric_value(record: Mapping[str, object], metric: str) -> float | None:
    """Finite ``metric`` from ``record["eval"]`` if present, else the top level."""
    evaluation = record.get("eval")
    if isinstance(evaluation, Mapping):
        value = finite(evaluation.get(metric))
        if value is not None:
            return value
    return finite(record.get(metric))


def _format(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.3g}"


def leaderboard_figure(
    records: Sequence[Mapping[str, object]],
    *,
    metric: str,
) -> FigureSpec:
    """Horizontal bar of the top :data:`MAX_LEADERBOARD_BARS` runs by ``metric``.

    Runs without a finite metric value are skipped; an empty result yields the
    themed "No data" figure.  Bars are ordered worst-to-best so the best run
    renders at the top of the chart.
    """
    scored: list[tuple[float, str, str]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        value = _metric_value(record, metric)
        if value is None:
            continue
        run_id = str(record.get("run_id") or "?")
        state = str(record.get("state") or "unknown")
        scored.append((value, run_id, state))

    if not scored:
        return empty_figure()

    scored.sort(key=lambda item: (item[0], item[1]))
    top = scored[-MAX_LEADERBOARD_BARS:]
    values = [item[0] for item in top]
    run_ids = [item[1] for item in top]
    colors = [_STATE_COLORS.get(item[2], TOKENS["sideways"]) for item in top]

    data: list[dict[str, Any]] = [
        {
            "type": "bar",
            "x": values,
            "y": run_ids,
            "orientation": "h",
            "name": metric,
            "marker": {"color": colors},
            "text": [_format(value) for value in values],
            "textposition": "outside",
            "textfont": {"color": TOKENS["text-primary"], "size": 11},
            "customdata": [[color] for color in colors],
            "hovertemplate": "%{y}<br>" + metric + " %{x:.4g}<extra></extra>",
        }
    ]
    layout: dict[str, Any] = {
        "title": {"text": f"Leaderboard — {metric}"},
        "xaxis": {"title": metric},
        "yaxis": {"title": "", "automargin": True},
        "margin": {"l": 140, "r": 48, "t": 48, "b": 44},
        "showlegend": False,
        "bargap": 0.35,
    }
    return build_spec(data, layout)
