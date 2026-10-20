"""Hero "ribbon" figure builder (U04, ``CONTRACTS.md`` §2/§6).

The Policy Replay chart: price on a log axis, the active position
``[tick_lower, tick_upper]`` as a filled band on a secondary tick axis, the band
segments recoloured by ``in_range``, ``gas``-sized rebalance markers, an optional
HODL counterfactual, and regime background bands.

Input rows are rollout-parquet-shaped mappings (plain dicts); the builder never
imports ``undertow.sim``/``undertow.data``.  A one-row or empty episode is
degenerate, so both render the themed "No data" figure rather than a broken
ribbon.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

from undertow.ui.figures.theme import (
    REGIME_TOKENS,
    TOKENS,
    FigureSpec,
    build_spec,
    empty_figure,
    finite,
    rgba,
)

__all__ = ["ribbon_figure"]


class _CleanRow(TypedDict):
    """One rollout row reduced to finite, typed values."""

    step: float
    price: float
    tick_lower: float | None
    tick_upper: float | None
    in_range: bool
    action_type: object
    center_offset: float
    width: float
    gas: float
    equity: float | None
    hodl: float | None
    regime: str


def _segments(values: Sequence[object]) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` runs of equal values (deterministic)."""
    if not values:
        return []
    segments: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            segments.append((start, index))
            start = index
    return segments


def _regime_color(regime: str) -> str:
    return TOKENS[REGIME_TOKENS.get(regime, "text-muted")]


def _marker_size(gas: float) -> float:
    """Marker area scales with ``sqrt(|gas|)`` so small costs stay legible."""
    return 7.0 + 6.0 * math.sqrt(abs(gas))


def _clean_rows(rows: Sequence[Mapping[str, object]]) -> list[_CleanRow]:
    """Finite price+step rows, sorted by step (CONTRACTS §0 ordering)."""
    cleaned: list[_CleanRow] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        step = finite(row.get("step", row.get("timestep")))
        price = finite(row.get("price"))
        if step is None or price is None or price <= 0:
            continue
        cleaned.append(
            {
                "step": step,
                "price": price,
                "tick_lower": finite(row.get("tick_lower")),
                "tick_upper": finite(row.get("tick_upper")),
                "in_range": bool(row.get("in_range")),
                "action_type": row.get("action_type"),
                "center_offset": finite(row.get("center_offset")) or 0.0,
                "width": finite(row.get("width")) or 0.0,
                "gas": finite(row.get("gas")) or 0.0,
                "equity": finite(row.get("equity")),
                "hodl": finite(row.get("hodl")),
                "regime": str(row.get("regime") or "unknown"),
            }
        )
    cleaned.sort(key=lambda item: item["step"])
    return cleaned


def _is_rebalance(row: Mapping[str, object]) -> bool:
    action_type = row.get("action_type")
    if action_type is not None:
        return str(action_type) == "rebalance"
    # Fallback for older rows that only carry the discrete action fields.
    return bool(row.get("center_offset")) or bool(row.get("width"))


def ribbon_figure(
    rows: Sequence[Mapping[str, object]],
    *,
    show_hodl: bool = True,
) -> FigureSpec:
    """Build the hero position-ribbon figure.

    ``rows`` are rollout rows (``CONTRACTS.md`` §2.1) as plain mappings.  The
    position band's lower/upper traces carry the raw ``tick_lower``/``tick_upper``
    values on the secondary tick axis; price stays on the primary log axis.
    """
    clean = _clean_rows(rows)
    if len(clean) < 2:
        return empty_figure()

    x = [row["step"] for row in clean]
    price = [row["price"] for row in clean]
    lower = [row["tick_lower"] for row in clean]
    upper = [row["tick_upper"] for row in clean]
    in_range = [row["in_range"] for row in clean]
    regimes = [row["regime"] for row in clean]
    gas = [row["gas"] for row in clean]

    data: list[dict[str, Any]] = [
        {
            "type": "scatter",
            "x": x,
            "y": price,
            "mode": "lines",
            "name": "price",
            "line": {"color": TOKENS["accent"], "width": 2},
            "hovertemplate": "step %{x}<br>price %{y:.2f} USDC/WETH<extra>price</extra>",
        }
    ]

    if show_hodl:
        hodl = _hodl_series(clean)
        if any(value is not None for value in hodl):
            data.append(
                {
                    "type": "scatter",
                    "x": x,
                    "y": hodl,
                    "mode": "lines",
                    "name": "HODL",
                    "line": {"color": TOKENS["text-muted"], "width": 1.5, "dash": "dot"},
                    "connectgaps": False,
                    "hovertemplate": "step %{x}<br>HODL %{y:.2f}<extra>HODL</extra>",
                }
            )

    band_legend = False
    for start, end in _segments(in_range):
        segment_lower = lower[start:end]
        segment_upper = upper[start:end]
        if all(value is None for value in segment_upper) and all(
            value is None for value in segment_lower
        ):
            continue
        color = TOKENS["bull"] if in_range[start] else TOKENS["bear"]
        label = "in range" if in_range[start] else "out of range"
        data.append(
            {
                "type": "scatter",
                "x": x[start:end],
                "y": segment_lower,
                "mode": "lines",
                "name": "position band lower",
                "legendgroup": "position-band",
                "line": {"width": 0, "color": rgba(color, 0.0)},
                "hoverinfo": "skip",
                "showlegend": False,
                "yaxis": "y2",
            }
        )
        data.append(
            {
                "type": "scatter",
                "x": x[start:end],
                "y": segment_upper,
                "mode": "lines",
                "name": f"position band ({label})",
                "legendgroup": "position-band",
                "fill": "tonexty",
                "fillcolor": rgba(color, 0.20),
                "line": {"width": 1, "color": color},
                "customdata": [[label] for _ in segment_upper],
                "hovertemplate": "step %{x}<br>tick %{y}<br>%{customdata[0]}<extra></extra>",
                "showlegend": not band_legend,
                "yaxis": "y2",
            }
        )
        band_legend = True

    rebalance_x: list[float] = []
    rebalance_y: list[float] = []
    rebalance_size: list[float] = []
    for index, row in enumerate(clean):
        if not _is_rebalance(row):
            continue
        rebalance_x.append(x[index])
        rebalance_y.append(price[index])
        rebalance_size.append(_marker_size(gas[index]))
    if rebalance_x:
        data.append(
            {
                "type": "scatter",
                "x": rebalance_x,
                "y": rebalance_y,
                "mode": "markers",
                "name": "rebalance",
                "marker": {
                    "symbol": "triangle-down",
                    "size": rebalance_size,
                    "color": TOKENS["rebalance"],
                    "line": {"color": TOKENS["bg-base"], "width": 1},
                },
                "customdata": [[size] for size in rebalance_size],
                "hovertemplate": "rebalance<br>step %{x}<br>price %{y:.2f}<extra></extra>",
            }
        )

    shapes = _regime_shapes(x, regimes)
    layout: dict[str, Any] = {
        "hovermode": "x unified",
        "xaxis": {"title": "step"},
        "yaxis": {"title": "price (USDC/WETH)", "type": "log"},
        # The primary axis is a log price scale; ``y2`` is a raw-tick axis in
        # protocol units. Overlaying them is intentional: the position band is
        # meaningless in price space, so only the price trace uses ``y``.
        "yaxis2": {
            "title": "tick",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "tickfont": {"color": TOKENS["text-muted"]},
        },
        "shapes": shapes,
        "margin": {"l": 64, "r": 64, "t": 48, "b": 44},
    }
    run_id = rows[0].get("run_id") if rows and isinstance(rows[0], Mapping) else None
    if isinstance(run_id, str) and run_id:
        layout["title"] = {"text": run_id}
    return build_spec(data, layout)


def _hodl_series(clean: Sequence[_CleanRow]) -> list[float | None]:
    """Explicit per-row ``hodl`` wealth, else ``equity[0] * price / price[0]``."""
    base_price = clean[0]["price"]
    base_equity = clean[0]["equity"]
    base = float(base_equity) if base_equity is not None else None
    series: list[float | None] = []
    for row in clean:
        explicit = row["hodl"]
        if explicit is not None:
            series.append(explicit)
            continue
        if base is None or base_price <= 0:
            series.append(None)
            continue
        series.append(base * row["price"] / base_price)
    return series


def _regime_shapes(x: Sequence[float], regimes: Sequence[str]) -> list[dict[str, Any]]:
    """One translucent background rectangle per contiguous regime run."""
    shapes: list[dict[str, Any]] = []
    step = x[1] - x[0] if len(x) > 1 else 1.0
    for start, end in _segments(regimes):
        x0 = float(x[start])
        x1 = float(x[end]) if end < len(x) else float(x[-1]) + max(step, 1.0)
        shapes.append(
            {
                "type": "rect",
                "xref": "x",
                "yref": "paper",
                "x0": x0,
                "x1": x1,
                "y0": 0.0,
                "y1": 1.0,
                "fillcolor": rgba(_regime_color(regimes[start]), 0.10),
                "line": {"width": 0},
                "layer": "below",
            }
        )
    return shapes
