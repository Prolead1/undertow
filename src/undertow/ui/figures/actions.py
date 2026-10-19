"""Action-distribution heatmap builder (U04, ``CONTRACTS.md`` §4/§6).

``action_heatmap_figure`` maps a sparse ``(center_offset, width) -> count``
mapping onto a dense ``z`` matrix over the requested grid, colours it with a
sequential scale derived from the ``accent`` token, and annotates each cell with
its value.  The always-available ``hold`` action (offset 0, width 0) is surfaced
as a labelled annotation when it is present in the input but outside the grid.
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
    rgba,
)

__all__ = ["action_heatmap_figure"]


def _format(value: float) -> str:
    """Compact cell label: integers without a decimal, floats to 3 significant digits."""
    if value.is_integer():
        return str(int(value))
    return f"{value:.3g}"


def action_heatmap_figure(
    counts: Mapping[tuple[int, int], float],
    *,
    offsets: Sequence[int],
    widths: Sequence[int],
    title: str = "Action distribution",
) -> FigureSpec:
    """Heatmap of action counts over ``offsets`` x ``widths`` (tick-spacings).

    ``z`` has shape ``(len(offsets), len(widths))``; missing cells are zero and
    non-finite counts are coerced to zero so the payload never carries ``NaN``.
    A ``(0, 0)`` entry (the ``hold`` action) that is not part of the grid is
    shown as a separate labelled annotation.
    """
    offs = [int(offset) for offset in offsets]
    wids = [int(width) for width in widths]
    if not offs or not wids:
        return empty_figure()

    z: list[list[float]] = []
    text: list[list[str]] = []
    for offset in offs:
        z_row: list[float] = []
        text_row: list[str] = []
        for width in wids:
            value = finite(counts.get((offset, width), 0.0))
            safe = value if value is not None else 0.0
            z_row.append(safe)
            text_row.append(_format(safe))
        z.append(z_row)
        text.append(text_row)

    data: list[dict[str, Any]] = [
        {
            "type": "heatmap",
            "x": wids,
            "y": offs,
            "z": z,
            "text": text,
            "texttemplate": "%{text}",
            "textfont": {"size": 10, "color": TOKENS["text-primary"]},
            "colorscale": [
                [0.0, rgba(TOKENS["bg-surface"], 0.0)],
                [0.5, rgba(TOKENS["accent"], 0.45)],
                [1.0, TOKENS["accent"]],
            ],
            "colorbar": {
                "title": {"text": title, "side": "right"},
                "outlinewidth": 0,
                "tickfont": {"color": TOKENS["text-muted"]},
            },
            "hovertemplate": "offset %{y}<br>width %{x}<br>%{z}<extra></extra>",
            "zmin": 0.0,
        }
    ]

    layout: dict[str, Any] = {
        "title": {"text": title},
        "xaxis": {"title": "width (tick-spacings)", "type": "category"},
        "yaxis": {"title": "center offset (tick-spacings)", "type": "category"},
        "margin": {"l": 64, "r": 24, "t": 48, "b": 48},
    }

    hold = finite(counts.get((0, 0)))
    if hold is not None and (0 not in wids or 0 not in offs):
        layout["annotations"] = [
            {
                "text": f"hold: {_format(hold)}",
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": 1.02,
                "xanchor": "left",
                "yanchor": "bottom",
                "showarrow": False,
                "font": {"color": TOKENS["hold"], "size": 12},
            }
        ]

    return build_spec(data, layout)
