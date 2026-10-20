"""Multi-policy equity overlay builder (U04, ``CONTRACTS.md`` §6).

``compare_equity_figure`` draws one line per policy from an ordered
``{policy_name: equity_values}`` mapping, with a shared unified hover and legend
toggles.  Non-finite values become ``null`` gaps (never ``NaN``) so a clipped
series breaks its line instead of corrupting the payload.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from undertow.ui.figures.theme import (
    COLORWAY,
    FigureSpec,
    build_spec,
    empty_figure,
    finite,
)

__all__ = ["compare_equity_figure"]


def _series(value: object) -> list[float | None]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [finite(item) for item in value]


def compare_equity_figure(series: Mapping[str, Sequence[float]]) -> FigureSpec:
    """Overlay one equity line per policy; trace names equal the mapping keys.

    An empty mapping yields the themed "No data" figure.  The first policy uses
    the ``accent`` token, the rest cycle through the shared :data:`COLORWAY`.
    """
    if not series:
        return empty_figure()

    data: list[dict[str, Any]] = []
    for index, (name, values) in enumerate(series.items()):
        numeric = _series(values)
        data.append(
            {
                "type": "scatter",
                "x": list(range(len(numeric))),
                "y": numeric,
                "mode": "lines",
                "name": str(name),
                "line": {"width": 2, "color": COLORWAY[index % len(COLORWAY)]},
                "connectgaps": False,
                "hovertemplate": f"{name}<br>step %{{x}}<br>%{{y:.2f}}<extra></extra>",
            }
        )

    layout: dict[str, Any] = {
        "hovermode": "x unified",
        "xaxis": {"title": "step"},
        "yaxis": {"title": "equity (USDC)"},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    }
    return build_spec(data, layout)
