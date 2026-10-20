"""Plotly dark theme + shared figure-spec helpers (U04, ``CONTRACTS.md`` §6/§9).

The design tokens in :data:`TOKENS` are the **single Python mirror** of the
``CONTRACTS.md`` §9 table (the CSS variables in ``web/src/theme/tokens.css`` are
the frontend mirror of the same source of truth).  :data:`DARK_TEMPLATE` is a
Plotly template dict built from them and registered as ``"undertow_dark"`` at
import time; :func:`apply_theme` applies it to a figure spec.

Builders return **plain, JSON-serialisable dicts** (``CONTRACTS.md`` §6).  This
module is therefore the only place in ``undertow.ui`` that imports ``plotly``:
the builders hand raw trace/layout dicts to :func:`build_spec`, which applies the
theme and materialises the final :class:`FigureSpec`.  Keeping the Plotly
dependency here means the rest of the figure package is pure data.
"""

from __future__ import annotations

import math
from typing import Any, TypedDict, cast

# Plotly ships no ``py.typed`` marker, so mypy cannot see its inline types.  The
# missing stubs are dependency packaging, not an interface gap: we confine the
# import to this one module and ignore the *import* only, never the calls.
import plotly.graph_objects as go  # type: ignore[import-untyped]
import plotly.io as pio  # type: ignore[import-untyped]

__all__ = [
    "COLORWAY",
    "DARK_TEMPLATE",
    "DEFAULT_CONFIG",
    "FONT_FAMILY",
    "REGIME_TOKENS",
    "TEMPLATE_NAME",
    "TOKENS",
    "FigureSpec",
    "apply_theme",
    "build_spec",
    "empty_figure",
    "finite",
    "rgba",
]


class FigureSpec(TypedDict):
    """JSON-serialisable Plotly figure (``CONTRACTS.md`` §6)."""

    data: list[dict[str, Any]]
    layout: dict[str, Any]
    config: dict[str, Any]


#: Design tokens — mirrors ``CONTRACTS.md`` §9 field-for-field.
TOKENS: dict[str, str] = {
    "bg-base": "#0B0E14",
    "bg-surface": "#11161F",
    "bg-elevated": "#161B26",
    "border": "#232A38",
    "text-primary": "#E6EAF2",
    "text-muted": "#8B93A7",
    "grid": "#1A2130",
    "accent": "#22D3EE",
    "accent-2": "#A78BFA",
    "bull": "#34D399",
    "bear": "#F87171",
    "sideways": "#94A3B8",
    "high-vol": "#FBBF24",
    "hold": "#64748B",
    "rebalance": "#38BDF8",
}

#: Plotly template name registered at import (``CONTRACTS.md`` §6).
TEMPLATE_NAME = "undertow_dark"

#: Plotly font family from ``CONTRACTS.md`` §9 typography.
FONT_FAMILY = "Inter, system-ui, sans-serif"

#: Series palette drawn from the tokens (agent first, pinned policies second).
COLORWAY: tuple[str, ...] = (
    TOKENS["accent"],
    TOKENS["accent-2"],
    TOKENS["bull"],
    TOKENS["bear"],
    TOKENS["sideways"],
    TOKENS["high-vol"],
    TOKENS["rebalance"],
    TOKENS["hold"],
)

#: Default ``config`` block returned with every figure spec (``CONTRACTS.md`` §6).
DEFAULT_CONFIG: dict[str, Any] = {
    "displayModeBar": True,
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
}

#: Regime label -> token name used for background bands and legends.
REGIME_TOKENS: dict[str, str] = {
    "bull": "bull",
    "bear": "bear",
    "sideways": "sideways",
    "high_vol": "high-vol",
    "high-vol": "high-vol",
    "unknown": "text-muted",
}


def _axis_template() -> dict[str, object]:
    """Shared x/y axis styling from the §9 ``grid``/``border``/``text-muted`` tokens."""
    return {
        "gridcolor": TOKENS["grid"],
        "zerolinecolor": TOKENS["border"],
        "linecolor": TOKENS["border"],
        "tickfont": {"color": TOKENS["text-muted"]},
        "title": {"font": {"color": TOKENS["text-muted"]}},
    }


#: Plotly template dict built from the §9 tokens (``CONTRACTS.md`` §6).
DARK_TEMPLATE: dict[str, object] = {
    "layout": {
        "paper_bgcolor": TOKENS["bg-surface"],
        "plot_bgcolor": TOKENS["bg-surface"],
        "font": {
            "family": FONT_FAMILY,
            "size": 12,
            "color": TOKENS["text-primary"],
        },
        "colorway": list(COLORWAY),
        "title": {"font": {"color": TOKENS["text-primary"], "size": 15}},
        "xaxis": _axis_template(),
        "yaxis": _axis_template(),
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": TOKENS["border"],
            "borderwidth": 1,
            "font": {"color": TOKENS["text-primary"], "size": 11},
        },
        "hoverlabel": {
            "bgcolor": TOKENS["bg-elevated"],
            "bordercolor": TOKENS["border"],
            "font": {"color": TOKENS["text-primary"], "family": FONT_FAMILY, "size": 11},
        },
        "margin": {"l": 56, "r": 56, "t": 48, "b": 44},
        "modebar": {
            "bgcolor": "rgba(0,0,0,0)",
            "color": TOKENS["text-muted"],
            "activecolor": TOKENS["accent"],
        },
    }
}

# Register the template at import so server-side rendering (and ``pio.to_json``)
# can resolve the ``"undertow_dark"`` reference that every spec carries.
pio.templates[TEMPLATE_NAME] = go.layout.Template(DARK_TEMPLATE)


def finite(value: object) -> float | None:
    """Return ``value`` as a finite float, or ``None`` for null/non-finite/non-numeric.

    ``bool`` is rejected even though it is an ``int`` subclass so that a stray
    boolean never becomes a data point (mirrors ``undertow.ui.registry._finite``).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def rgba(color: str, alpha: float) -> str:
    """Expand a ``#RRGGBB`` token into an ``rgba(r,g,b,a)`` string for translucent fills."""
    hex_color = color.lstrip("#")
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return f"rgba({red},{green},{blue},{alpha})"


def apply_theme(fig: dict[str, Any]) -> dict[str, Any]:
    """Apply the dark theme to a ``{"data": ..., "layout": ...}`` dict in place.

    Sets ``layout.template = "undertow_dark"`` (``CONTRACTS.md`` §6) and also
    inlines the concrete colours so the JSON the browser receives is dark even
    though it never registered the named template.  Responsive sizing is enabled
    via ``layout.autosize`` and the returned ``config``.
    """
    layout = fig.setdefault("layout", {})
    if not isinstance(layout, dict):  # pragma: no cover - defensive
        layout = {}
        fig["layout"] = layout

    layout["template"] = TEMPLATE_NAME
    layout["autosize"] = True
    layout["paper_bgcolor"] = TOKENS["bg-surface"]
    layout["plot_bgcolor"] = TOKENS["bg-surface"]
    layout.setdefault("colorway", list(COLORWAY))
    layout.setdefault("hovermode", "x unified")
    layout.setdefault("margin", {"l": 56, "r": 56, "t": 48, "b": 44})

    font = layout.setdefault("font", {})
    if isinstance(font, dict):
        font.setdefault("family", FONT_FAMILY)
        font.setdefault("size", 12)
        font.setdefault("color", TOKENS["text-primary"])

    title = layout.setdefault("title", {})
    if isinstance(title, dict):
        title.setdefault("font", {"color": TOKENS["text-primary"], "size": 15})

    for key in ("xaxis", "yaxis", "yaxis2", "yaxis3", "yaxis4"):
        if key not in layout and key not in ("xaxis", "yaxis"):
            continue
        axis = layout.setdefault(key, {})
        if not isinstance(axis, dict):  # pragma: no cover - defensive
            continue
        axis.setdefault("gridcolor", TOKENS["grid"])
        axis.setdefault("zerolinecolor", TOKENS["border"])
        axis.setdefault("linecolor", TOKENS["border"])
        tickfont = axis.setdefault("tickfont", {})
        if isinstance(tickfont, dict):
            tickfont.setdefault("color", TOKENS["text-muted"])
        axis_title = axis.setdefault("title", {})
        if isinstance(axis_title, dict):
            axis_title.setdefault("font", {"color": TOKENS["text-muted"]})

    legend = layout.setdefault("legend", {})
    if isinstance(legend, dict):
        legend.setdefault("bgcolor", "rgba(0,0,0,0)")
        legend.setdefault("bordercolor", TOKENS["border"])
        legend_font = legend.setdefault("font", {})
        if isinstance(legend_font, dict):
            legend_font.setdefault("color", TOKENS["text-primary"])

    config = dict(DEFAULT_CONFIG)
    existing = fig.get("config")
    if isinstance(existing, dict):
        config.update(existing)
    fig["config"] = config
    return fig


def build_spec(
    data: list[dict[str, Any]],
    layout: dict[str, Any] | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> FigureSpec:
    """Assemble a themed :class:`FigureSpec` from raw Plotly trace/layout dicts.

    ``None`` values survive as JSON ``null`` (Plotly renders them as gaps); the
    builders are responsible for never putting ``NaN``/``Infinity`` in a trace.
    """
    spec: dict[str, Any] = {
        "data": list(data),
        "layout": dict(layout) if layout else {},
    }
    if config is not None:
        spec["config"] = dict(config)
    apply_theme(spec)
    return cast(FigureSpec, spec)


def empty_figure(message: str = "No data") -> FigureSpec:
    """A themed, empty figure with a centred annotation (robustness contract)."""
    layout: dict[str, Any] = {
        "annotations": [
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "xanchor": "center",
                "yanchor": "middle",
                "showarrow": False,
                "font": {"color": TOKENS["text-muted"], "size": 14},
            }
        ],
        "xaxis": {"visible": False},
        "yaxis": {"visible": False},
    }
    return build_spec([], layout)
