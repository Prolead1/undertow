"""PPO diagnostics small-multiple builder (U04, ``CONTRACTS.md`` §4/§6).

``diagnostics_figure`` renders one metric at a time from a sequence of
``TelemetryPoint``-shaped mappings (plain dicts); U08 composes a grid of them.
The metric is validated against the SB3 ``train/`` keys the wire schema carries;
an unknown metric raises :class:`KeyError` rather than silently plotting nothing.
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

__all__ = ["DIAGNOSTIC_METRICS", "diagnostics_figure"]

#: Metrics readable from ``point["train"]`` (``CONTRACTS.md`` §1.1/§4).
DIAGNOSTIC_METRICS: tuple[str, ...] = (
    "policy_loss",
    "value_loss",
    "entropy_loss",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
    "learning_rate",
)


def diagnostics_figure(points: Sequence[Mapping[str, object]], metric: str) -> FigureSpec:
    """Line of ``point["train"][metric]`` versus ``point["timestep"]``.

    Null and non-finite metric values are skipped; a series with no finite point
    yields the themed "No data" figure.  An unknown ``metric`` raises
    :class:`KeyError`.
    """
    if metric not in DIAGNOSTIC_METRICS:
        raise KeyError(metric)

    samples: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, Mapping):
            continue
        timestep = finite(point.get("timestep"))
        train = point.get("train")
        if timestep is None or not isinstance(train, Mapping):
            continue
        value = finite(train.get(metric))
        if value is None:
            continue
        samples.append((timestep, value))

    if not samples:
        return empty_figure()

    samples.sort(key=lambda sample: sample[0])
    xs = [sample[0] for sample in samples]
    ys = [sample[1] for sample in samples]
    label = metric.replace("_", " ")

    data: list[dict[str, Any]] = [
        {
            "type": "scatter",
            "x": xs,
            "y": ys,
            "mode": "lines",
            "name": label,
            "line": {"width": 2, "color": TOKENS["accent"]},
            "hovertemplate": f"step %{{x}}<br>{label} %{{y:.4g}}<extra></extra>",
        }
    ]
    layout: dict[str, Any] = {
        "hovermode": "x unified",
        "xaxis": {"title": "timestep"},
        "yaxis": {"title": label},
        "showlegend": False,
        "margin": {"l": 56, "r": 24, "t": 24, "b": 40},
    }
    return build_spec(data, layout)
