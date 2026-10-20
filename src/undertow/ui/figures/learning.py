"""Learning-curve figure builder (U04, ``CONTRACTS.md`` §4/§6).

Handles a :class:`~undertow.ui.schemas.LearningCurve`-shaped mapping (plain
dict) and renders the mean line with a ``fill="toself"`` standard-deviation
band, the per-seed thin lines, and checkpoint markers.  It never imports the
training stack — the input is the wire shape only.
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

__all__ = ["learning_curve_figure"]


def _sequence(value: object) -> list[object]:
    """Return ``value`` as a list for a JSON-like sequence, else ``[]``."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _checkpoints(
    curve: Mapping[str, object],
    xs: list[float],
    means: list[float],
) -> tuple[list[float], list[float]]:
    """Explicit ``curve["checkpoints"]`` markers, else the best-mean point."""
    raw = curve.get("checkpoints")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        cx: list[float] = []
        cy: list[float] = []
        for entry in raw:
            if not isinstance(entry, Mapping):
                continue
            x_value = finite(entry.get("timestep"))
            y_value = finite(entry.get("value", entry.get("mean")))
            if x_value is None or y_value is None:
                continue
            cx.append(x_value)
            cy.append(y_value)
        if cx:
            return cx, cy

    best = max(range(len(means)), key=lambda index: means[index])
    return [xs[best]], [means[best]]


def learning_curve_figure(curve: Mapping[str, object], *, log_y: bool = False) -> FigureSpec:
    """Build the learning-curve figure (mean + std band + seed lines + checkpoints).

    Non-finite timesteps/means are dropped (never emitted as ``NaN``); a curve
    with no finite point yields the themed "No data" figure.  ``log_y`` switches
    the value axis to a log scale.
    """
    timesteps = _sequence(curve.get("timesteps"))
    mean = _sequence(curve.get("mean"))
    std = _sequence(curve.get("std"))

    points: list[tuple[float, float, float]] = []
    for index in range(min(len(timesteps), len(mean))):
        x_value = finite(timesteps[index])
        mean_value = finite(mean[index])
        if x_value is None or mean_value is None:
            continue
        std_value = finite(std[index]) if index < len(std) else 0.0
        points.append((x_value, mean_value, std_value if std_value is not None else 0.0))

    if not points:
        return empty_figure()

    # CONTRACTS §0: series are ordered by timestep ascending.
    points.sort(key=lambda point: point[0])
    xs = [point[0] for point in points]
    means = [point[1] for point in points]
    stds = [point[2] for point in points]

    data: list[dict[str, Any]] = []

    upper = [means[i] + stds[i] for i in range(len(xs))]
    lower = [means[i] - stds[i] for i in range(len(xs))]
    data.append(
        {
            "type": "scatter",
            "x": xs + list(reversed(xs)),
            "y": upper + list(reversed(lower)),
            "mode": "lines",
            "name": "\u00b11 std",
            "legendgroup": "mean",
            "fill": "toself",
            "fillcolor": rgba(TOKENS["accent"], 0.16),
            "line": {"width": 0, "color": rgba(TOKENS["accent"], 0.0)},
            "hoverinfo": "skip",
            "showlegend": True,
        }
    )

    per_seed = curve.get("per_seed")
    if isinstance(per_seed, Mapping):
        seeds: list[tuple[str, list[object]]] = [
            (str(seed), _sequence(values)) for seed, values in per_seed.items()
        ]
        for seed, values in sorted(seeds, key=lambda item: item[0]):
            # Align every seed to its ORIGINAL timestep, never to the mean's
            # filtered axis: dropping a non-finite mean must not shift later
            # seed values onto an earlier timestep.  A non-finite seed value
            # stays at its real timestep as a ``None`` gap.
            seed_points: list[tuple[float, float | None]] = []
            for index in range(min(len(timesteps), len(values))):
                x_value = finite(timesteps[index])
                if x_value is None:
                    continue
                seed_points.append((x_value, finite(values[index])))
            if not any(value is not None for _, value in seed_points):
                continue
            # CONTRACTS §0: series are ordered by timestep ascending (key-only
            # so a ``None`` gap at a duplicate timestep never gets compared).
            seed_points.sort(key=lambda point: point[0])
            data.append(
                {
                    "type": "scatter",
                    "x": [point[0] for point in seed_points],
                    "y": [point[1] for point in seed_points],
                    "mode": "lines",
                    "name": f"seed {seed}",
                    "legendgroup": "seeds",
                    "line": {"width": 1, "color": rgba(TOKENS["text-muted"], 0.55)},
                    "opacity": 0.7,
                    "connectgaps": False,
                    "hovertemplate": f"seed {seed}<br>%{{x}}<br>%{{y:.4g}}<extra></extra>",
                }
            )

    data.append(
        {
            "type": "scatter",
            "x": xs,
            "y": means,
            "mode": "lines",
            "name": "mean",
            "legendgroup": "mean",
            "line": {"width": 2.5, "color": TOKENS["accent"]},
            "hovertemplate": "step %{x}<br>mean %{y:.4g}<extra></extra>",
        }
    )

    checkpoint_x, checkpoint_y = _checkpoints(curve, xs, means)
    if checkpoint_x:
        data.append(
            {
                "type": "scatter",
                "x": checkpoint_x,
                "y": checkpoint_y,
                "mode": "markers",
                "name": "checkpoint",
                "marker": {
                    "symbol": "diamond",
                    "size": 9,
                    "color": TOKENS["accent-2"],
                    "line": {"color": TOKENS["bg-base"], "width": 1},
                },
                "hovertemplate": "checkpoint %{x}<br>%{y:.4g}<extra></extra>",
            }
        )

    run_id = curve.get("run_id")
    layout: dict[str, Any] = {
        "hovermode": "x unified",
        "xaxis": {"title": "timestep"},
        "yaxis": {"title": "return"},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    }
    if isinstance(run_id, str) and run_id:
        layout["title"] = {"text": run_id}
    if log_y:
        layout["yaxis"] = {"title": "return", "type": "log"}

    return build_spec(data, layout)
