"""U04 — figure builders + Plotly dark theme (``CONTRACTS.md`` §6/§9).

Structure/behaviour tests, not pixel tests: every builder must return a
JSON-serialisable :class:`FigureSpec` with the dark theme applied and no
``NaN``/``Infinity``; each chart asserts its specific shape (band fill, null
skipping, heatmap grid/text, ribbon traces, compare names) and the token table is
checked against the frozen contract.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import plotly.io as pio
import pytest

from undertow.ui.figures import (
    FIGURE_REGISTRY,
    TOKENS,
    action_heatmap_figure,
    compare_equity_figure,
    diagnostics_figure,
    leaderboard_figure,
    learning_curve_figure,
    ribbon_figure,
)
from undertow.ui.figures.actions import action_heatmap_figure as actions_builder
from undertow.ui.figures.theme import (
    DARK_TEMPLATE,
    TEMPLATE_NAME,
    apply_theme,
    build_spec,
    empty_figure,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONTRACTS = _REPO_ROOT / "docs" / "plans" / "ui" / "CONTRACTS.md"

_CURVE: dict[str, Any] = {
    "run_id": "ppo_s0_abc",
    "timesteps": [0, 100, 200],
    "mean": [0.0, 0.5, 1.0],
    "std": [0.1, 0.2, 0.3],
    "per_seed": {"0": [0.0, 0.4, 0.9], "1": [0.0, 0.6, 1.1]},
    "downsampled": False,
}

_POINTS: list[dict[str, Any]] = [
    {"timestep": 0, "train": {"policy_loss": 0.5, "approx_kl": None}},
    {"timestep": 100, "train": {"policy_loss": 0.3, "approx_kl": 0.01}},
    {"timestep": 200, "train": {"policy_loss": None, "approx_kl": 0.02}},
    {"timestep": 300, "train": {"policy_loss": float("nan"), "approx_kl": 0.03}},
    {"timestep": 400, "train": {}},
]

# (offset, width) -> count; (0, 0) is the separate `hold` action.
_COUNTS: dict[tuple[int, int], float] = {(0, 2): 3.0, (2, 4): 5.0, (0, 0): 7.0}
_OFFSETS = [0, 2]
_WIDTHS = [2, 4]

_RIBBON: list[dict[str, Any]] = [
    {
        "run_id": "ppo_s0",
        "step": 0,
        "price": 3000.0,
        "tick_lower": 199_000,
        "tick_upper": 201_000,
        "in_range": True,
        "action_type": "hold",
        "gas": 0.0,
        "equity": 10_000.0,
        "regime": "bull",
    },
    {
        "run_id": "ppo_s0",
        "step": 1,
        "price": 3010.0,
        "tick_lower": 199_000,
        "tick_upper": 201_000,
        "in_range": True,
        "action_type": "rebalance",
        "gas": -0.5,
        "equity": 10_010.0,
        "regime": "bull",
    },
    {
        "run_id": "ppo_s0",
        "step": 2,
        "price": 2980.0,
        "tick_lower": 199_500,
        "tick_upper": 201_500,
        "in_range": False,
        "action_type": "hold",
        "gas": 0.0,
        "equity": 9_990.0,
        "regime": "bear",
    },
    {
        "run_id": "ppo_s0",
        "step": 3,
        "price": 2970.0,
        "tick_lower": 199_500,
        "tick_upper": 201_500,
        "in_range": False,
        "action_type": "rebalance",
        "gas": -0.2,
        "equity": 9_980.0,
        "regime": "bear",
    },
]

_RECORDS: list[dict[str, Any]] = [
    {"run_id": "a", "state": "done", "eval": {"sharpe": 1.5}},
    {"run_id": "b", "state": "running", "eval": {"sharpe": 2.0}},
    {"run_id": "c", "state": "failed", "eval": None, "best_reward": 0.5},
]

_SERIES: dict[str, list[float]] = {
    "policy-a": [1.0, 2.0, 3.0],
    "policy-b": [1.0, 1.5, float("nan")],
}


def _all_specs() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("learning", learning_curve_figure(_CURVE)),
        ("diagnostics", diagnostics_figure(_POINTS, "policy_loss")),
        ("actions", action_heatmap_figure(_COUNTS, offsets=_OFFSETS, widths=_WIDTHS)),
        ("ribbon", ribbon_figure(_RIBBON)),
        ("leaderboard", leaderboard_figure(_RECORDS, metric="sharpe")),
        ("compare", compare_equity_figure(_SERIES)),
    ]


# 1. ---------------------------------------------------------------- structure
@pytest.mark.parametrize(("name", "spec"), _all_specs())
def test_every_builder_returns_themed_json_spec(name: str, spec: dict[str, Any]) -> None:
    assert set(spec) == {"data", "layout", "config"}, name
    assert isinstance(spec["data"], list)
    assert spec["layout"]["template"] == TEMPLATE_NAME
    assert spec["layout"]["paper_bgcolor"] == TOKENS["bg-surface"]
    assert spec["config"]["responsive"] is True
    # No NaN/Infinity literals anywhere in the payload.
    text = pio.to_json(spec)
    assert "NaN" not in text and "Infinity" not in text
    json.dumps(spec, allow_nan=False)


def test_figure_registry_matches_contract_names() -> None:
    assert set(FIGURE_REGISTRY) == {"learning", "diagnostics", "actions", "ribbon", "leaderboard"}
    assert all(callable(builder) for builder in FIGURE_REGISTRY.values())


def test_leaderboard_prefers_eval_then_falls_back_to_top_level() -> None:
    spec = leaderboard_figure(_RECORDS, metric="best_reward")
    bar = spec["data"][0]
    assert bar["y"] == ["c"]
    assert bar["x"] == [0.5]


def test_leaderboard_skips_records_without_the_metric() -> None:
    spec = leaderboard_figure(["junk", {"run_id": "x"}], metric="sharpe")  # type: ignore[list-item]
    assert spec["data"] == []
    assert spec["layout"]["annotations"][0]["text"] == "No data"


def test_apply_theme_registers_named_template() -> None:
    assert TEMPLATE_NAME == "undertow_dark"
    assert "layout" in DARK_TEMPLATE
    spec = apply_theme({"data": [], "layout": {}})
    assert spec["layout"]["template"] == TEMPLATE_NAME
    assert spec["layout"]["autosize"] is True


def test_empty_figure_has_centred_no_data_annotation() -> None:
    spec = empty_figure()
    annotation = spec["layout"]["annotations"][0]
    assert annotation["text"] == "No data"
    assert annotation["x"] == 0.5 and annotation["y"] == 0.5
    assert annotation["showarrow"] is False


def test_build_spec_merges_config_override() -> None:
    spec = build_spec([], {}, config={"displaylogo": True})
    assert spec["config"]["displaylogo"] is True
    assert spec["config"]["responsive"] is True


# 2. ------------------------------------------------------- learning mean/band
def test_learning_curve_emits_mean_and_toself_band() -> None:
    spec = learning_curve_figure(_CURVE)
    names = [trace.get("name") for trace in spec["data"]]
    assert "mean" in names
    band = next(trace for trace in spec["data"] if trace.get("fill") == "toself")
    assert band["name"] == "\u00b11 std"
    assert band["fill"] == "toself"


def test_learning_curve_per_seed_lines_present() -> None:
    spec = learning_curve_figure(_CURVE)
    names = {trace.get("name") for trace in spec["data"]}
    assert {"seed 0", "seed 1"} <= names


def test_learning_curve_log_y_sets_log_axis() -> None:
    spec = learning_curve_figure(_CURVE, log_y=True)
    assert spec["layout"]["yaxis"]["type"] == "log"


def test_learning_curve_empty_is_no_data() -> None:
    spec = learning_curve_figure({"timesteps": [], "mean": [], "std": []})
    assert spec["data"] == []
    assert spec["layout"]["annotations"][0]["text"] == "No data"


def test_learning_curve_non_sequence_input_is_no_data() -> None:
    spec = learning_curve_figure({"timesteps": "oops", "mean": 1.0})
    assert spec["data"] == []
    assert spec["layout"]["annotations"][0]["text"] == "No data"


def test_learning_curve_drops_non_finite_points() -> None:
    curve = {"timesteps": [0, 1, 2], "mean": [1.0, float("nan"), 3.0]}
    spec = learning_curve_figure(curve)
    mean_trace = next(trace for trace in spec["data"] if trace.get("name") == "mean")
    assert mean_trace["x"] == [0.0, 2.0]
    assert mean_trace["y"] == [1.0, 3.0]


def test_learning_curve_missing_std_defaults_to_zero() -> None:
    spec = learning_curve_figure({"timesteps": [0, 1], "mean": [1.0, 2.0], "std": [0.5]})
    band = next(trace for trace in spec["data"] if trace.get("fill") == "toself")
    assert band["y"] == [1.5, 2.0, 2.0, 0.5]


def test_learning_curve_uses_explicit_checkpoints() -> None:
    curve = {**_CURVE, "checkpoints": [{"timestep": 100, "value": 0.42}]}
    spec = learning_curve_figure(curve)
    checkpoint = next(trace for trace in spec["data"] if trace.get("name") == "checkpoint")
    assert checkpoint["x"] == [100.0]
    assert checkpoint["y"] == [0.42]


def test_learning_curve_skips_empty_seed_lines() -> None:
    curve = {**_CURVE, "per_seed": {"9": [None, None, None]}}
    spec = learning_curve_figure(curve)
    assert "seed 9" not in {trace.get("name") for trace in spec["data"]}


def test_learning_curve_seeds_stay_aligned_when_mean_has_gap() -> None:
    # A dropped non-finite mean must not shift later seed values onto an
    # earlier timestep: each seed is paired with its own original timestep.
    curve = {
        "timesteps": [0, 1, 2],
        "mean": [1.0, float("nan"), 3.0],
        "std": [0.1, 0.1, 0.1],
        "per_seed": {"0": [10.0, 20.0, 30.0], "1": [11.0, None, 31.0]},
    }
    spec = learning_curve_figure(curve)
    mean_trace = next(trace for trace in spec["data"] if trace.get("name") == "mean")
    seed_0 = next(trace for trace in spec["data"] if trace.get("name") == "seed 0")
    seed_1 = next(trace for trace in spec["data"] if trace.get("name") == "seed 1")
    # Mean still drops the gap; seeds keep the full original axis.
    assert mean_trace["x"] == [0.0, 2.0]
    assert seed_0["x"] == [0.0, 1.0, 2.0]
    assert seed_0["y"] == [10.0, 20.0, 30.0]  # step-2 value plotted at timestep 2
    # A non-finite seed value is a gap at its real timestep, not a shift.
    assert seed_1["x"] == [0.0, 1.0, 2.0]
    assert seed_1["y"] == [11.0, None, 31.0]


def test_learning_curve_sorts_series_by_timestep() -> None:
    curve = {
        "timesteps": [200, 0, 100],
        "mean": [1.0, 0.0, 0.5],
        "std": [0.3, 0.1, 0.2],
        "per_seed": {"0": [1.1, 0.0, 0.4]},
    }
    spec = learning_curve_figure(curve)
    mean_trace = next(trace for trace in spec["data"] if trace.get("name") == "mean")
    band = next(trace for trace in spec["data"] if trace.get("fill") == "toself")
    seed = next(trace for trace in spec["data"] if trace.get("name") == "seed 0")
    assert mean_trace["x"] == [0.0, 100.0, 200.0]
    assert mean_trace["y"] == [0.0, 0.5, 1.0]
    assert band["x"] == [0.0, 100.0, 200.0, 200.0, 100.0, 0.0]
    assert seed["x"] == [0.0, 100.0, 200.0]
    assert seed["y"] == [0.0, 0.4, 1.1]


def test_learning_curve_seed_skips_non_finite_timestep() -> None:
    curve = {
        "timesteps": [0, float("nan"), 2],
        "mean": [1.0, 2.0, 3.0],
        "std": [0.1, 0.1, 0.1],
        "per_seed": {"0": [10.0, 20.0, 30.0]},
    }
    spec = learning_curve_figure(curve)
    seed = next(trace for trace in spec["data"] if trace.get("name") == "seed 0")
    assert seed["x"] == [0.0, 2.0]
    assert seed["y"] == [10.0, 30.0]


def test_learning_curve_ignores_malformed_checkpoints() -> None:
    curve = {
        **_CURVE,
        "checkpoints": ["not-a-mapping", {"timestep": 0, "value": None}],
    }
    spec = learning_curve_figure(curve)
    checkpoint = next(trace for trace in spec["data"] if trace.get("name") == "checkpoint")
    # Falls back to the best-mean point (mean=1.0 at timestep 200).
    assert checkpoint["x"] == [200.0]
    assert checkpoint["y"] == [1.0]


# 3. ------------------------------------------------------------- band ordering
def test_learning_band_upper_is_never_below_lower() -> None:
    spec = learning_curve_figure(_CURVE)
    band = next(trace for trace in spec["data"] if trace.get("fill") == "toself")
    values = band["y"]
    half = len(values) // 2
    upper = values[:half]
    lower = list(reversed(values[half:]))
    assert upper == [0.1, 0.7, 1.3]
    assert lower == [-0.1, 0.3, 0.7]
    assert all(high >= low for high, low in zip(upper, lower, strict=True))  # noqa: B905


# 4. ------------------------------------------------------------- diagnostics
def test_diagnostics_skips_null_and_non_finite_points() -> None:
    policy = diagnostics_figure(_POINTS, "policy_loss")
    trace = next(iter(policy["data"]))
    assert trace["y"] == [0.5, 0.3]
    approx = diagnostics_figure(_POINTS, "approx_kl")
    approx_trace = next(iter(approx["data"]))
    assert approx_trace["y"] == [0.01, 0.02, 0.03]


def test_diagnostics_unknown_metric_raises_key_error() -> None:
    with pytest.raises(KeyError):
        diagnostics_figure(_POINTS, "not_a_metric")


def test_diagnostics_all_null_is_no_data() -> None:
    spec = diagnostics_figure(_POINTS, "learning_rate")
    assert spec["data"] == []
    assert spec["layout"]["annotations"][0]["text"] == "No data"


def test_diagnostics_skips_malformed_points() -> None:
    points: list[Any] = ["not-a-point", {"timestep": 0, "train": "not-a-mapping"}, {}]
    spec = diagnostics_figure(points, "policy_loss")
    assert spec["data"] == []
    assert spec["layout"]["annotations"][0]["text"] == "No data"


# 5. ------------------------------------------------------------- action heatmap
def test_action_heatmap_z_shape_and_cell_text() -> None:
    spec = action_heatmap_figure(_COUNTS, offsets=_OFFSETS, widths=_WIDTHS)
    heatmap = spec["data"][0]
    assert len(heatmap["z"]) == len(_OFFSETS)
    assert len(heatmap["z"][0]) == len(_WIDTHS)
    assert heatmap["z"] == [[3.0, 0.0], [0.0, 5.0]]
    assert heatmap["text"] == [["3", "0"], ["0", "5"]]


def test_action_heatmap_formats_fractional_counts() -> None:
    counts = {(0, 2): 1.25, (2, 4): 2.0}
    spec = action_heatmap_figure(counts, offsets=_OFFSETS, widths=_WIDTHS)
    assert spec["data"][0]["text"] == [["1.25", "0"], ["0", "2"]]


def test_action_heatmap_surfaces_hold_annotation() -> None:
    spec = action_heatmap_figure(_COUNTS, offsets=_OFFSETS, widths=_WIDTHS)
    annotations = spec["layout"].get("annotations", [])
    assert any("hold" in annotation["text"] for annotation in annotations)


def test_action_heatmap_empty_grid_is_no_data() -> None:
    spec = actions_builder({}, offsets=[], widths=[])
    assert spec["data"] == []
    assert spec["layout"]["annotations"][0]["text"] == "No data"


# 6. ------------------------------------------------------------------- ribbon
def test_ribbon_band_traces_carry_raw_ticks() -> None:
    spec = ribbon_figure(_RIBBON)
    lower = [
        value
        for trace in spec["data"]
        if trace.get("name") == "position band lower"
        for value in trace["y"]
    ]
    upper = [
        value
        for trace in spec["data"]
        if str(trace.get("name", "")).startswith("position band (")
        for value in trace["y"]
    ]
    assert lower == [row["tick_lower"] for row in _RIBBON]
    assert upper == [row["tick_upper"] for row in _RIBBON]


def test_ribbon_rebalance_marker_count_matches_fixture() -> None:
    spec = ribbon_figure(_RIBBON)
    marker = next(trace for trace in spec["data"] if trace.get("name") == "rebalance")
    expected = sum(1 for row in _RIBBON if row["action_type"] == "rebalance")
    assert expected == 2
    assert len(marker["x"]) == expected
    assert len(marker["marker"]["size"]) == expected


def test_ribbon_price_uses_log_axis_and_hodl_toggle() -> None:
    spec = ribbon_figure(_RIBBON)
    assert spec["layout"]["yaxis"]["type"] == "log"
    assert spec["layout"]["yaxis2"]["overlaying"] == "y"
    names = {trace.get("name") for trace in spec["data"]}
    assert "price" in names and "HODL" in names
    without = ribbon_figure(_RIBBON, show_hodl=False)
    assert "HODL" not in {trace.get("name") for trace in without["data"]}


def test_ribbon_regime_background_shapes() -> None:
    spec = ribbon_figure(_RIBBON)
    shapes = spec["layout"]["shapes"]
    assert len(shapes) == 2
    assert all(shape["type"] == "rect" and shape["yref"] == "paper" for shape in shapes)


@pytest.mark.parametrize("rows", [[], [_RIBBON[0]]])
def test_ribbon_one_row_or_empty_is_no_data(rows: list[dict[str, Any]]) -> None:
    spec = ribbon_figure(rows)
    assert spec["data"] == []
    assert spec["layout"]["annotations"][0]["text"] == "No data"


def test_ribbon_skips_rows_without_finite_price() -> None:
    rows = [{"step": 0, "price": None}, "not-a-mapping", dict(_RIBBON[0]), dict(_RIBBON[1])]
    spec = ribbon_figure(rows)  # type: ignore[arg-type]
    assert spec["data"]
    assert spec.get("layout", {}).get("annotations") is None


def test_ribbon_rebalance_fallback_without_action_type() -> None:
    rows = [dict(row) for row in _RIBBON]
    for row in rows:
        row.pop("action_type")
    rows[1]["center_offset"] = 2
    spec = ribbon_figure(rows)
    marker = next(trace for trace in spec["data"] if trace.get("name") == "rebalance")
    assert len(marker["x"]) == 1


def test_ribbon_missing_band_values_still_renders() -> None:
    rows = [
        {**_RIBBON[0], "tick_lower": None, "tick_upper": None},
        {**_RIBBON[1], "tick_lower": None, "tick_upper": None},
    ]
    spec = ribbon_figure(rows)
    assert spec["data"]
    assert not any(trace.get("name") == "position band lower" for trace in spec["data"])


def test_ribbon_uses_explicit_hodl_values() -> None:
    rows = [{**row, "hodl": 100.0 + index} for index, row in enumerate(_RIBBON)]
    spec = ribbon_figure(rows)
    hodl = next(trace for trace in spec["data"] if trace.get("name") == "HODL")
    assert hodl["y"] == [100.0, 101.0, 102.0, 103.0]


def test_ribbon_omits_hodl_without_equity() -> None:
    rows = [{**row, "equity": None} for row in _RIBBON]
    spec = ribbon_figure(rows)
    assert "HODL" not in {trace.get("name") for trace in spec["data"]}


# 7. -------------------------------------------------------------------- compare
def test_compare_trace_names_equal_mapping_keys() -> None:
    spec = compare_equity_figure(_SERIES)
    names = [trace["name"] for trace in spec["data"]]
    assert names == list(_SERIES)


def test_compare_non_finite_becomes_null_gap() -> None:
    spec = compare_equity_figure(_SERIES)
    policy_b = next(trace for trace in spec["data"] if trace["name"] == "policy-b")
    assert policy_b["y"][-1] is None


def test_compare_non_sequence_values_are_empty() -> None:
    spec = compare_equity_figure({"policy-a": None})  # type: ignore[dict-item]
    assert spec["data"][0]["x"] == []
    assert spec["data"][0]["y"] == []


def test_compare_empty_is_no_data() -> None:
    spec = compare_equity_figure({})
    assert spec["data"] == []
    assert spec["layout"]["annotations"][0]["text"] == "No data"


# 8. --------------------------------------------------------------------- tokens
def _contract_tokens() -> dict[str, str]:
    text = _CONTRACTS.read_text(encoding="utf-8")
    section = text.split("## 9. Design tokens", 1)[1]
    tokens: dict[str, str] = {}
    for line in section.splitlines():
        match = re.match(r"\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|", line)
        if match:
            tokens[match.group(1)] = match.group(2)
    return tokens


def test_tokens_match_contract_section_9() -> None:
    expected = _contract_tokens()
    assert len(expected) == 15
    assert expected == TOKENS
