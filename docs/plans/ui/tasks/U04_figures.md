# U04 — Figure builders + Plotly dark theme

**Wave 2 · size L · depends on: U02, U03 (contracts only) · blocks: U05**
**Branch:** `feature-ui-figures`

> Build against `CONTRACTS.md`, not against U02/U03 modules. Every builder takes **plain data**
> (dicts/lists), never a reader. Fixtures are yours; you do not wait for the loaders.

## Why this task exists

The dashboard's visuals are its product. Centralising them as pure, tested functions means the API
layer (U05) is a thin wrapper, the static/export path is free, and the dark theme is consistent. This
is also where the domain-specific hero charts live.

## Files you own

```
src/undertow/ui/figures/__init__.py        (new — re-export the builders + FIGURE_REGISTRY)
src/undertow/ui/figures/theme.py           (new — TOKENS, DARK_TEMPLATE, apply_theme)
src/undertow/ui/figures/learning.py        (new — learning_curve_figure)
src/undertow/ui/figures/diagnostics.py     (new — diagnostics_figure)
src/undertow/ui/figures/actions.py         (new — action_heatmap_figure)
src/undertow/ui/figures/ribbon.py          (new — ribbon_figure)
src/undertow/ui/figures/leaderboard.py     (new — leaderboard_figure)
src/undertow/ui/figures/compare.py         (new — compare_equity_figure)
tests/ui/test_figures.py                   (new)
```

## What to build

Read `CONTRACTS.md` §6 (figure contract) and §9 (tokens) first.

1. **`theme.py`.** `TOKENS: dict[str,str]` matching §9 exactly; `DARK_TEMPLATE` as a Plotly template
   dict (paper/plot bg, font family/size, gridcolor, zero line, legend style); `apply_theme(fig)`
   registers/applies `"undertow_dark"` and sets responsive sizing. Import `plotly.graph_objects` /
   `plotly.io` here only.

2. **Builders** — signatures frozen in §6, each returning `{"data": [...], "layout": {...},
   "config": {...}}`:
   - `learning_curve_figure(curve, log_y=False)` — mean line + std band (`fill="toself"`), per-seed
     thin lines toggleable via legend, checkpoint markers, hover unified.
   - `diagnostics_figure(points, metric)` — `metric` ∈ `policy_loss|value_loss|entropy_loss|approx_kl|
     clip_fraction|explained_variance|learning_rate`; reads `point["train"][metric]`, skips nulls;
     returns a single nicely-styled line. U08 renders a grid of these.
   - `action_heatmap_figure(counts, offsets, widths, title)` — x = width, y = center_offset, z =
     value; sequential colorscale built from `accent`; annotates cells with the value; includes the
     `hold` action as a separate labelled row/annotation when `counts` carries it.
   - `ribbon_figure(rows, show_hodl=True)` — **the hero chart**: `price` on a log y-axis; the position
     `[tick_lower, tick_upper]` as a filled band (two traces or a `fill="tonexty"` ribbon), recolored
     by `in_range`; rebalance steps marked with `gas`-sized markers; optional HODL counterfactual
     line; regime bands via `shapes`. Must survive a 1-row input.
   - `leaderboard_figure(records, metric)` — horizontal bar of the top N by `metric`, colored by
     state, with run ids as categories.
   - `compare_equity_figure(series)` — one line per policy, shared hover, legend toggles.
   - `FIGURE_REGISTRY: dict[str, Callable]` mapping the API names (`learning`, `diagnostics`,
     `actions`, `ribbon`, `leaderboard`) to builders.

3. **Robustness.** Every builder must handle empty input (return an empty figure with a centred
   "No data" annotation), null values, single points, and non-finite numbers (drop them, don't emit
   `NaN` into JSON). All figures apply the theme and set `name`.

## Tests (`tests/ui/test_figures.py`)

Assert structure and behavior, not pixels:
1. Each builder returns a dict with `data`/`layout`/`config`, `layout.template` themed, and is
   `json.dumps`-able (no NaN/Infinity literals — use `pio.to_json` and assert no `NaN`).
2. `learning_curve_figure` emits the mean trace + band trace and the band is `fill="toself"`.
3. Std band upper >= lower for every x on a fixture with known values.
4. `diagnostics_figure` skips null metric points and returns only finite values; unknown metric
   raises `KeyError`.
5. `action_heatmap_figure` z-matrix shape equals `(len(offsets), len(widths))` and cell text matches
   the input counts (hand-checked fixture).
6. `ribbon_figure` on a fixture: band lower/upper trace `y` values equal `tick_lower`/`tick_upper`;
   a rebalance row produces a marker trace whose count matches the fixture; 1-row and empty inputs
   do not raise and yield the "No data" annotation.
7. `compare_equity_figure` trace names equal the mapping keys.
8. `theme.TOKENS` contains every token in `CONTRACTS.md` §9 (parse the table or hard-code the
   expected key set).

## Definition of done

- `uv run pytest tests/ui/test_figures.py -q` green; full suite green at ≥90%.
- `ruff` + `mypy src/undertow` clean.
- Builder signatures + `FIGURE_REGISTRY` match `CONTRACTS.md` §6 exactly.
- `code-reviewer` APPROVE; PR from `feature-ui-figures`; `STATUS.md` updated.
