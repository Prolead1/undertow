# U01 — `undertow.ui` scaffold + repo plumbing + stack ADR

**Wave 0 · size S · depends on: — · blocks: all UI tasks**
**Branch:** `feature-ui-scaffold`

## Why this task exists

Create the third top-level package (`undertow.ui`), its dependency group and console script, and the
test scaffolding every later UI task builds on. Also record the stack decision so no later agent
re-litigates it.

## Files you own

```
src/undertow/ui/__init__.py                (new — exports UiConfig)
src/undertow/ui/__main__.py                (new — python -m undertow.ui)
src/undertow/ui/config.py                  (new — UiConfig, CONTRACTS §3)
src/undertow/ui/cli.py                     (new — argparse parser + `info`; `serve` is U05)
pyproject.toml                             (edit — ui extra, dev deps, undertow-ui script)
.gitignore                                 (edit — node_modules/, web/dist/, *.tsbuildinfo)
tests/ui/__init__.py                       (new — empty)
tests/ui/conftest.py                       (new — seeds/markers; shared run_dir factory lands U02)
docs/decisions/023-ui-stack-and-live-server.md   (new)
```

Do **not** create `src/undertow/ui/app.py`, `registry.py`, `telemetry.py`, `schemas.py`,
`figures/`, or `rollouts.py` — later tasks own them. Do not edit any `undertow.sim` / `undertow.data`
file.

## What to build

1. **`UiConfig`** exactly as `CONTRACTS.md` §3 (frozen dataclass, slots). Add a
   `from_toml(path) -> UiConfig` loader (uses `tomllib`, mirrors `undertow.sim.config.load_sim_config`
   style: unknown keys raise a clear error). Keep it dependency-free beyond stdlib.

2. **`cli.py`.** `build_parser()` + `main(argv=None) -> int` matching the `undertow-sim` CLI
   conventions (argparse, `--debug`, exit codes `0/1/2`, clean error messages to stderr). Implement
   `info` only: print the resolved config and the number of run directories discovered under
   `--runs` (a run directory = one containing `manifest.json` or `telemetry.jsonl`; implement the
   walk inline here — U02 replaces it with the registry). `serve` may be a stub that prints
   "serve is implemented in a later task" and returns `2`; U05 replaces it.

3. **`__init__.py`.** `from .config import UiConfig` and `__all__ = ["UiConfig"]`. `__main__.py`
   calls `raise SystemExit(main())`.

4. **`pyproject.toml`.**
   - `[project.optional-dependencies]`: add `ui = ["fastapi>=0.115", "uvicorn[standard]>=0.30", "plotly>=6.0"]`.
   - `[dependency-groups].dev`: add `fastapi`, `uvicorn`, `httpx` (FastAPI TestClient), `plotly`.
     (The uncommitted `plotly` line on `feature-sim-report` is superseded — remove/ignore it; this
     task is the source of truth.)
   - `[project.scripts]`: `undertow-ui = "undertow.ui.cli:main"`.
   - If coverage needs it, keep `--cov=undertow` as-is (do **not** lower the 90% floor). Do not add a
     `web/` path to the wheel; JS is not packaged.
   - Run `uv sync` and commit the updated `uv.lock`.

5. **`.gitignore`.** Append `node_modules/`, `web/dist/`, `web/node_modules/`, `*.tsbuildinfo`.

6. **ADR-023** (`docs/decisions/023-ui-stack-and-live-server.md`, use `TEMPLATE.md`): record the
   stack (FastAPI + WebSocket + React/Vite + Plotly.js), the third-package layout, why the backend
   owns figures, that Node is quarantined in `web/`, and the rejected alternatives (Dash, Streamlit,
   static-only report, separate distribution). Reference `docs/plans/ui/PLAN.md` §7.

7. **`tests/ui/conftest.py`.** Register nothing new (markers already exist); provide an
   `ui_config(tmp_path)` fixture returning a `UiConfig(runs_dir=tmp_path, results_dir=tmp_path)`. The
   shared `run_dir` factory is U02's.

## Tests

`tests/ui/test_cli.py` (new — you may add it to your owned files):

1. `info` on an empty runs dir exits `0` and reports `0 runs`.
2. `info` on a directory tree with one `manifest.json` reports `1 run`.
3. `UiConfig.from_toml` round-trips a minimal file and raises on an unknown key.
4. `python -m undertow.ui info` works (subprocess or `main([...])`).
5. Importing `undertow.ui` does **not** import `undertow.sim` (assert `"undertow.sim" not in
   sys.modules` after a fresh import) — this guards the boundary and keeps import cost low.

## Definition of done

- `uv run pytest tests/ui -q` green; full `uv run pytest` green at ≥90% coverage.
- `uv run ruff check .` and `uv run mypy src/undertow` clean.
- `undertow-ui info` runs from a fresh `uv sync`.
- ADR-023 merged into the branch; `code-reviewer` APPROVE; PR from `feature-ui-scaffold`;
  `STATUS.md` updated.
