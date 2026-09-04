# T00 — uv project scaffold + test harness

**Wave 0 · size S · depends on: nothing · blocks: everything**
**Branch:** `chore-data-scaffold`

## Why this task exists

Nothing else can be written until there is a package to write it into and a `uv run pytest` that runs.
Every other task's definition-of-done includes "pytest green", so this task defines what green means.
Keep it minimal: scaffolding only, no domain logic. Sixteen agents are about to branch off this — every
extra opinion you bake in here is one they all inherit.

## Files you own

```
pyproject.toml
uv.lock                              (generated)
src/undertow/__init__.py
src/undertow/data/__init__.py         (stub — T16 fills the real public API)
src/undertow/sim/__init__.py          (empty placeholder, so the split is visible from day one)
tests/__init__.py
tests/conftest.py
tests/data/__init__.py
tests/test_scaffold.py
configs/eth_usdc_3000.toml            (skeleton; T01 defines the real field set)
README.md                             (update the Status checklist only)
.gitignore                             (add anything missing)
```

Do **not** create any other module. Other tasks own them.

## What to build

1. `uv init` a src-layout package named `undertow`, requires-python `>=3.11`.
2. Dependencies — pin to major versions, keep the list tight:
   - runtime: `pyarrow`, `polars`, `numpy`, `requests`, `tenacity`, `tomli-w`, `platformdirs`,
     **`eth-utils`** (keccak256, needed by T06 to verify event topic0 constants — T06 cannot add it
     itself because `pyproject.toml` is a shared file only you and T14 may touch, so it goes in now)
   - dev (`[dependency-groups] dev`): `pytest`, `pytest-cov`, `pytest-mock`, `responses`, `ruff`, `mypy`
   - `polars` is the dataframe engine of record (fast as-of joins, no index footguns). If you prefer
     pandas, do not switch unilaterally — raise an ADR. Both `pandas` and `polars` in the same codebase
     is a rejection.
3. `[tool.pytest.ini_options]`:
   - `testpaths = ["tests"]`
   - `addopts = "-q --strict-markers"`
   - `markers = ["network: hits a live endpoint; deselected by default", "slow: > 5s"]`
   - **Do not put `-m 'not network'` in `addopts`.** It is prepended to every invocation, so a later
     `-m network` collides with it and the network tests become effectively unselectable. Deselect by
     default in `tests/conftest.py` instead: a `pytest_addoption` flag `--run-network` plus a
     `pytest_collection_modifyitems` hook that skips `network`-marked tests unless the flag is passed.
     Document the flag in the README — later tasks need it to run their live checks.
4. `[tool.ruff]` line-length 100, target py311, select `E,F,I,UP,B,SIM,ANN` (ANN for public API type
   hints — the reviewer checks these anyway, let the linter do it).
5. `[tool.mypy]`: `python_version = "3.11"`, `warn_unused_ignores`, `disallow_untyped_defs` scoped to
   `src/undertow/data`. Do not enable `strict` globally — it will bury the other 15 agents.
6. `src/undertow/data/__init__.py` — a stub with a module docstring stating the package boundary
   ("imports nothing from `undertow.sim`") and `__all__: list[str] = []`.
7. `configs/eth_usdc_3000.toml` — a skeleton with the section names T01 will populate
   (`[pool] [window] [regime] [endpoints] [paths]`) and commented-out `${ENV_VAR}` examples for
   `graph_url` / `rpc_url`. No real secrets, obviously.
8. Console script entry point declared but pointing at a `cli:main` that T14 will write — actually no:
   declaring an entry point to a nonexistent module breaks `uv sync`. **Leave the entry point out**;
   T14 adds `[project.scripts] undertow-data = "undertow.data.cli:main"` when it writes the CLI.

## Tests you must write (`tests/test_scaffold.py`)

These are not smoke tests — they are the architecture guards the whole plan leans on:

1. `test_data_package_importable` — `import undertow.data` succeeds.
2. `test_data_does_not_import_sim` — walk every `.py` file under `src/undertow/data/` with `ast` and
   assert no `import undertow.sim` / `from undertow.sim import ...` appears. **This test will be
   inherited by every later task and is the automated half of the architecture rule in
   `undertow/AGENTS.md`. Make it robust and well-named.**
3. `test_sim_does_not_import_data` — the mirror direction.
4. `test_python_version_supported` — `sys.version_info >= (3, 11)`.

## Acceptance criteria

- `uv sync` from a clean checkout succeeds.
- `uv run pytest` green, and the summary line shows ≥4 tests collected.
  `uv run pytest --run-network` also works — prove it with one trivially-passing `@pytest.mark.network`
  test that the default run skips and the flagged run executes.
- `uv run ruff check .` clean; `uv run mypy src/undertow/data` clean.
- README's Status checklist updated to reflect the scaffold landing.

## Handoff notes to record in your PR body

- The exact dataframe library chosen (polars vs pandas) — every transform task depends on this.
- The dev-dependency list, so fetcher tasks know `responses` is already available for HTTP mocking.
- Confirm `.gitignore` still excludes `.pi/`.

## Process (mandatory)

`git checkout -b chore-data-scaffold` off `main` → implement → `uv run pytest` → launch the
`code-reviewer` subagent → fix findings → commit → push → open PR against `main` → update
`.pi/plans/data-pipeline/STATUS.md`. Do not merge.
