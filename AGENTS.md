# Undertow — Repository Workflow (loaded automatically by pi)

Undertow is an RL-for-AMM-liquidity research codebase. Two packages:

- `undertow.data` — data ingestion & processing pipeline (on-chain Uniswap V3 data)
- `undertow.sim` — AMM / concentrated-liquidity simulator and RL environment

## Mandatory development workflow — for ALL code

1. **Every code change must be reviewed by the `code-reviewer` subagent before it is committed.** If you write code, launch the reviewer; incorporate its findings; only then commit. Do not skip this for small changes — it is the project's quality gate.
2. **Unit tests are non-negotiable.** Every new or changed logic ships with unit tests that assert real behavior and cover edge cases. Run `uv run pytest` until green.
3. **Respect the architecture:**
   - `undertow.data` and `undertow.sim` are separate packages with no imports across their boundaries except through stable public interfaces.
   - No god modules — split by responsibility (fetchers / parsers / transforms / schemas / storage / environment / agents).
   - Config and constants separate from logic; type hints on public functions; clean `__init__.py` exports.
4. Keep the change scoped: a commit does one thing.

## Tooling

- `uv` for project management (`uv init`, `uv add`, `uv sync`, `uv run pytest`)
- `pytest` for tests (with `pytest-cov` once coverage is configured)

## Reminders

- This vault (`fyp/`) is NOT part of the repo — only `undertow/` is versioned. Notes stay in the vault; code lives in the repo.