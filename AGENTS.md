# Undertow — Repository Workflow (loaded automatically by pi)

Undertow is an RL-for-AMM-liquidity research codebase. Two packages:

- `undertow.data` — data ingestion & processing pipeline (on-chain Uniswap V3 data)
- `undertow.sim` — AMM / concentrated-liquidity simulator and RL environment

## Mandatory development workflow — for ALL code

1. **Branch first — never commit to `main` directly.** All new code lives on a descriptively-named branch: `feature-xxx`, `bug-xxx`, `chore-xxx`, etc. Create it before writing anything; `main` only ever receives merged PRs.
2. **Every code change must be reviewed by the `code-reviewer` subagent before it is committed.** If you write code, launch the reviewer; incorporate its findings; only then commit. Do not skip this for small changes — it is the project's quality gate.
3. **Unit tests are non-negotiable.** Every new or changed logic ships with unit tests that assert real behavior and cover edge cases. Run `uv run pytest` until green.
4. **Respect the architecture:**
   - `undertow.data` and `undertow.sim` are separate packages with no imports across their boundaries except through stable public interfaces.
   - No god modules — split by responsibility (fetchers / parsers / transforms / schemas / storage / environment / agents).
   - Config and constants separate from logic; type hints on public functions; clean `__init__.py` exports.
5. Keep the change scoped: a commit does one thing.
6. **Raise a PR for every completed change.** After committing on the branch, push it and open a pull request against `main`. Do not merge it yourself — the human reviews and merges.

## What must never go into the repo

- `.pi/` — local pi agent configuration (agents, skills, workflows). It lives untracked in the vault; it is gitignored and must not be committed under any circumstances.
- Notes and research from the vault (`fyp/`). These stay untracked.

## Tooling

- `uv` for project management (`uv init`, `uv add`, `uv sync`, `uv run pytest`)
- `pytest` for tests (with `pytest-cov` once coverage is configured)

## Reminders

- This vault (`fyp/`) is NOT part of the repo — only `undertow/` is versioned. Notes stay in the vault; code lives in the repo.
- Only merge via PRs against `main`; `main` never receives direct commits.