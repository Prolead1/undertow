# Undertow — Repository Workflow (loaded automatically by pi)

Undertow is an RL-for-AMM-liquidity research codebase. Two packages:

- `undertow.data` — data ingestion & processing pipeline (on-chain Uniswap V3 data)
- `undertow.sim` — AMM / concentrated-liquidity simulator and RL environment

## Mandatory development workflow — for ALL code

1. **Branch first — never commit to `main` directly.** All new code lives on a descriptively-named branch: `feature-xxx`, `bug-xxx`, `chore-xxx`, etc. Create it before writing anything; `main` only ever receives merged PRs.
2. **Every code change must be reviewed by the `code-reviewer` subagent before it is committed.** If you write code, launch the reviewer; incorporate its findings; only then commit. Do not skip this for small changes — it is the project's quality gate.
   - The reviewer's prompt/definition is **versioned in the repo** at `docs/agents/code-reviewer.md` so any clone can reproduce it. To make it load into pi, symlink that file into the local (gitignored) config dir:

     ```bash
     mkdir -p .pi/agents
     ln -sf ../../docs/agents/code-reviewer.md .pi/agents/code-reviewer.md
     ```

     Always edit the **repo copy**, never a local copy — it is the single source of truth.
   - The thesis report has its own standard, versioned at `docs/agents/report-writer.md` (LaTeX, IEEE two-column, originality and voice rules, evidence discipline). Symlink it the same way:

     ```bash
     ln -sf ../../docs/agents/report-writer.md .pi/agents/report-writer.md
     ```
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

## Profiling and long-running experiments

- **Hard 3-minute cap.** Never launch a profiling, memory or wall-clock measurement with a timeout longer than **180 seconds**. If a question needs longer, split it into bounded stages or ask the human first.
- **Stream progress while it runs.** A profiling run must print phase boundaries and periodic RSS/timing samples line-buffered, and the agent must surface those samples to the human as they arrive (poll in short intervals and show the log tail) — never let a measurement run silently for minutes.
- **Bounded smoke first.** Prefer a tiny end-to-end smoke (a few thousand steps, one seed, reduced `parallel_envs`) before scaling to the real budget.

## Reminders

- This vault (`fyp/`) is NOT part of the repo — only `undertow/` is versioned. Notes stay in the vault; code lives in the repo.
- Only merge via PRs against `main`; `main` never receives direct commits.