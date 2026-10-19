"""``python -m undertow.ui`` — run the dashboard CLI as a module.

This is a process composition root (`PLAN.md` §4): it only wires the CLI entry
point, so it is excluded from coverage.  Its behavior is still exercised by the
subprocess test in ``tests/ui/test_cli.py``.
"""

from __future__ import annotations  # pragma: no cover - process entry point

from undertow.ui.cli import main  # pragma: no cover - process entry point

if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
