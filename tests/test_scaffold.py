"""Architecture guards for the Undertow scaffold.

These are not smoke tests: they encode the ``data`` / ``sim`` package-boundary
rule in ``AGENTS.md`` that every later task inherits. If a future module starts
importing across the boundary, these tests fail.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"


def _walk_py_files(package_path: Path) -> list[Path]:
    """Return all ``.py`` files under ``package_path`` (recursively)."""
    assert package_path.exists(), f"package path missing: {package_path}"
    return sorted(p for p in package_path.rglob("*.py") if p.is_file())


def _imports_across_boundary(package_path: Path, other_pkg: str) -> list[str]:
    """Return files (relative to ``package_path``) importing ``other_pkg``.

    Uses ``ast`` to inspect every module in ``package_path`` for either
    ``import <other_pkg>`` or ``from <other_pkg> import ...``. This is checked on
    source text rather than runtime imports, so it also catches imports hidden
    inside functions and under ``if TYPE_CHECKING``.
    """
    offenders: list[str] = []
    prefix = "undertow." + other_pkg
    for path in _walk_py_files(package_path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            imported_names: list[str]
            if isinstance(node, ast.Import):
                imported_names = [a.name for a in node.names]
            else:
                # node.module is None for `from . import x` — skip relative imports
                imported_names = [node.module] if node.module else []
            for name in imported_names:
                if name == prefix or name.startswith(prefix + "."):
                    offenders.append(str(path.relative_to(package_path)))
                    break
    return offenders


def test_data_package_importable() -> None:
    import undertow.data  # noqa: F401

    assert isinstance(undertow.data.__all__, list)
    # Public API must include at minimum load_dataset + Dataset.
    assert "load_dataset" in undertow.data.__all__
    assert "Dataset" in undertow.data.__all__


def test_data_does_not_import_sim() -> None:
    offenders = _imports_across_boundary(SRC / "undertow" / "data", "sim")
    assert offenders == [], (
        "undertow.data must never import undertow.sim; "
        f"offending files: {', '.join(offenders) or 'none'}"
    )


def test_sim_does_not_import_data() -> None:
    offenders = _imports_across_boundary(SRC / "undertow" / "sim", "data")
    assert offenders == [], (
        "undertow.sim must never import undertow.data; "
        f"offending files: {', '.join(offenders) or 'none'}"
    )


def test_python_version_supported() -> None:
    assert sys.version_info >= (3, 11)


@pytest.mark.network
def test_network_marker_proof() -> None:
    """Trivially-passing network-marked test.

    Skipped by the default ``uv run pytest`` run; executed only when
    ``--run-network`` is passed. See ``tests/conftest.py``.
    """
    assert True
