"""U01 — CLI scaffold, config loader and the ``ui`` → ``sim`` import boundary.

The CLI is exercised in-process via :func:`undertow.ui.cli.main` (fast, covers
the dispatch/exit-code paths) and once end-to-end via ``python -m undertow.ui``
so the documented module entry point is real.  The import boundary (ADR-023 §7)
is enforced twice: dynamically in a fresh interpreter (``import undertow.ui``
must not drag in ``undertow.sim`` / ``undertow.data``) and statically by scanning
the ``undertow/ui`` sources for any reverse import statement.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import uvicorn

import undertow.ui
from undertow.ui.cli import (
    EXIT_CONFIG,
    EXIT_ERROR,
    EXIT_OK,
    main,
)
from undertow.ui.config import UiConfig, UiConfigError

#: Repository root, resolved from ``tests/ui/test_cli.py``.
REPO_ROOT = Path(__file__).resolve().parents[2]
UI_SRC_DIR = REPO_ROOT / "src" / "undertow" / "ui"

#: Packages ``undertow.ui`` must never import (ADR-023 §7).
FORBIDDEN_IMPORT_ROOTS = ("undertow.sim", "undertow.data")


def _write_manifest(run_dir: Path) -> None:
    """Create ``run_dir`` with a minimal valid ``manifest.json``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "ppo_s0_deadbeef"}), encoding="utf-8"
    )


def test_info_empty_runs_dir_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()

    code = main(["info", "--runs", str(runs)])

    captured = capsys.readouterr()
    assert code == EXIT_OK
    assert "configuration summary" in captured.out
    assert "0 runs" in captured.out
    assert str(runs) in captured.out


def test_info_counts_run_directories(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_manifest(tmp_path / "experiment_1" / "seed_0")

    code = main(["info", "--runs", str(tmp_path)])

    captured = capsys.readouterr()
    assert code == EXIT_OK
    assert "1 run" in captured.out


def test_info_missing_runs_dir_is_not_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["info", "--runs", str(tmp_path / "does-not-exist")])

    assert code == EXIT_OK
    assert "0 runs" in capsys.readouterr().out


def test_info_discovers_both_manifest_and_telemetry_layouts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_manifest(tmp_path / "experiment_1" / "seed_0")
    bare = tmp_path / "seed_1"
    bare.mkdir()
    (bare / "telemetry.jsonl").write_text("", encoding="utf-8")

    code = main(["info", "--runs", str(tmp_path)])

    assert code == EXIT_OK
    assert "2 runs" in capsys.readouterr().out


def test_info_counts_a_run_once_when_both_markers_exist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    both = tmp_path / "experiment_1" / "seed_0"
    _write_manifest(both)
    (both / "telemetry.jsonl").write_text("", encoding="utf-8")

    code = main(["info", "--runs", str(tmp_path)])

    assert code == EXIT_OK
    assert "1 run" in capsys.readouterr().out


def test_from_toml_round_trips_and_rejects_unknown_key(tmp_path: Path) -> None:
    path = tmp_path / "ui.toml"
    path.write_text(
        "\n".join(
            [
                'runs_dir = "custom-runs"',
                'results_dir = "custom-results"',
                'host = "0.0.0.0"',
                "port = 9001",
                "telemetry_poll_seconds = 0.25",
                "max_points = 500",
                'cors_origins = ["http://localhost:5173"]',
                'web_dist = "web/dist"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = UiConfig.from_toml(path)

    assert config.runs_dir == Path("custom-runs")
    assert config.results_dir == Path("custom-results")
    assert config.host == "0.0.0.0"
    assert config.port == 9001
    assert config.telemetry_poll_seconds == 0.25
    assert config.max_points == 500
    assert config.cors_origins == ("http://localhost:5173",)
    assert config.web_dist == Path("web/dist")

    bad = tmp_path / "bad.toml"
    bad.write_text("nope = 1\n", encoding="utf-8")
    with pytest.raises(UiConfigError, match="unknown UI config key"):
        UiConfig.from_toml(bad)


def test_from_toml_defaults_for_omitted_keys(tmp_path: Path) -> None:
    path = tmp_path / "minimal.toml"
    path.write_text('runs_dir = "runs-x"\n', encoding="utf-8")

    config = UiConfig.from_toml(path)

    assert config.runs_dir == Path("runs-x")
    assert config.host == "127.0.0.1"
    assert config.port == 8050
    assert config.cors_origins == ()
    assert config.web_dist is None


def test_from_toml_missing_and_malformed(tmp_path: Path) -> None:
    with pytest.raises(UiConfigError, match="not found"):
        UiConfig.from_toml(tmp_path / "missing.toml")

    malformed = tmp_path / "malformed.toml"
    malformed.write_text("port = [\n", encoding="utf-8")
    with pytest.raises(UiConfigError, match="invalid TOML"):
        UiConfig.from_toml(malformed)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('runs_dir = 3', "runs_dir"),
        ('port = "high"', "port"),
        ("max_points = 1.5", "max_points"),
        ('telemetry_poll_seconds = "fast"', "telemetry_poll_seconds"),
        ('cors_origins = "http://x"', "cors_origins"),
        ("cors_origins = [1, 2]", "cors_origins"),
        ("host = 7", "host"),
    ],
)
def test_from_toml_rejects_bad_types(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(body + "\n", encoding="utf-8")

    with pytest.raises(UiConfigError, match=message):
        UiConfig.from_toml(path)


def test_python_m_undertow_ui_info(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "undertow.ui", "info", "--runs", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == EXIT_OK, result.stderr
    assert "0 runs" in result.stdout


def test_importing_ui_does_not_import_sim_or_data() -> None:
    code = (
        "import sys\n"
        "import undertow.ui\n"
        "leaked = sorted(n for n in sys.modules if n.startswith('undertow'))\n"
        "assert 'undertow.sim' not in sys.modules, leaked\n"
        "assert 'undertow.data' not in sys.modules, leaked\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == EXIT_OK, result.stderr


def _imported_modules(path: Path) -> set[str]:
    """Return every module named by an ``import``/``from`` statement in ``path``.

    Parsing beats grepping: a docstring or comment that *mentions*
    ``undertow.sim`` is prose, not a dependency, and must not fail the check.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_ui_sources_never_import_sim_or_data() -> None:
    sources = sorted(UI_SRC_DIR.rglob("*.py"))

    assert sources, f"no UI sources found under {UI_SRC_DIR}"

    offenders = [
        f"{path.relative_to(REPO_ROOT)}: {module}"
        for path in sources
        for module in sorted(_imported_modules(path))
        if any(
            module == root or module.startswith(root + ".")
            for root in FORBIDDEN_IMPORT_ROOTS
        )
    ]
    assert not offenders, (
        "undertow.ui must not import undertow.sim/undertow.data (ADR-023 §7): "
        + ", ".join(offenders)
    )


def test_ui_config_is_frozen() -> None:
    config = UiConfig()

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.port = 9001  # type: ignore[misc]


def test_ui_package_exports_public_surface() -> None:
    assert undertow.ui.__all__ == ["UiConfig", "create_app", "serve"]
    assert callable(undertow.ui.create_app)
    assert callable(undertow.ui.serve)


def test_serve_runs_uvicorn_with_the_app(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[object, dict[str, object]]] = []

    def fake_run(app: object, **kwargs: object) -> None:
        calls.append((app, kwargs))

    monkeypatch.setattr(uvicorn, "run", fake_run)

    code = main(
        [
            "serve",
            "--runs",
            str(tmp_path),
            "--results",
            str(tmp_path),
            "--port",
            "9001",
        ]
    )

    assert code == EXIT_OK
    captured = capsys.readouterr()
    assert "http://127.0.0.1:9001" in captured.out
    assert str(tmp_path) in captured.out
    assert len(calls) == 1
    app, kwargs = calls[0]
    assert kwargs == {"host": "127.0.0.1", "port": 9001, "reload": False}
    assert type(app).__name__ == "FastAPI"


def test_serve_reload_uses_the_env_factory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[object, dict[str, object]]] = []

    def fake_run(app: object, **kwargs: object) -> None:
        calls.append((app, kwargs))

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.delenv("UNDERTOW_UI_CONFIG_JSON", raising=False)

    code = main(["serve", "--runs", str(tmp_path), "--reload"])

    assert code == EXIT_OK
    assert len(calls) == 1
    app, kwargs = calls[0]
    assert app == "undertow.ui.app:create_app_from_env"
    assert kwargs["factory"] is True
    assert kwargs["reload"] is True
    payload = json.loads(os.environ["UNDERTOW_UI_CONFIG_JSON"])
    assert payload["runs_dir"] == str(tmp_path)
    monkeypatch.delenv("UNDERTOW_UI_CONFIG_JSON", raising=False)


def test_serve_rejects_bad_port(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    called: list[object] = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: called.append(args))

    code = main(["serve", "--runs", str(tmp_path), "--port", "70000"])

    assert code == EXIT_CONFIG
    assert called == []
    assert "port must be" in capsys.readouterr().err


def test_serve_rejects_missing_web_dist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    called: list[object] = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: called.append(args))

    code = main(
        ["serve", "--runs", str(tmp_path), "--web-dist", str(tmp_path / "missing")]
    )

    assert code == EXIT_CONFIG
    assert called == []
    assert "web-dist is not a directory" in capsys.readouterr().err


def test_serve_prints_the_configured_web_dist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dist = tmp_path / "web"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html>", encoding="utf-8")
    calls: list[object] = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append(args))

    code = main(
        [
            "serve",
            "--runs",
            str(tmp_path),
            "--results",
            str(tmp_path),
            "--web-dist",
            str(dist),
        ]
    )

    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert f"web dist:    {dist}" in out
    assert len(calls) == 1


def test_info_runs_path_not_a_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a_file = tmp_path / "not-a-dir"
    a_file.write_text("x", encoding="utf-8")

    code = main(["info", "--runs", str(a_file)])

    assert code == EXIT_CONFIG
    assert "not a directory" in capsys.readouterr().err


def test_main_reports_unexpected_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(runs_dir: Path) -> list[Path]:
        raise RuntimeError("kaboom")

    monkeypatch.setattr("undertow.ui.cli._discover_run_dirs", boom)

    code = main(["info", "--runs", str(tmp_path)])

    assert code == EXIT_ERROR
    assert "kaboom" in capsys.readouterr().err


def test_debug_reraises_unexpected_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(runs_dir: Path) -> list[Path]:
        raise RuntimeError("kaboom")

    monkeypatch.setattr("undertow.ui.cli._discover_run_dirs", boom)

    with pytest.raises(RuntimeError, match="kaboom"):
        main(["--debug", "info", "--runs", str(tmp_path)])


# ---------------------------------------------------------------------------
# WebSocket support detection (uvicorn[standard])
# ---------------------------------------------------------------------------
def test_websocket_library_available_true_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util

    from undertow.ui.cli import _websocket_library_available

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if name == "websockets" else None,
    )
    assert _websocket_library_available() is True


def test_websocket_library_available_false_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util

    from undertow.ui.cli import _websocket_library_available

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert _websocket_library_available() is False


def test_serve_warns_when_websocket_library_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr("undertow.ui.cli._websocket_library_available", lambda: False)

    code = main(["serve", "--runs", str(tmp_path)])

    assert code == EXIT_OK
    assert calls != []  # the warning must not block startup
    assert "no WebSocket library installed" in capsys.readouterr().err


def test_serve_does_not_warn_when_websocket_library_is_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr("undertow.ui.cli._websocket_library_available", lambda: True)

    code = main(["serve", "--runs", str(tmp_path)])

    assert code == EXIT_OK
    assert "WebSocket" not in capsys.readouterr().err
