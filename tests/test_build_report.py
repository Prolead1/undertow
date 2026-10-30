"""Hermetic tests for ``scripts/build_report.sh``.

The wrapper is exercised as a black box: the real script is copied into a
temporary repo tree, and stub ``tectonic``/``latexmk``/``pdflatex`` binaries on
an isolated ``PATH`` record which engine ran and produce (or skip) ``main.pdf``.
No real LaTeX toolchain and no network are involved.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_report.sh"

_ENGINE_STUB = """#!/usr/bin/env bash
set -euo pipefail
echo "{name}" >> engine_log.txt
echo "pdf" > main.pdf
"""


def _make_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _system_bin(tmp_path: Path) -> Path:
    """A PATH containing only the external tools the wrapper needs."""
    system = tmp_path / "system-bin"
    system.mkdir()
    for name in ("bash", "cp", "dirname", "mkdir"):
        resolved = shutil.which(name)
        assert resolved is not None, f"{name} is required for the test environment"
        (system / name).symlink_to(resolved)
    return system


def _prepare(
    tmp_path: Path,
    *,
    engines: tuple[str, ...] = ("tectonic",),
    with_report: bool = True,
) -> tuple[Path, Path, dict[str, str]]:
    """Create a temp repo, stub engines and an isolated environment."""
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    source = SCRIPT.read_text(encoding="utf-8")
    (scripts / "build_report.sh").write_text(source, encoding="utf-8")
    report = repo / "report"
    if with_report:
        report.mkdir()
        (report / "main.tex").write_text("stub", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in engines:
        _make_executable(bin_dir / name, _ENGINE_STUB.format(name=name))
    if "pdflatex" in engines:
        _make_executable(
            bin_dir / "bibtex",
            '#!/usr/bin/env bash\nset -euo pipefail\necho "bibtex" >> engine_log.txt\n',
        )
    home = tmp_path / "home"
    home.mkdir()
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{_system_bin(tmp_path)}",
        "HOME": str(home),
    }
    env.pop("TECTONIC", None)
    env.pop("UNDERTOW_REPORT_PDF", None)
    env.pop("BASH_ENV", None)
    return repo, report, env


def _run(
    repo: Path, env: dict[str, str], *args: str, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(repo / "scripts" / "build_report.sh"), *args],
        cwd=cwd or repo,
        env=env,
        capture_output=True,
        text=True,
    )


def test_prefers_tectonic_on_path(tmp_path: Path) -> None:
    repo, report, env = _prepare(tmp_path, engines=("tectonic", "latexmk"))
    result = _run(repo, env)
    assert result.returncode == 0
    assert (report / "engine_log.txt").read_text().split() == ["tectonic"]
    assert (report / "main.pdf").read_bytes() == b"pdf\n"


def test_tectonic_env_override_wins(tmp_path: Path) -> None:
    repo, report, env = _prepare(tmp_path, engines=("latexmk",))
    custom = tmp_path / "custom-tectonic"
    _make_executable(custom, _ENGINE_STUB.format(name="tectonic"))
    env["TECTONIC"] = str(custom)
    result = _run(repo, env)
    assert result.returncode == 0
    assert (report / "engine_log.txt").read_text().split() == ["tectonic"]


def test_non_executable_tectonic_warns_and_falls_back(tmp_path: Path) -> None:
    repo, report, env = _prepare(tmp_path, engines=("latexmk",))
    env["TECTONIC"] = str(tmp_path / "missing")
    result = _run(repo, env)
    assert result.returncode == 0
    assert "not executable" in result.stderr
    assert (report / "engine_log.txt").read_text().split() == ["latexmk"]


def test_falls_back_to_latexmk(tmp_path: Path) -> None:
    repo, report, env = _prepare(tmp_path, engines=("latexmk",))
    result = _run(repo, env)
    assert result.returncode == 0
    assert (report / "engine_log.txt").read_text().split() == ["latexmk"]


def test_falls_back_to_pdflatex_cycle(tmp_path: Path) -> None:
    repo, report, env = _prepare(tmp_path, engines=("pdflatex",))
    result = _run(repo, env)
    assert result.returncode == 0
    assert (report / "engine_log.txt").read_text().split() == [
        "pdflatex",
        "bibtex",
        "pdflatex",
        "pdflatex",
    ]


def test_no_engine_exits_2(tmp_path: Path) -> None:
    repo, _, env = _prepare(tmp_path, engines=())
    result = _run(repo, env)
    assert result.returncode == 2
    assert "no LaTeX engine found" in result.stderr


def test_engine_without_output_exits_2(tmp_path: Path) -> None:
    repo, _, env = _prepare(tmp_path, engines=())
    _make_executable(tmp_path / "bin" / "tectonic", "#!/usr/bin/env bash\nexit 0\n")
    result = _run(repo, env)
    assert result.returncode == 2
    assert "produced no" in result.stderr


def test_missing_tex_exits_2(tmp_path: Path) -> None:
    repo, _, env = _prepare(tmp_path, engines=("tectonic",), with_report=False)
    result = _run(repo, env)
    assert result.returncode == 2
    assert "main.tex not found" in result.stderr


def test_copies_to_argument_destination(tmp_path: Path) -> None:
    repo, report, env = _prepare(tmp_path, engines=("tectonic",))
    destination = tmp_path / "out" / "thesis.pdf"
    result = _run(repo, env, str(destination))
    assert result.returncode == 0
    assert destination.read_bytes() == (report / "main.pdf").read_bytes()
    assert f"copied {destination}" in result.stdout


def test_copies_to_env_destination(tmp_path: Path) -> None:
    repo, report, env = _prepare(tmp_path, engines=("tectonic",))
    destination = tmp_path / "viewer" / "thesis.pdf"
    env["UNDERTOW_REPORT_PDF"] = str(destination)
    result = _run(repo, env)
    assert result.returncode == 0
    assert destination.read_bytes() == (report / "main.pdf").read_bytes()


def test_relative_destination_resolves_against_caller_cwd(tmp_path: Path) -> None:
    repo, report, env = _prepare(tmp_path, engines=("tectonic",))
    caller = tmp_path / "caller"
    caller.mkdir()
    result = _run(repo, env, "out/thesis.pdf", cwd=caller)
    assert result.returncode == 0
    assert (caller / "out" / "thesis.pdf").exists()
    assert not (report / "out" / "thesis.pdf").exists()
