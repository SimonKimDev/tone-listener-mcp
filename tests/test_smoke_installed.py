import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.smoke_installed import resolve_executable

PYTHON = os.path.abspath(sys.executable)
ROOT = Path(__file__).resolve().parents[1]


def same(a: str, b: str) -> bool:
    return os.path.normcase(a) == os.path.normcase(b)


def test_absolute_executable_is_kept():
    assert same(resolve_executable(PYTHON), PYTHON)


def test_relative_executable_resolves_against_caller_cwd(monkeypatch, tmp_path):
    directory = Path(PYTHON).parent
    monkeypatch.chdir(directory.parent)
    resolved = resolve_executable(os.path.join(directory.name, Path(PYTHON).name))
    monkeypatch.chdir(tmp_path)
    assert os.path.isabs(resolved) and os.path.isfile(resolved)
    assert same(resolved, PYTHON)


def test_command_name_resolves_via_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(Path(PYTHON).parent))
    assert same(resolve_executable(Path(PYTHON).name), PYTHON)


@pytest.mark.parametrize(("executable", "message"), [
    (os.path.join("missing-dir", "tone-listener-mcp"), "path not found"),
    ("tone-listener-mcp-definitely-missing", "command on PATH not found"),
])
def test_missing_executable_fails_clearly(monkeypatch, tmp_path, executable, message):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match=message):
        resolve_executable(executable)


def test_installed_smoke_accepts_relative_executable():
    installed = shutil.which("tone-listener-mcp", path=str(Path(PYTHON).parent))
    if installed is None:
        pytest.skip("tone-listener-mcp entry point is not installed beside this Python")
    venv = Path(installed).parent.parent
    result = subprocess.run(
        [PYTHON, str(ROOT / "scripts" / "smoke_installed.py"), os.path.relpath(installed, venv)],
        check=False, cwd=venv, env={**os.environ, "GEMINI_API_KEY": ""},
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("ok: ")
