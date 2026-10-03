"""Seguridad: ningún archivo versionado contiene claves de API (ADR-010)."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

from rag.loaders import PROJECT_ROOT

_spec = importlib.util.spec_from_file_location(
    "secret_scan", PROJECT_ROOT / "scripts" / "secret_scan.py"
)
secret_scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(secret_scan)

# Claves falsas construidas en tiempo de ejecución (no aparecen literalmente en el repo)
FAKE_GOOGLE = "AI" + "za" + "X" * 35
FAKE_TOKEN = "AQ" + "." + "Y" * 24


def test_repository_has_no_secrets() -> None:
    """Ningún archivo de `git ls-files` contiene patrones de clave."""
    findings = secret_scan.scan(secret_scan.tracked_files(PROJECT_ROOT))
    assert findings == [], findings


def test_env_example_key_is_empty() -> None:
    """.env.example está versionado y GEMINI_API_KEY queda sin valor."""
    tracked = subprocess.run(
        ["git", "ls-files", ".env.example"], cwd=PROJECT_ROOT, capture_output=True, text=True
    ).stdout.strip()
    assert tracked == ".env.example"
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "GEMINI_API_KEY=\n" in text
    assert secret_scan.find_secrets(text) == []


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        (f"api_key = '{FAKE_GOOGLE}'", "clave Google (AIza…)"),
        (f"token: {FAKE_TOKEN}", "token (AQ.…)"),
        ("GEMINI_API_KEY=valor-real", "GEMINI_API_KEY con valor"),
        ('export GEMINI_API_KEY="valor-real"', "GEMINI_API_KEY con valor"),
    ],
    # ids fijos: sin ellos `pytest -v` imprime las claves falsas en el nombre del caso
    ids=["google", "token", "env-line", "env-export"],
)
def test_detects_keys(text: str, kind: str) -> None:
    """Detecta cada patrón y reporta línea y tipo, sin incluir la clave."""
    findings = secret_scan.find_secrets(f"línea 1\n{text}\n")
    assert findings == [(2, kind)]
    report = secret_scan.scan({"archivo.txt": text})
    assert all(
        FAKE_GOOGLE not in r and FAKE_TOKEN not in r and "valor-real" not in r for r in report
    )


@pytest.mark.parametrize(
    "text",
    [
        "GEMINI_API_KEY=",
        'GEMINI_API_KEY=""',
        "GEMINI_API_KEY=   # se completa en .env",
        "`GEMINI_API_KEY` (SecretStr) se lee del archivo .env",
        'monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)',
        "AIza corto no es clave",
    ],
)
def test_ignores_non_secrets(text: str) -> None:
    """Asignaciones vacías, menciones en documentación y textos cortos no son hallazgos."""
    assert secret_scan.find_secrets(text) == []


def test_pre_commit_hook_calls_scanner() -> None:
    """El hook versionado ejecuta el escáner sobre los archivos en stage."""
    hook = (PROJECT_ROOT / "scripts" / "pre-commit").read_text(encoding="utf-8")
    assert hook.startswith("#!/bin/sh")
    assert "secret_scan.py" in hook and "--staged" in hook


def test_staged_scan_blocks_key(tmp_path: Path) -> None:
    """En un repo temporal, un archivo en stage con clave hace fallar el escáner (código 1)."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "config.txt").write_text(f"key={FAKE_GOOGLE}\n", encoding="utf-8")
    subprocess.run(["git", "add", "config.txt"], cwd=tmp_path, check=True)
    script = PROJECT_ROOT / "scripts" / "secret_scan.py"
    result = subprocess.run(
        ["python3", str(script), "--staged"], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "config.txt:1" in result.stderr
    assert FAKE_GOOGLE not in result.stderr + result.stdout
