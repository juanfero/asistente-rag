"""Pruebas de scripts/check_gemini.py sin red: enmascarado y manejo sin key."""

import importlib.util

import pytest

from rag.config import Settings
from rag.loaders import PROJECT_ROOT

pytestmark = pytest.mark.usefixtures("clean_env")

_spec = importlib.util.spec_from_file_location(
    "check_gemini", PROJECT_ROOT / "scripts" / "check_gemini.py"
)
check_gemini = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_gemini)

SECRET = "clave-de-prueba-NO-REAL-1234"


def test_mask_shows_only_last_four() -> None:
    """La clave enmascarada solo deja ver los últimos 4 caracteres."""
    assert check_gemini.mask(SECRET) == "****1234"
    assert check_gemini.mask("corta") == "****"
    assert SECRET[:-4] not in check_gemini.mask(SECRET)


def test_safe_error_never_contains_key() -> None:
    """Los mensajes de error se limpian de la clave aunque la API la incluya."""
    exc = RuntimeError(f"Bad request for key {SECRET}")
    text = check_gemini.safe_error(exc, SECRET)
    assert SECRET not in text and "****1234" in text and "RuntimeError" in text


def test_check_without_key(capsys: pytest.CaptureFixture[str]) -> None:
    """Sin GEMINI_API_KEY: mensaje claro en español y código 1, sin traceback."""
    assert check_gemini.check(Settings(_env_file=None)) == 1
    err = capsys.readouterr().err
    assert "falta GEMINI_API_KEY" in err and "Traceback" not in err
