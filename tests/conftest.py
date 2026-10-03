"""Fixtures compartidas de pruebas."""

import os
from collections.abc import Iterator

import pytest

from rag.config import Settings, get_settings


def llm_skip_reason(environ: dict[str, str], has_key: bool) -> str | None:
    """Motivo para omitir pruebas `llm` (opt-in con RUN_LLM=1 y GEMINI_API_KEY), o None."""
    if environ.get("RUN_LLM") != "1":
        return "Prueba con costo (llama a Gemini): ejecútala con RUN_LLM=1 pytest -m llm"
    if not has_key:
        return "RUN_LLM=1 pero falta GEMINI_API_KEY en .env"
    return None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Omite las pruebas marcadas `llm` salvo opt-in explícito (evita gastar créditos)."""
    llm_items = [item for item in items if item.get_closest_marker("llm")]
    if not llm_items:
        return
    key = Settings().gemini_api_key
    reason = llm_skip_reason(dict(os.environ), bool(key and key.get_secret_value()))
    if reason:
        for item in llm_items:
            item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[None]:
    """Aísla la configuración: sin variables del proyecto, sin `.env` y sin caché."""
    for field_name in Settings.model_fields:
        monkeypatch.delenv(field_name.upper(), raising=False)
    monkeypatch.chdir(tmp_path)  # evita leer el .env real del repositorio
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
