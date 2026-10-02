"""Fixtures compartidas de pruebas."""

from collections.abc import Iterator

import pytest

from rag.config import Settings, get_settings


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[None]:
    """Aísla la configuración: sin variables del proyecto, sin `.env` y sin caché."""
    for field_name in Settings.model_fields:
        monkeypatch.delenv(field_name.upper(), raising=False)
    monkeypatch.chdir(tmp_path)  # evita leer el .env real del repositorio
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
