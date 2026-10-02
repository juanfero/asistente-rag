"""Prueba de integración M0-08: conexión real con Grok (xAI)."""

import pytest
from openai import OpenAI

from rag.config import Settings

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def settings() -> Settings:
    """Configuración real (variables de entorno o .env); omite si no hay key."""
    s = Settings()
    if s.xai_api_key is None or not s.xai_api_key.get_secret_value():
        pytest.skip("Sin XAI_API_KEY: se omite la prueba de conexión con xAI")
    return s


def test_xai_connection(settings: Settings) -> None:
    """M0-08: hay modelos, XAI_MODEL está entre ellos y una completion devuelve texto."""
    client = OpenAI(
        api_key=settings.xai_api_key.get_secret_value(),
        base_url=settings.xai_base_url,
        timeout=60,
    )
    model_ids = [m.id for m in client.models.list()]
    assert len(model_ids) >= 1
    assert settings.xai_model in model_ids

    response = client.chat.completions.create(
        model=settings.xai_model,
        messages=[{"role": "user", "content": "Responde solo: OK"}],
        temperature=0,
        max_tokens=20,
    )
    assert (response.choices[0].message.content or "").strip()
