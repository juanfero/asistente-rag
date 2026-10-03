"""Prueba de integración M0-08: conexión real con Gemini (endpoint compatible con OpenAI)."""

import pytest
from openai import OpenAI

from rag.config import Settings

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def settings() -> Settings:
    """Configuración real (variables de entorno o .env); omite si no hay key."""
    s = Settings()
    if s.gemini_api_key is None or not s.gemini_api_key.get_secret_value():
        pytest.skip("Sin GEMINI_API_KEY: se omite la prueba de conexión con Gemini")
    return s


def test_gemini_connection(settings: Settings) -> None:
    """M0-08: hay modelos, GEMINI_MODEL está entre ellos y una completion devuelve texto."""
    client = OpenAI(
        api_key=settings.gemini_api_key.get_secret_value(),
        base_url=settings.gemini_base_url,
        timeout=60,
    )
    model_ids = [m.id.split("/")[-1] for m in client.models.list()]
    assert len(model_ids) >= 1
    assert settings.gemini_model in model_ids

    response = client.chat.completions.create(
        model=settings.gemini_model,
        messages=[{"role": "user", "content": "Responde solo: OK"}],
        temperature=0,
        max_tokens=settings.llm_max_tokens,
    )
    choice = response.choices[0]
    assert (choice.message.content or "").strip()
    assert choice.finish_reason != "length"
