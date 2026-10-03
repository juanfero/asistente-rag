"""Prueba M6-07 con Gemini real (marcador `llm`: solo con RUN_LLM=1 y GEMINI_API_KEY)."""

import pytest

from rag.config import Settings
from rag.llm import GeminiClient

pytestmark = [pytest.mark.integration, pytest.mark.llm]

CONTEXT = (
    "[1] (fuente: politica_vacaciones_y_permisos.md)\n"
    "Cada colaborador tiene derecho a **15 días hábiles** de vacaciones remuneradas por cada "
    "año trabajado."
)


def test_gemini_real() -> None:
    """M6-07: llamada real tipo RAG devuelve texto no vacío, no cortado y en < 30 s."""
    client = GeminiClient(Settings())
    result = client.generate(
        "Responde solo con el contexto, en español, citando con [n].",
        f"Contexto:\n{CONTEXT}\n\nPregunta: ¿Cuántos días de vacaciones tengo al año?",
    )
    assert result.text.strip()
    assert "15" in result.text
    assert result.finish_reason != "length"
    assert result.latency_s < 30
    assert result.prompt_tokens and result.completion_tokens
