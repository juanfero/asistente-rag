"""Prueba M8 con e5 real + Gemini vía TestClient (marcador `llm`: solo con RUN_LLM=1)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag.api import create_app
from rag.config import Settings

pytestmark = [pytest.mark.integration, pytest.mark.llm]

Q2 = "¿Cuál es el tope diario de alimentación en viajes nacionales e internacionales?"


def test_api_ask_real(tmp_path: Path) -> None:
    """POST /ingest + POST /ask (Q2) reales: respuesta con ambos topes y citas del PDF."""
    settings = Settings(chroma_dir=tmp_path / "chroma", uploads_dir=tmp_path / "uploads")
    with TestClient(create_app(settings=settings)) as client:
        assert client.post("/ingest").json()["total_in_store"] == 16
        resp = client.post("/ask", json={"question": Q2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["grounded"] and body["llm_called"]
    assert "120.000" in body["answer"] and "90" in body["answer"]
    cited_pages = {s["page"] for s in body["sources"] if s["cited"]}
    assert {1, 2} <= cited_pages
