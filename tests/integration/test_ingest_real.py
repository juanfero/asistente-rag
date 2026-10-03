"""Prueba de integración M5-08: ingesta real (e5-small) del corpus en un Chroma temporal."""

from pathlib import Path

import pytest

from rag.config import Settings
from rag.ingest import build_store, ingest_paths
from rag.loaders import PROJECT_ROOT

pytestmark = pytest.mark.integration


def test_ingest_real(tmp_path: Path) -> None:
    """M5-08: ingesta real de data/docs (en tmp_path, no toca el índice del usuario)."""
    settings = Settings(_env_file=None, chroma_dir=tmp_path / "chroma")
    store = build_store(settings)
    report = ingest_paths([PROJECT_ROOT / "data" / "docs"], store, settings)

    assert (report.files_processed, report.files_skipped) == (3, 0)
    assert report.documents == 4  # md + txt + 2 páginas de PDF
    assert report.total_in_store == report.chunks_added == 16
    meta = store.collection_metadata
    assert (meta["embedding_model"], meta["embedding_dim"]) == (settings.embedding_model, 384)

    # Re-ingesta idempotente con el modelo real y consulta de control
    assert ingest_paths([PROJECT_ROOT / "data" / "docs"], store, settings).total_in_store == 16
    top = store.query("¿Qué VPN debo usar fuera de la oficina?", top_k=1)[0]
    assert top.metadata["source"] == "guia_onboarding_ti.txt" and "FortiClient" in top.text
