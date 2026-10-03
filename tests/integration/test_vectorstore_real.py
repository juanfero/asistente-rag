"""Prueba de integración M4: e5-small real + corpus en Chroma (tmp_path)."""

from pathlib import Path

import pytest

from rag.chunking import chunk_documents
from rag.config import Settings
from rag.embeddings import get_embedder
from rag.loaders import PROJECT_ROOT, load_directory
from rag.vectorstore import ChromaVectorStore

pytestmark = pytest.mark.integration

Q2 = "¿Cuál es el tope diario de alimentación en viajes nacionales e internacionales?"


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> ChromaVectorStore:
    """Corpus real indexado con el modelo y tamaño configurados (sin .env)."""
    settings = Settings(_env_file=None)
    docs = load_directory(PROJECT_ROOT / "data" / "docs")
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    path: Path = tmp_path_factory.mktemp("chroma")
    vs = ChromaVectorStore(
        path,
        settings.chroma_collection,
        get_embedder(settings),
        embedding_model=settings.embedding_model,
    )
    vs.add_chunks(chunks)
    return vs


def test_q2_both_pages_in_top4(store: ChromaVectorStore) -> None:
    """Q2 trae los chunks de "COP 120.000" (pág. 1) y "USD 90" (pág. 2) en el top-4."""
    top4 = store.query(Q2, top_k=4)
    texts = [r.text for r in top4]
    assert any("COP 120.000" in t for t in texts)
    assert any("USD 90" in t for t in texts)
    pages = {r.metadata.get("page") for r in top4 if "manual_reembolso" in r.metadata["source"]}
    assert {1, 2} <= pages


def test_collection_records_model(store: ChromaVectorStore) -> None:
    """La colección registra el modelo real, la dimensión 384 y el prefijo de pasaje."""
    meta = store.collection_metadata
    assert meta["embedding_model"] == "intfloat/multilingual-e5-small"
    assert meta["embedding_dim"] == 384
    assert meta["passage_prefix"] == "passage: "
    assert meta["hnsw:space"] == "cosine"
