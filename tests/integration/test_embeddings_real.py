"""Pruebas de integración M3 con el modelo real (descarga ~460 MB la primera vez)."""

import numpy as np
import pytest

from rag.chunking import chunk_documents
from rag.config import Settings
from rag.embeddings import SentenceTransformerEmbedder
from rag.loaders import PROJECT_ROOT, load_directory

pytestmark = pytest.mark.integration

MAX_EXCEED_PCT = 5.0


@pytest.fixture(scope="module")
def settings() -> Settings:
    """Configuración por defecto (sin .env) para que la prueba sea reproducible."""
    return Settings(_env_file=None)


@pytest.fixture(scope="module")
def embedder(settings: Settings) -> SentenceTransformerEmbedder:
    """Embedder real compartido por las pruebas del módulo (se carga una sola vez)."""
    return SentenceTransformerEmbedder(settings.embedding_model)


@pytest.fixture(scope="module")
def chunks(settings: Settings):
    """Chunks del corpus real con el tamaño configurado."""
    docs = load_directory(PROJECT_ROOT / "data" / "docs")
    return chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)


def test_real_embedder_shape(embedder: SentenceTransformerEmbedder) -> None:
    """M3-04: dimensión 384, vectores normalizados; query y documentos consistentes."""
    texts = ["¿Cuántos días de vacaciones tengo?", "La VPN corporativa es FortiClient."]
    docs = embedder.embed_documents(texts)
    query = embedder.embed_query(texts[0])
    assert embedder.dimension == 384
    assert len(docs) == 2 and all(len(v) == 384 for v in docs)
    assert np.allclose(np.linalg.norm(np.array(docs), axis=1), 1.0, atol=1e-5)
    assert np.allclose(query, docs[0], atol=1e-5)
    assert all(isinstance(x, float) for x in query)


def test_semantic_sanity(embedder: SentenceTransformerEmbedder, chunks) -> None:
    """M3-05: la pregunta de vacaciones se parece más al chunk de vacaciones que al de VPN."""
    vacation = next(c.text for c in chunks if "15 días hábiles" in c.text)
    vpn = next(c.text for c in chunks if "FortiClient" in c.text)
    q = np.array(embedder.embed_query("¿cuántos días de vacaciones tengo?"))
    v_vac, v_vpn = (np.array(v) for v in embedder.embed_documents([vacation, vpn]))
    assert float(q @ v_vac) > float(q @ v_vpn)


def test_chunks_fit_model(embedder: SentenceTransformerEmbedder, chunks) -> None:
    """M3-06: ≤ 5 % de los chunks del corpus excede max_seq_length (con el tamaño final)."""
    tokenizer, max_len = embedder.model.tokenizer, embedder.max_seq_length
    lengths = [
        len(
            tokenizer(c.text, add_special_tokens=True, truncation=False, verbose=False)["input_ids"]
        )
        for c in chunks
    ]
    exceed_pct = 100 * sum(n > max_len for n in lengths) / len(lengths)
    assert max_len == 128
    assert exceed_pct <= MAX_EXCEED_PCT, f"{exceed_pct:.1f} % de chunks truncados"
