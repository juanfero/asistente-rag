"""Pruebas de integración M3 con el modelo real (descarga ~460 MB la primera vez)."""

import numpy as np
import pytest

from rag.chunking import chunk_documents
from rag.config import Settings
from rag.embeddings import SentenceTransformerEmbedder, get_embedder
from rag.loaders import PROJECT_ROOT, load_directory

pytestmark = pytest.mark.integration

MAX_EXCEED_PCT = 5.0


@pytest.fixture(scope="module")
def settings() -> Settings:
    """Configuración por defecto (sin .env) para que la prueba sea reproducible."""
    return Settings(_env_file=None)


@pytest.fixture(scope="module")
def embedder(settings: Settings) -> SentenceTransformerEmbedder:
    """Embedder real configurado (modelo y prefijos de settings), cargado una sola vez."""
    return get_embedder(settings)


@pytest.fixture(scope="module")
def chunks(settings: Settings):
    """Chunks del corpus real con el tamaño configurado."""
    docs = load_directory(PROJECT_ROOT / "data" / "docs")
    return chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)


def test_real_embedder_shape(embedder: SentenceTransformerEmbedder) -> None:
    """M3-04 (redefinido por ADR-009): dimensión 384, norma 1 y consistencia con prefijos.

    Consistencia = embed_query(t) ≡ codificar query_prefix + t; embed_documents([t]) ≡
    codificar passage_prefix + t; ambas versiones del mismo texto muy similares (> 0,9).
    """
    texts = ["¿Cuántos días de vacaciones tengo?", "La VPN corporativa es FortiClient."]
    docs = embedder.embed_documents(texts)
    query = embedder.embed_query(texts[0])
    assert embedder.dimension == 384
    assert len(docs) == 2 and all(len(v) == 384 for v in docs)
    assert np.allclose(np.linalg.norm(np.array(docs), axis=1), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(query), 1.0, atol=1e-5)
    assert all(isinstance(x, float) for x in query)

    # 1. Cada método aplica su prefijo
    raw_query, raw_passage = embedder.model.encode(
        [embedder.query_prefix + texts[0], embedder.passage_prefix + texts[0]],
        normalize_embeddings=True,
    )
    assert np.allclose(query, raw_query, atol=1e-5)
    assert np.allclose(docs[0], raw_passage, atol=1e-5)
    # 2. La versión pregunta y la versión pasaje del mismo texto son muy similares
    assert float(np.dot(query, docs[0])) > 0.9
    # 3. Aserción inversa: con prefijos distintos (e5) los vectores NO son iguales
    if embedder.query_prefix != embedder.passage_prefix:
        assert not np.allclose(query, docs[0], atol=1e-5)


def test_semantic_sanity(embedder: SentenceTransformerEmbedder, chunks) -> None:
    """M3-05: la pregunta de vacaciones se parece más al chunk de vacaciones que al de VPN."""
    vacation = next(c.text for c in chunks if "15 días hábiles" in c.text)
    vpn = next(c.text for c in chunks if "FortiClient" in c.text)
    q = np.array(embedder.embed_query("¿cuántos días de vacaciones tengo?"))
    v_vac, v_vpn = (np.array(v) for v in embedder.embed_documents([vacation, vpn]))
    assert float(q @ v_vac) > float(q @ v_vpn)


def test_chunks_fit_model(embedder: SentenceTransformerEmbedder, chunks) -> None:
    """M3-06: ≤ 5 % de los chunks (con prefijo de pasaje) excede el max_seq_length del modelo."""
    tokenizer, max_len = embedder.model.tokenizer, embedder.max_seq_length
    lengths = [
        len(
            tokenizer(
                embedder.passage_prefix + c.text,
                add_special_tokens=True,
                truncation=False,
                verbose=False,
            )["input_ids"]
        )
        for c in chunks
    ]
    exceed_pct = 100 * sum(n > max_len for n in lengths) / len(lengths)
    assert max_len > 0  # se toma del modelo (128 MiniLM, 512 e5), no se fija aquí
    assert exceed_pct <= MAX_EXCEED_PCT, f"{exceed_pct:.1f} % de chunks truncados"
