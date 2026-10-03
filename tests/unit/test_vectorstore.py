"""Pruebas de M4: ChromaVectorStore con FakeEmbedder y directorios temporales."""

from pathlib import Path

import numpy as np
import pytest

from rag.chunking import chunk_documents
from rag.embeddings import FakeEmbedder
from rag.loaders import PROJECT_ROOT, load_directory
from rag.models import Chunk, Document
from rag.vectorstore import (
    ChromaVectorStore,
    EmbeddingModelMismatchError,
    reset_collection,
)

COLLECTION = "pruebas"
TEXTS = {
    "vacaciones.md": [
        "Cada colaborador tiene derecho a quince días hábiles de vacaciones por año.",
        "La solicitud de vacaciones se hace en el portal NexaPeople con anticipación.",
    ],
    "ti.txt": [
        "La VPN corporativa es FortiClient y se usa fuera de la oficina.",
        "La contraseña debe tener mínimo doce caracteres y cambiarse cada noventa días.",
        "El soporte técnico atiende en la extensión 4040 de lunes a viernes.",
    ],
}


def _chunks() -> list[Chunk]:
    """Chunks sintéticos de dos fuentes (una con páginas)."""
    docs = [
        Document(t, {"source": s, "doc_type": s.split(".")[1], "path": f"data/docs/{s}"})
        for s, texts in TEXTS.items()
        for t in texts
    ]
    docs[0].metadata["page"] = 1
    return chunk_documents(docs, 800, 120)


def _store(path: Path, dim: int = 32, model: str = "fake-32") -> ChromaVectorStore:
    return ChromaVectorStore(path, COLLECTION, FakeEmbedder(dim), embedding_model=model)


@pytest.fixture
def store(tmp_path: Path) -> ChromaVectorStore:
    return _store(tmp_path)


# --- Criterios M4-01 … M4-09 -------------------------------------------------------


def test_add_and_count(store: ChromaVectorStore) -> None:
    """M4-01: add_chunks + count reflejan el nº de chunks."""
    chunks = _chunks()
    assert store.count() == 0
    assert store.add_chunks(chunks) == len(chunks) == 5
    assert store.count() == 5
    assert store.add_chunks([]) == 0


def test_idempotent_upsert(store: ChromaVectorStore) -> None:
    """M4-02: insertar dos veces los mismos chunks no duplica."""
    store.add_chunks(_chunks())
    store.add_chunks(_chunks())
    assert store.count() == 5


def test_query_order_and_scores(store: ChromaVectorStore) -> None:
    """M4-03: ≤ top_k resultados, orden por score desc y score en [0, 1]."""
    store.add_chunks(_chunks())
    results = store.query("vacaciones días hábiles por año", top_k=3)
    assert 0 < len(results) <= 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_exact_match_first(store: ChromaVectorStore) -> None:
    """M4-04: el chunk idéntico a la consulta queda primero con score ≈ 1."""
    chunks = _chunks()
    store.add_chunks(chunks)
    target = chunks[3]
    results = store.query(target.text, top_k=5)
    assert results[0].chunk_id == target.chunk_id
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_metadata_roundtrip(store: ChromaVectorStore) -> None:
    """M4-05: source, page, chunk_index y demás metadatos vuelven intactos."""
    chunks = _chunks()
    store.add_chunks(chunks)
    by_id = {c.chunk_id: c for c in chunks}
    for r in store.query("vacaciones", top_k=5):
        assert r.metadata == by_id[r.chunk_id].metadata
        assert r.text == by_id[r.chunk_id].text
    with_page = next(r for r in store.query(chunks[0].text, top_k=5) if "page" in r.metadata)
    assert with_page.metadata["page"] == 1 and isinstance(with_page.metadata["page"], int)


def test_persistence(tmp_path: Path) -> None:
    """M4-06: una nueva instancia sobre el mismo directorio ve los datos."""
    _store(tmp_path).add_chunks(_chunks())
    reopened = _store(tmp_path)
    assert reopened.count() == 5
    assert reopened.query(_chunks()[0].text, top_k=1)[0].chunk_id == _chunks()[0].chunk_id


def test_delete_by_source(store: ChromaVectorStore) -> None:
    """M4-07: delete_by_source elimina solo esa fuente; list_sources lo refleja."""
    store.add_chunks(_chunks())
    assert store.list_sources() == ["ti.txt", "vacaciones.md"]
    assert store.delete_by_source("vacaciones.md") == 2
    assert store.list_sources() == ["ti.txt"]
    assert store.count() == 3
    assert store.delete_by_source("no_existe.md") == 0


def test_reset_and_empty_query(store: ChromaVectorStore) -> None:
    """M4-08: reset deja la colección vacía; query en colección vacía → []."""
    assert store.query("cualquier cosa", top_k=4) == []
    store.add_chunks(_chunks())
    store.reset()
    assert store.count() == 0
    assert store.query("vacaciones", top_k=4) == []
    assert store.collection_metadata["embedding_model"] == "fake-32"  # metadata conservada


def test_none_metadata(store: ChromaVectorStore) -> None:
    """M4-09: metadatos con None no rompen la inserción ni borran el resto de la metadata."""
    chunk = Chunk(
        text="Texto con metadata incompleta para probar valores None en Chroma.",
        metadata={"source": "x.md", "page": None, "chunk_index": 0, "extra": Path("a/b")},
        chunk_id="id-none-0001",
    )
    assert store.add_chunks([chunk]) == 1
    [result] = store.query(chunk.text, top_k=1)
    assert result.metadata == {"source": "x.md", "chunk_index": 0, "extra": "a/b"}


# --- B. Metadata de la colección y validación de modelo ----------------------------


def test_collection_metadata(tmp_path: Path) -> None:
    """La colección guarda modelo, dimensión, prefijo de pasaje y métrica coseno."""
    emb = FakeEmbedder(32)
    emb.passage_prefix = "passage: "
    store = ChromaVectorStore(tmp_path, COLLECTION, emb, embedding_model="fake-32")
    assert store.collection_metadata == {
        "hnsw:space": "cosine",
        "embedding_model": "fake-32",
        "embedding_dim": 32,
        "passage_prefix": "passage: ",
    }


def test_model_mismatch(tmp_path: Path) -> None:
    """Abrir con otro modelo → EmbeddingModelMismatchError que sugiere el reset."""
    _store(tmp_path, model="modelo-a").add_chunks(_chunks())
    with pytest.raises(EmbeddingModelMismatchError, match="reset") as exc:
        _store(tmp_path, model="modelo-b")
    assert "modelo-a" in str(exc.value) and "modelo-b" in str(exc.value)


def test_dimension_mismatch(tmp_path: Path) -> None:
    """Abrir con otra dimensión → EmbeddingModelMismatchError."""
    _store(tmp_path, dim=32, model="fake").add_chunks(_chunks())
    with pytest.raises(EmbeddingModelMismatchError, match="embedding_dim"):
        _store(tmp_path, dim=16, model="fake")


def test_reset_collection_after_mismatch(tmp_path: Path) -> None:
    """reset_collection permite abrir con el nuevo modelo (índice vacío)."""
    _store(tmp_path, model="modelo-a").add_chunks(_chunks())
    reset_collection(tmp_path, COLLECTION)
    store = _store(tmp_path, model="modelo-b")
    assert store.count() == 0
    reset_collection(tmp_path, "no-existe")  # no falla si no existe


# --- C. top_k y colección vacía -----------------------------------------------------


def test_top_k_larger_than_count(store: ChromaVectorStore) -> None:
    """top_k > count → devuelve count resultados, sin error."""
    store.add_chunks(_chunks())
    assert len(store.query("vacaciones", top_k=50)) == 5


def test_top_k_invalid(store: ChromaVectorStore) -> None:
    """top_k < 1 → ValueError."""
    with pytest.raises(ValueError):
        store.query("vacaciones", top_k=0)


# --- D. source_stats ---------------------------------------------------------------


def test_source_stats(store: ChromaVectorStore) -> None:
    """source_stats devuelve chunks por fuente; list_sources se deriva de ahí."""
    assert store.source_stats() == {}
    store.add_chunks(_chunks())
    assert store.source_stats() == {"ti.txt": 3, "vacaciones.md": 2}
    assert store.list_sources() == list(store.source_stats())


# --- E. Lotes ----------------------------------------------------------------------


def test_upsert_in_batches(store: ChromaVectorStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """El upsert se parte según el tamaño máximo de lote de Chroma."""
    monkeypatch.setattr(store._client, "get_max_batch_size", lambda: 2)
    calls: list[int] = []
    original = store._collection.upsert

    def spy(**kwargs):
        calls.append(len(kwargs["ids"]))
        return original(**kwargs)

    monkeypatch.setattr(store._collection, "upsert", spy)
    assert store.add_chunks(_chunks()) == 5
    assert calls == [2, 2, 1]
    assert store.count() == 5


# --- F. Consistencia con coseno por fuerza bruta -----------------------------------


def test_cosine_consistency_with_numpy(tmp_path: Path) -> None:
    """Ranking y scores de Chroma coinciden (±1e-5) con coseno por fuerza bruta en numpy."""
    docs = load_directory(PROJECT_ROOT / "data" / "docs")
    chunks = chunk_documents(docs, 800, 120)
    emb = FakeEmbedder(64)
    store = ChromaVectorStore(tmp_path, COLLECTION, emb, embedding_model="fake-64")
    store.add_chunks(chunks)
    matrix = np.array(emb.embed_documents([c.text for c in chunks]))

    for question in ["días de vacaciones por año", "tope de alimentación viajes", "VPN"]:
        cos = matrix @ np.array(emb.embed_query(question))
        expected = sorted(
            ((chunks[i].chunk_id, float(np.clip(cos[i], 0, 1))) for i in range(len(chunks))),
            key=lambda x: -x[1],
        )
        got = store.query(question, top_k=len(chunks))
        assert len(got) == len(chunks)
        got_scores = {r.chunk_id: r.score for r in got}
        for cid, score in expected:
            assert got_scores[cid] == pytest.approx(score, abs=1e-5)
        # Mismo orden donde los scores son distintos (los empates pueden permutarse)
        positive = [cid for cid, s in expected if s > 0]
        assert [r.chunk_id for r in got[: len(positive)]] == positive


# --- Prefijos: el texto guardado no lleva prefijo ----------------------------------


class _PrefixRecordingEmbedder(FakeEmbedder):
    """FakeEmbedder que antepone un prefijo al codificar y registra lo que codifica."""

    passage_prefix = "passage: "

    def __init__(self) -> None:
        super().__init__(32)
        self.encoded: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [self.passage_prefix + t for t in texts]
        self.encoded.extend(prefixed)
        return super().embed_documents(prefixed)


def test_stored_text_has_no_prefix(tmp_path: Path) -> None:
    """El prefijo solo se usa al codificar: documento e id guardados son los del chunk."""
    emb = _PrefixRecordingEmbedder()
    store = ChromaVectorStore(tmp_path, COLLECTION, emb, embedding_model="fake-prefix")
    chunks = _chunks()
    store.add_chunks(chunks)
    stored = store._collection.get(include=["documents"])
    by_id = dict(zip(stored["ids"], stored["documents"], strict=True))
    assert by_id == {c.chunk_id: c.text for c in chunks}
    assert not any(d.startswith("passage: ") for d in by_id.values())
    assert all(e.startswith("passage: ") for e in emb.encoded)
    assert store.collection_metadata["passage_prefix"] == "passage: "
