"""Vector store local persistente con ChromaDB (métrica coseno, embeddings propios)."""

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from rag.embeddings import Embedder
from rag.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

_PRIMITIVES = (str, int, float, bool)


class EmbeddingModelMismatchError(RuntimeError):
    """La colección existente se creó con otro modelo de embeddings o dimensión."""


def _client(persist_dir: str | Path) -> Any:
    """Cliente persistente de Chroma sin telemetría (evita ruido y llamadas de red)."""
    return chromadb.PersistentClient(
        path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
    )


def reset_collection(persist_dir: str | Path, collection_name: str) -> None:
    """Elimina la colección (si existe) sin validar el modelo; útil tras un cambio de modelo."""
    try:
        _client(persist_dir).delete_collection(collection_name)
    except NotFoundError:
        pass


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Chroma solo admite valores primitivos y descarta la metadata completa si hay `None`."""
    return {
        k: (v if isinstance(v, _PRIMITIVES) else str(v))
        for k, v in metadata.items()
        if v is not None
    }


class ChromaVectorStore:
    """Guarda chunks + embeddings + metadatos y consulta por similitud coseno.

    Los embeddings se calculan con el `embedder` inyectado (no con la función de Chroma). La
    colección guarda en su metadata el modelo, la dimensión y el prefijo de pasaje; al abrir
    una colección existente se valida que el modelo y la dimensión coincidan.
    """

    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: str,
        embedder: Embedder,
        embedding_model: str | None = None,
    ) -> None:
        self.embedder = embedder
        self.embedding_model = embedding_model or getattr(
            embedder, "model_name", type(embedder).__name__
        )
        self.collection_name = collection_name
        self._client = _client(persist_dir)
        self._collection = self._open_collection()

    def _expected_metadata(self) -> dict[str, Any]:
        """Metadata que identifica cómo se generaron los vectores de la colección."""
        return {
            "hnsw:space": "cosine",
            "embedding_model": self.embedding_model,
            "embedding_dim": int(self.embedder.dimension),
            "passage_prefix": getattr(self.embedder, "passage_prefix", ""),
        }

    def _open_collection(self) -> Any:
        """Abre la colección validando modelo y dimensión, o la crea si no existe."""
        expected = self._expected_metadata()
        try:
            collection = self._client.get_collection(self.collection_name)
        except NotFoundError:
            return self._client.create_collection(
                self.collection_name, metadata=expected, embedding_function=None
            )
        found = collection.metadata or {}
        for key in ("embedding_model", "embedding_dim"):
            if found.get(key) != expected[key]:
                raise EmbeddingModelMismatchError(
                    f"La colección '{self.collection_name}' se creó con "
                    f"{key}={found.get(key)!r}, pero la configuración actual usa "
                    f"{expected[key]!r}. Ejecuta el reset del índice (p. ej. "
                    f"`python -m rag.cli reset --yes`, o `reset_collection()`) y vuelve a "
                    "ingerir los documentos."
                )
        return collection

    @property
    def collection_metadata(self) -> dict[str, Any]:
        """Metadata guardada en la colección."""
        return dict(self._collection.metadata or {})

    def count(self) -> int:
        """Número de chunks guardados."""
        return int(self._collection.count())

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Inserta o actualiza (upsert por `chunk_id`) en lotes; devuelve cuántos procesó."""
        if not chunks:
            return 0
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        batch = int(self._client.get_max_batch_size())
        for start in range(0, len(chunks), batch):
            part = chunks[start : start + batch]
            self._collection.upsert(
                ids=[c.chunk_id for c in part],
                embeddings=embeddings[start : start + batch],
                documents=[c.text for c in part],  # texto sin prefijo
                metadatas=[_clean_metadata(c.metadata) for c in part],
            )
        logger.info("Upsert de %d chunks en '%s'", len(chunks), self.collection_name)
        return len(chunks)

    def query(self, text: str, top_k: int) -> list[RetrievedChunk]:
        """Los `top_k` chunks más similares (score = 1 − distancia coseno, en [0, 1])."""
        if top_k < 1:
            raise ValueError("top_k debe ser ≥ 1")
        total = self.count()
        if total == 0:
            return []
        result = self._collection.query(
            query_embeddings=[self.embedder.embed_query(text)],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )
        items = zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        )
        retrieved = [
            RetrievedChunk(
                chunk_id=cid,
                text=doc,
                metadata=dict(meta or {}),
                score=min(1.0, max(0.0, 1.0 - float(dist))),
            )
            for cid, doc, meta, dist in items
        ]
        return sorted(retrieved, key=lambda r: r.score, reverse=True)

    def delete_by_source(self, source: str) -> int:
        """Elimina todos los chunks de una fuente; devuelve cuántos borró."""
        ids = self._collection.get(where={"source": source}, include=[])["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def reset(self) -> None:
        """Vacía la colección (la elimina y la vuelve a crear con la metadata actual)."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            self.collection_name, metadata=self._expected_metadata(), embedding_function=None
        )

    def source_stats(self) -> dict[str, int]:
        """Chunks por fuente, ordenado por nombre de fuente."""
        metadatas = self._collection.get(include=["metadatas"])["metadatas"] or []
        counts = Counter(str((m or {}).get("source", "")) for m in metadatas)
        return dict(sorted(counts.items()))

    def list_sources(self) -> list[str]:
        """Fuentes indexadas (derivado de `source_stats`)."""
        return list(self.source_stats())
