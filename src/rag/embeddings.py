"""Embeddings: interfaz común, modelo local de sentence-transformers y fake para pruebas."""

import hashlib
import logging
import math
import re
import time
from typing import Any, Protocol

from rag.config import Settings

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"\w+")


class Embedder(Protocol):
    """Convierte textos en vectores normalizados (norma 1 → producto punto = coseno)."""

    @property
    def dimension(self) -> int:
        """Dimensión de los vectores."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vectoriza fragmentos de documentos."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Vectoriza una pregunta."""
        ...


class SentenceTransformerEmbedder:
    """Embeddings locales con sentence-transformers en CPU; el modelo se carga en el primer uso.

    Prefijos opcionales: algunos modelos de recuperación (p. ej. e5) esperan "query: " en las
    preguntas y "passage: " en los documentos. Con prefijos vacíos (default) preguntas y
    documentos se codifican igual, como requiere `paraphrase-multilingual-MiniLM-L12-v2`.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        device: str = "cpu",
        query_prefix: str = "",
        passage_prefix: str = "",
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self._model: Any = None

    @property
    def model(self) -> Any:
        """Modelo cargado (carga perezosa)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            start = time.perf_counter()
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(
                "Modelo de embeddings '%s' cargado en %.1f s",
                self.model_name,
                time.perf_counter() - start,
            )
        return self._model

    @property
    def dimension(self) -> int:
        """Dimensión de los vectores del modelo."""
        return int(self.model.get_embedding_dimension())

    @property
    def max_seq_length(self) -> int:
        """Máximo de tokens que el modelo procesa (el resto se trunca)."""
        return int(self.model.max_seq_length)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """Codifica textos (ya prefijados) en lotes, normalizados."""
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in row] for row in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vectoriza fragmentos (con `passage_prefix`); lista vacía → [] sin cargar el modelo."""
        if not texts:
            return []
        return self._encode([self.passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        """Vectoriza una pregunta (con `query_prefix`)."""
        return self._encode([self.query_prefix + text])[0]


class FakeEmbedder:
    """Embedder determinista para pruebas: bolsa de palabras hasheada con sha1 y normalizada.

    No usa `hash()` de Python (varía entre procesos por PYTHONHASHSEED). Textos que comparten
    palabras tienen mayor similitud coseno.
    """

    def __init__(self, dimension: int = 16) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Dimensión de los vectores."""
        return self._dimension

    def _embed(self, text: str) -> list[float]:
        """Vector normalizado de un texto."""
        vector = [0.0] * self._dimension
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            return [1.0 / math.sqrt(self._dimension)] * self._dimension
        return [x / norm for x in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vectoriza una lista de textos."""
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """Vectoriza una pregunta."""
        return self._embed(text)


def get_embedder(settings: Settings) -> SentenceTransformerEmbedder:
    """Crea el embedder real configurado (sin cargar el modelo todavía)."""
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        query_prefix=settings.embedding_query_prefix,
        passage_prefix=settings.embedding_passage_prefix,
    )
