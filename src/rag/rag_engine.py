"""Motor RAG: recupera contexto, filtra por umbral, genera con el LLM y mapea citas."""

import re
import time
import unicodedata
from dataclasses import dataclass, field

from rag.config import Settings
from rag.llm import LLMClient
from rag.models import RetrievedChunk
from rag.prompts import NOT_FOUND_MESSAGE, SYSTEM_PROMPT, build_user_prompt
from rag.vectorstore import ChromaVectorStore

MAX_QUESTION_CHARS = 1000
EXCERPT_CHARS = 300
NOT_FOUND_CORE = "no encontre informacion"
EMPTY_INDEX_MESSAGE = "El índice está vacío. Ejecuta python -m rag.cli ingest data/docs."

_CITATION = re.compile(r"\[(\s*\d+\s*(?:[,\-–]\s*\d+\s*)*)\]")


class EmptyIndexError(ValueError):
    """No hay documentos indexados."""


@dataclass
class SourceRef:
    """Fragmento usado como contexto, con su posición `[n]` en el prompt."""

    source: str
    page: int | None
    chunk_id: str
    score: float
    excerpt: str
    cited: bool
    index: int


@dataclass
class RAGAnswer:
    """Respuesta del motor.

    `context`: todos los fragmentos que pasaron el filtro (con `cited`); `sources`: solo los
    citados o, si no hubo citas, todo el contexto con `cited=False`; `retrieved`: nº de
    fragmentos devueltos por el índice antes del filtro `MIN_SCORE`.
    """

    question: str
    answer: str
    grounded: bool
    sources: list[SourceRef] = field(default_factory=list)
    context: list[SourceRef] = field(default_factory=list)
    model: str | None = None
    latency_s: float = 0.0
    retrieved: int = 0
    llm_called: bool = False
    prompt: str | None = None  # prompt de usuario enviado al LLM (para --show-context)


def normalize(text: str) -> str:
    """Minúsculas, sin tildes y con espacios colapsados."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    without_accents = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return " ".join(without_accents.split())


def parse_citations(answer: str, n_chunks: int) -> set[int]:
    """Índices citados válidos (1..n): `[1]`, `[1][3]`, `[1, 3]`, `[1,3]`, `[1-3]`."""
    cited: set[int] = set()
    for match in _CITATION.finditer(answer):
        for part in match.group(1).split(","):
            bounds = re.split(r"[\-–]", part)
            numbers = [int(b) for b in bounds if b.strip()]
            if len(numbers) == 2 and numbers[0] <= numbers[1]:
                cited.update(range(numbers[0], numbers[1] + 1))
            elif len(numbers) == 1:
                cited.add(numbers[0])
    return {i for i in cited if 1 <= i <= n_chunks}


def is_not_found(answer: str, cited: set[int]) -> bool:
    """La respuesta es de "no encontrado": contiene el núcleo de la frase y no cita nada."""
    return NOT_FOUND_CORE in normalize(answer) and not cited


def validate_question(question: str) -> str:
    """Pregunta no vacía y de máximo 1.000 caracteres."""
    question = (question or "").strip()
    if not question:
        raise ValueError("La pregunta no puede estar vacía.")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"La pregunta supera el máximo de {MAX_QUESTION_CHARS} caracteres.")
    return question


def _to_ref(index: int, chunk: RetrievedChunk, cited: bool) -> SourceRef:
    return SourceRef(
        source=str(chunk.metadata.get("source", "")),
        page=chunk.metadata.get("page"),
        chunk_id=chunk.chunk_id,
        score=chunk.score,
        excerpt=chunk.text[:EXCERPT_CHARS],
        cited=cited,
        index=index,
    )


class RAGEngine:
    """Pregunta → recuperación → filtro por umbral → LLM → respuesta con citas."""

    def __init__(self, store: ChromaVectorStore, llm: LLMClient, settings: Settings) -> None:
        self.store = store
        self.llm = llm
        self.settings = settings

    def retrieve(self, question: str, top_k: int) -> tuple[list[RetrievedChunk], int]:
        """Fragmentos con score ≥ MIN_SCORE y nº de recuperados antes del filtro."""
        results = self.store.query(question, top_k=top_k)
        return [r for r in results if r.score >= self.settings.min_score], len(results)

    def ask(self, question: str, top_k: int | None = None) -> RAGAnswer:
        """Responde solo con los documentos; sin contexto relevante no llama al LLM."""
        start = time.perf_counter()
        question = validate_question(question)
        if self.store.count() == 0:
            raise EmptyIndexError(EMPTY_INDEX_MESSAGE)

        chunks, retrieved = self.retrieve(question, top_k or self.settings.top_k)
        if not chunks:
            return RAGAnswer(
                question=question,
                answer=NOT_FOUND_MESSAGE,
                grounded=False,
                latency_s=time.perf_counter() - start,
                retrieved=retrieved,
            )

        user_prompt = build_user_prompt(question, chunks)
        result = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        cited = parse_citations(result.text, len(chunks))
        context = [_to_ref(i, c, i in cited) for i, c in enumerate(chunks, start=1)]
        sources = [ref for ref in context if ref.cited] or context
        return RAGAnswer(
            question=question,
            answer=result.text,
            grounded=not is_not_found(result.text, cited),
            sources=sources,
            context=context,
            model=result.model,
            latency_s=time.perf_counter() - start,
            retrieved=retrieved,
            llm_called=True,
            prompt=user_prompt,
        )
