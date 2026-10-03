"""Esquemas Pydantic de request/response de la API."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class AskRequest(BaseModel):
    """Pregunta (1–1.000 caracteres, sin contar espacios extremos) y top_k opcional."""

    question: Question
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceOut(BaseModel):
    """Fragmento usado como contexto, con su número [n] en el prompt."""

    source: str
    page: int | None
    chunk_id: str
    score: float
    excerpt: str
    cited: bool
    index: int


class AskResponse(BaseModel):
    """Respuesta del motor RAG."""

    question: str
    answer: str
    grounded: bool
    sources: list[SourceOut]
    context: list[SourceOut]
    model: str | None
    latency_s: float
    retrieved: int
    llm_called: bool


class HealthOut(BaseModel):
    """Estado del servicio (no llama a Gemini)."""

    status: Literal["ok", "index_model_mismatch"]
    llm_model: str
    llm_configured: bool
    embedding_model: str
    chunks: int | None
    documents: int | None
    min_score: float
    top_k: int


class DocumentOut(BaseModel):
    """Fuente indexada con su nº de chunks y su origen."""

    source: str
    chunks: int
    origin: Literal["corpus", "upload", "desconocido"]


class IngestRequest(BaseModel):
    """Opciones de re-ingesta."""

    reset: bool = False


class IngestReportOut(BaseModel):
    """Reporte de ingesta."""

    files_processed: int
    files_skipped: int
    documents: int
    chunks_added: int
    total_in_store: int
    sources: list[str]
    skipped: list[str]
    duration_s: float


class DeleteOut(BaseModel):
    """Resultado de eliminar un documento."""

    source: str
    deleted_chunks: int
    file_deleted: bool


class ErrorOut(BaseModel):
    """Formato único de error."""

    error: str
    detail: str
