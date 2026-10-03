"""API REST (FastAPI) del asistente documental RAG.

Ejecutar con un solo worker (Chroma local no admite escritores concurrentes entre procesos):
    uvicorn rag.api:app --workers 1
Swagger en /docs. Errores con formato único {"error": "<CODIGO>", "detail": "<mensaje>"}.
"""

import logging
import threading
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from rag.config import Settings, get_settings, log_effective_settings
from rag.embeddings import Embedder, get_embedder
from rag.ingest import IngestReport, build_store, ingest_paths
from rag.llm import (
    LazyLLM,
    LLMAuthError,
    LLMClient,
    LLMError,
    LLMQuotaExhaustedError,
    LLMRateLimitError,
    get_llm,
)
from rag.loaders import SUPPORTED_EXTENSIONS
from rag.logging_conf import setup_logging
from rag.rag_engine import EmptyIndexError, RAGEngine
from rag.schemas import (
    AskRequest,
    AskResponse,
    DeleteOut,
    DocumentOut,
    ErrorOut,
    HealthOut,
    IngestReportOut,
    IngestRequest,
)
from rag.vectorstore import ChromaVectorStore, EmbeddingModelMismatchError

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
CORS_ORIGINS = ["http://localhost:8501", "http://127.0.0.1:8501"]
INTERNAL_ERROR_MESSAGE = "Error interno del servidor. Revisa los logs de la API."
ERROR_RESPONSES = {
    code: {"model": ErrorOut} for code in (400, 403, 404, 409, 413, 415, 422, 429, 500, 502, 503)
}


class APIError(Exception):
    """Error HTTP con código propio y mensaje en español."""

    def __init__(
        self, status: int, code: str, detail: str, headers: dict[str, str] | None = None
    ) -> None:
        super().__init__(detail)
        self.status, self.code, self.detail, self.headers = status, code, detail, headers


def _error(status: int, code: str, detail: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": code, "detail": detail}, headers=headers
    )


# Mapeo de excepciones del dominio → (HTTP, código, cabeceras). El orden importa (más específicas
# primero) porque se recorre con isinstance.
EXCEPTION_MAP: list[tuple[type[Exception], int, str, dict[str, str] | None]] = [
    (EmptyIndexError, 409, "EMPTY_INDEX", None),
    (EmbeddingModelMismatchError, 409, "INDEX_MODEL_MISMATCH", None),
    (LLMQuotaExhaustedError, 503, "LLM_QUOTA_EXHAUSTED", None),
    (LLMAuthError, 503, "LLM_AUTH_ERROR", None),
    (LLMRateLimitError, 429, "LLM_RATE_LIMIT", {"Retry-After": "60"}),
    (LLMError, 502, "LLM_ERROR", None),
    (ValueError, 400, "BAD_REQUEST", None),
]


def map_exception(exc: Exception) -> JSONResponse:
    """Traduce una excepción del dominio a la respuesta JSON de error."""
    for exc_type, status, code, headers in EXCEPTION_MAP:
        if isinstance(exc, exc_type):
            return _error(status, code, str(exc), headers)
    logger.exception("Error no controlado", exc_info=exc)
    return _error(500, "INTERNAL_ERROR", INTERNAL_ERROR_MESSAGE)


def _validation_message(exc: RequestValidationError) -> str:
    """Mensaje en español a partir de los errores de validación de Pydantic."""
    parts = []
    for err in exc.errors():
        field = ".".join(str(loc) for loc in err.get("loc", ()) if loc not in ("body", "query"))
        kind = err.get("type", "")
        ctx = err.get("ctx") or {}
        if kind == "string_too_short":
            msg = "no puede estar vacío"
        elif kind == "string_too_long":
            msg = f"supera el máximo de {ctx.get('max_length')} caracteres"
        elif kind == "missing":
            msg = "es obligatorio"
        elif kind in ("greater_than_equal", "less_than_equal"):
            msg = f"fuera de rango ({err.get('msg')})"
        else:
            msg = f"valor inválido ({err.get('msg')})"
        parts.append(f"'{field or 'cuerpo'}' {msg}")
    return "Petición inválida: " + "; ".join(parts) + "."


def _is_text(data: bytes) -> bool:
    """True si los bytes son texto: sin bytes nulos y decodificables (UTF-8 o latin-1 limpio)."""
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        text = data.decode("latin-1")
        controls = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\r\t")
        return controls <= 0.01 * max(len(text), 1)


def _validate_upload(name: str, data: bytes) -> None:
    """Extensión soportada, tamaño ≤ 10 MB y contenido coherente con la extensión."""
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise APIError(
            415,
            "UNSUPPORTED_FILE_TYPE",
            f"Tipo de archivo no soportado: '{name}'. Usa .txt, .md o .pdf.",
        )
    if len(data) > MAX_UPLOAD_BYTES:
        raise APIError(413, "FILE_TOO_LARGE", f"'{name}' supera el máximo de 10 MB.")
    if suffix == ".pdf" and not data.startswith(b"%PDF"):
        raise APIError(415, "INVALID_CONTENT", f"'{name}' no es un PDF válido.")
    if suffix != ".pdf" and not _is_text(data):
        raise APIError(415, "INVALID_CONTENT", f"'{name}' no contiene texto legible.")


def _safe_name(raw: str | None) -> str:
    """Nombre de archivo sin rutas (evita path traversal); rechaza vacíos u ocultos."""
    name = Path((raw or "").replace("\\", "/")).name
    if not name or name.startswith(".") or name in ("..",):
        raise APIError(400, "INVALID_FILENAME", f"Nombre de archivo inválido: '{raw}'.")
    return name


def _origin(settings: Settings, source: str) -> str:
    if (settings.docs_dir / source).is_file():
        return "corpus"
    if (settings.uploads_dir / source).is_file():
        return "upload"
    return "desconocido"


def _report_out(report: IngestReport) -> IngestReportOut:
    data = asdict(report)
    data.pop("skipped_files", None)
    return IngestReportOut(**data)


def get_store(request: Request) -> ChromaVectorStore:
    """Store abierto en el arranque; si el índice es de otro modelo, lanza el error (409)."""
    if request.app.state.store_error is not None:
        raise request.app.state.store_error
    return request.app.state.store


def get_cfg(request: Request) -> Settings:
    """Configuración efectiva de la app."""
    return request.app.state.settings


StoreDep = Annotated[ChromaVectorStore, Depends(get_store)]
SettingsDep = Annotated[Settings, Depends(get_cfg)]
UploadFiles = Annotated[
    list[UploadFile], File(description="Archivos .txt, .md o .pdf (≤ 10 MB cada uno)")
]


def create_app(
    engine: RAGEngine | None = None,
    store: ChromaVectorStore | None = None,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
) -> FastAPI:
    """Crea la app; las dependencias pesadas se construyen una sola vez en el arranque."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Iterator[None]:
        cfg = settings or (engine.settings if engine else get_settings())
        if not logging.getLogger().handlers:  # bajo uvicorn; en pruebas lo maneja pytest
            setup_logging(cfg.log_level)
        log_effective_settings(cfg)
        app.state.settings = cfg
        app.state.write_lock = threading.Lock()
        app.state.store_error = None
        app.state.store = engine.store if engine else store
        if app.state.store is None:
            try:
                app.state.store = build_store(cfg, embedder or get_embedder(cfg))
            except EmbeddingModelMismatchError as exc:
                logger.error("Índice incompatible: %s", exc)
                app.state.store_error = exc
        if app.state.store is not None:
            app.state.store.embedder.embed_query("calentamiento del modelo")  # warm-up
            app.state.engine = engine or RAGEngine(
                app.state.store, llm or LazyLLM(lambda: get_llm(cfg)), cfg
            )
        cfg.uploads_dir.mkdir(parents=True, exist_ok=True)
        logger.info("API lista (1 worker; escrituras serializadas con un lock)")
        yield

    app = FastAPI(
        title="Asistente documental RAG",
        description="Responde preguntas solo con los documentos internos, citando las fuentes.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"]
    )

    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: APIError) -> JSONResponse:
        return _error(exc.status, exc.code, exc.detail, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(422, "VALIDATION_ERROR", _validation_message(exc))

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = "NOT_FOUND" if exc.status_code == 404 else f"HTTP_{exc.status_code}"
        return _error(exc.status_code, code, str(exc.detail))

    async def _domain(_: Request, exc: Exception) -> JSONResponse:
        return map_exception(exc)

    # Excepciones del dominio: se registran por clase (las maneja ExceptionMiddleware); el
    # handler de Exception solo cubre lo no controlado (500 genérico, detalle solo en el log).
    for exc_type, *_ in EXCEPTION_MAP:
        app.add_exception_handler(exc_type, _domain)
    app.add_exception_handler(Exception, _domain)

    @app.get("/health", response_model=HealthOut, tags=["estado"])
    def health(request: Request) -> HealthOut:
        """Estado del servicio, modelos configurados y tamaño del índice (no llama a Gemini)."""
        cfg: Settings = request.app.state.settings
        store_ = request.app.state.store
        key = cfg.gemini_api_key
        return HealthOut(
            status="ok" if request.app.state.store_error is None else "index_model_mismatch",
            llm_model=cfg.gemini_model,
            llm_configured=bool(key and key.get_secret_value()),
            embedding_model=cfg.embedding_model,
            chunks=store_.count() if store_ else None,
            documents=len(store_.source_stats()) if store_ else None,
            min_score=cfg.min_score,
            top_k=cfg.top_k,
        )

    @app.get("/documents", response_model=list[DocumentOut], tags=["documentos"])
    def list_documents(store_: StoreDep, cfg: SettingsDep) -> list[DocumentOut]:
        """Fuentes indexadas con su nº de chunks y su origen (corpus o subida)."""
        return [
            DocumentOut(source=s, chunks=n, origin=_origin(cfg, s))
            for s, n in store_.source_stats().items()
        ]

    @app.post(
        "/documents", response_model=IngestReportOut, responses=ERROR_RESPONSES, tags=["documentos"]
    )
    def upload_documents(
        request: Request,
        files: UploadFiles,
        store_: StoreDep,
        cfg: SettingsDep,
    ) -> IngestReportOut:
        """Sube 1..n archivos a data/uploads/ y los ingiere (una subida previa con el mismo nombre
        se reemplaza; un nombre del corpus devuelve 409)."""
        validated: dict[str, bytes] = {}
        for upload in files:
            name = _safe_name(upload.filename)
            data = upload.file.read(MAX_UPLOAD_BYTES + 1)
            _validate_upload(name, data)
            if (cfg.docs_dir / name).exists():
                raise APIError(
                    409,
                    "DUPLICATE_SOURCE",
                    f"'{name}' ya existe en el corpus (data/docs). Renombra el archivo.",
                )
            if name in validated:
                raise APIError(409, "DUPLICATE_SOURCE", f"'{name}' viene repetido en la petición.")
            validated[name] = data

        with request.app.state.write_lock:
            cfg.uploads_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for name, data in validated.items():
                path = cfg.uploads_dir / name
                path.write_bytes(data)
                paths.append(path)
            report = ingest_paths(paths, store_, cfg)
            for name in report.skipped_files:  # no conservar subidas sin texto útil
                (cfg.uploads_dir / name).unlink(missing_ok=True)
        return _report_out(report)

    @app.post(
        "/ingest", response_model=IngestReportOut, responses=ERROR_RESPONSES, tags=["documentos"]
    )
    def reingest(
        request: Request,
        store_: StoreDep,
        cfg: SettingsDep,
        body: IngestRequest | None = None,
    ) -> IngestReportOut:
        """Re-ingiere data/docs + data/uploads (con {"reset": true} vacía antes el índice)."""
        with request.app.state.write_lock:
            if body and body.reset:
                store_.reset()
            paths = [cfg.docs_dir] + ([cfg.uploads_dir] if cfg.uploads_dir.is_dir() else [])
            report = ingest_paths(paths, store_, cfg)
        return _report_out(report)

    @app.delete(
        "/documents/{source}",
        response_model=DeleteOut,
        responses=ERROR_RESPONSES,
        tags=["documentos"],
    )
    def delete_document(
        request: Request,
        source: str,
        store_: StoreDep,
        cfg: SettingsDep,
        delete_file: bool = False,
    ) -> DeleteOut:
        """Quita un documento del índice; con ?delete_file=true borra también el archivo, solo si
        es una subida (los del corpus son protegidos → 403)."""
        name = _safe_name(source)
        if name != source:
            raise APIError(400, "INVALID_FILENAME", f"Nombre de documento inválido: '{source}'.")
        if delete_file and (cfg.docs_dir / name).exists():
            raise APIError(
                403,
                "PROTECTED_SOURCE",
                f"'{name}' pertenece al corpus (data/docs) y no se puede borrar del "
                "disco. Puedes quitarlo del índice sin delete_file.",
            )
        with request.app.state.write_lock:
            deleted = store_.delete_by_source(name)
            file_deleted = False
            upload = cfg.uploads_dir / name
            if delete_file and upload.is_file():
                upload.unlink()
                file_deleted = True
        if deleted == 0 and not file_deleted:
            raise APIError(404, "DOCUMENT_NOT_FOUND", f"No existe el documento '{name}'.")
        return DeleteOut(source=name, deleted_chunks=deleted, file_deleted=file_deleted)

    @app.post("/ask", response_model=AskResponse, responses=ERROR_RESPONSES, tags=["preguntas"])
    def ask(request: Request, body: AskRequest, _store: StoreDep) -> AskResponse:
        """Responde solo con los documentos, citando fragmentos [n]."""
        answer = request.app.state.engine.ask(body.question, top_k=body.top_k)
        data = asdict(answer)
        data.pop("prompt", None)
        return AskResponse(**data)

    return app


app = create_app()
