"""Pruebas de M8: API FastAPI con TestClient, FakeEmbedder, FakeLLM y directorios temporales."""

import shutil
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fpdf import FPDF

from rag.api import create_app
from rag.config import Settings
from rag.embeddings import FakeEmbedder
from rag.ingest import build_store
from rag.llm import FakeLLM, LLMAuthError, LLMError, LLMQuotaExhaustedError, LLMRateLimitError
from rag.loaders import PROJECT_ROOT

pytestmark = pytest.mark.usefixtures("clean_env")

CORPUS = PROJECT_ROOT / "data" / "docs"
TEXT = "La cafetería de la sede principal abre de lunes a viernes de 7:00 a 15:00 horas."


def make_settings(tmp_path: Path, **overrides) -> Settings:
    docs = tmp_path / "docs"
    if not docs.exists():
        shutil.copytree(CORPUS, docs)
    values = {
        "docs_dir": docs,
        "uploads_dir": tmp_path / "uploads",
        "chroma_dir": tmp_path / "chroma",
        "min_score": 0.0,  # FakeEmbedder da similitudes bajas
    }
    return Settings(_env_file=None, **(values | overrides))


def make_client(tmp_path: Path, llm=None, ingest: bool = True, **overrides) -> TestClient:
    settings = make_settings(tmp_path, **overrides)
    app = create_app(
        settings=settings,
        embedder=FakeEmbedder(64),
        llm=llm or FakeLLM(lambda s, u: "Respuesta [1]."),
    )
    client = TestClient(app)
    client.__enter__()  # ejecuta el lifespan
    if ingest:
        assert client.post("/ingest").status_code == 200
    return client


@pytest.fixture
def client(tmp_path: Path):
    c = make_client(tmp_path)
    yield c
    c.__exit__(None, None, None)


def pdf_bytes(text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, text)
    return bytes(pdf.output())


def upload(client: TestClient, *files: tuple[str, bytes]):
    return client.post("/documents", files=[("files", (n, d)) for n, d in files])


# --- M8-01 … M8-10 -----------------------------------------------------------------


def test_health(client: TestClient) -> None:
    """M8-01 + D: /health → 200 con los campos esperados y sin llamar a Gemini."""
    body = client.get("/health").json()
    assert body == {
        "status": "ok",
        "llm_model": "gemini-3.1-flash-lite",
        "llm_configured": False,
        "embedding_model": "intfloat/multilingual-e5-small",
        "chunks": 16,
        "documents": 3,
        "min_score": 0.0,
        "top_k": 4,
    }
    assert client.app.state.engine.llm.calls == []  # FakeLLM: no se llamó


def test_health_llm_configured(tmp_path: Path) -> None:
    """llm_configured refleja si hay GEMINI_API_KEY (sin exponerla)."""
    c = make_client(tmp_path, gemini_api_key="clave-falsa-1234")
    body = c.get("/health").json()
    assert body["llm_configured"] is True and "clave-falsa" not in str(body)


def test_upload_documents(client: TestClient, tmp_path: Path) -> None:
    """M8-02: subir .md y .pdf → 200, chunks > 0 y archivos guardados en data/uploads/."""
    resp = upload(client, ("cafeteria.md", TEXT.encode()), ("horario.pdf", pdf_bytes(TEXT)))
    assert resp.status_code == 200
    report = resp.json()
    assert report["files_processed"] == 2 and report["chunks_added"] > 0
    assert (tmp_path / "uploads" / "cafeteria.md").is_file()
    assert (tmp_path / "uploads" / "horario.pdf").is_file()
    assert not (tmp_path / "docs" / "cafeteria.md").exists()  # el corpus no se toca


@pytest.mark.parametrize("name", ["programa.exe", "informe.docx"])
def test_upload_validation(client: TestClient, name: str) -> None:
    """M8-03: .exe/.docx → 415 UNSUPPORTED_FILE_TYPE."""
    resp = upload(client, (name, b"MZ\x00binario"))
    assert resp.status_code == 415
    assert resp.json()["error"] == "UNSUPPORTED_FILE_TYPE"


def test_upload_too_large(client: TestClient) -> None:
    """M8-03: archivo > 10 MB → 413 FILE_TOO_LARGE."""
    resp = upload(client, ("grande.txt", b"a" * (10 * 1024 * 1024 + 1)))
    assert resp.status_code == 413 and resp.json()["error"] == "FILE_TOO_LARGE"


@pytest.mark.parametrize("raw", ["../../evil.txt", "..\\..\\evil.txt", "sub/../evil.txt"])
def test_path_traversal(client: TestClient, tmp_path: Path, raw: str) -> None:
    """M8-04: nombres con ../ se sanean y nunca se escribe fuera de data/uploads/."""
    resp = upload(client, (raw, TEXT.encode()))
    assert resp.status_code == 200
    assert (tmp_path / "uploads" / "evil.txt").is_file()
    assert not (tmp_path / "evil.txt").exists() and not (tmp_path.parent / "evil.txt").exists()


def test_list_documents(client: TestClient) -> None:
    """M8-05: /documents lista las fuentes con chunks y origen (corpus o subida)."""
    upload(client, ("cafeteria.md", TEXT.encode()))
    docs = {d["source"]: d for d in client.get("/documents").json()}
    assert docs["cafeteria.md"]["origin"] == "upload" and docs["cafeteria.md"]["chunks"] >= 1
    assert docs["guia_onboarding_ti.txt"] == {
        "source": "guia_onboarding_ti.txt",
        "chunks": 6,
        "origin": "corpus",
    }


def test_ask(client: TestClient) -> None:
    """M8-06: /ask → 200 con answer, sources, grounded, context (sin el prompt interno)."""
    resp = client.post("/ask", json={"question": "¿Tope de alimentación?", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Respuesta [1]." and body["grounded"] is True
    assert body["sources"][0]["cited"] is True and body["sources"][0]["index"] == 1
    assert len(body["context"]) == 3 and body["llm_called"] is True
    assert "prompt" not in body


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 1001},
        {},
        {"question": "a", "top_k": 0},
    ],
    ids=["vacia", "espacios", "larga", "sin-campo", "top-k-0"],
)
def test_ask_validation(client: TestClient, payload: dict) -> None:
    """M8-07: pregunta vacía/larga/ausente o top_k inválido → 422 con formato único."""
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert set(body) == {"error", "detail"} and body["error"] == "VALIDATION_ERROR"
    assert body["detail"].startswith("Petición inválida")


def test_llm_error_502(tmp_path: Path) -> None:
    """M8-08: LLMError genérico → 502 LLM_ERROR con detalle legible."""
    c = make_client(tmp_path, llm=FakeLLM([LLMError("Gemini no respondió a tiempo.")]))
    resp = c.post("/ask", json={"question": "¿Algo?"})
    assert resp.status_code == 502
    assert resp.json() == {"error": "LLM_ERROR", "detail": "Gemini no respondió a tiempo."}


def test_delete_document(client: TestClient, tmp_path: Path) -> None:
    """M8-09: DELETE elimina del índice (y de disco si es subida); inexistente → 404."""
    upload(client, ("cafeteria.md", TEXT.encode()))
    resp = client.delete("/documents/cafeteria.md", params={"delete_file": "true"})
    assert resp.status_code == 200
    assert resp.json()["deleted_chunks"] >= 1 and resp.json()["file_deleted"] is True
    assert not (tmp_path / "uploads" / "cafeteria.md").exists()
    resp = client.delete("/documents/no_existe.md")
    assert resp.status_code == 404 and resp.json()["error"] == "DOCUMENT_NOT_FOUND"


def test_openapi(client: TestClient) -> None:
    """M8-10: Swagger /docs y /openapi.json disponibles."""
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json").json()
    assert {"/health", "/documents", "/documents/{source}", "/ingest", "/ask"} <= set(spec["paths"])


# --- A. Mapeo único de errores (incluye M8-11) -------------------------------------


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (LLMQuotaExhaustedError("Se agotaron los créditos…"), 503, "LLM_QUOTA_EXHAUSTED"),
        (LLMAuthError("Key inválida…"), 503, "LLM_AUTH_ERROR"),
        (LLMRateLimitError("Límite…"), 429, "LLM_RATE_LIMIT"),
        (LLMError("Otro error del LLM"), 502, "LLM_ERROR"),
    ],
    ids=["creditos", "auth", "rate-limit", "llm-generico"],
)
def test_llm_specific_errors(tmp_path: Path, error, status: int, code: str) -> None:
    """M8-11 / A: errores del LLM → código HTTP y código propio; 429 con Retry-After: 60."""
    c = make_client(tmp_path, llm=FakeLLM([error]))
    resp = c.post("/ask", json={"question": "¿Algo?"})
    assert resp.status_code == status
    assert resp.json() == {"error": code, "detail": str(error)}
    if status == 429:
        assert resp.headers["retry-after"] == "60"


def test_empty_index_409(tmp_path: Path) -> None:
    """A: índice vacío → 409 EMPTY_INDEX."""
    c = make_client(tmp_path, ingest=False)
    resp = c.post("/ask", json={"question": "¿Algo?"})
    assert resp.status_code == 409 and resp.json()["error"] == "EMPTY_INDEX"
    assert "python -m rag.cli ingest" in resp.json()["detail"]


def test_index_model_mismatch_409(tmp_path: Path) -> None:
    """A: índice creado con otro modelo → 409 INDEX_MODEL_MISMATCH; /health lo informa."""
    settings = make_settings(tmp_path)
    build_store(settings.model_copy(update={"embedding_model": "otro/modelo"}), FakeEmbedder(64))
    c = make_client(tmp_path, ingest=False)
    assert c.get("/health").json()["status"] == "index_model_mismatch"
    for resp in (c.post("/ask", json={"question": "¿Algo?"}), c.get("/documents")):
        assert resp.status_code == 409 and resp.json()["error"] == "INDEX_MODEL_MISMATCH"


def test_unhandled_error_500(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A: excepción no controlada → 500 INTERNAL_ERROR genérico; el detalle solo va al log."""
    settings = make_settings(tmp_path)
    app = create_app(
        settings=settings,
        embedder=FakeEmbedder(64),
        llm=FakeLLM([RuntimeError("secreto-interno ruta /home/x")]),
    )
    with TestClient(app, raise_server_exceptions=False) as c:
        c.post("/ingest")
        resp = c.post("/ask", json={"question": "¿Algo?"})
    assert resp.status_code == 500
    assert resp.json()["error"] == "INTERNAL_ERROR"
    assert "secreto-interno" not in resp.text
    assert "secreto-interno" in caplog.text


def test_unknown_route_format(client: TestClient) -> None:
    """Rutas inexistentes también usan el formato único de error."""
    resp = client.get("/no-existe")
    assert resp.status_code == 404 and resp.json()["error"] == "NOT_FOUND"


# --- B. Subidas fuera del corpus versionado (riesgo H9) -----------------------------


def test_duplicate_corpus_name_409(client: TestClient, tmp_path: Path) -> None:
    """Subir un archivo con el nombre de uno del corpus → 409 DUPLICATE_SOURCE."""
    resp = upload(client, ("guia_onboarding_ti.txt", TEXT.encode()))
    assert resp.status_code == 409 and resp.json()["error"] == "DUPLICATE_SOURCE"
    assert not (tmp_path / "uploads" / "guia_onboarding_ti.txt").exists()


def test_same_upload_name_replaces(client: TestClient) -> None:
    """Subir de nuevo el mismo nombre reemplaza la subida anterior (sin chunks viejos)."""
    upload(client, ("cafeteria.md", TEXT.encode()))
    upload(client, ("cafeteria.md", b"La cafeteria ahora abre de 8:00 a 14:00 los dias habiles."))
    store = client.app.state.store
    texts = [
        r.text for r in store.query("cafeteria", top_k=50) if r.metadata["source"] == "cafeteria.md"
    ]
    assert texts and all("8:00" in t for t in texts)


def test_delete_corpus_file_protected(client: TestClient, tmp_path: Path) -> None:
    """delete_file=true sobre un archivo del corpus → 403 PROTECTED_SOURCE (nada cambia);
    sin delete_file sí se quita del índice."""
    resp = client.delete("/documents/guia_onboarding_ti.txt", params={"delete_file": "true"})
    assert resp.status_code == 403 and resp.json()["error"] == "PROTECTED_SOURCE"
    assert (tmp_path / "docs" / "guia_onboarding_ti.txt").exists()
    assert client.app.state.store.source_stats()["guia_onboarding_ti.txt"] == 6
    resp = client.delete("/documents/guia_onboarding_ti.txt")
    assert resp.status_code == 200 and resp.json()["file_deleted"] is False
    assert (tmp_path / "docs" / "guia_onboarding_ti.txt").exists()


def test_reingest_includes_uploads(client: TestClient) -> None:
    """POST /ingest re-ingiere data/docs + data/uploads (reset opcional)."""
    upload(client, ("cafeteria.md", TEXT.encode()))
    report = client.post("/ingest", json={"reset": True}).json()
    assert "cafeteria.md" in report["sources"] and report["files_processed"] == 4


@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("falso.pdf", b"esto no es un pdf"),
        ("binario.txt", b"texto\x00\x01\x02binario"),
        ("binario.md", bytes(range(256)) * 4),
    ],
    ids=["pdf-sin-cabecera", "txt-con-nulos", "md-binario"],
)
def test_content_validation(client: TestClient, name: str, data: bytes) -> None:
    """Se valida el contenido además de la extensión → 415 INVALID_CONTENT."""
    resp = upload(client, (name, data))
    assert resp.status_code == 415 and resp.json()["error"] == "INVALID_CONTENT"


def test_latin1_text_accepted(client: TestClient) -> None:
    """Un .txt en latin-1 (texto legítimo) se acepta."""
    resp = upload(
        client,
        ("antiguo.txt", "Contraseña de más de doce caracteres en la sede.".encode("latin-1")),
    )
    assert resp.status_code == 200 and resp.json()["files_processed"] == 1


def test_empty_upload_not_kept(client: TestClient, tmp_path: Path) -> None:
    """Una subida sin texto útil se informa como omitida y no queda en data/uploads/."""
    resp = upload(client, ("vacio.txt", b"   "))
    assert resp.status_code == 200 and resp.json()["files_skipped"] == 1
    assert not (tmp_path / "uploads" / "vacio.txt").exists()


# --- C. Arranque: warm-up, Gemini lazy y lock --------------------------------------


class CountingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__(64)
        self.queries = 0

    def embed_query(self, text: str) -> list[float]:
        self.queries += 1
        return super().embed_query(text)


def test_startup_warmup_lazy_llm_and_lock(tmp_path: Path) -> None:
    """El arranque hace warm-up del embedder, no crea Gemini y prepara un threading.Lock."""
    embedder = CountingEmbedder()
    app = create_app(settings=make_settings(tmp_path, min_score=0.99), embedder=embedder)
    with TestClient(app) as c:
        assert embedder.queries == 1  # warm-up antes de cualquier petición
        assert isinstance(app.state.write_lock, type(threading.Lock()))
        c.post("/ingest")
        # Sin key: la pregunta cortada por el umbral no necesita a Gemini → 200
        resp = c.post("/ask", json={"question": "¿Capital de Francia?"})
        assert resp.status_code == 200 and resp.json()["llm_called"] is False
        # Con contexto sí lo necesita: sin key → 503 LLM_AUTH_ERROR
        app.state.settings = app.state.settings.model_copy(update={"min_score": 0.0})
        app.state.engine.settings = app.state.settings
        resp = c.post("/ask", json={"question": "¿Tope de alimentación?"})
        assert resp.status_code == 503 and resp.json()["error"] == "LLM_AUTH_ERROR"


def test_not_grounded_has_no_sources(tmp_path: Path) -> None:
    """Respuesta "no encontré" → sources vacía; context conserva lo recuperado."""
    from rag.prompts import NOT_FOUND_MESSAGE

    c = make_client(tmp_path, llm=FakeLLM([NOT_FOUND_MESSAGE]))
    body = c.post("/ask", json={"question": "¿Trabajo remoto?"}).json()
    assert body["grounded"] is False
    assert body["sources"] == [] and len(body["context"]) == 4
