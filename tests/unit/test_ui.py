"""Pruebas de M9: UI Streamlit con AppTest y un cliente de API falso (sin red ni Gemini)."""

import pytest
from streamlit.testing.v1 import AppTest

from rag import api_client
from rag.api_client import APIClientError, UploadResult
from rag.loaders import PROJECT_ROOT
from rag.ui_streamlit import (
    EXAMPLES,
    NO_MEMORY_CAPTION,
    SPINNER_TEXT,
    error_level,
    error_text,
    upload_messages,
)

APP = str(PROJECT_ROOT / "src" / "rag" / "ui_streamlit.py")
Q2 = EXAMPLES[0][1]

HEALTH = {
    "status": "ok",
    "llm_model": "gemini-3.1-flash-lite",
    "llm_configured": True,
    "embedding_model": "intfloat/multilingual-e5-small",
    "chunks": 17,
    "documents": 4,
    "min_score": 0.805,
    "top_k": 4,
}
DOCS = [
    {"source": "guia_onboarding_ti.txt", "chunks": 6, "origin": "corpus"},
    {"source": "horario_cafeteria.txt", "chunks": 1, "origin": "upload"},
]
SOURCE = {
    "source": "manual_reembolso_gastos.pdf",
    "page": 1,
    "chunk_id": "a1",
    "score": 0.8667,
    "excerpt": "Alimentación: COP 120.000 por día.",
    "cited": True,
    "index": 1,
}
UNCITED = SOURCE | {
    "chunk_id": "b2",
    "page": None,
    "source": "politica.md",
    "cited": False,
    "index": 2,
    "excerpt": "Vacaciones.",
}
GROUNDED = {
    "question": Q2,
    "answer": "El tope es COP 120.000 [1].",
    "grounded": True,
    "sources": [SOURCE],
    "context": [SOURCE, UNCITED],
    "model": "gemini-3.1-flash-lite",
    "latency_s": 6.54,
    "retrieved": 4,
    "llm_called": True,
}
NOT_FOUND = GROUNDED | {
    "answer": "No encontré información sobre eso en los documentos cargados.",
    "grounded": False,
    "sources": [],
}


class FakeClient:
    """Cliente de API falso: respuestas programadas y registro de llamadas."""

    def __init__(self, ask=GROUNDED, health=HEALTH, docs=DOCS) -> None:
        self._ask, self._health, self._docs = ask, health, docs
        self.asked: list[str] = []
        self.deleted: list[str] = []
        self.ingested = 0

    @staticmethod
    def _maybe_raise(value):
        if isinstance(value, Exception):
            raise value
        return value

    def health(self):
        return self._maybe_raise(self._health)

    def list_documents(self):
        return self._maybe_raise(self._docs)

    def ask(self, question, top_k=None):
        self.asked.append(question)
        return self._maybe_raise(self._ask)

    def ingest(self, reset=False):
        self.ingested += 1
        return {"total_in_store": 16}

    def delete_document(self, source, delete_file=True):
        self.deleted.append(source)
        return {"source": source, "deleted_chunks": 1, "file_deleted": True}

    def upload(self, files):
        return []


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch):
    def install(**kwargs) -> FakeClient:
        client = FakeClient(**kwargs)
        monkeypatch.setattr(api_client, "get_client", lambda *a, **k: client)
        return client

    return install


def run_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=15).run()


def ask(at: AppTest, question: str = Q2) -> AppTest:
    return at.chat_input[0].set_value(question).run()


def markdown_text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


# --- M9-02 … M9-04 ------------------------------------------------------------------


def test_app_renders(fake) -> None:
    """M9-02: la app renderiza sin excepciones con la API mockeada."""
    fake()
    at = run_app()
    assert not at.exception
    assert at.title[0].value == "Asistente documental"
    assert NO_MEMORY_CAPTION in [c.value for c in at.caption]
    assert "API conectada · gemini-3.1-flash-lite" in at.sidebar.success[0].value
    assert {b.key for b in at.button} >= {
        "ex_contestable",
        "ex_parcial",
        "ex_sin_respuesta",
        "clear",
    }
    assert at.sidebar.toggle(key="show_context").value is False


def test_app_ask_flow(fake) -> None:
    """M9-03: al preguntar se muestra la respuesta, el badge verde y el expander de fuentes."""
    client = fake()
    at = ask(run_app())
    assert client.asked == [Q2]
    text = markdown_text(at)
    assert "El tope es COP 120.000 [1]." in text
    assert ":green-badge[:material/check: Basada en documentos]" in text
    assert [e.label for e in at.expander] == ["Fuentes citadas"]
    assert "**[1] manual_reembolso_gastos.pdf, pág. 1** · score 0.867" in text
    assert any("gemini-3.1-flash-lite · 6.5 s" == c.value for c in at.caption)
    assert not at.exception


def test_app_api_down(fake) -> None:
    """M9-04: API caída → aviso con el comando para levantarla, sin traceback."""
    fake(
        health=APIClientError(
            "API_UNAVAILABLE",
            "No se pudo conectar con la API en "
            "http://localhost:8011. Levántala con: uvicorn rag.api:app "
            "--workers 1 --port 8011",
        )
    )
    at = run_app()
    assert not at.exception
    assert "uvicorn rag.api:app --workers 1 --port 8011" in at.sidebar.error[0].value


# --- C. Respuestas --------------------------------------------------------------------


def test_not_grounded_has_no_sources_expander(fake) -> None:
    """grounded=False → badge gris y SIN expander de fuentes."""
    fake(ask=NOT_FOUND)
    at = ask(run_app())
    assert ":gray-badge[Sin información en los documentos]" in markdown_text(at)
    assert "Fuentes citadas" not in [e.label for e in at.expander]


def test_context_toggle(fake) -> None:
    """El toggle de depuración (apagado por defecto) muestra el contexto con citado/no citado."""
    fake()
    at = ask(run_app())
    assert "Contexto recuperado (depuración)" not in [e.label for e in at.expander]
    at = at.sidebar.toggle(key="show_context").set_value(True).run()
    assert "Contexto recuperado (depuración)" in [e.label for e in at.expander]
    text = markdown_text(at)
    assert "✅ citado" in text and "▫️ no citado" in text


def test_no_llm_call_caption(fake) -> None:
    """Si la pregunta se cortó por el umbral, el pie lo indica."""
    fake(ask=NOT_FOUND | {"llm_called": False, "model": None, "latency_s": 0.04})
    at = ask(run_app())
    assert any("sin llamada al LLM" in c.value for c in at.caption)


# --- D/E. Historial y ejemplos ----------------------------------------------------------


def test_example_buttons_and_clear(fake) -> None:
    """Los botones de ejemplo envían Q2/Q4/Q8; "Limpiar conversación" vacía el historial."""
    client = fake()
    at = run_app()
    for _, _question, key in EXAMPLES:
        at = at.button(key=key).click().run()
    assert client.asked == [q for _, q, _ in EXAMPLES]
    assert len(at.chat_message) == 6  # 3 preguntas + 3 respuestas
    at = at.button(key="clear").click().run()
    assert len(at.chat_message) == 0


# --- B. Errores por código -------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "detail", "kind"),
    [
        (
            "LLM_QUOTA_EXHAUSTED",
            "Se agotaron los créditos de la API key de Gemini. Cambia "
            "GEMINI_API_KEY en el archivo .env y reinicia la aplicación.",
            "error",
        ),
        ("LLM_AUTH_ERROR", "La API key de Gemini no es válida o fue revocada.", "error"),
        ("LLM_RATE_LIMIT", "Límite de solicitudes de Gemini alcanzado.", "warning"),
        ("INDEX_MODEL_MISMATCH", "La colección se creó con otro modelo.", "error"),
        ("API_UNAVAILABLE", "No se pudo conectar con la API. Levántala con: uvicorn …", "error"),
        ("LLM_ERROR", "Gemini respondió con un error.", "error"),
    ],
    ids=["creditos", "auth", "rate-limit", "mismatch", "api-caida", "llm-generico"],
)
def test_error_codes(fake, code: str, detail: str, kind: str) -> None:
    """Cada código de error se muestra con su tipo de aviso y sin traceback."""
    fake(ask=APIClientError(code, detail))
    at = ask(run_app())
    assert not at.exception
    shown = [e.value for e in getattr(at, kind)]
    assert any(detail in s for s in shown), shown
    if code == "INDEX_MODEL_MISMATCH":
        assert any("rag.cli reset --yes" in s for s in shown)


def test_empty_index_offers_ingest(fake) -> None:
    """EMPTY_INDEX → st.info con botón "Indexar documentos" que llama POST /ingest."""
    client = fake(ask=APIClientError("EMPTY_INDEX", "El índice está vacío."))
    at = ask(run_app())
    assert any("El índice está vacío." in i.value for i in at.info)
    button = at.button(key="ingest_1")
    assert button.label == "Indexar documentos"
    at = button.click().run()
    assert client.ingested == 1
    assert any("Índice creado: 16 fragmentos" in s.value for s in at.success)


# --- F. Documentos ----------------------------------------------------------------------


def test_delete_only_for_uploads(fake) -> None:
    """Solo las subidas tienen botón eliminar; las del corpus llevan candado y no se borran."""
    client = fake()
    at = run_app()
    keys = {b.key for b in at.sidebar.button}
    assert "del_horario_cafeteria.txt" in keys
    assert "del_guia_onboarding_ti.txt" not in keys
    assert any("🔒 guia_onboarding_ti.txt" in m.value for m in at.sidebar.markdown)
    at.sidebar.button(key="del_horario_cafeteria.txt").click().run()
    assert client.deleted == ["horario_cafeteria.txt"]


# --- Lógica pura ----------------------------------------------------------------------


def test_error_level_and_text() -> None:
    """Mapeo de códigos a tipo de aviso y mensajes con la instrucción adecuada."""
    assert error_level("LLM_RATE_LIMIT") == "warning"
    assert error_level("EMPTY_INDEX") == "info"
    assert error_level("LLM_QUOTA_EXHAUSTED") == "error"
    assert "rag.cli reset --yes" in error_text("INDEX_MODEL_MISMATCH", "Otro modelo.")
    assert "GEMINI_API_KEY" in error_text("LLM_QUOTA_EXHAUSTED", "Sin créditos.")
    assert SPINNER_TEXT == "Buscando en los documentos y generando la respuesta…"


def test_upload_messages_per_file() -> None:
    """415/409/413 en subidas → un mensaje por archivo con su motivo."""
    messages = upload_messages(
        [
            UploadResult("ok.md", True, {"files_processed": 1, "chunks_added": 2, "skipped": []}),
            UploadResult(
                "vacio.txt", True, {"files_processed": 0, "skipped": ["vacio.txt: vacío"]}
            ),
            UploadResult("x.exe", False, code="UNSUPPORTED_FILE_TYPE", detail="No soportado."),
            UploadResult(
                "guia_onboarding_ti.txt", False, code="DUPLICATE_SOURCE", detail="Existe."
            ),
            UploadResult("big.txt", False, code="FILE_TOO_LARGE", detail="Supera 10 MB."),
        ]
    )
    assert [level for level, _ in messages] == ["success", "warning", "error", "error", "error"]
    assert "ok.md: 2 fragmentos indexados" in messages[0][1]
    assert "[UNSUPPORTED_FILE_TYPE]" in messages[2][1] and "x.exe" in messages[2][1]
    assert "[DUPLICATE_SOURCE]" in messages[3][1] and "[FILE_TOO_LARGE]" in messages[4][1]
