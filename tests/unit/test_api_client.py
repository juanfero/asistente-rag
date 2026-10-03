"""Pruebas de M9-01: api_client con una sesión HTTP falsa (sin red)."""

import pytest
import requests

from rag.api_client import (
    TIMEOUT_ASK,
    TIMEOUT_DOCUMENTS,
    TIMEOUT_HEALTH,
    APIClient,
    APIClientError,
    UISettings,
    get_client,
    start_command,
)

pytestmark = pytest.mark.usefixtures("clean_env")


class FakeResponse:
    def __init__(self, status: int, payload=None, text_only: bool = False) -> None:
        self.status_code = status
        self._payload = payload
        self._text_only = text_only

    def json(self):
        if self._text_only:
            raise ValueError("no es JSON")
        return self._payload


class FakeSession:
    """Registra cada request y devuelve respuestas o excepciones programadas."""

    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def request(self, method, url, timeout, **kwargs):
        self.calls.append({"method": method, "url": url, "timeout": timeout, **kwargs})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


BASE = "http://localhost:8011"


@pytest.mark.parametrize(
    ("call", "method", "path", "timeout"),
    [
        (lambda c: c.health(), "GET", "/health", TIMEOUT_HEALTH),
        (lambda c: c.list_documents(), "GET", "/documents", TIMEOUT_DOCUMENTS),
        (lambda c: c.ingest(reset=True), "POST", "/ingest", TIMEOUT_DOCUMENTS),
        (lambda c: c.delete_document("a.md"), "DELETE", "/documents/a.md", TIMEOUT_DOCUMENTS),
        (lambda c: c.ask("¿Hola?", top_k=3), "POST", "/ask", TIMEOUT_ASK),
    ],
    ids=["health", "documents", "ingest", "delete", "ask"],
)
def test_requests_and_timeouts(call, method: str, path: str, timeout: int) -> None:
    """Cada método arma la petición correcta con su timeout (5 / 120 / 60 s)."""
    payload = {"llm_model": "m", "chunks": 1, "top_k": 4}
    session = FakeSession(FakeResponse(200, payload))
    assert call(APIClient(BASE + "/", session)) == payload
    req = session.calls[0]
    assert (req["method"], req["url"], req["timeout"]) == (method, BASE + path, timeout)


def test_payloads() -> None:
    """Cuerpos y parámetros: ask con/sin top_k, ingest con reset, delete con delete_file."""
    session = FakeSession(*[FakeResponse(200, {}) for _ in range(4)])
    client = APIClient(BASE, session)
    client.ask("¿P?", top_k=2)
    client.ask("¿P?")
    client.ingest()
    client.delete_document("x.txt", delete_file=True)
    assert session.calls[0]["json"] == {"question": "¿P?", "top_k": 2}
    assert session.calls[1]["json"] == {"question": "¿P?"}
    assert session.calls[2]["json"] == {"reset": False}
    assert session.calls[3]["params"] == {"delete_file": "true"}


def test_api_error_uses_error_code() -> None:
    """Errores de la API → APIClientError con el código del campo "error" y el detalle."""
    session = FakeSession(
        FakeResponse(503, {"error": "LLM_QUOTA_EXHAUSTED", "detail": "Sin créditos."})
    )
    with pytest.raises(APIClientError) as exc:
        APIClient(BASE, session).ask("¿P?")
    assert (exc.value.code, exc.value.detail, exc.value.status) == (
        "LLM_QUOTA_EXHAUSTED",
        "Sin créditos.",
        503,
    )


def test_non_json_error() -> None:
    """Un error sin cuerpo JSON → API_ERROR con el código HTTP."""
    with pytest.raises(APIClientError, match="HTTP 500") as exc:
        APIClient(BASE, FakeSession(FakeResponse(500, text_only=True))).health()
    assert exc.value.code == "API_ERROR"


def test_connection_error_message() -> None:
    """API caída → API_UNAVAILABLE con el comando para levantarla (puerto de API_URL)."""
    session = FakeSession(requests.ConnectionError("refused"))
    with pytest.raises(APIClientError) as exc:
        APIClient(BASE, session).health()
    assert exc.value.code == "API_UNAVAILABLE"
    assert "uvicorn rag.api:app --workers 1 --port 8011" in exc.value.detail


def test_timeout_message() -> None:
    """Timeout → API_TIMEOUT con mensaje en español."""
    session = FakeSession(requests.Timeout("lento"))
    with pytest.raises(APIClientError, match="no respondió en 60 s") as exc:
        APIClient(BASE, session).ask("¿P?")
    assert exc.value.code == "API_TIMEOUT"


def test_upload_one_request_per_file() -> None:
    """Las subidas van una por archivo y cada una informa su resultado (415/409/413)."""
    session = FakeSession(
        FakeResponse(200, {"files_processed": 1, "chunks_added": 2}),
        FakeResponse(415, {"error": "UNSUPPORTED_FILE_TYPE", "detail": "No soportado."}),
        FakeResponse(409, {"error": "DUPLICATE_SOURCE", "detail": "Ya existe en el corpus."}),
        FakeResponse(413, {"error": "FILE_TOO_LARGE", "detail": "Supera 10 MB."}),
    )
    results = APIClient(BASE, session).upload(
        [("a.md", b"x"), ("b.exe", b"x"), ("guia_onboarding_ti.txt", b"x"), ("big.txt", b"x")]
    )
    assert [r.ok for r in results] == [True, False, False, False]
    assert [r.code for r in results] == [
        None,
        "UNSUPPORTED_FILE_TYPE",
        "DUPLICATE_SOURCE",
        "FILE_TOO_LARGE",
    ]
    assert all(call["files"][0][0] == "files" for call in session.calls)
    assert len(session.calls) == 4


def test_ui_settings_reads_only_api_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """UISettings lee API_URL y no tiene campo para la API key de Gemini."""
    monkeypatch.setenv("API_URL", "http://localhost:8011")
    monkeypatch.setenv("GEMINI_API_KEY", "clave-que-no-debe-leerse")
    settings = UISettings(_env_file=None)
    assert settings.api_url == "http://localhost:8011"
    assert set(UISettings.model_fields) == {"api_url"}
    assert "clave-que-no-debe-leerse" not in repr(settings)
    assert get_client().base_url == "http://localhost:8011"
    assert start_command("http://localhost:8011") == "uvicorn rag.api:app --workers 1 --port 8011"


def test_health_detects_other_service() -> None:
    """Si API_URL apunta a otro servicio (p. ej. /health sin llm_model) → API_UNEXPECTED."""
    session = FakeSession(FakeResponse(200, {"status": "healthy", "database": "ok"}))
    with pytest.raises(APIClientError, match="no es la API del asistente") as exc:
        APIClient("http://localhost:8000", session).health()
    assert exc.value.code == "API_UNEXPECTED"
    assert "--port 8000" in exc.value.detail


def test_run_demo_script() -> None:
    """scripts/run_demo.sh: sintaxis válida, ejecutable, PIDs + trap y sin pkill."""
    import subprocess

    from rag.loaders import PROJECT_ROOT

    script = PROJECT_ROOT / "scripts" / "run_demo.sh"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    assert script.stat().st_mode & 0o111
    text = script.read_text(encoding="utf-8")
    assert "API_PID=$!" in text and "UI_PID=$!" in text and "trap cleanup" in text
    assert "pkill" not in text.replace("(sin pkill)", "")
