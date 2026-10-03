"""Pruebas de M6: GeminiClient con respuestas HTTP simuladas en el formato de error de Gemini."""

import json
import logging

import httpx
import openai
import pytest

from rag.config import Settings
from rag.llm import (
    AUTH_MESSAGE,
    EMPTY_RESPONSE_MESSAGE,
    QUOTA_EXHAUSTED_MESSAGE,
    RATE_LIMIT_MESSAGE,
    RATE_LIMIT_UNKNOWN_MESSAGE,
    FakeLLM,
    GeminiClient,
    LLMAuthError,
    LLMError,
    LLMQuotaExhaustedError,
    LLMRateLimitError,
    LLMResult,
    get_llm,
)

pytestmark = pytest.mark.usefixtures("clean_env")

FAKE_KEY = "clave-falsa-de-prueba-NO-REAL-9876"
BASE_URL = "https://gemini.test/v1beta/openai/"
QUOTA_MSG = (
    "You exceeded your current quota, please check your plan and billing details. For more "
    "information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits."
)


def quota_failure(quota_id: str) -> list[dict]:
    """`details` de un 429 de Gemini con QuotaFailure.violations[].quotaId."""
    return [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [
                {
                    "quotaMetric": "generativelanguage.googleapis.com/generate_content_requests",
                    "quotaId": quota_id,
                    "quotaDimensions": {"location": "global", "model": "gemini-3.1-flash-lite"},
                }
            ],
        },
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "19s"},
    ]


def gemini_error(code: int, message: str, status: str, details: list | None = None) -> dict:
    """Cuerpo de error documentado de Gemini: {"error": {code, message, status, details}}."""
    error = {"code": code, "message": message, "status": status}
    if details is not None:
        error["details"] = details
    return {"error": error}


KEY_INVALID = gemini_error(
    400,
    "API key not valid. Please pass a valid API key.",
    "INVALID_ARGUMENT",
    [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}],
)
OK_BODY = {
    "id": "resp-1",
    "object": "chat.completion",
    "created": 0,
    "model": "gemini-3.1-flash-lite",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Son 15 días hábiles [1]."},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 120,
        "completion_tokens": 9,
        "total_tokens": 129,
        "completion_tokens_details": {"reasoning_tokens": 4},
    },
}


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, gemini_api_key=FAKE_KEY, llm_max_tokens=321)


def make_client(settings: Settings, handler, max_retries: int = 0) -> GeminiClient:
    """GeminiClient con el SDK openai real y un transporte HTTP simulado."""
    sdk = openai.OpenAI(
        api_key=FAKE_KEY,
        base_url=BASE_URL,
        max_retries=max_retries,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return GeminiClient(settings, client=sdk)


def respond(status: int, body, headers: dict | None = None):
    """Handler que siempre responde con el estado y cuerpo JSON dados."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body, headers=headers or {})

    return handler


# --- M6-01 … M6-07 -----------------------------------------------------------------


def test_missing_key() -> None:
    """M6-01: sin API key → LLMError con mensaje claro."""
    with pytest.raises(LLMAuthError, match="Falta GEMINI_API_KEY") as exc:
        GeminiClient(Settings(_env_file=None))
    assert isinstance(exc.value, LLMError)


def test_request_payload(settings: Settings) -> None:
    """M6-02: se envían model, temperature, max_tokens y los mensajes system/user."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers["authorization"]
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=OK_BODY)

    make_client(settings, handler).generate("Eres un asistente.", "¿Cuántos días?")
    body = captured["json"]
    assert captured["url"].endswith("/chat/completions")
    assert captured["auth"] == f"Bearer {FAKE_KEY}"
    assert body["model"] == "gemini-3.1-flash-lite"
    assert body["temperature"] == 0.1
    assert body["max_tokens"] == 321
    assert body["messages"] == [
        {"role": "system", "content": "Eres un asistente."},
        {"role": "user", "content": "¿Cuántos días?"},
    ]


def test_default_sdk_client(settings: Settings) -> None:
    """Sin cliente inyectado: SDK openai con base_url de Gemini, timeout 60 y 2 reintentos."""
    client = GeminiClient(settings)._client
    assert str(client.base_url) == settings.gemini_base_url
    assert client.max_retries == 2
    assert client.timeout == 60
    assert isinstance(get_llm(settings), GeminiClient)


@pytest.mark.parametrize(
    ("handler", "expected", "sdk_error"),
    [
        (
            respond(401, gemini_error(401, "Unauthenticated", "UNAUTHENTICATED")),
            LLMAuthError,
            openai.AuthenticationError,
        ),
        (
            respond(429, gemini_error(429, "Too many requests", "RESOURCE_EXHAUSTED")),
            LLMRateLimitError,
            openai.RateLimitError,
        ),
        (
            respond(404, gemini_error(404, "models/x is not found", "NOT_FOUND")),
            LLMError,
            openai.NotFoundError,
        ),
    ],
    ids=["auth", "rate-limit", "not-found"],
)
def test_error_mapping(settings: Settings, handler, expected, sdk_error) -> None:
    """M6-03: AuthenticationError, RateLimitError y NotFoundError → subclase de LLMError."""
    with pytest.raises(expected) as exc:
        make_client(settings, handler).generate("s", "u")
    assert isinstance(exc.value, LLMError)
    assert isinstance(exc.value.__cause__, sdk_error)


def test_timeout_mapping(settings: Settings) -> None:
    """M6-03: APITimeoutError → LLMError con mensaje en español."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    with pytest.raises(LLMError, match="no respondió a tiempo") as exc:
        make_client(settings, handler).generate("s", "u")
    assert isinstance(exc.value.__cause__, openai.APITimeoutError)


def test_not_found_message_names_model(settings: Settings) -> None:
    """404: el mensaje nombra el modelo y sugiere revisar GEMINI_MODEL."""
    handler = respond(404, gemini_error(404, "not found", "NOT_FOUND"))
    with pytest.raises(LLMError, match="gemini-3.1-flash-lite.*GEMINI_MODEL"):
        make_client(settings, handler).generate("s", "u")


def test_result_fields(settings: Settings) -> None:
    """M6-04: LLMResult incluye texto, modelo, tokens, latencia, finish_reason y razonamiento."""
    result = make_client(settings, respond(200, OK_BODY)).generate("s", "u")
    assert isinstance(result, LLMResult)
    assert result.text == "Son 15 días hábiles [1]."
    assert result.model == "gemini-3.1-flash-lite"
    assert (result.prompt_tokens, result.completion_tokens) == (120, 9)
    assert result.reasoning_tokens == 4
    assert result.finish_reason == "stop"
    assert result.latency_s >= 0


@pytest.mark.parametrize("content", ["", "   ", None], ids=["vacio", "espacios", "none"])
def test_empty_response(settings: Settings, content) -> None:
    """Respuesta vacía → LLMError con mensaje en español."""
    body = json.loads(json.dumps(OK_BODY))
    body["choices"][0]["message"]["content"] = content
    with pytest.raises(LLMError) as exc:
        make_client(settings, respond(200, body)).generate("s", "u")
    assert str(exc.value) == EMPTY_RESPONSE_MESSAGE


def test_fake_llm() -> None:
    """M6-05: FakeLLM registra los prompts y responde en orden (texto, función, excepción)."""
    fake = FakeLLM(["uno", lambda s, u: f"eco: {u}", LLMQuotaExhaustedError("sin créditos")])
    assert fake.generate("sys", "p1").text == "uno"
    assert fake.generate("sys", "p2").text == "eco: p2"
    with pytest.raises(LLMQuotaExhaustedError):
        fake.generate("sys", "p3")
    assert fake.calls == [("sys", "p1"), ("sys", "p2"), ("sys", "p3")]
    with pytest.raises(LLMError, match="no quedan respuestas"):
        fake.generate("sys", "p4")
    assert FakeLLM(lambda s, u: "siempre").generate("a", "b").finish_reason == "stop"


def test_no_key_leak(settings: Settings, caplog: pytest.LogCaptureFixture) -> None:
    """M6-06: la key no aparece en logs (incluidos los DEBUG de openai/httpx) ni en los errores."""
    caplog.set_level(logging.DEBUG)
    client = make_client(settings, respond(401, gemini_error(401, "bad key", "UNAUTHENTICATED")))
    with pytest.raises(LLMAuthError) as exc:
        client.generate("s", "u")
    make_client(settings, respond(200, OK_BODY)).generate("s", "u")
    assert caplog.records
    assert FAKE_KEY not in caplog.text
    assert FAKE_KEY not in str(exc.value) and FAKE_KEY not in repr(exc.value)
    assert FAKE_KEY not in str(exc.value.__cause__)


# --- M6-08 … M6-10: créditos agotados, límite por minuto y key inválida -------------


def test_quota_exhausted_402(settings: Settings) -> None:
    """M6-08: HTTP 402 → LLMQuotaExhaustedError con el mensaje de créditos agotados."""
    handler = respond(402, gemini_error(402, "Payment required", "FAILED_PRECONDITION"))
    with pytest.raises(LLMQuotaExhaustedError) as exc:
        make_client(settings, handler).generate("s", "u")
    assert str(exc.value) == QUOTA_EXHAUSTED_MESSAGE
    assert "cambia gemini_api_key en el archivo .env" in str(exc.value).lower()


PER_DAY = gemini_error(
    429,
    QUOTA_MSG,
    "RESOURCE_EXHAUSTED",
    quota_failure("GenerateRequestsPerDayPerProjectPerModel-FreeTier"),
)
PER_MINUTE = gemini_error(
    429,
    QUOTA_MSG,
    "RESOURCE_EXHAUSTED",
    quota_failure("GenerateRequestsPerMinutePerProjectPerModel-FreeTier"),
)


@pytest.mark.parametrize(
    ("body", "expected", "message"),
    [
        (PER_DAY, LLMQuotaExhaustedError, QUOTA_EXHAUSTED_MESSAGE),  # regla b
        ([PER_DAY], LLMQuotaExhaustedError, QUOTA_EXHAUSTED_MESSAGE),  # cuerpo en lista
        (PER_MINUTE, LLMRateLimitError, RATE_LIMIT_MESSAGE),  # regla c (antes que "billing")
        ([PER_MINUTE], LLMRateLimitError, RATE_LIMIT_MESSAGE),
        (
            gemini_error(
                429, "Resource has been exhausted (e.g. check quota).", "RESOURCE_EXHAUSTED"
            ),
            LLMRateLimitError,
            RATE_LIMIT_UNKNOWN_MESSAGE,
        ),  # regla e
        (
            gemini_error(429, QUOTA_MSG, "RESOURCE_EXHAUSTED"),
            LLMRateLimitError,
            RATE_LIMIT_UNKNOWN_MESSAGE,
        ),  # mensaje real con "billing details" SIN details → regla e (no es créditos)
        (
            gemini_error(429, "Your prepayment credits are depleted.", "RESOURCE_EXHAUSTED"),
            LLMQuotaExhaustedError,
            QUOTA_EXHAUSTED_MESSAGE,
        ),  # regla d (prepay)
        (
            gemini_error(429, "You have no remaining credits.", "RESOURCE_EXHAUSTED"),
            LLMQuotaExhaustedError,
            QUOTA_EXHAUSTED_MESSAGE,
        ),  # regla d (credits)
    ],
    ids=[
        "dia",
        "dia-lista",
        "minuto",
        "minuto-lista",
        "sin-details",
        "mensaje-real-billing-sin-details",
        "prepay",
        "credits",
    ],
)
def test_429_daily_vs_per_minute(settings: Settings, body, expected, message) -> None:
    """M6-09: 429 diario/créditos → QuotaExhausted; por minuto o sin detalle → RateLimit."""
    with pytest.raises(expected) as exc:
        make_client(settings, respond(429, body)).generate("s", "u")
    assert str(exc.value) == message


def test_429_classified_after_sdk_retries(settings: Settings) -> None:
    """El 429 se clasifica después de los reintentos del SDK (1 intento + 2 reintentos)."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, json=PER_DAY, headers={"retry-after-ms": "1"})

    with pytest.raises(LLMQuotaExhaustedError):
        make_client(settings, handler, max_retries=2).generate("s", "u")
    assert len(calls) == 3


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (400, KEY_INVALID),
        (400, [KEY_INVALID]),
        (
            401,
            gemini_error(401, "Request had invalid authentication credentials.", "UNAUTHENTICATED"),
        ),
        (403, gemini_error(403, "Permission denied: API key was revoked.", "PERMISSION_DENIED")),
    ],
    ids=["400-api-key-invalid", "400-lista", "401", "403"],
)
def test_invalid_key(settings: Settings, status: int, body) -> None:
    """M6-10: 400 API_KEY_INVALID, 401 o 403 → LLMAuthError con mensaje claro."""
    with pytest.raises(LLMAuthError) as exc:
        make_client(settings, respond(status, body)).generate("s", "u")
    assert str(exc.value) == AUTH_MESSAGE


def test_other_400_is_generic(settings: Settings) -> None:
    """Un 400 que no es de API key → LLMError genérico (no LLMAuthError)."""
    handler = respond(400, gemini_error(400, "Invalid JSON payload", "INVALID_ARGUMENT"))
    with pytest.raises(LLMError, match="HTTP 400") as exc:
        make_client(settings, handler).generate("s", "u")
    assert not isinstance(exc.value, LLMAuthError)


def test_lazy_llm_creates_client_on_first_use() -> None:
    """LazyLLM no crea el cliente hasta la primera generación y lo reutiliza."""
    from rag.llm import LazyLLM

    created: list[int] = []

    def factory() -> FakeLLM:
        created.append(1)
        return FakeLLM(lambda s, u: "ok")

    lazy = LazyLLM(factory)
    assert created == []
    assert lazy.generate("s", "u").text == "ok"
    lazy.generate("s", "u")
    assert created == [1]
