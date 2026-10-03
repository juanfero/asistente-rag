"""Cliente LLM: interfaz común, Gemini (endpoint compatible con OpenAI) y FakeLLM para pruebas."""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import openai

from rag.config import Settings

logger = logging.getLogger(__name__)

QUOTA_EXHAUSTED_MESSAGE = (
    "Se agotaron los créditos de la API key de Gemini. Cambia GEMINI_API_KEY en el archivo .env "
    "y reinicia la aplicación."
)
RATE_LIMIT_MESSAGE = (
    "Límite de solicitudes por minuto alcanzado. Espera unos segundos e intenta de nuevo."
)
AUTH_MESSAGE = "La API key de Gemini no es válida o fue revocada. Revisa GEMINI_API_KEY en .env."
EMPTY_RESPONSE_MESSAGE = (
    "El modelo devolvió una respuesta vacía. Intenta de nuevo o reformula la pregunta."
)


class LLMError(Exception):
    """Error al generar texto con el LLM (mensaje en español, apto para el usuario)."""


class LLMAuthError(LLMError):
    """API key inválida o revocada."""


class LLMRateLimitError(LLMError):
    """Límite transitorio de solicitudes (por minuto)."""


class LLMQuotaExhaustedError(LLMError):
    """Créditos o cuota diaria de la API key agotados."""


@dataclass
class LLMResult:
    """Respuesta del LLM con métricas de uso."""

    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float
    finish_reason: str | None = None
    reasoning_tokens: int | None = None


class LLMClient(Protocol):
    """Genera texto a partir de un prompt de sistema y uno de usuario."""

    def generate(self, system: str, user: str) -> LLMResult:
        """Genera la respuesta."""
        ...


def _error_text(exc: openai.APIStatusError) -> str:
    """Mensaje + cuerpo del error (dict o lista) en minúsculas, para clasificarlo."""
    parts = [str(getattr(exc, "message", "") or "")]
    if exc.body is not None:
        parts.append(json.dumps(exc.body, ensure_ascii=False))
    return " ".join(parts).lower()


def map_api_error(exc: openai.APIStatusError, model: str) -> LLMError:
    """Traduce un error HTTP de Gemini a la excepción propia (regla de ADR-010).

    429: "PerDay" → cuota agotada; "PerMinute" → límite por minuto; "billing"/"credit"/"prepay"
    → cuota agotada; otro 429 → límite por minuto. "quota" no se usa: aparece en casi todos.
    """
    status = exc.status_code
    text = _error_text(exc)
    if status == 402:
        return LLMQuotaExhaustedError(QUOTA_EXHAUSTED_MESSAGE)
    if status == 429:
        if "perday" in text:
            return LLMQuotaExhaustedError(QUOTA_EXHAUSTED_MESSAGE)
        if "perminute" in text:
            return LLMRateLimitError(RATE_LIMIT_MESSAGE)
        if any(word in text for word in ("billing", "credit", "prepay")):
            return LLMQuotaExhaustedError(QUOTA_EXHAUSTED_MESSAGE)
        return LLMRateLimitError(RATE_LIMIT_MESSAGE)
    if status in (401, 403) or (status == 400 and "api_key_invalid" in text):
        return LLMAuthError(AUTH_MESSAGE)
    if status == 404:
        return LLMError(
            f"El modelo '{model}' no existe o no está disponible para esta API key. "
            "Revisa GEMINI_MODEL en .env."
        )
    return LLMError(f"Gemini respondió con un error (HTTP {status}). Intenta de nuevo más tarde.")


class GeminiClient:
    """Cliente de Gemini con el SDK `openai` (reintentos del SDK: `max_retries=2`)."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else ""
        if not key:
            raise LLMError(
                "Falta GEMINI_API_KEY. Defínela en el archivo .env de la raíz del proyecto "
                "(ver .env.example)."
            )
        self.model = settings.gemini_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self._client = client or openai.OpenAI(
            api_key=key, base_url=settings.gemini_base_url, timeout=60, max_retries=2
        )

    def generate(self, system: str, user: str) -> LLMResult:
        """Genera la respuesta; los errores se traducen a subclases de `LLMError`."""
        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except openai.APITimeoutError as exc:
            raise LLMError("Gemini no respondió a tiempo (timeout). Intenta de nuevo.") from exc
        except openai.APIConnectionError as exc:
            raise LLMError("No se pudo conectar con Gemini. Revisa tu conexión.") from exc
        except openai.APIStatusError as exc:
            raise map_api_error(exc, self.model) from exc
        latency = time.perf_counter() - start

        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        if not text:
            raise LLMError(EMPTY_RESPONSE_MESSAGE)
        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        result = LLMResult(
            text=text,
            model=self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            latency_s=latency,
            finish_reason=choice.finish_reason,
            reasoning_tokens=getattr(details, "reasoning_tokens", None) if details else None,
        )
        logger.info(
            "LLM %s: %.2f s, tokens %s/%s, finish_reason=%s",
            self.model,
            latency,
            result.prompt_tokens,
            result.completion_tokens,
            result.finish_reason,
        )
        return result


Response = str | Exception | Callable[[str, str], str]


class FakeLLM:
    """LLM programado para pruebas: devuelve respuestas en orden y registra los prompts.

    Cada respuesta puede ser texto, una excepción (se lanza) o una función (system, user) → texto.
    """

    def __init__(self, responses: list[Response] | Callable[[str, str], str], model: str = "fake"):
        self._responses = responses
        self.model = model
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> LLMResult:
        """Devuelve la siguiente respuesta programada."""
        self.calls.append((system, user))
        if callable(self._responses):
            item: Response = self._responses
        else:
            index = len(self.calls) - 1
            if index >= len(self._responses):
                raise LLMError("FakeLLM: no quedan respuestas programadas.")
            item = self._responses[index]
        if isinstance(item, Exception):
            raise item
        text = item(system, user) if callable(item) else item
        return LLMResult(
            text=text,
            model=self.model,
            prompt_tokens=len(system + user) // 4,
            completion_tokens=len(text) // 4,
            latency_s=0.0,
            finish_reason="stop",
        )


def get_llm(settings: Settings) -> GeminiClient:
    """Crea el cliente LLM configurado."""
    return GeminiClient(settings)
