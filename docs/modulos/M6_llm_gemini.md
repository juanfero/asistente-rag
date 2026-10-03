# M6 — Cliente LLM Gemini

**Estado:** ⬜ Pendiente · **Estimado:** 2 h · **Depende de:** M0 · **Requisito:** R8
> Antes "Cliente LLM Grok (xAI)"; el LLM cambió a Gemini (ADR-010).

## Objetivo
Encapsular la generación de texto con **Gemini** (endpoint compatible con OpenAI) detrás de una interfaz simple, con manejo de errores claro (incluidos créditos agotados) y un `FakeLLM` para pruebas sin costo.

## Diseño
- `src/rag/llm.py`:
  - `class LLMClient(Protocol)`: `generate(system: str, user: str) -> LLMResult`.
  - `LLMResult(text: str, model: str, prompt_tokens: int | None, completion_tokens: int | None, latency_s: float)`.
  - `GeminiClient(settings)`: `openai.OpenAI(api_key=settings.gemini_api_key, base_url=settings.gemini_base_url, timeout=60, max_retries=2)`; `chat.completions.create(model=settings.gemini_model, messages=[system, user], temperature, max_tokens)`.
  - Si falta `GEMINI_API_KEY` → `LLMError("Falta GEMINI_API_KEY…")` al instanciar. Nunca loguear la key.
  - **Jerarquía de errores** (todas hijas de `LLMError`, mensajes en español, sin traceback para el usuario):
    | Excepción | Cuándo | Mensaje |
    |---|---|---|
    | `LLMQuotaExhaustedError` | HTTP 402, o 429 cuyo mensaje menciona cuota diaria/agotada (`PerDay`, `quota`, `exhausted`, `billing`) **después de los reintentos** | "Se agotaron los créditos de la API key de Gemini. Cambia GEMINI_API_KEY en el archivo .env y reinicia la aplicación." |
    | `LLMRateLimitError` | 429 transitorio (límite por minuto) | "Límite de solicitudes por minuto alcanzado. Espera unos segundos e intenta de nuevo." |
    | `LLMAuthError` | 400 con `API_KEY_INVALID`, 401 o 403 | "La API key de Gemini no es válida o fue revocada. Revisa GEMINI_API_KEY en .env." |
    | `LLMError` (genérica) | timeout, modelo inexistente (404), otros | mensaje claro en español |
  - **Tokens de razonamiento:** Gemini 3.x puede "pensar" y esos tokens consumen `max_tokens`; ver ADR-003 (se valida con `scripts/check_gemini.py`).
  - `FakeLLM(responses | callable)`: devuelve respuestas programadas y guarda los prompts recibidos (para asserts en M7).
  - `get_llm(settings)` fábrica.

## Criterios de aceptación
| ID | Criterio | Test |
|---|---|---|
| M6-01 | Sin API key → `LLMError` con mensaje claro | `test_missing_key` |
| M6-02 | `GeminiClient` envía `model`, `temperature`, `max_tokens` y mensajes system/user correctos (cliente OpenAI mockeado) | `test_request_payload` |
| M6-03 | Mapea `AuthenticationError`, `RateLimitError`, `APITimeoutError`, `NotFoundError` a la subclase de `LLMError` correspondiente | `test_error_mapping` (parametrizado) |
| M6-04 | `LLMResult` incluye tokens de uso y latencia | `test_result_fields` |
| M6-05 | `FakeLLM` registra prompts y devuelve respuestas en orden | `test_fake_llm` |
| M6-06 | La key no aparece en logs ni en `str(LLMError)` | `test_no_key_leak` (caplog) |
| M6-07 *(integration)* | Llamada real a Gemini devuelve texto no vacío en < 30 s | `test_gemini_real` |
| M6-08 | HTTP 402 → `LLMQuotaExhaustedError` con el mensaje de créditos agotados | `test_quota_exhausted_402` |
| M6-09 | 429 con cuota diaria agotada (tras reintentos) → `LLMQuotaExhaustedError`; 429 por minuto → `LLMRateLimitError` | `test_429_daily_vs_per_minute` |
| M6-10 | Key inválida (400 `API_KEY_INVALID`, 401, 403) → `LLMAuthError` | `test_invalid_key` |

Las pruebas M6-08…M6-10 usan respuestas HTTP simuladas que reproducen el formato de error de Gemini (cuerpo JSON `{"error": {"code", "message", "status", "details"}}`).

## Verificación
```bash
pytest tests/unit/test_llm.py -q
pytest -m integration tests/integration/test_gemini_real.py -q
```

## Registro de ejecución
| Fecha | Comando / acción | Resultado | Notas |
|---|---|---|---|
| | | | |
