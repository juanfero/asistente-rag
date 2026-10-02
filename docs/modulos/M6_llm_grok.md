# M6 — Cliente LLM Grok (xAI)

**Estado:** ⬜ Pendiente · **Estimado:** 2 h · **Depende de:** M0 · **Requisito:** R8

## Objetivo
Encapsular la generación de texto con **Grok (xAI)** detrás de una interfaz simple, con manejo de errores y un `FakeLLM` para pruebas sin costo.

## Diseño
- `src/rag/llm.py`:
  - `class LLMClient(Protocol)`: `generate(system: str, user: str) -> LLMResult`.
  - `LLMResult(text: str, model: str, prompt_tokens: int | None, completion_tokens: int | None, latency_s: float)`.
  - `GrokClient(settings)`: `openai.OpenAI(api_key=..., base_url=settings.XAI_BASE_URL, timeout=60, max_retries=2)`; `chat.completions.create(model, messages=[system, user], temperature, max_tokens)`.
  - Errores → excepción propia `LLMError` con mensaje en español (auth inválida, rate limit, timeout, modelo inexistente). Nunca loguear la key.
  - Si falta `XAI_API_KEY` → `LLMError("Falta XAI_API_KEY…")` al instanciar.
  - `FakeLLM(responses | callable)`: devuelve respuestas programadas y guarda los prompts recibidos (para asserts en M7).
  - `get_llm(settings)` fábrica.

## Criterios de aceptación
| ID | Criterio | Test |
|---|---|---|
| M6-01 | Sin API key → `LLMError` con mensaje claro | `test_missing_key` |
| M6-02 | `GrokClient` envía `model`, `temperature`, `max_tokens` y mensajes system/user correctos (cliente OpenAI mockeado) | `test_request_payload` |
| M6-03 | Mapea `AuthenticationError`, `RateLimitError`, `APITimeoutError`, `NotFoundError` a `LLMError` | `test_error_mapping` (parametrizado) |
| M6-04 | `LLMResult` incluye tokens de uso y latencia | `test_result_fields` |
| M6-05 | `FakeLLM` registra prompts y devuelve respuestas en orden | `test_fake_llm` |
| M6-06 | La key no aparece en logs ni en `str(LLMError)` | `test_no_key_leak` (caplog) |
| M6-07 *(integration)* | Llamada real a Grok devuelve texto no vacío en < 30 s | `test_grok_real` |

## Verificación
```bash
pytest tests/unit/test_llm.py -q
pytest -m integration tests/integration/test_grok_real.py -q
```

## Registro de ejecución
| Fecha | Comando / acción | Resultado | Notas |
|---|---|---|---|
| | | | |
