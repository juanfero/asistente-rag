# M6 — Cliente LLM Gemini

**Estado:** ✅ Completado · **Estimado:** 2 h · **Depende de:** M0 · **Requisito:** R8
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
  - **Regla de clasificación de errores 402/429** (en orden, sin distinguir mayúsculas, sobre el mensaje + el cuerpo/detalles del error, incluido el cuerpo envuelto en lista `[{"error": …}]`):
    a. HTTP 402 → `LLMQuotaExhaustedError`.
    b. 429 que contiene `PerDay` (p. ej. `quotaId` `GenerateRequestsPerDayPerProjectPerModel-FreeTier`) → `LLMQuotaExhaustedError`.
    c. 429 que contiene `PerMinute` → `LLMRateLimitError`.
    d. 429 que contiene `billing`, `credit` o `prepay` → `LLMQuotaExhaustedError`.
    e. Cualquier otro 429 → `LLMRateLimitError`.
    `quota` **no** se usa: Gemini lo incluye en casi todos los 429 ("You exceeded your current quota"), también en los de límite por minuto. Se clasifica después de los reintentos del SDK (`max_retries=2`).
  - `LLMResult` incluye además `finish_reason` y `reasoning_tokens` (si vienen en `usage`). Respuesta vacía → `LLMError("El modelo devolvió una respuesta vacía…")`.
  - Pruebas que llaman a Gemini: marcador `llm` (solo con `RUN_LLM=1` y `GEMINI_API_KEY`).
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
RUN_LLM=1 pytest -m llm -q          # llama a Gemini
python scripts/gemini_smoke.py      # prompt RAG real (evidencias/M6_gemini_smoke.txt)
```

## Registro de ejecución
| Fecha | Comando / acción | Resultado | Notas |
|---|---|---|---|
| 2026-10-03 | `secret_scan.py --stdin` + M11-06 actualizado | `git log --all -p \| … --stdin` → 0 coincidencias | Solo informa el conteo; nueva prueba `test_git_history_has_no_secrets` |
| 2026-10-03 | Marcador `llm` (opt-in `RUN_LLM=1`) aplicado a M0-08 y M6-07 | OK | `pytest -m integration` sin costo (las `llm` se omiten); `RUN_LLM=1 pytest -m llm` las ejecuta |
| 2026-10-03 | `pytest tests/unit/test_llm.py -q` | 27 passed + 1 failed → 28 passed | El fallo era de la prueba (comparaba mayúsculas contra el mensaje en minúsculas) |
| 2026-10-03 | Mutaciones de `map_api_error`: "billing" antes que "PerMinute" / usar "quota" / sin control de vacío | 2 / 1 / 3 fallan | Una cuarta mutación (no leer `exc.body`) **sobrevive**: el SDK `openai` ya incluye el cuerpo del error en `exc.message`; se mantiene la lectura explícita del cuerpo por robustez |
| 2026-10-03 | `python scripts/gemini_smoke.py` (prompt RAG: system + 4 chunks, 2.800 car. de contexto + Q2) | exit 0 | `finish_reason=stop`, prompt 899 / completion 36 tokens, `reasoning_tokens` no informado, 0,98 s; respuesta con COP 120.000 [1] y USD 90 [2] → **no hace falta tocar `reasoning_effort` ni `LLM_MAX_TOKENS`** |
| 2026-10-03 | `pytest -m "not integration" -v` | 257 passed, 8 deselected | `evidencias/M6_pytest_ruff.txt` |
| 2026-10-03 | `pytest -m integration -v -rs` / `RUN_LLM=1 pytest -m llm -v -rs` | 6 passed + 2 skipped / 2 passed | `evidencias/M6_pytest_integration.txt` |
| 2026-10-03 | `ruff check src tests && ruff format --check src tests` | All checks passed! / 28 files already formatted | |

**Estado de criterios:** M6-01 ✅ · M6-02 ✅ · M6-03 ✅ · M6-04 ✅ · M6-05 ✅ · M6-06 ✅ · M6-07 ✅ (`RUN_LLM=1`) · M6-08 ✅ · M6-09 ✅ · M6-10 ✅.

**Pruebas adicionales:** `test_default_sdk_client`, `test_timeout_mapping`, `test_not_found_message_names_model`, `test_empty_response` (×3), `test_429_classified_after_sdk_retries` (3 llamadas = 1 + 2 reintentos), `test_other_400_is_generic`; `test_stdin_mode_only_counts`, `test_git_history_has_no_secrets`, `test_llm_tests_are_opt_in`.

**Observación para decidir más adelante:** el mensaje real de los 429 de Gemini dice "…check your plan and **billing** details…". Si un 429 llega **sin** `details` pero con ese mensaje, la regla (d) lo clasificará como créditos agotados aunque sea por minuto. Con `details` (caso habitual) la regla (c) lo clasifica bien.
