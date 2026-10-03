# M8 — API REST con FastAPI

**Estado:** ✅ Completado · **Estimado:** 3 h · **Depende de:** M7 · **Requisito:** R6

## Objetivo
Exponer ingesta y consultas como API REST documentada (Swagger en `/docs`), reutilizando `ingest.py` y `RAGEngine`.

## Endpoints
| Método | Ruta | Descripción | Respuesta |
|---|---|---|---|
| GET | `/health` | Estado del servicio (no llama a Gemini) | `{status, llm_model, llm_configured, embedding_model, chunks, documents, min_score, top_k}` |
| GET | `/documents` | Fuentes indexadas con nº de chunks y origen | `[{source, chunks, origin: corpus\|upload\|desconocido}]` |
| POST | `/documents` | Subir 1..n archivos (`multipart/form-data`, campo `files`), guardarlos en **`UPLOADS_DIR` (`data/uploads/`)** e ingerirlos | `IngestReport` |
| POST | `/ingest` | Re-ingerir `DOCS_DIR` + `UPLOADS_DIR` (body opcional `{reset: bool}`) | `IngestReport` |
| DELETE | `/documents/{source}` | Eliminar documento del índice (y del disco con `?delete_file=true`, solo subidas) | `{source, deleted_chunks, file_deleted}` |
| POST | `/ask` | Body `{question, top_k?}` | `RAGAnswer` (sin el prompt interno) |

## Diseño
- `src/rag/api.py`: `create_app(engine=None, store=None, settings=None) -> FastAPI` (inyección para tests) y `app = create_app()`.
- Dependencias pesadas (embedder, Chroma, LLM) se crean en `lifespan` una sola vez.
- Esquemas Pydantic de request/response en `src/rag/schemas.py`.
- Validaciones: extensión permitida, tamaño máx. 10 MB por archivo, nombre saneado (`Path(name).name`, sin `..`).
- **Formato único de error** `{"error": "<CODIGO>", "detail": "<mensaje en español>"}` (ADR-011):

| Situación | HTTP | `error` |
|---|---|---|
| Validación del body (pregunta vacía, > 1.000 car., `top_k` fuera de 1–20) | 422 | `VALIDATION_ERROR` |
| Nombre de archivo inválido | 400 | `INVALID_FILENAME` |
| Borrar del disco un archivo del corpus | 403 | `PROTECTED_SOURCE` |
| Documento inexistente / ruta inexistente | 404 | `DOCUMENT_NOT_FOUND` / `NOT_FOUND` |
| Índice vacío (`EmptyIndexError`) | 409 | `EMPTY_INDEX` |
| Índice creado con otro modelo (`EmbeddingModelMismatchError`) | 409 | `INDEX_MODEL_MISMATCH` |
| Subida con el nombre de un archivo del corpus o repetida en la petición | 409 | `DUPLICATE_SOURCE` |
| Archivo > 10 MB | 413 | `FILE_TOO_LARGE` |
| Extensión no soportada / contenido que no corresponde (PDF sin `%PDF`, binario como .txt) | 415 | `UNSUPPORTED_FILE_TYPE` / `INVALID_CONTENT` |
| `LLMRateLimitError` | 429 (+ `Retry-After: 60`) | `LLM_RATE_LIMIT` |
| Otro `LLMError` | 502 | `LLM_ERROR` |
| `LLMQuotaExhaustedError` | 503 | `LLM_QUOTA_EXHAUSTED` |
| `LLMAuthError` (key inválida o ausente) | 503 | `LLM_AUTH_ERROR` |
| Cualquier excepción no controlada | 500 | `INTERNAL_ERROR` (mensaje genérico; el detalle solo va al log) |

- CORS habilitado para `localhost:8501` (Streamlit).
- *Decisiones del autor (2026-10-03):*
  - **Subidas fuera del corpus (riesgo H9 resuelto):** `POST /documents` escribe en `data/uploads/` (ignorado por git, con `.gitkeep`); `data/docs/` sigue siendo el corpus congelado. Un nombre del corpus → 409 `DUPLICATE_SOURCE`; el mismo nombre que una subida previa → la reemplaza. `DELETE …?delete_file=true` solo borra del disco subidas; sobre el corpus → 403 `PROTECTED_SOURCE` (quitarlo del índice sin `delete_file` sí se permite). Se valida el **contenido** además de la extensión (PDF debe empezar con `%PDF`; txt/md deben ser texto) → 415 `INVALID_CONTENT`. Subidas sin texto útil se informan como omitidas y no se conservan.
  - **Arranque (lifespan):** embedder y store se crean una sola vez + *warm-up* del modelo (la primera pregunta no paga la carga); Gemini se instancia de forma *lazy* (`LazyLLM`); `threading.Lock` serializa ingesta, subidas y borrados. **La API corre con 1 worker** (`uvicorn rag.api:app --workers 1`): Chroma local no admite escritores concurrentes entre procesos. Se registra la configuración efectiva (sin key).
  - **`/health` no llama a Gemini**; si el índice es de otro modelo responde `status: index_model_mismatch` y los demás endpoints devuelven 409.
  - Sin `GEMINI_API_KEY`, `GeminiClient` lanza `LLMAuthError` (→ 503 `LLM_AUTH_ERROR`; código 2 en la CLI).

## Criterios de aceptación (TestClient + FakeEmbedder + FakeLLM + `tmp_path`)
| ID | Criterio | Test |
|---|---|---|
| M8-01 | `GET /health` → 200 con campos esperados | `test_health` |
| M8-02 | `POST /documents` con .md y .pdf → 200, chunks > 0, archivos guardados | `test_upload_documents` |
| M8-03 | Subir `.exe`/`.docx` → 415; archivo > 10 MB → 413 | `test_upload_validation` |
| M8-04 | Nombre de archivo con `../` se sanea (no escribe fuera de `DOCS_DIR`) | `test_path_traversal` |
| M8-05 | `GET /documents` lista las fuentes subidas | `test_list_documents` |
| M8-06 | `POST /ask` → 200 con `answer`, `sources`, `grounded` | `test_ask` |
| M8-07 | `POST /ask` con pregunta vacía → 422 | `test_ask_validation` |
| M8-08 | `LLMError` → 502 con detalle legible | `test_llm_error_502` |
| M8-09 | `DELETE /documents/{source}` elimina; inexistente → 404 | `test_delete_document` |
| M8-10 | Swagger `/docs` y `/openapi.json` disponibles | `test_openapi` |
| M8-11 | `LLMQuotaExhaustedError` → 503 `{"error":"LLM_QUOTA_EXHAUSTED","detail":…}`; `LLMAuthError` → 503 `"LLM_AUTH_ERROR"`; `LLMRateLimitError` → 429 | `test_llm_specific_errors` |

## Verificación
```bash
pytest tests/unit/test_api.py -q
uvicorn rag.api:app --workers 1          # http://localhost:8000/docs
curl -s localhost:8000/health | python -m json.tool
curl -s -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question":"¿Qué VPN debo usar fuera de la oficina?"}' | python -m json.tool
```
Captura de Swagger y salidas de curl en `evidencias/`.

## Registro de ejecución
| Fecha | Comando / acción | Resultado | Notas |
|---|---|---|---|
| 2026-10-03 | `pytest tests/unit/test_api.py -q` | 29 passed + 8 failed → 37 passed | Los 8 fallos: Starlette re-lanza las excepciones manejadas por el handler genérico de `Exception`; se registró el handler por cada clase del dominio (ExceptionMiddleware) y se dejó el de `Exception` solo para el 500 |
| 2026-10-03 | Mutaciones: sin validar contenido / sin 409 corpus / sin 403 corpus / sin `Retry-After` / sin sanear el nombre | 2 / 1 / 1 / 1 / 3 fallan | Las pruebas detectan cada defecto |
| 2026-10-03 | `RUN_LLM=1 pytest tests/integration/test_api_real.py -m llm` | 1 passed | `POST /ingest` + `POST /ask` (Q2) reales: ambos topes y citas de las págs. 1 y 2 |
| 2026-10-03 | `uvicorn rag.api:app --workers 1 --port 8011` + 13 llamadas curl | Códigos esperados (200×9, 415, 409, 422, 403) | `evidencias/M8_curl.txt`, `evidencias/M8_openapi.json`. Puerto 8011 porque el 8000 lo ocupaba **otro servicio local** (no se tocó). La subida de prueba se borró al final: índice de nuevo con 16 chunks |
| 2026-10-03 | `pytest -m "not integration" -v` | 337 passed, 12 deselected | `evidencias/M8_pytest_ruff.txt` |
| 2026-10-03 | `pytest -m integration -v -rs` / `RUN_LLM=1 pytest -m llm -v -rs` | 6 passed + 6 skipped / 6 passed | `evidencias/M8_pytest_integration.txt` |
| 2026-10-03 | `ruff check src tests && ruff format --check src tests` | All checks passed! / 37 files already formatted | |

**Estado de criterios:** M8-01 ✅ · M8-02 ✅ · M8-03 ✅ · M8-04 ✅ · M8-05 ✅ · M8-06 ✅ · M8-07 ✅ · M8-08 ✅ · M8-09 ✅ · M8-10 ✅ · M8-11 ✅.

**Pruebas adicionales (A–E):** `test_empty_index_409`, `test_index_model_mismatch_409`, `test_unhandled_error_500`, `test_unknown_route_format`, `test_duplicate_corpus_name_409`, `test_same_upload_name_replaces`, `test_delete_corpus_file_protected`, `test_reingest_includes_uploads`, `test_content_validation` (×3), `test_latin1_text_accepted`, `test_empty_upload_not_kept`, `test_startup_warmup_lazy_llm_and_lock`, `test_health_llm_configured`; `test_api_ask_real` (llm).
