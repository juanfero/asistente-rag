# M8 — API REST con FastAPI

**Estado:** ⬜ Pendiente · **Estimado:** 3 h · **Depende de:** M7 · **Requisito:** R6

## Objetivo
Exponer ingesta y consultas como API REST documentada (Swagger en `/docs`), reutilizando `ingest.py` y `RAGEngine`.

## Endpoints
| Método | Ruta | Descripción | Respuesta |
|---|---|---|---|
| GET | `/health` | Estado del servicio, modelo LLM configurado, nº chunks | `{status, llm_model, embedding_model, chunks}` |
| GET | `/documents` | Fuentes indexadas con nº de chunks | `[{source, chunks}]` |
| POST | `/documents` | Subir 1..n archivos (`multipart/form-data`), guardarlos en `DOCS_DIR` e ingerirlos | `IngestReport` |
| POST | `/ingest` | Re-ingerir `DOCS_DIR` (body opcional `{reset: bool}`) | `IngestReport` |
| DELETE | `/documents/{source}` | Eliminar documento del índice (y del disco con `?delete_file=true`) | `{deleted_chunks}` |
| POST | `/ask` | Body `{question, top_k?}` | `RAGAnswer` |

## Diseño
- `src/rag/api.py`: `create_app(engine=None, store=None, settings=None) -> FastAPI` (inyección para tests) y `app = create_app()`.
- Dependencias pesadas (embedder, Chroma, LLM) se crean en `lifespan` una sola vez.
- Esquemas Pydantic de request/response en `src/rag/schemas.py`.
- Validaciones: extensión permitida, tamaño máx. 10 MB por archivo, nombre saneado (`Path(name).name`, sin `..`).
- Errores: `EmptyIndexError` (índice vacío, M7) → código HTTP específico a definir en M8; 400 validación, 413 archivo grande, 415 tipo no soportado, 502 `LLMError` genérico con mensaje claro; `LLMQuotaExhaustedError` → **503** `{"error":"LLM_QUOTA_EXHAUSTED","detail":<mensaje>}`, `LLMAuthError` → **503** `"LLM_AUTH_ERROR"`, `LLMRateLimitError` → **429** (ADR-010), 404 documento inexistente.
- CORS habilitado para `localhost:8501` (Streamlit).

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
uvicorn rag.api:app --reload
curl -s localhost:8000/health | python -m json.tool
curl -s -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question":"¿Qué VPN debo usar fuera de la oficina?"}' | python -m json.tool
```
Captura de Swagger y salidas de curl en `evidencias/`.

## Registro de ejecución
| Fecha | Comando / acción | Resultado | Notas |
|---|---|---|---|
| | | | |
