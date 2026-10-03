# M5 — Pipeline de ingesta + CLI

**Estado:** ✅ Completado · **Estimado:** 2 h · **Depende de:** M1–M4 · **Requisitos:** R1–R5

## Objetivo
Orquestar **cargar → fragmentar → embeber → guardar** en una sola función reutilizable por CLI, API y UI, con un reporte de lo ingerido.

## Diseño
- `src/rag/ingest.py`:
  - `IngestReport(files_processed, files_skipped, documents, chunks_added, total_in_store, sources, duration_s)`.
  - `ingest_paths(paths: list[Path], store, settings, replace=True) -> IngestReport`: acepta archivos o directorios. Con `replace=True`, antes de insertar un archivo ejecuta `delete_by_source` (si el archivo cambió, no quedan chunks viejos).
  - `build_store(settings, embedder=None) -> ChromaVectorStore` (fábrica compartida).
- `src/rag/cli.py` (argparse, `python -m rag.cli`):
  - `ingest <ruta...> [--reset]` → imprime tabla del reporte.
  - `stats` → nº chunks y fuentes.
  - `reset` → vacía la colección (pide confirmación salvo `--yes`).
  - (`ask` se añade en M7.)
  - Código de salida ≠ 0 si no se ingirió ningún documento.
- *Decisiones aprobadas por el autor (2026-10-03):*
  - **Nombres duplicados:** `source` = nombre del archivo; si dos archivos se llaman igual en una misma ingesta, se ingiere el primero y el segundo se omite con *warning* (`files_skipped`).
  - **Vacíos / PDF sin texto** (sin chunks útiles) cuentan como `files_skipped`.
  - **Directorios:** no recursivo por defecto; `--recursive` opcional. Archivos ocultos (p. ej. `.gitkeep`) se ignoran sin contarse.
  - **`reset` y `ingest --reset`** usan `reset_collection()`: funcionan aunque el índice sea de otro modelo de embeddings.
  - **Índice de otro modelo** (`EmbeddingModelMismatchError`): mensaje claro con la solución, código 1, sin traceback.
  - **`stats`** muestra total de chunks, modelo y dimensión de la colección y chunks por fuente.
  - **M5-08** se ejecuta en `tmp_path`; la evidencia usa el índice real `data/chroma/` (ignorado por git).
  - **Limitación documentada:** si un archivo se borra de `data/docs`, sus chunks siguen en el índice hasta `ingest --reset` (sin `--prune`).
  - `IngestReport` incluye además `skipped` (lista "archivo: motivo") para mostrar por qué se omitió cada archivo.

## Criterios de aceptación
| ID | Criterio | Test |
|---|---|---|
| M5-01 | Ingerir `data/docs` (FakeEmbedder, `tmp_path`) procesa los 3 archivos y ≥ 2 documentos (R5) | `test_ingest_corpus` |
| M5-02 | Re-ingesta del mismo directorio no cambia `total_in_store` | `test_reingest_idempotent` |
| M5-03 | Modificar un archivo y re-ingerir reemplaza sus chunks (no quedan los viejos) | `test_replace_modified_file` |
| M5-04 | Archivos no soportados se cuentan en `files_skipped` sin abortar | `test_skips_unsupported` |
| M5-05 | `--reset` vacía antes de ingerir | `test_cli_reset` |
| M5-06 | CLI `ingest` y `stats` devuelven código 0 e imprimen fuentes | `test_cli_ingest_stats` (subprocess o `main(argv)`) |
| M5-07 | Ruta inexistente → mensaje claro y código ≠ 0 | `test_cli_bad_path` |
| M5-08 *(integration)* | Ingesta real con el modelo de embeddings sobre `data/docs` → `data/chroma` | `test_ingest_real` |

## Verificación
```bash
pytest tests/unit/test_ingest.py tests/unit/test_cli.py -q
python -m rag.cli ingest data/docs --reset
python -m rag.cli stats
```
Guardar la salida de estos dos comandos en `evidencias/M5_ingesta.txt`.

## Registro de ejecución
| Fecha | Comando / acción | Resultado | Notas |
|---|---|---|---|
| 2026-10-03 | `pytest tests/unit/test_ingest.py tests/unit/test_cli.py -q` | 13 passed + 1 failed → 14 passed | El fallo era de la prueba (supuso 2 espacios fijos en la tabla de `stats`; la columna se alinea al nombre más largo) → se usa regex |
| 2026-10-03 | Mutaciones: sin `replace` / sin control de duplicados / sin capturar mismatch / sin código de error | 1 / 1 / 1 / 3 fallan | Las pruebas detectan cada defecto |
| 2026-10-03 | `pytest tests/integration/test_ingest_real.py -m integration` | 1 passed | e5 real en `tmp_path`: 3 archivos, 4 documentos, 16 chunks, re-ingesta idempotente, consulta de control (VPN → FortiClient) |
| 2026-10-03 | `python -m rag.cli ingest data/docs --reset` y `stats` | exit 0 / exit 0 | `evidencias/M5_ingesta.txt`; 16 chunks (6 txt, 4 pdf, 6 md), e5 dim 384; `data/chroma/` ignorado por git |
| 2026-10-03 | Ruido de logs: Hugging Face registraba cada petición HTTP (`INFO httpx …`) | Corregido | `setup_logging` deja `httpx`, `huggingface_hub`, `sentence_transformers`, etc. en WARNING (+ prueba). La evidencia filtra además la barra de progreso "Loading weights" y el aviso `HF_TOKEN` (stderr de terceros) |
| 2026-10-03 | `pytest -m "not integration" -v` | 222 passed, 7 deselected | `evidencias/M5_pytest_ruff.txt` |
| 2026-10-03 | `pytest -m integration -v -rs` | 7 passed | `evidencias/M5_pytest_integration.txt` (incluye M0-08 con Gemini) |
| 2026-10-03 | `ruff check src tests && ruff format --check src tests` | All checks passed! / 25 files already formatted | |
| 2026-10-03 | Primer intento de commit | **Bloqueado por el hook de pre-commit** | `evidencias/M5_pytest_ruff.txt` contenía las claves *falsas* de `test_detects_keys` (pytest -v las imprime en el id del caso). Corregido con `ids` fijos y evidencia regenerada; ninguna evidencia tiene hallazgos. Además, el tag `M5-ok` se había creado y subido sobre el commit anterior por un encadenamiento de comandos incorrecto: se borró (local y remoto) y se recreó sobre el commit correcto |

**Estado de criterios:** M5-01 ✅ · M5-02 ✅ · M5-03 ✅ · M5-04 ✅ · M5-05 ✅ · M5-06 ✅ · M5-07 ✅ · M5-08 ✅.

**Pruebas adicionales:** `test_skips_empty_and_textless_pdf`, `test_duplicate_name_skipped`, `test_recursive_flag`, `test_single_file_and_missing_path`, `test_cli_nothing_ingested`, `test_cli_reset_command`, `test_cli_model_mismatch`.
