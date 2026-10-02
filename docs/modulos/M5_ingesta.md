# M5 — Pipeline de ingesta + CLI

**Estado:** ⬜ Pendiente · **Estimado:** 2 h · **Depende de:** M1–M4 · **Requisitos:** R1–R5

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
| | | | |
