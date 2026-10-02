# M2 — Chunking

**Estado:** ⬜ Pendiente · **Estimado:** 2 h · **Depende de:** M1 · **Requisito:** R2

## Objetivo
Dividir cada `Document` en fragmentos (`Chunk`) de tamaño controlado, con solapamiento, respetando en lo posible los límites naturales del texto (párrafo → línea → oración → palabra), y con **IDs estables** para poder citar y re-ingestar sin duplicar.

## Diseño
- Algoritmo **recursivo por separadores**: intentar partir por `"\n\n"`; si una parte excede `chunk_size`, partir por `"\n"`, luego `". "`, luego `" "`, y en último caso por caracteres. Luego **fusionar** piezas pequeñas hasta acercarse a `chunk_size` y aplicar `chunk_overlap` (las últimas ~`overlap` caracteres del chunk anterior, cortando en límite de palabra).
- `Chunk(text, metadata, chunk_id)` en `models.py`. Metadata hereda la del Document + `chunk_index` (por documento/página), `char_count`.
- `chunk_id` = `sha1(f"{source}|{page}|{chunk_index}|{text}")[:16]` → determinista.
- Descartar chunks con < 30 caracteres no vacíos (ruido).
- Por qué 800/120 caracteres: los documentos son cortos y con secciones; ~800 caracteres ≈ 1 sección pequeña ≈ 150–200 tokens, cabe holgado en el modelo de embeddings (límite 128 *word pieces* recomendado → **verificar** en M3 y ajustar si trunca; registrar en ADR).

## Tareas
1. `src/rag/chunking.py`: `split_text(text, chunk_size, chunk_overlap) -> list[str]` y `chunk_documents(docs, chunk_size, chunk_overlap) -> list[Chunk]`.
2. Script/notebook rápido `scripts/inspeccionar_chunks.py` que imprime nº de chunks por documento, tamaño medio/min/max y los 2 primeros chunks (para evidencia).
3. Registrar estadísticas del corpus en el registro de ejecución.

## Criterios de aceptación
| ID | Criterio | Test |
|---|---|---|
| M2-01 | Ningún chunk supera `chunk_size` (+ tolerancia 0) | `test_max_size` (parametrizado con varios tamaños) |
| M2-02 | Chunks consecutivos comparten solapamiento > 0 cuando `overlap > 0` y el texto se partió | `test_overlap` |
| M2-03 | No se pierde contenido: toda palabra del texto original aparece en algún chunk | `test_no_content_loss` |
| M2-04 | Se prefiere cortar en párrafo: texto con 3 párrafos cortos y `chunk_size` grande → 1 chunk; con `chunk_size` pequeño → cortes en `\n\n` | `test_respects_paragraphs` |
| M2-05 | Palabra más larga que `chunk_size` no causa bucle infinito | `test_long_token` |
| M2-06 | Metadatos heredados + `chunk_index` secuencial por documento/página | `test_metadata` |
| M2-07 | `chunk_id` es determinista (mismo input → mismo id) y único en el corpus | `test_stable_ids` |
| M2-08 | Texto vacío/espacios → 0 chunks; chunks < 30 caracteres descartados | `test_empty_and_tiny` |
| M2-09 | Con el corpus real: ≥ 1 chunk por documento y por página PDF | `test_corpus_chunking` |

## Verificación
```bash
pytest tests/unit/test_chunking.py -q
python scripts/inspeccionar_chunks.py
pytest -m "not integration" -q && ruff check src tests
```

## Registro de ejecución
| Fecha | Comando / acción | Resultado | Notas |
|---|---|---|---|
| | | | |

**Estadísticas del corpus:** _(nº chunks por documento, tamaño medio)_
