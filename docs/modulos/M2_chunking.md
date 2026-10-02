# M2 — Chunking

**Estado:** ✅ Completado · **Estimado:** 2 h · **Depende de:** M1 · **Requisito:** R2

## Objetivo
Dividir cada `Document` en fragmentos (`Chunk`) de tamaño controlado, con solapamiento, respetando en lo posible los límites naturales del texto (párrafo → línea → oración → palabra), y con **IDs estables** para poder citar y re-ingestar sin duplicar.

## Diseño
- Algoritmo **recursivo por separadores**: intentar partir por `"\n\n"`; si una parte excede `chunk_size`, partir por `"\n"`, luego `". "`, luego `" "`, y en último caso por caracteres. Luego **fusionar** piezas pequeñas hasta acercarse a `chunk_size` y aplicar `chunk_overlap` (las últimas ~`overlap` caracteres del chunk anterior, cortando en límite de palabra).
- `Chunk(text, metadata, chunk_id)` en `models.py`. Metadata hereda la del Document + `chunk_index` (por documento/página), `char_count`.
- `chunk_id` = `sha1(f"{source}|{page}|{chunk_index}|{text}")[:16]` → determinista.
- Descartar chunks con < 30 caracteres no vacíos (ruido).
- *Decisiones del autor (2026-10-02):*
  - Los tests no dependen de 800/120: M2-01, M2-02 y M2-09 se ejecutan con **(800,120) y (500,80)** (tamaño probable tras M3).
  - Se fragmenta **por Document** (en PDF, por página): ningún chunk mezcla páginas. Cada chunk es una subcadena contigua de su Document.
  - Ningún chunk del corpus puede ser solo un encabezado: los encabezados (líneas `#`, títulos numerados cortos sin punto final, líneas en MAYÚSCULAS) se **pegan a la parte siguiente** antes de fusionar (ver ADR-008).
  - La tabla Markdown de permisos (md §3) queda completa, con su encabezado, en un mismo chunk.
  - En el `chunk_id`, `page` vale `""` para md/txt (no tienen página).
  - *Ajuste aprobado:* dentro de la ventana de overlap, el chunk siguiente empieza en el primer límite natural disponible, por preferencia: salto de línea, fin de oración (`. `, `? `, `! `, `: `) y, si no hay, límite de palabra.
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
| 2026-10-02 | Vista previa del chunking sobre el corpus real (800/120 y 500/80) | OK | Ningún chunk solo-encabezado ni terminado en encabezado; máx. 785 / 496 caracteres |
| 2026-10-02 | `pytest tests/unit/test_chunking.py -q` | 48 passed | Pasaron al primer intento → se validaron con 3 mutaciones del código (ver notas) |
| 2026-10-02 | Mutaciones: sin pegado de encabezados / sin solapamiento / tamaño +50 | 2 / 5 / 11 tests fallan | Las pruebas detectan cada defecto inyectado |
| 2026-10-02 | `python scripts/inspeccionar_chunks.py --chunk-size 800 --chunk-overlap 120` y `500 80` | OK | `evidencias/M2_estadisticas.txt` |
| 2026-10-02 | `python scripts/inspeccionar_chunks.py --chunk-size 800 --chunk-overlap 120 --dump` | 16 chunks | `evidencias/M2_chunks_corpus.txt` |
| 2026-10-02 | `pytest -m "not integration" -v` | 149 passed, 1 deselected | Sin regresiones en M0/M1 |
| 2026-10-02 | `ruff check src tests && ruff format --check src tests` | All checks passed! / 12 files already formatted | |
| 2026-10-02 | `git diff --quiet -- data/docs` | Sin cambios | Corpus congelado intacto |
| 2026-10-02 | **fix(M2)**: solapamiento en límite natural (línea > oración > palabra) | 58 passed en `test_chunking.py`; 159 passed en total | 10 pruebas nuevas; mutación "solo palabra" → 6 fallan. Evidencias `M2_*.txt` regeneradas |

**Estado de criterios:** M2-01 ✅ · M2-02 ✅ · M2-03 ✅ · M2-04 ✅ · M2-05 ✅ · M2-06 ✅ · M2-07 ✅ · M2-08 ✅ · M2-09 ✅.

**Pruebas adicionales:** `test_no_cross_page_chunks`, `test_no_heading_only_chunks`, `test_is_heading`, `test_markdown_table_in_one_chunk`, `test_no_content_loss_corpus`, `test_overlap_corpus`, `test_no_overlap_when_zero`, `test_prefers_line_then_sentence`, `test_tiny_chunks_discarded`, `test_invalid_parameters`.

**Estadísticas del corpus:**

| Config | txt | pdf p.1 | pdf p.2 | md | Total | Medio | Mín | Máx |
|---|---|---|---|---|---|---|---|---|
| 800/120 | 6 | 2 | 2 | 6 | **16** | 564 | 133 | 785 |
| 500/80 | 8 | 3 | 2 | 9 | **22** | 414 | 245 | 496 |

_(Valores tras el fix(M2) del solapamiento; antes: 800/120 medio 573, mín 157.)_

**Observaciones:**
- Tras el fix, los chunks empiezan en inicio de línea u oración cuando la ventana de overlap lo permite (p. ej. "Cada colaborador tiene derecho…", "8. Gastos no reembolsables…"); si la ventana no contiene ninguno, empiezan en límite de palabra.
- **Comportamiento conocido (aceptado por el autor):** el último chunk de la página 2 del PDF (133 caracteres con 800/120) es casi todo solapamiento y solo añade "- Gastos personales o de acompañantes.". Se deja así.
