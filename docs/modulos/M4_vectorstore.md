# M4 — Vector store local (ChromaDB)

**Estado:** ✅ Completado · **Estimado:** 2.5 h · **Depende de:** M2, M3 · **Requisitos:** R4, R7

## Objetivo
Persistir chunks + embeddings + metadatos en ChromaDB y consultar por similitud devolviendo **score de similitud coseno** y metadatos para citar.

## Diseño
- `src/rag/vectorstore.py` → `class ChromaVectorStore`:
  - `__init__(persist_dir, collection_name, embedder)`: `chromadb.PersistentClient(path=...)`, colección con `metadata={"hnsw:space": "cosine"}`. **Los embeddings los calculamos nosotros** (no se usa la embedding function de Chroma) → control total y testeable con `FakeEmbedder`.
  - `add_chunks(chunks) -> int`: `upsert` por lotes (ids = `chunk_id`) → idempotente.
  - `query(text, top_k) -> list[RetrievedChunk]`: `RetrievedChunk(chunk_id, text, metadata, score)` con `score = 1 - distance` (coseno), ordenado desc.
  - `delete_by_source(source) -> int`, `reset()`, `count() -> int`, `list_sources() -> list[str]`.
  - Metadatos en Chroma solo admiten tipos primitivos → no guardar `None` (omitir `page` si no aplica).
- `RetrievedChunk` en `models.py`.
- *Decisiones del autor (2026-10-03):*
  - **A.** `PersistentClient` con `Settings(anonymized_telemetry=False)` (sin ruido ni llamadas de red).
  - **B.** La metadata de la colección guarda `embedding_model`, `embedding_dim`, `passage_prefix` y `hnsw:space=cosine`. Al abrir una colección existente con otro modelo o dimensión → `EmbeddingModelMismatchError` con mensaje que sugiere el reset; `reset_collection(persist_dir, name)` borra la colección sin validar.
  - **C.** Colección vacía → `[]`; `top_k > count` → `min(top_k, count)`; `top_k < 1` → `ValueError`.
  - **D.** `source_stats() -> dict[str, int]` (chunks por fuente, para `GET /documents` en M8); `list_sources()` se deriva de ahí.
  - **E.** Upsert por lotes de `client.get_max_batch_size()` (5461 en chromadb 1.5.9).
  - **F.** Consistencia con coseno por fuerza bruta (numpy) con `FakeEmbedder`.
  - **G.** Integración con e5 real: Q2 trae "COP 120.000" y "USD 90" en el top-4.
  - `score = 1 − distancia coseno`, acotado a [0, 1]. El texto guardado y el id no llevan prefijo (el prefijo lo aplica el embedder al codificar).
  - Hallazgo: chromadb **descarta toda la metadata del registro** si un valor es `None`; `_clean_metadata` omite los `None` y convierte tipos no primitivos a `str`.

## Criterios de aceptación (usar `tmp_path` + `FakeEmbedder`)
| ID | Criterio | Test |
|---|---|---|
| M4-01 | `add_chunks` + `count` reflejan el nº de chunks | `test_add_and_count` |
| M4-02 | Insertar dos veces los mismos chunks no duplica (upsert) | `test_idempotent_upsert` |
| M4-03 | `query` devuelve ≤ `top_k` resultados ordenados por score desc, score en [0,1] | `test_query_order_and_scores` |
| M4-04 | El chunk con texto idéntico a la consulta queda primero con score ≈ 1 | `test_exact_match_first` |
| M4-05 | Metadatos (`source`, `page`, `chunk_index`) se recuperan intactos | `test_metadata_roundtrip` |
| M4-06 | Persistencia: nueva instancia sobre el mismo directorio ve los datos | `test_persistence` |
| M4-07 | `delete_by_source` elimina solo ese documento; `list_sources` lo refleja | `test_delete_by_source` |
| M4-08 | `reset` deja la colección vacía; `query` en colección vacía → lista vacía | `test_reset_and_empty_query` |
| M4-09 | Metadatos con `None` no rompen la inserción | `test_none_metadata` |

## Verificación
```bash
pytest tests/unit/test_vectorstore.py -q
pytest -m "not integration" -q && ruff check src tests
```

## Registro de ejecución
| Fecha | Comando / acción | Resultado | Notas |
|---|---|---|---|
| 2026-10-03 | Sondeo de la API de chromadb 1.5.9 | OK | Nombre de colección ≥ 3 caracteres; lote máx. 5461; `n_results > count` no falla; `None` en metadata borra toda la metadata del registro |
| 2026-10-03 | `pytest tests/unit/test_vectorstore.py -q` | 19 passed | Pasaron al primer intento → validadas con 4 mutaciones |
| 2026-10-03 | Mutaciones: métrica `l2` / sin limpiar metadata / sin validar modelo / sin lotes | 2 / 1 / 2 / 1 fallan | `test_cosine_consistency_with_numpy` detecta el cambio de métrica |
| 2026-10-03 | `pytest tests/integration/test_vectorstore_real.py -m integration` | 2 passed | Q2: pág. 1 (0,867) y pág. 2 (0,863) en el top-2 |
| 2026-10-03 | `python scripts/consultar_chroma.py` | OK | `evidencias/M4_consulta_chroma.txt` (Chroma temporal; la ingesta a `data/chroma` es de M5). Scores idénticos a la fuerza bruta de M3 |
| 2026-10-03 | `pytest -m "not integration" -v` | 191 passed, 6 deselected | `evidencias/M4_pytest_ruff.txt` |
| 2026-10-03 | `pytest -m integration -v -rs` | 5 passed, 1 skipped | `evidencias/M4_pytest_integration.txt`; skip = M0-08 (sin API key) |
| 2026-10-03 | `ruff check src tests && ruff format --check src tests` | All checks passed! / 18 files already formatted | |

**Estado de criterios:** M4-01 ✅ · M4-02 ✅ · M4-03 ✅ · M4-04 ✅ · M4-05 ✅ · M4-06 ✅ · M4-07 ✅ · M4-08 ✅ · M4-09 ✅.

**Pruebas adicionales (A–G):** `test_collection_metadata`, `test_model_mismatch`, `test_dimension_mismatch`, `test_reset_collection_after_mismatch`, `test_top_k_larger_than_count`, `test_top_k_invalid`, `test_source_stats`, `test_upsert_in_batches`, `test_cosine_consistency_with_numpy`, `test_stored_text_has_no_prefix`; integración `test_q2_both_pages_in_top4`, `test_collection_records_model`.
