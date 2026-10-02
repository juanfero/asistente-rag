# M4 — Vector store local (ChromaDB)

**Estado:** ⬜ Pendiente · **Estimado:** 2.5 h · **Depende de:** M2, M3 · **Requisitos:** R4, R7

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
| | | | |
