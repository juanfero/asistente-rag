# M3 — Embeddings

**Estado:** ⬜ Pendiente · **Estimado:** 2 h · **Depende de:** M0 · **Requisito:** R3

## Objetivo
Convertir textos en vectores con un modelo **local multilingüe** de `sentence-transformers`, detrás de una interfaz que permita usar un *fake* en pruebas unitarias.

## Diseño
- `src/rag/embeddings.py`:
  - `class Embedder(Protocol)`: `dimension: int`, `embed_documents(texts: list[str]) -> list[list[float]]`, `embed_query(text: str) -> list[float]`.
  - `SentenceTransformerEmbedder(model_name, batch_size=32)`: carga perezosa (*lazy*) del modelo en el primer uso; `normalize_embeddings=True` (así producto punto = coseno); log del tiempo de carga.
  - `FakeEmbedder(dimension=16)`: determinista basado en hash de tokens (bolsa de palabras hasheada + normalización) → textos con palabras en común tienen similitud mayor. Solo para tests.
  - `get_embedder(settings)` fábrica.
- Revisar `model.max_seq_length` (MiniLM-L12 multilingüe = 128 tokens). Medir con el tokenizer cuántos tokens tienen los chunks de M2; si > 128 se truncan → **reducir `CHUNK_SIZE`** (p. ej. 500/80) y documentar en ADR-004.

## Tareas
1. Implementar interfaz, implementación real, fake y fábrica.
2. `scripts/medir_tokens_chunks.py`: % de chunks que exceden `max_seq_length`.
3. Ajustar `CHUNK_SIZE/CHUNK_OVERLAP` si aplica y re-correr tests de M2.

## Criterios de aceptación
| ID | Criterio | Test |
|---|---|---|
| M3-01 | `FakeEmbedder` determinista, dimensión correcta, vectores normalizados (norma ≈ 1) | `test_fake_embedder` |
| M3-02 | Lista vacía → lista vacía sin cargar el modelo | `test_empty_input` |
| M3-03 | El modelo real no se carga al instanciar (lazy) | `test_lazy_loading` (mock) |
| M3-04 *(integration)* | Modelo real: dimensión 384, vectores normalizados, `embed_query` y `embed_documents` consistentes | `test_real_embedder_shape` |
| M3-05 *(integration)* | Sanidad semántica: sim("¿cuántos días de vacaciones tengo?", chunk de vacaciones) > sim(misma pregunta, chunk de VPN) | `test_semantic_sanity` |
| M3-06 *(integration)* | ≤ 5% de los chunks del corpus excede `max_seq_length` (o se ajustó el tamaño) | `test_chunks_fit_model` |

## Verificación
```bash
pytest tests/unit/test_embeddings.py -q
pytest -m integration tests/integration/test_embeddings_real.py -q   # descarga ~470 MB la 1ª vez
python scripts/medir_tokens_chunks.py
```

## Registro de ejecución
| Fecha | Comando / acción | Resultado | Notas |
|---|---|---|---|
| | | | |

**Tiempo de carga del modelo / embeddings del corpus:** _(completar)_ · **CHUNK_SIZE final:** _(completar)_
