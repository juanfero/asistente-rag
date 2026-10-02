# M3 — Embeddings

**Estado:** ✅ Completado · **Estimado:** 2 h · **Depende de:** M0 · **Requisito:** R3

## Objetivo
Convertir textos en vectores con un modelo **local multilingüe** de `sentence-transformers`, detrás de una interfaz que permita usar un *fake* en pruebas unitarias.

## Diseño
- `src/rag/embeddings.py`:
  - `class Embedder(Protocol)`: `dimension: int`, `embed_documents(texts: list[str]) -> list[list[float]]`, `embed_query(text: str) -> list[float]`.
  - `SentenceTransformerEmbedder(model_name, batch_size=32)`: carga perezosa (*lazy*) del modelo en el primer uso; `normalize_embeddings=True` (así producto punto = coseno); log del tiempo de carga.
  - `FakeEmbedder(dimension=16)`: determinista basado en hash de tokens (bolsa de palabras hasheada + normalización) → textos con palabras en común tienen similitud mayor. Solo para tests.
  - `get_embedder(settings)` fábrica.
- *Decisiones del autor (2026-10-02):*
  - `FakeEmbedder` usa `hashlib.sha1` por token (no `hash()` de Python, que cambia con `PYTHONHASHSEED`); verificado en subprocesos con semillas distintas.
  - `SentenceTransformerEmbedder`: `device="cpu"` explícito, `normalize_embeddings=True`, carga perezosa, `batch_size` configurable; sin prefijos query/passage (este modelo no los usa), `embed_query` = `embed_documents([q])[0]`.
  - Medición con el tokenizer real (incluye 2 tokens especiales) en 800/120, 600/90, 500/80, 400/60 y 350/50; se elige la mayor con ≤ 5 % truncado; si quedara < 400 caracteres se consultaría un cambio de modelo (no fue necesario).
  - Diagnóstico temprano de recuperación con las 9 preguntas de M10 (`scripts/similitud_preguntas.py`, coseno por fuerza bruta en numpy).
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
| 2026-10-02 | `pytest tests/unit/test_embeddings.py -q` | 8 passed | Mutación `hash()` en lugar de sha1 → falla `test_fake_embedder_deterministic_across_processes` |
| 2026-10-02 | Primera descarga + carga del modelo | 40,4 s (+ 11,8 s de `import sentence_transformers`) | Sin `HF_TOKEN` (aviso de rate limit, no bloquea) |
| 2026-10-02 | Carga desde caché | 3,4 s (7,0 s medido dentro del script de similitud) | |
| 2026-10-02 | Caché en disco | **458 MB** | `~/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2` (`du -shL`) |
| 2026-10-02 | Embeddings de todo el corpus (22 chunks, 500/80) | **0,85 s** en CPU | |
| 2026-10-02 | `python scripts/medir_tokens_chunks.py` | Recomendación 500/80 | `evidencias/M3_tokens_chunks.txt`; `max_seq_length` = 128 |
| 2026-10-02 | Chunk truncado con 500/80 | 1 de 22 (136 tokens) | `politica_vacaciones…md` idx 0: se pierden 8 tokens ("las sedes y centros de distribución."), que sí están al inicio del chunk siguiente por el solapamiento |
| 2026-10-02 | Defaults `CHUNK_SIZE=500`, `CHUNK_OVERLAP=80` en `config.py` y `.env.example` | OK | ADR-004; `test_defaults` actualizado; evidencias M2 regeneradas con 500/80 |
| 2026-10-02 | `python scripts/similitud_preguntas.py` | OK | `evidencias/M3_similitud_preguntas.txt` (diagnóstico, no criterio) |
| 2026-10-02 | `pytest -m integration -v -rs` | 3 passed, 1 skipped | `evidencias/M3_pytest_integration.txt`; skip = M0-08 (sin API key) |
| 2026-10-02 | `pytest -m "not integration" -v` | 167 passed, 4 deselected | `evidencias/M3_pytest_ruff.txt` |
| 2026-10-02 | `ruff check src tests && ruff format --check src tests` | All checks passed! / 15 files already formatted | |

**Estado de criterios:** M3-01 ✅ · M3-02 ✅ · M3-03 ✅ · M3-04 ✅ · M3-05 ✅ · M3-06 ✅ (4,5 % con 500/80).

**Medición de tokens (tokenizer real, con tokens especiales; `max_seq_length` = 128):**

| Config | Chunks | Tokens medio | p95 | Máx | > 128 | % excede |
|---|---|---|---|---|---|---|
| 800/120 | 16 | 140.8 | 195.5 | 203 | 10 | 62.5 % |
| 600/90 | 19 | 117.9 | 145.6 | 151 | 6 | 31.6 % |
| **500/80** | **22** | **103.8** | **121.0** | **136** | **1** | **4.5 %** |
| 400/60 | 30 | 75.7 | 105.5 | 114 | 0 | 0.0 % |
| 350/50 | 34 | 67.0 | 94.7 | 109 | 0 | 0.0 % |

**Hallazgos del diagnóstico de similitud (para M7/M10):**
- Q2 recupera la página 2 (USD 90) en 1.er lugar (0,534), pero el chunk de alimentación nacional (pág. 1, COP 120.000) queda en el puesto **7 de 22** (0,355): con `TOP_K=4` no se recuperaría → riesgo de respuesta incompleta en Q2.
- Q8 (no contestable, trabajo remoto) obtiene 0,591, más que Q2 (0,534): un umbral único de similitud **no separa** contestables de no contestables; el rechazo dependerá también del prompt del LLM. Q7 (0,344) y Q9 (0,322) quedan apenas por debajo de `MIN_SCORE=0,35`.

**Tiempo de carga del modelo / embeddings del corpus:** 40,4 s primera vez (descarga) · 3,4 s desde caché · 0,85 s para 22 chunks · **CHUNK_SIZE final:** 500 (overlap 80)
