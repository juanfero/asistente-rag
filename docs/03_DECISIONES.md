# Registro de Decisiones de Arquitectura (ADR)

Formato: contexto → decisión → alternativas → consecuencias. Una entrada por decisión no trivial.

---

## ADR-001 — Pipeline RAG sin frameworks (sin LangChain/LlamaIndex)
- **Fecha:** 2026-10-01 · **Estado:** Aceptada
- **Contexto:** el caso evalúa que el candidato entienda y explique lo que construyó; el prototipo es pequeño.
- **Decisión:** implementar loaders, chunking, vector store y motor RAG con componentes propios + librerías puntuales (`pypdf`, `sentence-transformers`, `chromadb`, `openai`).
- **Alternativas:** LangChain, LlamaIndex (más rápidos de armar, pero ocultan el mecanismo y añaden dependencias).
- **Consecuencias:** más código propio, pero cada paso es testeable y explicable en el video.

## ADR-002 — Stack: LLM vía API compatible con OpenAI + sentence-transformers + ChromaDB + FastAPI/Streamlit, en Linux
- **Fecha:** 2026-10-01 · **Estado:** Aceptada
- **Decisión:** LLM **Gemini** vía endpoint compatible con OpenAI (`https://generativelanguage.googleapis.com/v1beta/openai/`) (**actualizado por ADR-010**; inicialmente Grok, `https://api.x.ai/v1`); embeddings locales multilingües `intfloat/multilingual-e5-small` con prefijos `query: `/`passage: ` (**actualizado por ADR-009**; inicialmente `paraphrase-multilingual-MiniLM-L12-v2`); Chroma persistente con coseno; FastAPI como núcleo y Streamlit como UI; desarrollo en Linux.
- **Motivo:** elección del candidato para el LLM; embeddings gratis/offline y buenos en español; Chroma guarda metadatos para citar; FastAPI y Streamlit son las opciones recomendadas por el caso.
- **Consecuencias:** requiere `GEMINI_API_KEY` con créditos; primera ejecución descarga ~490 MB del modelo de embeddings (e5-small).

## ADR-003 — Modelo LLM concreto
- **Fecha:** 2026-10-03 · **Estado:** Aceptada
- **Decisión:** `GEMINI_MODEL=gemini-3.1-flash-lite` (elegido por el autor). `scripts/check_gemini.py` confirmó que está entre los 61 modelos disponibles para la key (`evidencias/M0_check_gemini.txt`).
- **Tokens de razonamiento (verificación pedida):** con `max_tokens=20` y con `max_tokens=700` la respuesta a "Responde solo: OK" fue `'OK'`, `finish_reason=stop`, `completion_tokens=1`; el endpoint compatible no reporta `reasoning_tokens` (`None`). **No se observó** que el razonamiento consuma `max_tokens` en respuestas cortas, así que no se cambia ninguna configuración.
- **Pendiente para M6 (propuesta, no decidida):** la prueba usó un prompt trivial. En M6 conviene verificar `finish_reason` con un prompt RAG real y, si aparecen respuestas vacías o cortadas (`length`), evaluar subir `LLM_MAX_TOKENS` o enviar `reasoning_effort` (p. ej. `"low"`/`"none"`) por el endpoint compatible. Cualquier cambio se consulta con el autor.
- **Historial:** con Grok el default era `grok-3-mini` y nunca se verificó (sin key).

## ADR-004 — Tamaño de chunk definitivo
- **Fecha:** 2026-10-02 · **Estado:** Aceptada — **actualizada por ADR-009: `CHUNK_SIZE=800`, `CHUNK_OVERLAP=120`**
- **Actualización (M3.1, e5-small, 512 tokens; medición con prefijo `passage: `):**

| Config | Chunks | Tokens medio | p95 | Máx | > 512 | % excede |
|---|---|---|---|---|---|---|
| **800/120** | **16** | **142.8** | **197.5** | **205** | **0** | **0.0 %** |
| 600/90 | 19 | 119.9 | 147.6 | 153 | 0 | 0.0 % |
| 500/80 | 22 | 105.8 | 123.0 | 138 | 0 | 0.0 % |
| 400/60 | 30 | 77.7 | 107.5 | 116 | 0 | 0.0 % |
| 350/50 | 34 | 69.0 | 96.7 | 111 | 0 | 0.0 % |

  Con e5 ningún tamaño se trunca; se elige el mayor (800/120, 16 chunks) siguiendo la misma regla. La medición original con MiniLM se conserva abajo como historial.

**Historial — medición inicial con MiniLM (128 tokens):**
- **Contexto:** `paraphrase-multilingual-MiniLM-L12-v2` tiene `max_seq_length` = 128 tokens (incluye 2 especiales); lo que exceda se trunca y no se representa en el embedding. Con 800/120, el 62,5 % de los chunks se truncaba.
- **Medición** (`scripts/medir_tokens_chunks.py`, tokenizer real, corpus de M1):

| Config | Chunks | Tokens medio | p95 | Máx | > 128 | % excede |
|---|---|---|---|---|---|---|
| 800/120 | 16 | 140.8 | 195.5 | 203 | 10 | 62.5 % |
| 600/90 | 19 | 117.9 | 145.6 | 151 | 6 | 31.6 % |
| **500/80** | **22** | **103.8** | **121.0** | **136** | **1** | **4.5 %** |
| 400/60 | 30 | 75.7 | 105.5 | 114 | 0 | 0.0 % |
| 350/50 | 34 | 67.0 | 94.7 | 109 | 0 | 0.0 % |

- **Regla:** elegir la configuración más grande con ≤ 5 % de chunks truncados; si quedara por debajo de 400 caracteres, evaluar un modelo con contexto de 512 tokens (p. ej. `intfloat/multilingual-e5-small`).
- **Decisión inicial (sustituida por ADR-009):** `CHUNK_SIZE=500`, `CHUNK_OVERLAP=80`.
- **Consecuencias (de la decisión inicial):** 22 chunks en el corpus. Solo 1 chunk (136 tokens) se trunca y lo perdido (8 tokens) está al inicio del chunk siguiente gracias al solapamiento. Se mantiene el modelo (sin cambio de stack). `test_chunks_fit_model` (integración) protege la regla si cambian el corpus o el tamaño.

## ADR-005 — Umbral de relevancia (MIN_SCORE) y top_k
- **Fecha:** _(M7)_ · **Estado:** Pendiente
- **Criterio acordado (M3.1):** el margen medido es negativo con todos los modelos (ADR-009), así que `MIN_SCORE` **no** separa contestable de no contestable. Se calibrará como **filtro de ruido**: `MIN_SCORE = (mínimo top-1 de Q1–Q6) − 0.05`; la abstención la decide el LLM con el prompt. Valor provisional: `0.80`.

## ADR-006 — PyTorch CPU-only
- **Fecha:** 2026-10-02 · **Estado:** Aceptada
- **Contexto:** `sentence-transformers` depende de `torch`; desde PyPI se instala la variante con CUDA (varios GB más paquetes `nvidia-*`). El prototipo solo calcula embeddings de un corpus pequeño y no requiere GPU.
- **Decisión:** instalar `torch` desde el índice CPU de PyTorch **antes** del paquete: `pip install torch --index-url https://download.pytorch.org/whl/cpu`, luego `pip install -e ".[dev]"`. `requirements.txt` se genera con `pip freeze --exclude-editable` y su primera línea es `--extra-index-url https://download.pytorch.org/whl/cpu`, para que el pin `torch==X.Y.Z+cpu` se resuelva.
- **Alternativas:** torch con CUDA desde PyPI (descarga mucho mayor, sin beneficio aquí); fijar torch en `pyproject.toml` con una URL directa (rompe la portabilidad entre plataformas).
- **Consecuencias:** instalación más liviana y rápida; no hay `nvidia-*` en `requirements.txt` (verificado). Quien instale con `pip install -e .` sin el paso previo obtendría torch con CUDA; el README (M11) documentará el orden de instalación.

## ADR-007 — Convención de nombres de configuración
- **Fecha:** 2026-10-02 · **Estado:** Aceptada
- **Contexto:** los documentos de módulo nombraban la configuración como `settings.XAI_MODEL`, `settings.XAI_BASE_URL`, etc. (hoy `GEMINI_*`, ADR-010), pero en Python la convención para atributos es snake_case minúscula.
- **Decisión:** los atributos de `Settings` van en **snake_case minúscula** (`settings.gemini_api_key`, `settings.chunk_size`, …). Las variables de entorno y `.env.example` se mantienen en **MAYÚSCULAS**; `pydantic-settings` las mapea porque no distingue mayúsculas/minúsculas (`case_sensitive=False`).
- **Equivalencia:** toda referencia en la documentación del tipo `settings.GEMINI_MODEL` equivale a `settings.gemini_model` (y así para cada variable).
- **Consecuencias:** código idiomático sin cambiar la interfaz de configuración (`.env`) que se documenta al usuario.

## ADR-008 — Encabezados pegados a su contenido en el chunking
- **Fecha:** 2026-10-02 · **Estado:** Aceptada
- **Contexto:** el autor exige que ningún chunk del corpus sea solo un encabezado (p. ej. "## 5. Día de cumpleaños") sin el contenido que le sigue. Un splitter recursivo puro puede dejar el título al final de un chunk y su contenido en el siguiente.
- **Decisión:** antes de fusionar, en cada nivel de separación las partes que son encabezado (línea Markdown `#`, título numerado corto sin puntuación final o línea en MAYÚSCULAS, ≤ 80 caracteres) se pegan a la siguiente parte con contenido (`is_heading` + `_glue_headings` en `chunking.py`).
- **Alternativas:** chunking por secciones Markdown (no sirve para txt/pdf); post-procesar moviendo títulos huérfanos (más complejo y puede romper el límite de tamaño).
- **Consecuencias:** heurística simple y genérica para los 3 formatos, validada sobre el corpus real con 800/120 y 500/80. Una línea corta en MAYÚSCULAS o numerada sin punto se trataría como título; si aparecen falsos positivos en otros corpus se ajusta la regla.

## ADR-009 — Modelo de embeddings: intfloat/multilingual-e5-small (800/120, prefijos)
- **Fecha:** 2026-10-02 · **Estado:** Aceptada (sustituye la elección de embeddings de ADR-002 y el tamaño de ADR-004)
- **Contexto:** el diagnóstico de M3 mostró baja discriminación con `paraphrase-multilingual-MiniLM-L12-v2`: Q8 (sin respuesta) 0,591 > Q2 (contestable) 0,534, y el chunk de "COP 120.000" quedaba en el puesto 7 para Q2. MiniLM está entrenado para paráfrasis/similitud semántica (STS), no para recuperar pasajes a partir de una pregunta, y solo admite 128 tokens.
- **Evaluación** (`scripts/comparar_embeddings.py`, gold set por texto literal; `evidencias/M3_comparacion_embeddings.txt`):

| Var. | Modelo y chunk | Chunks | Recall@4 | Rango COP 120.000 (Q2) | MRR Q1–Q6 | mín top-1 Q1–Q6 | máx top-1 Q7–Q9 | Margen | Carga (caché) | Disco |
|---|---|---|---|---|---|---|---|---|---|---|
| A | paraphrase-multilingual-MiniLM-L12-v2 500/80 | 22 | 87.5 % | 7 | 0.833 | 0.534 | 0.591 | -0.057 | 7.1 s* | 480 MB |
| B | multilingual-e5-small 500/80 | 22 | 100.0 % | 1 | 0.917 | 0.869 | 0.877 | -0.008 | 4.9 s | 493 MB |
| **C** | **multilingual-e5-small 800/120** | **16** | **100.0 %** | **1** | **1.000** | **0.867** | **0.871** | **-0.004** | **4.4 s** | **493 MB** |

| Var. | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 | Q9 |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.699 | 0.534 | 0.621 | 0.545 | 0.737 | 0.546 | 0.344 | 0.591 | 0.322 |
| B | 0.902 | 0.877 | 0.869 | 0.883 | 0.890 | 0.874 | 0.842 | 0.877 | 0.833 |
| C | 0.900 | 0.867 | 0.885 | 0.883 | 0.884 | 0.870 | 0.861 | 0.871 | 0.834 |

\*La carga de A incluye la importación de `sentence_transformers` (fue la primera variante evaluada).

- **Decisión:** variante **C** — `EMBEDDING_MODEL=intfloat/multilingual-e5-small`, `EMBEDDING_QUERY_PREFIX="query: "`, `EMBEDDING_PASSAGE_PREFIX="passage: "`, `CHUNK_SIZE=800`, `CHUNK_OVERLAP=120`.
- **Motivo:** e5 está entrenado para recuperación pregunta→pasaje (con prefijos) y admite 512 tokens: Recall@4 100 %, MRR 1,000, Q2 recupera ambas páginas en el top-2 y 0 % de chunks truncados. Misma dimensión (384) y tamaño similar en disco.
- **Alternativas:** A (MiniLM 500/80, peor recuperación y truncado); B (e5 500/80, MRR 0,917 y más chunks).
- **Consecuencias:**
  - Los prefijos **solo** se aplican al codificar; el texto guardado y el `chunk_id` no los llevan (`test_prefix_only_at_encoding`).
  - e5 concentra las similitudes en ~0,81–0,90 y ningún modelo deja margen positivo: `MIN_SCORE` pasa a ser filtro de ruido (provisional 0,80) y la abstención la decide el LLM (ADR-005).
  - El criterio M3-04 se redefine: `embed_query`/`embed_documents` son consistentes con su prefijo respectivo, muy similares (> 0,9) y **distintos** entre sí.
  - Cualquier índice creado con otro modelo debe regenerarse (M4 valida modelo y dimensión de la colección).
  - Descartado el "encabezado de sección en cada chunk" (recall ya 100 %): queda en Mejoras futuras (M11).

## ADR-010 — LLM: Gemini en lugar de Grok
- **Fecha:** 2026-10-03 · **Estado:** Aceptada (sustituye la elección de LLM de ADR-002)
- **Contexto:** Grok (xAI) no funcionó para el autor; Gemini sí.
- **Decisión:** usar **Gemini** a través de su endpoint compatible con OpenAI (`https://generativelanguage.googleapis.com/v1beta/openai/`) con el SDK `openai` ya presente. Variables `GEMINI_API_KEY` (`SecretStr`), `GEMINI_BASE_URL`, `GEMINI_MODEL` (default `gemini-3.1-flash-lite`, elegido por el autor; se confirma en ADR-003). Sin compatibilidad con `XAI_*`.
- **Alternativas:** SDK oficial `google-genai` (más completo, pero añade dependencia y obliga a reescribir el mapeo de errores de M6, que hoy usa las excepciones del SDK de OpenAI).
- **Consecuencias:**
  - `scripts/check_xai.py` → `scripts/check_gemini.py`; `test_xai_connection.py` → `test_gemini_connection.py`; `M6_llm_grok.md` → `M6_llm_gemini.md`.
  - Nuevos criterios de errores de créditos/autenticación/límite (M6-08…M6-10) y su propagación a CLI (código 2), API (503/429) y UI (`st.warning`) en M7, M8 y M9.
  - Seguridad reforzada: `tests/unit/test_no_secrets.py`, `scripts/secret_scan.py` y hook `scripts/pre-commit`; la key nunca se imprime (solo `****` + últimos 4).
  - Gemini 3.x puede consumir `max_tokens` en razonamiento: se verifica en `check_gemini.py` y se documenta en ADR-003.
  - **Regla de clasificación de errores 402/429** (en orden, sin distinguir mayúsculas, sobre el mensaje + el cuerpo/detalles del error, incluido el cuerpo envuelto en lista `[{"error": …}]`):
    a. HTTP 402 → `LLMQuotaExhaustedError`.
    b. 429 que contiene `PerDay` (p. ej. `quotaId` `GenerateRequestsPerDayPerProjectPerModel-FreeTier`) → `LLMQuotaExhaustedError`.
    c. 429 que contiene `PerMinute` → `LLMRateLimitError`.
    d. 429 que contiene `billing`, `credit` o `prepay` → `LLMQuotaExhaustedError`.
    e. Cualquier otro 429 → `LLMRateLimitError`.
    `quota` **no** se usa: Gemini lo incluye en casi todos los 429 ("You exceeded your current quota"), también en los de límite por minuto. Se clasifica después de los reintentos del SDK (`max_retries=2`).
  - Pruebas que llaman a Gemini: marcador `llm`, opt-in con `RUN_LLM=1` (sin costo por defecto).
