# Registro de Decisiones de Arquitectura (ADR)

Formato: contexto → decisión → alternativas → consecuencias. Una entrada por decisión no trivial.

---

## ADR-001 — Pipeline RAG sin frameworks (sin LangChain/LlamaIndex)
- **Fecha:** 2026-10-01 · **Estado:** Aceptada
- **Contexto:** el caso evalúa que el candidato entienda y explique lo que construyó; el prototipo es pequeño.
- **Decisión:** implementar loaders, chunking, vector store y motor RAG con componentes propios + librerías puntuales (`pypdf`, `sentence-transformers`, `chromadb`, `openai`).
- **Alternativas:** LangChain, LlamaIndex (más rápidos de armar, pero ocultan el mecanismo y añaden dependencias).
- **Consecuencias:** más código propio, pero cada paso es testeable y explicable en el video.

## ADR-002 — Stack: Grok (xAI) + sentence-transformers + ChromaDB + FastAPI/Streamlit, en Linux
- **Fecha:** 2026-10-01 · **Estado:** Aceptada
- **Decisión:** LLM Grok vía API compatible con OpenAI (`https://api.x.ai/v1`); embeddings locales multilingües `paraphrase-multilingual-MiniLM-L12-v2`; Chroma persistente con coseno; FastAPI como núcleo y Streamlit como UI; desarrollo en Linux.
- **Motivo:** elección del candidato para el LLM; embeddings gratis/offline y buenos en español; Chroma guarda metadatos para citar; FastAPI y Streamlit son las opciones recomendadas por el caso.
- **Consecuencias:** requiere `XAI_API_KEY` con saldo; primera ejecución descarga ~470 MB del modelo de embeddings.

## ADR-003 — Modelo Grok concreto
- **Fecha:** _(M0)_ · **Estado:** Pendiente — sin `XAI_API_KEY` al cierre de M0 (2026-10-02); se mantiene el default `grok-3-mini` hasta ejecutar `scripts/check_xai.py`.
- **Decisión:** _(modelo elegido tras `scripts/check_xai.py`, y por qué: costo/latencia/calidad)_

## ADR-004 — Tamaño de chunk definitivo
- **Fecha:** 2026-10-02 · **Estado:** Aceptada
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
- **Decisión:** `CHUNK_SIZE=500`, `CHUNK_OVERLAP=80` (defaults en `config.py` y `.env.example`).
- **Consecuencias:** 22 chunks en el corpus. Solo 1 chunk (136 tokens) se trunca y lo perdido (8 tokens) está al inicio del chunk siguiente gracias al solapamiento. Se mantiene el modelo (sin cambio de stack). `test_chunks_fit_model` (integración) protege la regla si cambian el corpus o el tamaño.

## ADR-005 — Umbral de relevancia (MIN_SCORE) y top_k
- **Fecha:** _(M7)_ · **Estado:** Pendiente

## ADR-006 — PyTorch CPU-only
- **Fecha:** 2026-10-02 · **Estado:** Aceptada
- **Contexto:** `sentence-transformers` depende de `torch`; desde PyPI se instala la variante con CUDA (varios GB más paquetes `nvidia-*`). El prototipo solo calcula embeddings de un corpus pequeño y no requiere GPU.
- **Decisión:** instalar `torch` desde el índice CPU de PyTorch **antes** del paquete: `pip install torch --index-url https://download.pytorch.org/whl/cpu`, luego `pip install -e ".[dev]"`. `requirements.txt` se genera con `pip freeze --exclude-editable` y su primera línea es `--extra-index-url https://download.pytorch.org/whl/cpu`, para que el pin `torch==X.Y.Z+cpu` se resuelva.
- **Alternativas:** torch con CUDA desde PyPI (descarga mucho mayor, sin beneficio aquí); fijar torch en `pyproject.toml` con una URL directa (rompe la portabilidad entre plataformas).
- **Consecuencias:** instalación más liviana y rápida; no hay `nvidia-*` en `requirements.txt` (verificado). Quien instale con `pip install -e .` sin el paso previo obtendría torch con CUDA; el README (M11) documentará el orden de instalación.

## ADR-007 — Convención de nombres de configuración
- **Fecha:** 2026-10-02 · **Estado:** Aceptada
- **Contexto:** los documentos de módulo nombran la configuración como `settings.XAI_MODEL`, `settings.XAI_BASE_URL`, etc., pero en Python la convención para atributos es snake_case minúscula.
- **Decisión:** los atributos de `Settings` van en **snake_case minúscula** (`settings.xai_api_key`, `settings.chunk_size`, …). Las variables de entorno y `.env.example` se mantienen en **MAYÚSCULAS**; `pydantic-settings` las mapea porque no distingue mayúsculas/minúsculas (`case_sensitive=False`).
- **Equivalencia:** toda referencia en la documentación del tipo `settings.XAI_MODEL` equivale a `settings.xai_model` (y así para cada variable).
- **Consecuencias:** código idiomático sin cambiar la interfaz de configuración (`.env`) que se documenta al usuario.

## ADR-008 — Encabezados pegados a su contenido en el chunking
- **Fecha:** 2026-10-02 · **Estado:** Aceptada
- **Contexto:** el autor exige que ningún chunk del corpus sea solo un encabezado (p. ej. "## 5. Día de cumpleaños") sin el contenido que le sigue. Un splitter recursivo puro puede dejar el título al final de un chunk y su contenido en el siguiente.
- **Decisión:** antes de fusionar, en cada nivel de separación las partes que son encabezado (línea Markdown `#`, título numerado corto sin puntuación final o línea en MAYÚSCULAS, ≤ 80 caracteres) se pegan a la siguiente parte con contenido (`is_heading` + `_glue_headings` en `chunking.py`).
- **Alternativas:** chunking por secciones Markdown (no sirve para txt/pdf); post-procesar moviendo títulos huérfanos (más complejo y puede romper el límite de tamaño).
- **Consecuencias:** heurística simple y genérica para los 3 formatos, validada sobre el corpus real con 800/120 y 500/80. Una línea corta en MAYÚSCULAS o numerada sin punto se trataría como título; si aparecen falsos positivos en otros corpus se ajusta la regla.
