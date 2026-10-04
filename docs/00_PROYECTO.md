# Asistente Documental RAG — Documento General del Proyecto

> Prueba técnica **AI Developer Engineer Junior — I Cloud Seven (IC7)**
> Perfil: *Generative AI, RAG & Cloud Development Foundations* · Tiempo sugerido: 24–36 h
> Autor: Juan Felipe Rojas · Inicio: 2026-10-01
> Fuente oficial: `caso_tecnico_original.md` (convertido del .docx entregado; archivo local, no versionado)

---

## 1. Resumen ejecutivo

Un cliente necesita un **asistente que responda preguntas usando únicamente la información de sus documentos internos**. Construiremos un **prototipo RAG (Retrieval-Augmented Generation) en Python** que:

1. Carga documentos **.txt, .md y .pdf** (mínimo 2).
2. Los divide en fragmentos (*chunking*).
3. Genera **embeddings locales** con `sentence-transformers`.
4. Los guarda en un **vector store local persistente (ChromaDB)**.
5. Recibe una pregunta, recupera los fragmentos más similares y genera la respuesta con **Gemini** (`gemini-3.1-flash-lite`, ADR-010).
6. Devuelve la respuesta **con referencia al documento y fragmento usados**, y responde explícitamente *"no encontré esa información en los documentos"* cuando no hay contexto suficiente.

Se expone mediante una **API FastAPI** (núcleo) y una **interfaz Streamlit** (demo), más una **CLI** de consola.

## 2. Objetivo de la prueba (qué evalúan)

Validar capacidad técnica, criterio de solución, claridad de comunicación y **uso responsable de herramientas AI-assisted**. El candidato debe construir una solución funcional básica, **entender lo que hizo, documentarlo y explicarlo**.

### Criterios de evaluación y cómo los cubrimos

| Criterio | Peso | Cómo lo cubrimos | Módulos |
|---|---|---|---|
| Funcionamiento básico de la solución | 25% | Flujo completo carga → pregunta → respuesta, ejecutable con 3 comandos | M5, M7, M8, M9 |
| Implementación RAG básica | 25% | Chunking propio, embeddings, Chroma, similitud, prompt *grounded*, citas, umbral de relevancia | M2–M7 |
| Calidad y orden del código | 15% | Paquete `src/rag`, tipado, config centralizada, interfaces (LLM/embeddings), ruff + pytest | Todos |
| Documentación | 15% | README, docs por módulo, decisiones (ADR), bitácora AI | Todos, M11 |
| Pruebas realizadas | 10% | Tests unitarios + integración por módulo; evaluación de preguntas (contestable/parcial/no contestable) | Todos, M10 |
| Claridad en la explicación | 10% | Guion de video ≤5 min, diagrama de arquitectura | M11 |

## 3. Alcance

### 3.1 Requisitos obligatorios (trazabilidad con el caso)

| ID | Requisito (sección del caso) | Módulo |
|---|---|---|
| R1 | Cargar documentos texto, PDF o Markdown (4.1) | M1 |
| R2 | Dividir contenido en fragmentos (4.1, 4.2) | M2 |
| R3 | Generar embeddings (4.1) | M3 |
| R4 | Guardar embeddings en vector store local — Chroma (4.1, 4.2) | M4 |
| R5 | Carga de mínimo 2 documentos (4.2) | M1, M5 |
| R6 | Realizar preguntas (4.1) | M7, M8, M9 |
| R7 | Consulta por similitud / recuperar contexto relevante (4.1, 4.2) | M4, M7 |
| R8 | Respuesta generada por un modelo basada en lo recuperado (4.1, 4.2) | M6, M7 |
| R9 | Referencia al fragmento o documento usado (4.2) | M7 |
| R10 | Documentar uso de AI-assisted dev: para qué, qué se revisó manualmente, qué se aprendió (4.3) | Transversal → `docs/04_USO_AI_ASSISTED.md` |
| R11 | Mínimo 3 preguntas de prueba: contestable, parcial, no contestable; registrar pregunta, respuesta, correcta/no, observación (4.4) | M10 |
| R12 | Documentación: descripción, instalación, ejecución, dependencias, cómo cargar docs, cómo preguntar, limitaciones, mejoras (4.5) | M11 (README) |

### 3.2 Entregables (sección 5)

| Entregable | Dónde quedará |
|---|---|
| Repositorio de código | GitHub (repo público o compartido) |
| README claro | `README.md` |
| Evidencia de ejecución | `evidencias/` (capturas, logs, salida de tests) |
| Archivo de preguntas de prueba y resultados | `evaluacion/resultados.md` + `evaluacion/resultados.json` |
| Video ≤ 5 min | Enlace en README; guion en `docs/05_GUION_VIDEO.md` |
| Explicación del uso de herramientas AI | `docs/04_USO_AI_ASSISTED.md` |

### 3.3 Fuera de alcance (prototipo)

- Autenticación/usuarios, multi-tenant, OCR de PDFs escaneados, documentos Word/Excel, historial conversacional multi-turno, despliegue productivo.
- **Cloud**: deseable, no obligatorio → queda como módulo opcional **M12** (Docker + despliegue) solo si sobra tiempo.

## 4. Decisiones tecnológicas

| Componente | Elección | Motivo |
|---|---|---|
| Lenguaje | Python ≥ 3.10 (Linux) | Requerido por el caso |
| LLM | **Gemini** (`gemini-3.1-flash-lite`) vía SDK `openai` con `base_url=https://generativelanguage.googleapis.com/v1beta/openai/` | Endpoint compatible con OpenAI → cliente simple y estándar (ADR-010). Modelo configurable en `.env` (`GEMINI_MODEL`) |
| Embeddings | `sentence-transformers` — `intfloat/multilingual-e5-small` (384 dim, 512 tokens, prefijos `query: `/`passage: `) | Local, gratis, offline, multilingüe y entrenado para recuperar pasajes; elegido tras comparar 3 variantes (ADR-009) |
| Vector store | **ChromaDB** persistente (`data/chroma/`), métrica coseno | Local, persistente, guarda metadatos junto al vector |
| Lectura PDF | `pypdf` | Ligero, extrae texto por página (permite citar página) |
| Chunking | Implementación propia (recursivo por separadores + solapamiento), **800/120 caracteres**, encabezados pegados a su contenido | Demuestra comprensión; sin dependencia pesada; 0 % de chunks truncados con e5 (ADR-004, ADR-008) |
| Recuperación | `TOP_K=4`, `MIN_SCORE=0.805` (filtro de ruido) | Calibrado con preguntas legítimas, paráfrasis y fuera de dominio (ADR-005) |
| API | FastAPI + Uvicorn | Opción recomendada por el caso; Swagger automático |
| UI | Streamlit (consume la API) | Opción recomendada; ideal para el video |
| Config | `pydantic-settings` + `.env` | Config tipada y centralizada |
| Pruebas | `pytest`, `pytest-cov`; marcador `integration` (modelos reales, Chroma) y `llm` (llama a Gemini; opt-in con `RUN_LLM=1`) | Unitarias sin red/costo; integración opcional |
| Calidad | `ruff` (lint + format) | Rápido, estándar |

> Por qué **no** LangChain/LlamaIndex: el caso evalúa *entender lo que hiciste*. Implementar el pipeline con piezas pequeñas y explícitas facilita explicarlo en el video y en la sustentación. Se documenta en `docs/03_DECISIONES.md` (ADR-001).

### 4.1 Cambios respecto al plan original

El plan del 2026-10-01 se ajustó con mediciones y problemas reales. Cada cambio quedó registrado como ADR en [`03_DECISIONES.md`](03_DECISIONES.md); los valores del plan solo se conservan allí como historial.

| Tema | Plan original | Final | Motivo | ADR |
|---|---|---|---|---|
| LLM | Grok (xAI), `grok-3-mini` | Gemini `gemini-3.1-flash-lite` (endpoint compatible con OpenAI) | Grok no funcionó con la cuenta del autor; Gemini sí y reutiliza el SDK `openai` | ADR-010, ADR-003 |
| Modelo de embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (128 tokens) | `intfloat/multilingual-e5-small` (512 tokens, prefijos `query: `/`passage: `) | Comparación de 3 variantes: Recall@4 87,5 % → 100 %, MRR 0,833 → 1,000 | ADR-009 |
| Tamaño de chunk | 800/120 → 500/80 en M3 (MiniLM truncaba el 62,5 %) | 800/120 | Con e5 no se trunca ningún chunk | ADR-004, ADR-009 |
| Umbral `MIN_SCORE` | 0,35 (valor inicial sin calibrar) | 0,805, como filtro de ruido | e5 concentra los scores en 0,81–0,90: el umbral no separa las preguntas del dominio sin respuesta; esa abstención la decide el LLM | ADR-005 |
| Subidas por la API | En `data/docs/` | En `data/uploads/` (ignorado por git) | El corpus versionado queda congelado como verdad de la evaluación | ADR-011 |
| Errores de la API | Sin definir | Formato único `{"error","detail"}` con códigos por caso | Errores claros y consistentes en API y UI | ADR-011 |
| Pruebas con costo | Marcador `integration` | `integration` (sin costo) + `llm` (opt-in con `RUN_LLM=1`) | Evitar gastar créditos por accidente | ADR-010 |
| PyTorch | Sin especificar | Solo CPU | Instalación liviana, sin paquetes CUDA | ADR-006 |
| `GET /stats` | Previsto en M8 | Eliminado | `/health` da el conteo y `GET /documents` el detalle | — (M0) |

## 5. Arquitectura

```
                ┌───────────────────────── INGESTA ─────────────────────────┐
 data/docs/     │                                                           │
 data/uploads/  │                                                           │
 .txt .md .pdf ─┼─► loaders.py ─► chunking.py ─► embeddings.py ─► vectorstore.py ──► data/chroma/
                │   (Document)    (Chunk 800/120  (vectores        (Chroma upsert     (persistente)
                │                  metadatos)      384-d)           ids estables)
                └───────────────────────────────────────────────────────────┘

                ┌───────────────────────── CONSULTA ────────────────────────┐
 Pregunta ──────┼─► embeddings.py ─► vectorstore.query(top_k) ─► filtro por umbral
                │                                  (score ≥ MIN_SCORE 0,805)
                │                                                  │
                │          ¿hay contexto relevante? ── no ──► "No encontré esa información…"
                │                     │ sí
                │                     ▼
                │   prompts.py (contexto numerado [1]..[k] + reglas) ─► llm.py (Gemini)   
                │                     ▼
                │   RespuestaRAG { answer, sources:[doc, página, chunk_id, score, extracto] }
                └───────────────────────────────────────────────────────────┘

 Interfaces:  cli.py (consola)   ·   api.py (FastAPI)   ◄── ui_streamlit.py (Streamlit)
```

### Estructura del repositorio

```
asistente-rag-ic7/
├── CLAUDE.md                  # Reglas de trabajo para Claude Code
├── README.md                  # Documentación final (M11)
├── pyproject.toml / requirements.txt
├── .env.example
├── src/rag/
│   ├── config.py              # M0  Settings (pydantic-settings)
│   ├── logging_conf.py        # M0
│   ├── models.py              # M1  Document, Chunk, RetrievedChunk, RAGAnswer
│   ├── loaders.py             # M1  txt / md / pdf
│   ├── chunking.py            # M2
│   ├── embeddings.py          # M3
│   ├── vectorstore.py         # M4
│   ├── ingest.py              # M5  pipeline de ingesta
│   ├── llm.py                 # M6  cliente Gemini + FakeLLM para tests
│   ├── prompts.py             # M7
│   ├── rag_engine.py          # M7  retrieve + generate + citas
│   ├── cli.py                 # M5/M7
│   ├── api.py                 # M8
│   ├── api_client.py          # M9  cliente HTTP de la UI
│   ├── ui_streamlit.py        # M9
│   └── evaluation.py          # M10 reglas de verificación y reporte
├── data/docs/                 # Corpus de documentos internos (M1)
├── data/uploads/              # Subidas por la API (ignorado en git, ADR-011)
├── data/chroma/               # Vector store persistente (ignorado en git)
├── tests/unit/ · tests/integration/
├── evaluacion/                # M10 preguntas + resultados
├── evidencias/                # capturas y logs
├── scripts/                   # utilidades (generar PDF, correr evaluación)
└── docs/                      # esta documentación
```

## 6. Corpus de documentos (cliente ficticio)

Como el caso no entrega documentos, se crea un corpus de una empresa **ficticia: "Nexa Logística S.A.S."** con 3 documentos internos (uno por formato) diseñados para que existan preguntas **contestables, parcialmente contestables y no contestables**:

| Archivo | Formato | Contenido |
|---|---|---|
| `politica_vacaciones_y_permisos.md` | Markdown | Días de vacaciones, solicitud, permisos remunerados, licencias |
| `manual_reembolso_gastos.pdf` | PDF (generado por script) | Topes de viáticos, plazos, soportes, aprobaciones |
| `guia_onboarding_ti.txt` | Texto | Accesos, VPN, contraseñas, soporte técnico, equipos |

El detalle de los hechos clave de cada documento se define en **M1** y es la "verdad" contra la cual se evalúa en **M10**.

## 7. Metodología de trabajo (módulo por módulo)

1. Cada módulo tiene su documento en `docs/modulos/Mx_*.md` con: objetivo, requisitos que cubre, tareas, archivos, **criterios de aceptación (pruebas)**, definición de terminado y un **registro de ejecución**.
2. **Regla de avance:** no se inicia el módulo *N+1* hasta que el módulo *N* tenga **todas sus pruebas en verde** (`pytest`), `ruff` sin errores y su documentación actualizada.
3. Al cerrar un módulo: actualizar el registro del módulo, `docs/BITACORA.md`, `docs/04_USO_AI_ASSISTED.md` y hacer commit `feat(Mx): …` + tag `Mx-ok`.
4. Toda decisión no trivial se registra como ADR en `docs/03_DECISIONES.md`.

Plan de módulos detallado: [`01_PLAN_MODULOS.md`](01_PLAN_MODULOS.md).

## 8. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Nombre de modelo Gemini cambia / no disponible | Modelo en `.env`; `scripts/check_gemini.py` lista los modelos de la key y confirma el configurado |
| Créditos de la API key agotados | `LLMQuotaExhaustedError` con mensaje claro en CLI/API/UI para cambiar `GEMINI_API_KEY` (M6–M9) |
| Fuga de la API key | `.env` ignorado, `test_no_secrets.py`, hook `scripts/pre-commit`, la key nunca se imprime (solo `****` + últimos 4) |
| Costo o falta de saldo en Gemini | Tests unitarios usan `FakeLLM`; solo los tests `llm` (opt-in con `RUN_LLM=1`) llaman a la API |
| Alucinaciones | Prompt estricto, umbral de similitud, respuesta de "no encontrado", citas obligatorias |
| Primera descarga del modelo de embeddings (~490 MB) | Se descarga una vez en M3 y queda en caché; documentado en README |
| PDF sin texto extraíble | Fuera de alcance (OCR); se advierte en logs y limitaciones |
| Tiempo (24–36 h) | Módulos pequeños; M12 (cloud) solo opcional |

## 9. Estimación de tiempo

| Módulo | Horas |
|---|---|
| M0 Setup | 2 |
| M1 Corpus + loaders | 3 |
| M2 Chunking | 2 |
| M3 Embeddings | 2 |
| M4 Vector store | 2.5 |
| M5 Pipeline de ingesta + CLI | 2 |
| M6 Cliente LLM Gemini | 2 |
| M7 Motor RAG | 4 |
| M8 API FastAPI | 3 |
| M9 UI Streamlit | 2.5 |
| M10 Evaluación | 2.5 |
| M11 Documentación + video | 3.5 |
| **Total** | **~31 h** (dentro de 24–36 h) |
| M12 Docker/Cloud (opcional) | +3 |

## 10. Glosario rápido

- **RAG**: técnica que recupera fragmentos relevantes de una base documental y se los pasa al LLM como contexto para responder.
- **Chunk**: fragmento de texto de tamaño controlado, con solapamiento para no cortar ideas.
- **Embedding**: vector numérico que representa el significado de un texto; textos similares → vectores cercanos.
- **Similitud coseno**: medida de cercanía entre vectores (1 = idénticos).
- **Grounding**: obligar al modelo a responder solo con el contexto entregado.
