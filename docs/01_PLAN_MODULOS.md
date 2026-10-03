# Plan de Módulos

Regla de oro: **un módulo solo se cierra cuando todas sus pruebas pasan** (`pytest` verde + `ruff check` limpio + documentación actualizada). Solo entonces se inicia el siguiente.

| # | Módulo | Objetivo | Requisitos | Depende de | Estado |
|---|---|---|---|---|---|
| M0 | [Setup y configuración](modulos/M0_setup.md) | Entorno, dependencias, config `.env`, logging, pytest, conexión al LLM verificada (Gemini, ADR-010) | base | — | ✅ Completado |
| M1 | [Corpus y carga de documentos](modulos/M1_corpus_loaders.md) | 3 documentos de ejemplo (md, pdf, txt) + loaders con metadatos | R1, R5 | M0 | ✅ Completado |
| M2 | [Chunking](modulos/M2_chunking.md) | Fragmentación recursiva con solapamiento e IDs estables | R2 | M1 | ✅ Completado |
| M3 | [Embeddings](modulos/M3_embeddings.md) | Embeddings locales multilingües normalizados | R3 | M0 | ✅ Completado (M3.1: e5-small 800/120, ADR-009) |
| M4 | [Vector store (Chroma)](modulos/M4_vectorstore.md) | Persistencia, upsert idempotente, consulta por similitud con score | R4, R7 | M2, M3 | ✅ Completado |
| M5 | [Pipeline de ingesta + CLI](modulos/M5_ingesta.md) | `rag ingest` de punta a punta, re-ingesta idempotente | R1–R5 | M1–M4 | ⬜ Pendiente |
| M6 | [Cliente LLM Gemini](modulos/M6_llm_gemini.md) | Interfaz LLM, cliente Gemini, FakeLLM, errores (créditos agotados, key inválida, límite por minuto) | R8 | M0 | ⬜ Pendiente |
| M7 | [Motor RAG](modulos/M7_motor_rag.md) | Retrieve + umbral + prompt grounded + respuesta con citas + `rag ask` | R6–R9 | M4–M6 | ⬜ Pendiente |
| M8 | [API FastAPI](modulos/M8_api.md) | `/health`, `/documents`, `/ingest`, `/ask` | R6 | M7 | ⬜ Pendiente |
| M9 | [UI Streamlit](modulos/M9_ui_streamlit.md) | Subir documentos, preguntar, ver fuentes | R6 | M8 | ⬜ Pendiente |
| M10 | [Evaluación con preguntas de prueba](modulos/M10_evaluacion.md) | Set de preguntas (contestable/parcial/no contestable), runner y reporte | R11 | M7 | ⬜ Pendiente |
| M11 | [Documentación final y entregables](modulos/M11_documentacion_entrega.md) | README completo, evidencias, uso AI, guion de video, release | R10, R12 | M0–M10 | ⬜ Pendiente |
| M12 | [Opcional: Docker + Cloud](modulos/M12_opcional_cloud.md) | Contenedor y despliegue (deseable, no obligatorio) | deseable | M11 | ⬜ Opcional |

Leyenda de estado: ⬜ Pendiente · 🟨 En progreso · ✅ Completado (pruebas en verde) · ❌ Bloqueado

## Orden y dependencias

```
M0 ─┬─► M1 ─► M2 ─┐
    ├─► M3 ───────┼─► M4 ─► M5 ─┐
    └─► M6 ───────────────────── ┴─► M7 ─┬─► M8 ─► M9 ─┐
                                         └─► M10 ──────┴─► M11 ─► (M12)
```

Se ejecutan en orden numérico (aunque M3 y M6 podrían adelantarse) para mantener una historia de commits limpia.

## Cómo arrancar cada módulo en Claude Code

Prompt sugerido (cambiar `Mx`):

```
Lee CLAUDE.md, docs/00_PROYECTO.md y docs/modulos/Mx_*.md.
Desarrolla el módulo Mx siguiendo exactamente sus tareas y criterios de aceptación.
Escribe primero las pruebas, luego la implementación. Ejecuta pytest y ruff.
No avances a otro módulo. Al terminar, actualiza el "Registro de ejecución" del módulo,
docs/BITACORA.md y docs/04_USO_AI_ASSISTED.md, y muéstrame el resumen de pruebas.
```

## Definición de terminado (aplica a todos los módulos)

- [ ] Todas las pruebas del módulo pasan (`pytest tests/ -m "not integration"`) y las de integración del módulo, si las tiene y hay API key.
- [ ] Las pruebas de módulos anteriores siguen en verde (no hay regresiones).
- [ ] `ruff check src tests` y `ruff format --check src tests` sin errores.
- [ ] Funciones públicas con type hints y docstring breve.
- [ ] Registro de ejecución del módulo completado (fecha, comandos, resultado de tests, notas).
- [ ] `docs/BITACORA.md` y `docs/04_USO_AI_ASSISTED.md` actualizados.
- [ ] Commit `feat(Mx): <descripción>` y tag `Mx-ok`.
