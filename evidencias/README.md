# Evidencias de ejecución

Salidas reales, generadas al cerrar cada módulo. Las rutas locales se reemplazaron por `<repo>` y la API key nunca aparece (solo enmascarada como `****`). Las salidas de pytest de módulos anteriores reflejan el número de pruebas de ese momento.

| Archivo | Contenido |
|---|---|
| `M0_entorno.txt` | Versión de Python, torch solo CPU (sin paquetes `nvidia-*`) y cabecera de `requirements.txt` |
| `M0_check_gemini.txt` | `check_gemini.py`: modelos disponibles para la key y prueba de `gemini-3.1-flash-lite` (key enmascarada) |
| `M0_M1_pytest.txt` | Pruebas unitarias al cierre de M0 y M1 |
| `M0_M1_ruff.txt` | `ruff check` y `ruff format --check` al cierre de M0 y M1 |
| `M1_documentos_cargados.txt` | Documentos del corpus cargados por los loaders, con metadatos (fuente, tipo, página, tamaño) |
| `M2_chunks_corpus.txt` | Volcado completo de los 16 chunks del corpus (800/120) |
| `M2_estadisticas.txt` | Estadísticas de chunks por documento y página |
| `M2_pytest_ruff.txt` | Pruebas unitarias y ruff al cierre de M2 |
| `M3_comparacion_embeddings.txt` | Comparación de las variantes A/B/C (MiniLM vs e5-small; Recall@4, MRR, márgenes) que justificó ADR-009 |
| `M3_similitud_preguntas.txt` | Similitud de las 9 preguntas de evaluación con el corpus (top-k por pregunta) |
| `M3_tokens_chunks.txt` | Tokens por chunk con el tokenizer real de e5 (0 % truncado con 800/120) |
| `M3_pytest_integration.txt` | Pruebas de integración con el modelo de embeddings real |
| `M3_pytest_ruff.txt` | Pruebas unitarias y ruff al cierre de M3 |
| `M4_consulta_chroma.txt` | Consulta a ChromaDB: metadata de la colección y top-4 de preguntas de ejemplo |
| `M4_pytest_integration.txt` | Pruebas de integración con Chroma real |
| `M4_pytest_ruff.txt` | Pruebas unitarias y ruff al cierre de M4 |
| `M5_ingesta.txt` | `rag.cli ingest data/docs --reset` y `stats` sobre el índice real (16 chunks) |
| `M5_pytest_integration.txt` | Pruebas de integración de la ingesta |
| `M5_pytest_ruff.txt` | Pruebas unitarias y ruff al cierre de M5 |
| `M6_gemini_smoke.txt` | Prueba de humo RAG con Gemini: respuesta, `finish_reason` y tokens |
| `M6_pytest_integration.txt` | Pruebas de integración sin costo (las `llm` se omiten) |
| `M6_pytest_ruff.txt` | Pruebas unitarias y ruff al cierre de M6 |
| `M7_calibracion_umbral.txt` | Calibración de `MIN_SCORE` con preguntas legítimas, paráfrasis y fuera de dominio (ADR-005) |
| `M7_cli_ask.txt` | `rag.cli ask` con preguntas contestable, parcial, no contestable y fuera de dominio |
| `M7_prompt_injection.txt` | Prueba de *prompt injection* con un documento malicioso: Gemini no obedece |
| `M7_pytest_integration.txt` | Pruebas de integración al cierre de M7 |
| `M7_pytest_ruff.txt` | Pruebas unitarias y ruff al cierre de M7 |
| `M8_curl.txt` | Llamadas reales a la API con `curl` (comandos copiables) y sus respuestas, incluidos los errores |
| `M8_openapi.json` | Especificación OpenAPI generada por FastAPI |
| `M8_pytest_integration.txt` | Pruebas de integración de la API |
| `M8_pytest_ruff.txt` | Pruebas unitarias y ruff al cierre de M8 |
| `M9_pytest_integration.txt` | Pruebas de integración al cierre de M9 |
| `M9_pytest_ruff.txt` | Pruebas unitarias (incluida la UI con `AppTest`) y ruff al cierre de M9 |
| `M10_ejecucion.txt` | Corrida real de la evaluación (9 preguntas × 3 repeticiones + robustez) con Gemini |
| `M10_pytest_ruff.txt` | Pruebas unitarias y ruff al cierre de M10 |
| `M11_coverage.txt` | Cobertura de `src/rag` (98 %, sin `ui_streamlit.py`) con `--cov-fail-under=80` |
| `M11_seguridad.txt` | Seguridad final: historial sin claves, `.env`/`data/chroma`/`data/uploads` ignorados, sin rutas locales, hook instalado |
| `M11_instalacion.txt` | Instalación desde cero en un clon limpio siguiendo el README, sin API key |
| `M11_pytest_ruff.txt` | Pruebas unitarias y ruff al cierre de M11 |

Resultados de la evaluación (tabla completa con pregunta, respuesta, ¿correcta? y observación): [`../evaluacion/resultados.md`](../evaluacion/resultados.md).
