# M11 — Documentación final y entregables

**Estado:** ✅ Completado · **Estimado:** 3.5 h · **Depende de:** M0–M10 · **Requisitos:** R10, R12 + sección 5

## Objetivo
Dejar el repositorio listo para entregar: README completo según la sección 4.5, evidencias, explicación del uso de AI y video de ≤ 5 min.

## Tareas
1. **README.md** con estas secciones (las 8 primeras son exigidas por el caso):
   1. Descripción de la solución (+ diagrama de arquitectura)
   2. Dependencias (stack y por qué)
   3. Pasos de instalación (venv, `pip install`, `.env`, primera descarga del modelo de embeddings)
   4. Pasos de ejecución (CLI, API, UI)
   5. Cómo cargar documentos (carpeta + `ingest`, `POST /documents`, uploader de Streamlit)
   6. Cómo hacer preguntas (CLI, curl, Streamlit) con ejemplo de respuesta y fuentes
   7. Limitaciones conocidas (ver lista base abajo)
   8. Mejoras futuras (ver lista base abajo)
   9. Pruebas (cómo correrlas, cobertura) y resultados de evaluación (resumen + enlace a `evaluacion/resultados.md`)
   10. Uso de herramientas AI-assisted (resumen + enlace a `docs/04_USO_AI_ASSISTED.md`)
   11. Estructura del repositorio y enlace al video
2. **Evidencias** (`evidencias/`): salida de `pytest --cov`, salida de `ingest`/`stats`/`ask`, capturas de Swagger y Streamlit, `evaluacion/resultados.md`.
3. **`docs/04_USO_AI_ASSISTED.md`** consolidado: herramienta (Claude Code), para qué se usó por módulo, qué se revisó/corrigió manualmente (con ejemplos concretos), qué se aprendió.
4. **`docs/05_GUION_VIDEO.md`** (≤ 5 min): 0:00 problema y solución · 0:40 arquitectura · 1:30 demo ingesta · 2:15 demo preguntas (contestable, parcial, no contestable con fuentes) · 3:30 pruebas y evaluación · 4:15 uso de AI, limitaciones y mejoras. Grabar y enlazar.
5. Revisión final: clonar el repo en carpeta limpia y seguir el README al pie de la letra (prueba de instalación desde cero).
6. Release: tag `v1.0.0`.

### Limitaciones (base)
Archivos borrados de `data/docs` siguen en el índice hasta `ingest --reset` (sin `--prune`, M5); dos archivos con el mismo nombre en una ingesta: solo se indexa el primero (M5); PDFs escaneados sin OCR; sin memoria conversacional; chunking por caracteres (no semántico); umbral de relevancia calibrado con un corpus pequeño; evaluación heurística + manual (no métricas tipo RAGAS); embeddings con límite de 128 tokens; dependencia de la API de xAI (costo, latencia, disponibilidad); *(lista base del plan: en el README se actualizó al stack final — e5 con 512 tokens, Gemini — y se añadieron las limitaciones medidas)*; sin autenticación; un solo índice/colección.

### Mejoras futuras (base)
OCR (Tesseract) y soporte DOCX; búsqueda híbrida (BM25 + vectorial) y *re-ranking* con cross-encoder; chunking semántico; memoria conversacional; streaming de respuestas; evaluación con RAGAS/LLM-as-judge; autenticación y multi-colección por cliente; despliegue cloud (contenedor + Chroma server/pgvector); observabilidad (trazas, costo por consulta); encabezado de sección en cada chunk (contexto jerárquico; evaluado y descartado en M3.1 porque el recall ya es 100 %).

## Criterios de aceptación
| ID | Criterio | Verificación |
|---|---|---|
| M11-01 | README contiene las 8 secciones exigidas en 4.5 | `test_readme_sections` (busca encabezados) |
| M11-02 | Instalación desde cero siguiendo el README funciona | manual, registrar en bitácora |
| M11-03 | `pytest -m "not integration"` verde y cobertura ≥ 80% en `src/rag` (excluye `ui_streamlit.py`) | `pytest --cov=rag` |
| M11-04 | Existen `evaluacion/resultados.md`, evidencias y `docs/04_USO_AI_ASSISTED.md` completo | checklist |
| M11-05 | Video ≤ 5 min grabado y enlazado | checklist |
| M11-06 | `.env` y `data/chroma` no están en el repo; no hay claves en el historial | `git check-ignore .env`; `git log --all -p \| python scripts/secret_scan.py --stdin` → **"0 coincidencias"** (patrones estrictos `AIza[0-9A-Za-z_-]{35}`, `AQ\.[0-9A-Za-z_-]{20,}` y `GEMINI_API_KEY=` con valor; solo informa el conteo, nunca el contenido); `pytest tests/unit/test_no_secrets.py` (incluye `test_git_history_has_no_secrets`); hook `scripts/pre-commit` instalado |

## Checklist de entrega (sección 5 del caso)
- [x] Repositorio de código — `https://github.com/juanfero/asistente-rag`
- [x] README claro — `README.md` (8 secciones de 4.5 + arquitectura, evaluación, embeddings, seguridad, video)
- [x] Evidencia de ejecución — `evidencias/` (índice en `evidencias/README.md`)
- [x] Archivo con preguntas de prueba y resultados — `evaluacion/preguntas.yaml`, `evaluacion/resultados.md`
- [x] Video ≤ 5 min — Google Drive, enlazado en el README (sección *Video*)
- [x] Explicación del uso de herramientas AI-assisted development — `docs/04_USO_AI_ASSISTED.md`

## Registro de ejecución
| Fecha | Comando / acción | Resultado | Notas |
|---|---|---|---|
| 2026-10-04 | Coherencia de la documentación (A) | OK | `00_PROYECTO.md` sin Grok/xAI/MiniLM/500-80 fuera del historial; nueva §4.1 *Cambios respecto al plan original* (tabla plan → final → ADR). `CLAUDE.md` y `01_PLAN_MODULOS.md` ya estaban coherentes |
| 2026-10-04 | README final (B, C) | OK | Secciones de 4.5, diagrama, resultados de M10, cobertura, *Cómo se eligió el modelo de embeddings* (tabla A/B/C), seguridad, limitaciones medidas, mejoras, sección *Video*; `tests/unit/test_readme.py` (secciones, entregables y enlaces locales) |
| 2026-10-04 | `docs/04_USO_AI_ASSISTED.md` (D) y `docs/05_GUION_VIDEO.md` (E) | OK | 10 ejemplos concretos de revisión/corrección y conclusiones; guion ≤ 5 min con la demo real |
| 2026-10-04 | Instalación desde cero (F): clon limpio, venv, torch CPU, `pip install -e ".[dev]"`, `.env` de `.env.example` **sin key**, hook, `check_gemini`, `ingest`, `stats`, `ask`, pytest, ruff | OK | `evidencias/M11_instalacion.txt`: 389 passed; integración 6 passed + 6 skipped; ruff limpio; `torch 2.14.1+cpu`; ingest 16 chunks. Sin key, `check_gemini.py` (exit 1) y `ask` (exit 2) terminan con mensaje claro. Ajustes al README: clon por HTTPS (el SSH requiere llave configurada), nota de qué funciona sin key y del aviso `HF_TOKEN` (informativo). El modelo ya estaba en la caché compartida: la descarga inicial no se re-midió |
| 2026-10-04 | `pytest -m "not integration" --cov=rag --cov-fail-under=80` (G) | 98,33 % | `evidencias/M11_coverage.txt`; `ui_streamlit.py` excluido vía `[tool.coverage.run] omit` en `pyproject.toml` |
| 2026-10-04 | Seguridad final (H) | 0 coincidencias | `evidencias/M11_seguridad.txt`: historial sin claves, `.env`/`data/chroma`/`data/uploads` ignorados (solo `.gitkeep` versionados), 0 rutas locales en evidencias/docs, hook instalado, `test_no_secrets` 17 passed |
| 2026-10-04 | `evidencias/README.md` (I) y horas por módulo en `BITACORA.md` (J) | OK | Una línea por archivo; plan 31 h vs ≈7,8 h en sesión |
| 2026-10-04 | Enlace del video en el README + tag `v1.0.0` | OK | Video del autor en Google Drive (acceso con el enlace) |
| 2026-10-04 | `pytest -m "not integration" -q` y ruff | 392 passed, 12 deselected · All checks passed! | `evidencias/M11_pytest_ruff.txt` |

**Estado de criterios:** M11-01 ✅ (`test_readme_sections`) · M11-02 ✅ (instalación desde cero) · M11-03 ✅ (392 verdes, cobertura 98 %) · M11-04 ✅ · M11-05 ✅ (video en Google Drive, enlazado en el README) · M11-06 ✅.
