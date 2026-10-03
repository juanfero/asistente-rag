# CLAUDE.md — Reglas del proyecto "Asistente Documental RAG (IC7)"

## Contexto
Prueba técnica AI Developer Engineer Junior (I Cloud Seven). Asistente RAG en Python que responde **solo** con la información de documentos internos. Documento general: `docs/00_PROYECTO.md`. Caso original: `docs/caso_tecnico_original.md` (archivo local, no versionado). Plan: `docs/01_PLAN_MODULOS.md`.

## Stack fijo (no cambiar sin ADR en docs/03_DECISIONES.md)
- Python ≥ 3.10, Linux. Paquete en `src/rag/`.
- LLM: **Gemini** con SDK `openai` (endpoint compatible), `base_url=https://generativelanguage.googleapis.com/v1beta/openai/`, `GEMINI_API_KEY`, modelo en `GEMINI_MODEL` (ADR-010).
- Embeddings: `sentence-transformers` `intfloat/multilingual-e5-small` (local, prefijos `query: `/`passage: `, chunks 800/120; ADR-009).
- Vector store: ChromaDB persistente en `data/chroma/`, métrica coseno.
- PDF: `pypdf`. Chunking: implementación propia.
- API: FastAPI. UI: Streamlit (consume la API). Config: `pydantic-settings`.
- Tests: pytest (marcador `integration` para todo lo que llame a Gemini o descargue modelos). Lint: ruff.
- **No usar** LangChain ni LlamaIndex.

## Reglas de trabajo
1. Se trabaja **un módulo a la vez**, en el orden de `docs/01_PLAN_MODULOS.md`. No tocar código de módulos futuros.
2. Antes de implementar, leer `docs/modulos/Mx_*.md`. Seguir sus tareas y criterios de aceptación **literalmente**; si algo es ambiguo o conviene desviarse, **preguntar** antes.
3. Escribir pruebas primero (o en paralelo) y cubrir cada criterio de aceptación con al menos un test.
4. Un módulo se cierra solo si: `pytest -m "not integration"` en verde (todos los módulos), integración del módulo en verde si aplica, `ruff check` y `ruff format --check` limpios.
5. Al cerrar: completar el "Registro de ejecución" del módulo, actualizar `docs/BITACORA.md`, `docs/04_USO_AI_ASSISTED.md` (qué generó la IA, qué se revisó a mano, qué se aprendió), estado en `docs/01_PLAN_MODULOS.md`, y proponer commit `feat(Mx): ...` + tag `Mx-ok`.
6. Nunca commitear `.env`, `data/chroma/` ni claves. Nunca imprimir la API key en logs ni respuestas (enmascarar: `****` + últimos 4). Hook `scripts/pre-commit` instalado y `tests/unit/test_no_secrets.py` en verde.
7. Código: type hints, docstrings breves en español, funciones pequeñas, sin estado global oculto (inyectar dependencias para poder testear con fakes).
8. Documentación y mensajes al usuario en **español**; identificadores de código en inglés.

## Comandos
```bash
source .venv/bin/activate
pytest -m "not integration" -q          # rápido, sin red
pytest -m integration -q                # requiere GEMINI_API_KEY / descarga de modelos
ruff check src tests && ruff format --check src tests
python -m rag.cli ingest data/docs      # desde M5
python -m rag.cli ask "pregunta"        # desde M7
uvicorn rag.api:app --reload            # desde M8
streamlit run src/rag/ui_streamlit.py   # desde M9
```
