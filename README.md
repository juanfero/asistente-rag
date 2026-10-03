# Asistente Documental RAG — Prueba técnica IC7

> 🚧 En construcción. El README completo se escribe en el módulo M11.

Asistente RAG en Python que responde preguntas **solo** con la información de documentos internos (txt, md, pdf), citando el documento y fragmento usado. Stack: Gemini (endpoint compatible con OpenAI) · sentence-transformers · ChromaDB · FastAPI · Streamlit.

- Documento general: [docs/00_PROYECTO.md](docs/00_PROYECTO.md)
- Plan de módulos: [docs/01_PLAN_MODULOS.md](docs/01_PLAN_MODULOS.md)
- Caso técnico original: `docs/caso_tecnico_original.md` (archivo local, no versionado)

## Seguridad: hook de pre-commit (obligatorio para contribuir)

La API key va **solo** en `.env` (ignorado por git). Para bloquear commits que incluyan claves:

```bash
cp scripts/pre-commit .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
python scripts/secret_scan.py      # revisión manual de todos los archivos versionados
```
