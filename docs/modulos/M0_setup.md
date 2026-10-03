# M0 — Setup y configuración

**Estado:** ✅ Completado (M0-08 ✅ con Gemini, ADR-010) · **Estimado:** 2 h · **Depende de:** —

## Objetivo
Dejar un esqueleto de proyecto instalable y testeable: entorno virtual, dependencias fijadas, configuración centralizada por `.env`, logging, pytest/ruff configurados y **conexión al LLM verificada** con un modelo válido (Gemini desde ADR-010; originalmente Grok).

## Tareas
1. `git init`, `.gitignore` (Python, `.venv/`, `.env`, `data/chroma/*` excepto `.gitkeep`, `__pycache__`, `.pytest_cache`, `.ruff_cache`).
2. `pyproject.toml` con paquete `rag` en layout `src/`, instalable con `pip install -e ".[dev]"`. Dependencias:
   - runtime: `openai`, `sentence-transformers`, `chromadb`, `pypdf`, `fastapi`, `uvicorn[standard]`, `python-multipart`, `streamlit`, `requests`, `pydantic-settings`, `python-dotenv`, `pyyaml` (para `evaluacion/preguntas.yaml` en M10; añadido en M0 por decisión del 2026-10-02)
   - dev: `pytest`, `pytest-cov`, `ruff`, `httpx`, `fpdf2` (para generar el PDF de ejemplo en M1)
   - Generar además `requirements.txt` (pin de versiones instaladas) para el README.
   - PyTorch se instala **CPU-only** antes del paquete (ver ADR-006): `pip install torch --index-url https://download.pytorch.org/whl/cpu`. `requirements.txt` se genera con `pip freeze --exclude-editable` y lleva como primera línea `--extra-index-url https://download.pytorch.org/whl/cpu`.
3. Config pytest en `pyproject.toml`: `testpaths=["tests"]`, `pythonpath=["src"]`, marcador `integration`.
4. Config ruff: `line-length=100`, reglas `E,F,I,UP,B`.
5. `src/rag/config.py`: clase `Settings(BaseSettings)` con (atributos en snake_case minúscula, p. ej. `settings.gemini_model`; variables de entorno en MAYÚSCULAS — ver ADR-007):
   | Variable | Default |
   |---|---|
   | `GEMINI_API_KEY` | (obligatoria para LLM; `SecretStr`, opcional al cargar) — antes `XAI_API_KEY` (ADR-010) |
   | `GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` — antes `XAI_BASE_URL` |
   | `GEMINI_MODEL` | `gemini-3.1-flash-lite` (confirmado con `check_gemini.py`) — antes `XAI_MODEL=grok-3-mini` |
   | `LLM_TEMPERATURE` | `0.1` |
   | `LLM_MAX_TOKENS` | `700` |
   | `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` → **`intfloat/multilingual-e5-small` desde M3.1 (ADR-009)**; más `EMBEDDING_QUERY_PREFIX`/`EMBEDDING_PASSAGE_PREFIX` |
   | `DOCS_DIR` | `data/docs` |
   | `CHROMA_DIR` | `data/chroma` |
   | `CHROMA_COLLECTION` | `documentos` |
   | `CHUNK_SIZE` | `800` (caracteres) → `500` en M3 → **`800` desde M3.1 (ADR-004/ADR-009)** |
   | `CHUNK_OVERLAP` | `120` → `80` en M3 → **`120` desde M3.1 (ADR-004/ADR-009)** |
   | `TOP_K` | `4` |
   | `MIN_SCORE` | `0.35` → **`0.80` provisional desde M3.1** (filtro de ruido; se calibra en M7, ADR-005) |
   | `API_URL` | `http://localhost:8000` (para Streamlit) |
   | `LOG_LEVEL` | `INFO` |
   Función `get_settings()` con `lru_cache`. Validaciones: `CHUNK_OVERLAP < CHUNK_SIZE`, `TOP_K ≥ 1`, `0 ≤ MIN_SCORE ≤ 1`.
6. `src/rag/logging_conf.py`: `setup_logging(level)` formato `%(asctime)s %(levelname)s %(name)s - %(message)s`.
7. `.env.example` con todas las variables (sin la key real).
8. `scripts/check_gemini.py` (antes `check_xai.py`, ADR-010): lista modelos (`client.models.list()`), imprime los disponibles, confirma `GEMINI_MODEL` y hace una llamada mínima ("Responde solo: OK") revisando texto no vacío y `finish_reason`; la key solo se muestra enmascarada. Con el resultado, fijar `XAI_MODEL` en `.env` y documentarlo en ADR-003.
9. Crear `docs/BITACORA.md`, `docs/03_DECISIONES.md`, `docs/04_USO_AI_ASSISTED.md` (ya existen como plantilla — completar entrada M0).

## Archivos
`pyproject.toml`, `requirements.txt`, `.gitignore`, `.env.example`, `src/rag/__init__.py`, `src/rag/config.py`, `src/rag/logging_conf.py`, `scripts/check_gemini.py`, `tests/conftest.py`, `tests/unit/test_config.py`, `tests/integration/test_gemini_connection.py` (antes `test_xai_connection.py`).

## Criterios de aceptación
| ID | Criterio | Test |
|---|---|---|
| M0-01 | `pip install -e ".[dev]"` funciona en un venv limpio e `import rag` no falla | manual + `test_import_package` |
| M0-02 | `Settings` carga defaults sin `.env` | `test_defaults` |
| M0-03 | Variables de entorno sobreescriben defaults (`monkeypatch.setenv`) | `test_env_override` |
| M0-04 | `CHUNK_OVERLAP >= CHUNK_SIZE` lanza `ValidationError` | `test_invalid_overlap` |
| M0-05 | `MIN_SCORE` fuera de [0,1] o `TOP_K < 1` lanza error | `test_invalid_ranges` |
| M0-06 | La API key no aparece en `repr(settings)` ni en logs | `test_secret_not_exposed` |
| M0-07 | `pytest -m "not integration"` y `ruff` corren limpios | CI local |
| M0-08 *(integration)* | Con `GEMINI_API_KEY` válida, `models.list()` devuelve ≥1 modelo y `GEMINI_MODEL` está entre ellos; una completion devuelve texto no vacío | `test_gemini_connection` (skip si no hay key) |

## Verificación
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest -m "not integration" -q
python scripts/check_gemini.py
pytest -m integration -q
ruff check src tests && ruff format --check src tests
```

## Registro de ejecución
| Fecha | Comando / acción | Resultado | Notas |
|---|---|---|---|
| 2026-10-02 | `git init` + `git branch -m main` | OK | Repositorio creado; commits solo con confirmación del autor |
| 2026-10-02 | `python3 -m venv .venv && source .venv/bin/activate && pip install -U pip` | OK | pip 22.0.2 (sistema) → 26.2.1 (venv) |
| 2026-10-02 | `pip install torch --index-url https://download.pytorch.org/whl/cpu` | OK | `torch==2.14.1+cpu`, sin paquetes `nvidia-*` (ADR-006) |
| 2026-10-02 | `pip install -e ".[dev]"` + `pip check` | OK | "No broken requirements found." |
| 2026-10-02 | `(echo "--extra-index-url https://download.pytorch.org/whl/cpu"; pip freeze --exclude-editable) > requirements.txt` | OK | 132 líneas; 0 coincidencias `nvidia` |
| 2026-10-02 | **M0-01 venv limpio** (venv nuevo fuera del repo): `python3 -m venv <tmp>/venv_limpio && <tmp>/venv_limpio/bin/pip install -U pip && <tmp>/venv_limpio/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu && <tmp>/venv_limpio/bin/pip install -e ".[dev]"`; luego `cd /tmp && <tmp>/venv_limpio/bin/python -c "import rag"` | OK | install exit=0; `import rag OK 0.1.0`; `torch 2.14.1+cpu`; `pytest -m "not integration"` en ese venv: 15 passed |
| 2026-10-02 | Venv vacío: `pip install -r requirements.txt --dry-run` | OK (exit 0) | El pin `torch==2.14.1+cpu` se resuelve gracias al `--extra-index-url` |
| 2026-10-02 | `pytest -m "not integration" -v` | 15 passed, 1 deselected | 1ª ejecución: 1 fallo en `test_secret_not_exposed` (ver notas) → corregido |
| 2026-10-02 | `pytest -m integration -v -rs` | 1 skipped | "Sin XAI_API_KEY: se omite la prueba de conexión con xAI" |
| 2026-10-02 | `ruff check src tests scripts && ruff format --check src tests scripts` | All checks passed! / 7 files already formatted | |
| 2026-10-02 | `python scripts/check_xai.py` (sin key) | exit 1, sin traceback | "ERROR: falta XAI_API_KEY. Defínela en el archivo .env (ver .env.example)…" |
| 2026-10-03 | **ADR-010** Cambio de LLM Grok → Gemini (endpoint compatible con OpenAI) | OK | `GEMINI_*` en config y `.env.example`; `check_gemini.py`, `test_gemini_connection.py` |
| 2026-10-03 | `python scripts/check_gemini.py` (key en `.env`, enmascarada) | exit 0 | 61 modelos; `gemini-3.1-flash-lite` disponible; respuesta `'OK'`, `finish_reason=stop` con 20 y 700 tokens (`evidencias/M0_check_gemini.txt`) |
| 2026-10-03 | `pytest -m integration -v -rs` | 6 passed | `test_gemini_connection` en verde → **M0-08 cerrado** |
| 2026-10-03 | Seguridad: `test_no_secrets.py`, `secret_scan.py`, hook `pre-commit` instalado | OK | `git check-ignore .env` ✔; historial `AIza\|AQ\.`: vacío |

**Estado de criterios:** M0-01 ✅ · M0-02 ✅ · M0-03 ✅ · M0-04 ✅ · M0-05 ✅ · M0-06 ✅ · M0-07 ✅ · M0-08 ✅ (cerrado el 2026-10-03 con Gemini; estuvo ⏸ sin API key desde el 2026-10-02).

**Notas:**
- Fallo inicial de M0-06: `setup_logging()` usa `logging.basicConfig(force=True)`, que elimina el handler de `caplog`. El test ahora verifica por separado los registros capturados (`caplog`) y la salida real formateada por el handler del proyecto (`capsys`).
- No se creó `.env` (sin key no se inventa ninguna); sin `.env` aplican los defaults.
- Pruebas extra: `test_setup_logging` (aprobada por el autor) y, como complemento de M0-02/03/05, `test_env_file_is_read`, `test_get_settings_is_cached`, `test_valid_range_limits`.
- **Riesgo anotado (H9, se resuelve en M8):** `POST /documents` guarda los archivos subidos en `data/docs/`, que está versionado como corpus; las subidas de prueba podrían terminar en git.

**Modelo LLM fijado:** `gemini-3.1-flash-lite` (ADR-003, ADR-010). Antes: `grok-3-mini`, nunca verificado · **Versión de Python:** 3.10.12 (venv `.venv`)
