# M9 — Interfaz Streamlit

**Estado:** ✅ Completado (M9-05 ⏸ pendiente: capturas manuales del autor) · **Estimado:** 2.5 h · **Depende de:** M8 · **Requisito:** R6

## Objetivo
Interfaz sencilla para la demo y el video: cargar documentos, ver qué está indexado y hacer preguntas viendo las fuentes usadas. **Consume la API** (no importa el motor directamente) → separación cliente/servidor.

## Diseño
- `src/rag/ui_streamlit.py` + `src/rag/api_client.py` (funciones `health()`, `list_documents()`, `upload(files)`, `ask(question, top_k)` con `requests`, timeout y manejo de errores).
- **Sidebar:** estado de la API (✅/❌, modelo, nº chunks), *file uploader* múltiple (txt/md/pdf) + botón "Indexar", lista de documentos indexados, slider `top_k` (1–8), botón "Re-indexar todo".
- **Principal:** `st.chat_input` + historial de la sesión (`st.session_state`); cada respuesta muestra:
  - Texto de la respuesta.
  - Badge "Basada en documentos" / "Sin información en los documentos".
  - `st.expander("Fuentes")` con documento, página, score y extracto de cada fragmento citado.
  - Latencia y modelo.
- Si la API no responde: mensaje claro con el comando para levantarla.
- *Decisiones del autor (2026-10-03):*
  - **A. Configuración:** la UI lee solo `API_URL` (`UISettings`, sin campo para la key); nunca lee ni muestra `GEMINI_API_KEY`.
  - **B. Errores por código** (campo `error` de la API, sin traceback): `LLM_QUOTA_EXHAUSTED` y `LLM_AUTH_ERROR` → `st.error`; `LLM_RATE_LIMIT` → `st.warning`; `EMPTY_INDEX` → `st.info` + botón **Indexar documentos** (`POST /ingest`); `INDEX_MODEL_MISMATCH` → `st.error` con la instrucción de reset; API caída → mensaje con el comando para levantarla; si `API_URL` responde pero no es la API del asistente (otro servicio en el puerto) → `API_UNEXPECTED`; subidas: un mensaje por archivo (415/409/413).
  - **C. Respuestas:** spinner "Buscando en los documentos y generando la respuesta…"; badge verde "Basada en documentos" o gris "Sin información en los documentos"; expander **Fuentes citadas** solo si `grounded=True`; toggle lateral **Mostrar contexto recuperado (depuración)** (apagado por defecto) con citado/no citado; latencia y modelo en letra pequeña.
  - **D. Historial** en `st.session_state` + botón **Limpiar conversación**; caption: cada pregunta es independiente (sin memoria).
  - **E. Ejemplos:** 3 botones (contestable Q2, parcial Q4, sin respuesta Q8).
  - **F. Documentos:** lista con origen; 🗑️ solo para subidas (`delete_file=true`); corpus con 🔒 y sin botón.
  - **G. Timeouts:** `/ask` 60 s; `/documents` e `/ingest` 120 s; `/health` 5 s.
  - **H.** `scripts/run_demo.sh`: API (puerto de `API_URL`) + Streamlit; guarda los PID y con `trap` los detiene con Ctrl+C (sin `pkill`); verifica antes que los puertos estén libres (`UI_PORT` configurable).

## Criterios de aceptación
| ID | Criterio | Test |
|---|---|---|
| M9-01 | `api_client` arma bien las peticiones y maneja errores HTTP/timeouts con mensajes en español (requests mockeado) | `test_api_client` |
| M9-02 | La app renderiza sin excepciones con la API mockeada (`streamlit.testing.v1.AppTest`) | `test_app_renders` |
| M9-03 | Al enviar una pregunta (AppTest) se muestra la respuesta y el expander de fuentes | `test_app_ask_flow` |
| M9-04 | Con la API caída se muestra el aviso y no hay traceback | `test_app_api_down` |
| M9-06 | Errores del LLM: créditos agotados y key inválida → `st.error`; límite por minuto → `st.warning`; mensaje en español, sin traceback (ADR-010) | `test_error_codes` |
| M9-05 *(manual)* | Flujo completo en navegador: subir un documento nuevo → preguntar sobre él → ver fuente | captura en `evidencias/M9_*.png` |

## Verificación
```bash
pytest tests/unit/test_api_client.py tests/unit/test_ui.py -q
scripts/run_demo.sh                     # API + UI; Ctrl+C detiene ambas
UI_PORT=8502 scripts/run_demo.sh        # si el puerto 8501 está ocupado
```

## Registro de ejecución
| Fecha | Comando / acción | Resultado | Notas |
|---|---|---|---|
| 2026-10-03 | Sondeo de `streamlit.testing.v1.AppTest` (Streamlit 1.64) | OK | `st.badge` se renderiza como markdown `:green-badge[…]`; AppTest ejecuta el script como `__main__` |
| 2026-10-03 | `pytest tests/unit/test_api_client.py tests/unit/test_ui.py -q` | Error de colección → corregido | Importar `ui_streamlit` ejecutaba `main()`, que consultó el `API_URL` real (`:8000`, **otro servicio** del autor) y falló con `KeyError`. Se protegió `main()` con `__name__ == "__main__"` y el cliente ahora detecta un servicio ajeno (`API_UNEXPECTED`) |
| 2026-10-03 | Mutaciones: fuentes con `grounded=False` / borrar corpus / límite como error | 1 / 1 / 2 fallan | Las pruebas detectan cada defecto |
| 2026-10-03 | Prueba de humo de `scripts/run_demo.sh` (API 8011, UI 8512) | OK | Puerto de UI ocupado → mensaje y código 1; arranque normal → ambos responden; SIGTERM (equivalente a Ctrl+C) → API, UI y script detenidos, puertos libres |
| 2026-10-03 | `pytest -m "not integration" -v` | 369 passed, 12 deselected | `evidencias/M9_pytest_ruff.txt` |
| 2026-10-03 | `pytest -m integration -v -rs` | 6 passed, 6 skipped | `evidencias/M9_pytest_integration.txt` (M9 no añade pruebas `llm`: la UI se prueba con la API simulada) |
| 2026-10-03 | `ruff check src tests && ruff format --check src tests` | All checks passed! / 41 files already formatted | |

**Estado de criterios:** M9-01 ✅ · M9-02 ✅ · M9-03 ✅ · M9-04 ✅ · M9-06 ✅ · M9-05 ⏸ (manual, capturas del autor).

**Pruebas adicionales:** `test_not_grounded_has_no_sources_expander`, `test_context_toggle`, `test_no_llm_call_caption`, `test_example_buttons_and_clear`, `test_error_codes` (×6), `test_empty_index_offers_ingest`, `test_delete_only_for_uploads`, `test_error_level_and_text`, `test_upload_messages_per_file`, `test_health_detects_other_service`, `test_ui_settings_reads_only_api_url`, `test_run_demo_script`.

**Capturas para M9-05 y el video** (las toma el autor; guardarlas en `evidencias/`). Levantar la demo con `UI_PORT=8502 scripts/run_demo.sh` (8501 ocupado en la máquina del autor; `API_URL=http://localhost:8011` en `.env`):

1. `M9_01_inicio.png` — pantalla inicial: barra lateral con "API conectada", documentos del corpus con 🔒 y los 3 botones de ejemplo.
2. `M9_02_contestable.png` — botón **💰 Contestable**: respuesta con COP 120.000 [1] y USD 90 [2], badge verde y el expander **Fuentes citadas** abierto.
3. `M9_03_parcial.png` — botón **🧩 Parcial**: "5 días hábiles [1]. Los documentos no incluyen información sobre… horas extra".
4. `M9_04_sin_respuesta.png` — botón **🚫 Sin respuesta**: badge gris y sin expander de fuentes.
5. `M9_05_contexto_depuracion.png` — con el toggle **Mostrar contexto recuperado (depuración)** activado: contexto con ✅ citado / ▫️ no citado.
6. `M9_06_subida.png` — subir `horario_cafeteria.txt` (crearlo con `printf 'Horario de la cafetería\n\nLa cafetería abre de lunes a viernes de 7:00 a 15:00.\n' > horario_cafeteria.txt`) y pulsar **Indexar archivos**: mensaje ✔ y el documento en la lista con 📄 y botón 🗑️.
7. `M9_07_pregunta_documento_nuevo.png` — **(criterio M9-05)** preguntar "¿A qué hora abre la cafetería?": respuesta con la fuente `horario_cafeteria.txt` en **Fuentes citadas**.
8. `M9_08_errores_subida.png` — subir a la vez un `.exe` y un archivo llamado `guia_onboarding_ti.txt`: un mensaje por archivo (415 y 409).
9. `M9_09_api_caida.png` — con la UI abierta, detener la API (Ctrl+C del script o `kill <PID de la API>`) y recargar: aviso con el comando para levantarla.
10. `M8_swagger.png` — `http://localhost:8011/docs` con los endpoints (captura de M8 para el video).

Al terminar, eliminar `horario_cafeteria.txt` con 🗑️ para dejar el índice como estaba.
