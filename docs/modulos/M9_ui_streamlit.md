# M9 — Interfaz Streamlit

**Estado:** ⬜ Pendiente · **Estimado:** 2.5 h · **Depende de:** M8 · **Requisito:** R6

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

## Criterios de aceptación
| ID | Criterio | Test |
|---|---|---|
| M9-01 | `api_client` arma bien las peticiones y maneja errores HTTP/timeouts con mensajes en español (requests mockeado) | `test_api_client` |
| M9-02 | La app renderiza sin excepciones con la API mockeada (`streamlit.testing.v1.AppTest`) | `test_app_renders` |
| M9-03 | Al enviar una pregunta (AppTest) se muestra la respuesta y el expander de fuentes | `test_app_ask_flow` |
| M9-04 | Con la API caída se muestra el aviso y no hay traceback | `test_app_api_down` |
| M9-06 | Errores del LLM (créditos agotados, key inválida, límite por minuto) se muestran con `st.warning` y el mensaje en español, sin traceback (ADR-010) | `test_app_llm_errors` |
| M9-05 *(manual)* | Flujo completo en navegador: subir un documento nuevo → preguntar sobre él → ver fuente | captura en `evidencias/M9_*.png` |

## Verificación
```bash
pytest tests/unit/test_api_client.py tests/unit/test_ui.py -q
uvicorn rag.api:app &      # terminal 1
streamlit run src/rag/ui_streamlit.py   # terminal 2 → http://localhost:8501
```

## Registro de ejecución
| Fecha | Comando / acción | Resultado | Notas |
|---|---|---|---|
| | | | |
