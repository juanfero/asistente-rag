"""Interfaz Streamlit del asistente documental (consume la API REST; no importa el motor).

Ejecutar: streamlit run src/rag/ui_streamlit.py   (la API debe estar levantada; ver API_URL)
"""

from typing import Any

import streamlit as st

from rag import api_client
from rag.api_client import APIClientError, UploadResult

SPINNER_TEXT = "Buscando en los documentos y generando la respuesta…"
NO_MEMORY_CAPTION = (
    "Cada pregunta se responde de forma independiente (sin memoria conversacional) y solo con "
    "la información de los documentos indexados."
)
EXAMPLES = [
    (
        "💰 Contestable",
        "¿Cuál es el tope diario de alimentación en viajes nacionales e internacionales?",
        "ex_contestable",
    ),
    (
        "🧩 Parcial",
        "¿Cuántos días de permiso me dan por matrimonio y cómo se pagan las horas extra?",
        "ex_parcial",
    ),
    ("🚫 Sin respuesta", "¿Cuántos días a la semana puedo trabajar remoto?", "ex_sin_respuesta"),
]
RESET_HINT = (
    "Ejecuta `python -m rag.cli reset --yes` y luego pulsa **Re-indexar todo** (o "
    "`python -m rag.cli ingest data/docs`)."
)


# --- Lógica pura (probada sin Streamlit) --------------------------------------------


def error_level(code: str) -> str:
    """Tipo de aviso de Streamlit para cada código de error de la API."""
    return {"LLM_RATE_LIMIT": "warning", "EMPTY_INDEX": "info", "API_TIMEOUT": "warning"}.get(
        code, "error"
    )


def error_text(code: str, detail: str) -> str:
    """Mensaje a mostrar para un código de error."""
    if code == "INDEX_MODEL_MISMATCH":
        return f"{detail}\n\n{RESET_HINT}"
    if code == "LLM_QUOTA_EXHAUSTED" and "GEMINI_API_KEY" not in detail:
        return f"{detail} Cambia GEMINI_API_KEY en el archivo .env y reinicia la API."
    return detail


def upload_messages(results: list[UploadResult]) -> list[tuple[str, str]]:
    """(nivel, mensaje) por archivo subido, p. ej. 415/409/413 con su motivo."""
    messages = []
    for r in results:
        if r.ok and r.report and r.report.get("files_processed"):
            messages.append(
                ("success", f"✔ {r.name}: {r.report['chunks_added']} fragmentos indexados.")
            )
        elif r.ok:
            skipped = "; ".join((r.report or {}).get("skipped", [])) or "sin texto útil"
            messages.append(("warning", f"⚠ {r.name}: no se indexó ({skipped})."))
        else:
            messages.append(("error", f"✘ {r.name}: {r.detail} [{r.code}]"))
    return messages


# --- Renderizado ---------------------------------------------------------------------


def show(level: str, text: str) -> None:
    getattr(st, level)(text)


def render_sources(sources: list[dict[str, Any]]) -> None:
    with st.expander("Fuentes citadas"):
        for s in sources:
            page = f", pág. {s['page']}" if s.get("page") is not None else ""
            st.markdown(f"**[{s['index']}] {s['source']}{page}** · score {s['score']:.3f}")
            st.caption(s["excerpt"])


def render_context(context: list[dict[str, Any]]) -> None:
    with st.expander("Contexto recuperado (depuración)"):
        for c in context:
            page = f", pág. {c['page']}" if c.get("page") is not None else ""
            mark = "✅ citado" if c["cited"] else "▫️ no citado"
            st.markdown(f"[{c['index']}] {c['source']}{page} · score {c['score']:.3f} · {mark}")
            st.caption(c["excerpt"])


def render_answer(data: dict[str, Any], show_context: bool) -> None:
    st.markdown(data["answer"])
    if data["grounded"]:
        st.badge("Basada en documentos", icon=":material/check:", color="green")
        if data["sources"]:
            render_sources(data["sources"])
    else:
        st.badge("Sin información en los documentos", color="gray")
    if show_context and data.get("context"):
        render_context(data["context"])
    llm = data["model"] if data.get("llm_called") else "sin llamada al LLM (fuera del umbral)"
    st.caption(f"{llm} · {data['latency_s']:.1f} s")


def render_error(msg: dict[str, Any], index: int, client: api_client.APIClient) -> None:
    show(error_level(msg["code"]), error_text(msg["code"], msg["detail"]))
    if msg["code"] == "EMPTY_INDEX" and st.button("Indexar documentos", key=f"ingest_{index}"):
        try:
            report = client.ingest()
            st.success(f"Índice creado: {report['total_in_store']} fragmentos. Vuelve a preguntar.")
        except APIClientError as exc:
            show(error_level(exc.code), error_text(exc.code, exc.detail))


def ask(client: api_client.APIClient, question: str, top_k: int) -> None:
    """Envía la pregunta y guarda la respuesta (o el error) en el historial."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner(SPINNER_TEXT):
        try:
            data = client.ask(question, top_k=top_k)
            st.session_state.messages.append({"role": "assistant", "data": data})
        except APIClientError as exc:
            st.session_state.messages.append(
                {"role": "assistant", "code": exc.code, "detail": exc.detail}
            )


def sidebar(client: api_client.APIClient) -> bool:
    """Estado de la API, opciones y gestión de documentos. Devuelve si la API responde."""
    st.sidebar.header("📚 Asistente documental")
    try:
        health = client.health()
    except APIClientError as exc:
        st.sidebar.error(exc.detail)
        return False

    if health["status"] == "index_model_mismatch":
        st.sidebar.error(f"El índice se creó con otro modelo de embeddings. {RESET_HINT}")
    else:
        st.sidebar.success(f"API conectada · {health['llm_model']}")
    llm_state = "configurado" if health["llm_configured"] else "SIN key"
    st.sidebar.caption(
        f"{health['chunks']} fragmentos · {health['documents']} documentos · "
        f"MIN_SCORE {health['min_score']} · LLM {llm_state}"
    )
    st.sidebar.slider("Fragmentos a recuperar (top_k)", 1, 8, health["top_k"], key="top_k")
    st.sidebar.toggle("Mostrar contexto recuperado (depuración)", value=False, key="show_context")

    st.sidebar.subheader("Documentos indexados")
    try:
        documents = client.list_documents()
    except APIClientError as exc:
        show(error_level(exc.code), exc.detail)
        documents = []
    for doc in documents:
        cols = st.sidebar.columns([5, 1])
        if doc["origin"] == "corpus":
            cols[0].markdown(
                f"🔒 {doc['source']}  \n<small>{doc['chunks']} fragmentos · corpus</small>",
                unsafe_allow_html=True,
            )
        else:
            cols[0].markdown(
                f"📄 {doc['source']}  \n<small>{doc['chunks']} fragmentos · subida</small>",
                unsafe_allow_html=True,
            )
            if cols[1].button("🗑️", key=f"del_{doc['source']}", help="Eliminar esta subida"):
                try:
                    client.delete_document(doc["source"], delete_file=True)
                    st.rerun()
                except APIClientError as exc:
                    show(error_level(exc.code), exc.detail)

    files = st.sidebar.file_uploader(
        "Subir documentos (.txt, .md, .pdf)", type=["txt", "md", "pdf"], accept_multiple_files=True
    )
    if st.sidebar.button("Indexar archivos", disabled=not files, key="upload"):
        results = client.upload([(f.name, f.getvalue()) for f in files])
        st.session_state.upload_messages = upload_messages(results)
        st.rerun()
    for level, text in st.session_state.get("upload_messages", []):
        getattr(st.sidebar, level)(text)

    if st.sidebar.button("Re-indexar todo", key="reindex"):
        try:
            report = client.ingest()
            st.sidebar.success(f"Re-indexado: {report['total_in_store']} fragmentos.")
        except APIClientError as exc:
            show(error_level(exc.code), error_text(exc.code, exc.detail))
    return True


def main() -> None:
    st.set_page_config(page_title="Asistente documental RAG", page_icon="📚")
    st.session_state.setdefault("messages", [])
    client = api_client.get_client()
    sidebar(client)

    st.title("Asistente documental")
    st.caption(NO_MEMORY_CAPTION)

    pending: str | None = None
    cols = st.columns(len(EXAMPLES) + 1)
    for col, (label, question, key) in zip(cols, EXAMPLES, strict=False):
        if col.button(label, key=key, help=question, use_container_width=True):
            pending = question
    if cols[-1].button("Limpiar conversación", key="clear", use_container_width=True):
        st.session_state.messages = []

    typed = st.chat_input("Escribe tu pregunta sobre los documentos…")
    if typed or pending:
        ask(client, typed or pending, st.session_state.get("top_k", 4))

    show_context = st.session_state.get("show_context", False)
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            elif "data" in msg:
                render_answer(msg["data"], show_context)
            else:
                render_error(msg, i, client)


if __name__ == "__main__":  # Streamlit (y AppTest) ejecutan el script como __main__
    main()
