"""Prueba de humo de Gemini con un prompt tipo RAG real (consume créditos).

Recupera 4 chunks del índice real (data/chroma, creado con `python -m rag.cli ingest`) para una
pregunta, arma system + contexto (~3.000 caracteres) + pregunta y reporta respuesta,
finish_reason, uso de tokens (incluido razonamiento si viene) y latencia. La key solo se muestra
enmascarada. Si la respuesta sale vacía o cortada (finish_reason="length") sale con código 1.

Uso: python scripts/gemini_smoke.py ["pregunta"]
"""

import sys

from rag.config import get_settings
from rag.ingest import build_store
from rag.llm import GeminiClient, LLMError

QUESTION = "¿Cuál es el tope diario de alimentación en viajes nacionales e internacionales?"
SYSTEM = (
    "Eres un asistente que responde ÚNICAMENTE con la información del contexto. Responde en "
    "español, de forma concisa, y cita cada afirmación con [n] según el fragmento usado. Si el "
    'contexto no contiene la respuesta, di: "No encontré información sobre eso en los '
    'documentos cargados."'
)


def mask(secret: str) -> str:
    """Solo los últimos 4 caracteres visibles."""
    return "****" + secret[-4:] if len(secret) >= 8 else "****"


def main() -> int:
    """Punto de entrada del script."""
    settings = get_settings()
    question = sys.argv[1] if len(sys.argv) > 1 else QUESTION
    store = build_store(settings)
    chunks = store.query(question, top_k=4)
    if not chunks:
        print("ERROR: índice vacío. Ejecuta `python -m rag.cli ingest data/docs`.", file=sys.stderr)
        return 1

    context = "\n\n".join(
        f"[{i}] (fuente: {c.metadata['source']}"
        + (f", pág. {c.metadata['page']}" if "page" in c.metadata else "")
        + f")\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    )
    user = f"Contexto:\n{context}\n\nPregunta: {question}"
    key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else ""

    print(f"API key: {mask(key)} · modelo: {settings.gemini_model}")
    print(f"LLM_MAX_TOKENS: {settings.llm_max_tokens} · temperatura: {settings.llm_temperature}")
    print(f"Pregunta: {question}")
    print("Chunks recuperados:")
    for i, c in enumerate(chunks, start=1):
        page = c.metadata.get("page", "-")
        print(f"  [{i}] {c.score:.3f} · {c.metadata['source']} pág={page} · {len(c.text)} car.")
    print(
        f"Tamaño del prompt: system {len(SYSTEM)} + user {len(user)} caracteres "
        f"(contexto {len(context)})"
    )

    try:
        result = GeminiClient(settings).generate(SYSTEM, user)
    except LLMError as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1

    print(f"\nRespuesta:\n{result.text}\n")
    print(f"finish_reason: {result.finish_reason}")
    print(
        f"prompt_tokens: {result.prompt_tokens} · completion_tokens: {result.completion_tokens}"
        f" · reasoning_tokens: {result.reasoning_tokens}"
    )
    print(f"latencia: {result.latency_s:.2f} s")
    if result.finish_reason == "length":
        print(
            "DETENER: respuesta cortada (finish_reason=length). Consultar antes de cambiar "
            "LLM_MAX_TOKENS o reasoning_effort.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
