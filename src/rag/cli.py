"""Interfaz de consola: `python -m rag.cli {ingest,stats,reset,ask}`."""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from rag.config import Settings, get_settings, log_effective_settings
from rag.embeddings import Embedder
from rag.ingest import IngestReport, build_store, ingest_paths
from rag.llm import LazyLLM, LLMClient, LLMError, get_llm
from rag.logging_conf import setup_logging
from rag.rag_engine import RAGAnswer, RAGEngine
from rag.vectorstore import ChromaVectorStore, EmbeddingModelMismatchError, reset_collection

EXIT_OK = 0
EXIT_ERROR = 1  # entrada inválida, índice vacío o de otro modelo
EXIT_LLM_ERROR = 2  # créditos agotados, key inválida, límite, etc. (ADR-010)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="Asistente documental RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="ingiere archivos o directorios (txt, md, pdf)")
    ingest.add_argument("paths", nargs="+", type=Path, help="archivos o directorios")
    ingest.add_argument("--reset", action="store_true", help="vacía el índice antes de ingerir")
    ingest.add_argument("--recursive", action="store_true", help="recorre subdirectorios")

    sub.add_parser("stats", help="muestra chunks, modelo y fuentes del índice")

    reset = sub.add_parser("reset", help="vacía el índice")
    reset.add_argument("--yes", action="store_true", help="no pedir confirmación")

    ask = sub.add_parser("ask", help="pregunta sobre los documentos indexados")
    ask.add_argument("question", help="pregunta en lenguaje natural")
    ask.add_argument("--top-k", type=int, default=None, help="fragmentos a recuperar")
    ask.add_argument("--show-context", action="store_true", help="muestra el contexto usado")
    return parser


def _print_answer(answer: RAGAnswer, settings: Settings, show_context: bool) -> None:
    """Imprime respuesta, estado, fuentes y (opcional) el contexto enviado al LLM."""
    print("Respuesta:")
    print(answer.answer)
    print()
    print("✔ Basada en documentos" if answer.grounded else "✘ Sin información en los documentos")
    if answer.sources:
        print("Fuentes:")
        for ref in answer.sources:
            page = f", pág. {ref.page}" if ref.page is not None else ""
            mark = "citada" if ref.cited else "no citada"
            print(f"  [{ref.index}] {ref.source}{page} · score {ref.score:.3f} · {mark}")
    llm = f"modelo {answer.model}" if answer.llm_called else "sin llamada al LLM"
    print(
        f"({llm} · {answer.latency_s:.2f} s · recuperados {answer.retrieved}, "
        f"sobre MIN_SCORE={settings.min_score}: {len(answer.context)})"
    )
    if show_context:
        print("\nContexto enviado al LLM:")
        if answer.prompt is None:
            print(
                f"  (ninguno: ningún fragmento supera MIN_SCORE={settings.min_score}; "
                "no se llamó al LLM)"
            )
        else:
            print(answer.prompt)


def _run_ask(
    args: argparse.Namespace, store: ChromaVectorStore, settings: Settings, llm: LLMClient | None
) -> int:
    """Ejecuta `ask` y traduce los errores a códigos de salida (0/1/2) sin traceback."""
    engine = RAGEngine(store, llm or LazyLLM(lambda: get_llm(settings)), settings)
    try:
        answer = engine.ask(args.question, top_k=args.top_k)
    except LLMError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_LLM_ERROR
    except ValueError as exc:  # pregunta inválida o EmptyIndexError
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR
    _print_answer(answer, settings, args.show_context)
    return EXIT_OK


def _print_report(report: IngestReport) -> None:
    """Imprime el reporte de ingesta como tabla."""
    rows = [
        ("Archivos procesados", report.files_processed),
        ("Archivos omitidos", report.files_skipped),
        ("Documentos (págs. PDF incluidas)", report.documents),
        ("Chunks añadidos", report.chunks_added),
        ("Total de chunks en el índice", report.total_in_store),
        ("Duración", f"{report.duration_s:.2f} s"),
    ]
    width = max(len(label) for label, _ in rows)
    print("Reporte de ingesta")
    for label, value in rows:
        print(f"  {label:<{width}}  {value}")
    print("Fuentes ingeridas:")
    for source in report.sources or ["(ninguna)"]:
        print(f"  - {source}")
    if report.skipped:
        print("Omitidos:")
        for item in report.skipped:
            print(f"  - {item}")


def _print_stats(store: ChromaVectorStore) -> None:
    """Imprime chunks totales, modelo/dimensión de la colección y chunks por fuente."""
    meta = store.collection_metadata
    stats = store.source_stats()
    print(f"Colección: {store.collection_name}")
    print(f"Modelo de embeddings: {meta.get('embedding_model')} (dim {meta.get('embedding_dim')})")
    print(f"Total de chunks: {store.count()}")
    print("Fuentes:")
    if not stats:
        print("  (índice vacío)")
    width = max((len(s) for s in stats), default=0)
    for source, count in stats.items():
        print(f"  - {source:<{width}}  {count} chunks")


def _open_store(settings: Settings, embedder: Embedder | None) -> ChromaVectorStore | None:
    """Abre el store; si el índice es de otro modelo, explica cómo resolverlo."""
    try:
        return build_store(settings, embedder)
    except EmbeddingModelMismatchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Solución: `python -m rag.cli reset --yes` o "
            "`python -m rag.cli ingest <ruta> --reset`.",
            file=sys.stderr,
        )
        return None


def main(
    argv: list[str] | None = None,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    confirm: Callable[[str], str] = input,
    llm: LLMClient | None = None,
) -> int:
    """Ejecuta la CLI y devuelve el código de salida (dependencias inyectables para pruebas)."""
    args = _parser().parse_args(argv)
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    log_effective_settings(settings)

    if args.command == "reset":
        if not args.yes:
            answer = confirm(f"¿Vaciar el índice '{settings.chroma_collection}'? [s/N] ")
            if answer.strip().lower() not in {"s", "si", "sí", "y", "yes"}:
                print("Cancelado: el índice no se modificó.")
                return EXIT_OK
        reset_collection(settings.chroma_dir, settings.chroma_collection)
        print(f"Índice '{settings.chroma_collection}' vaciado.")
        return EXIT_OK

    if args.command == "ingest" and args.reset:
        reset_collection(settings.chroma_dir, settings.chroma_collection)
        print(f"Índice '{settings.chroma_collection}' vaciado antes de ingerir.")

    store = _open_store(settings, embedder)
    if store is None:
        return EXIT_ERROR

    if args.command == "stats":
        _print_stats(store)
        return EXIT_OK

    if args.command == "ask":
        return _run_ask(args, store, settings, llm)

    try:
        report = ingest_paths(args.paths, store, settings, recursive=args.recursive)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}. Revisa la ruta e intenta de nuevo.", file=sys.stderr)
        return EXIT_ERROR
    _print_report(report)
    if report.files_processed == 0:
        print("ERROR: no se ingirió ningún documento.", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
