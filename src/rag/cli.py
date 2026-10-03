"""Interfaz de consola: `python -m rag.cli {ingest,stats,reset}`."""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from rag.config import Settings, get_settings
from rag.embeddings import Embedder
from rag.ingest import IngestReport, build_store, ingest_paths
from rag.logging_conf import setup_logging
from rag.vectorstore import ChromaVectorStore, EmbeddingModelMismatchError, reset_collection

EXIT_OK = 0
EXIT_ERROR = 1


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
    return parser


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
) -> int:
    """Ejecuta la CLI y devuelve el código de salida (dependencias inyectables para pruebas)."""
    args = _parser().parse_args(argv)
    settings = settings or get_settings()
    setup_logging(settings.log_level)

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
