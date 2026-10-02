"""Inspecciona la fragmentación del corpus: estadísticas por documento/página y chunks.

Uso:
  python scripts/inspeccionar_chunks.py [--chunk-size N] [--chunk-overlap M] [--dir data/docs]
  python scripts/inspeccionar_chunks.py --dump      # imprime TODOS los chunks completos
"""

import argparse
from collections import defaultdict
from statistics import mean

from rag.chunking import chunk_documents
from rag.config import get_settings
from rag.loaders import load_directory
from rag.models import Chunk

SEPARATOR = "-----"


def _page(chunk: Chunk) -> str:
    """Página del chunk como texto ('-' si no aplica)."""
    return str(chunk.metadata.get("page", "-"))


def print_stats(chunks: list[Chunk], size: int, overlap: int) -> None:
    """Imprime nº de chunks y tamaño medio/mín/máx por documento/página y total."""
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for c in chunks:
        groups[(c.metadata["source"], _page(c))].append(len(c.text))

    print(f"=== CHUNK_SIZE={size} · CHUNK_OVERLAP={overlap} ===")
    print(f"{'documento':<36} {'pág':>3} {'chunks':>6} {'medio':>6} {'mín':>5} {'máx':>5}")
    for (source, page), sizes in sorted(groups.items()):
        print(
            f"{source:<36} {page:>3} {len(sizes):>6} {mean(sizes):>6.0f} "
            f"{min(sizes):>5} {max(sizes):>5}"
        )
    sizes = [len(c.text) for c in chunks]
    print(
        f"{'TOTAL':<36} {'':>3} {len(sizes):>6} {mean(sizes):>6.0f} {min(sizes):>5} {max(sizes):>5}"
    )


def print_first(chunks: list[Chunk], n: int = 2) -> None:
    """Imprime los `n` primeros chunks del corpus."""
    print(f"\nPrimeros {n} chunks:")
    for c in chunks[:n]:
        print(
            f"[{c.chunk_id}] {c.metadata['source']} pág={_page(c)} idx={c.metadata['chunk_index']}"
        )
        print(c.text)
        print(SEPARATOR)


def print_dump(chunks: list[Chunk]) -> None:
    """Imprime todos los chunks completos separados por '-----'."""
    for c in chunks:
        print(f"chunk_id: {c.chunk_id}")
        print(f"source: {c.metadata['source']}")
        print(f"page: {_page(c)}")
        print(f"chunk_index: {c.metadata['chunk_index']}")
        print(f"caracteres: {len(c.text)}")
        print("texto:")
        print(c.text)
        print(SEPARATOR)


def main() -> None:
    """Punto de entrada del script."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--dir", default=str(settings.docs_dir))
    parser.add_argument("--dump", action="store_true", help="imprime todos los chunks")
    args = parser.parse_args()

    chunks = chunk_documents(load_directory(args.dir), args.chunk_size, args.chunk_overlap)
    if args.dump:
        print(
            f"# {len(chunks)} chunks · CHUNK_SIZE={args.chunk_size} · "
            f"CHUNK_OVERLAP={args.chunk_overlap}\n{SEPARATOR}"
        )
        print_dump(chunks)
    else:
        print_stats(chunks, args.chunk_size, args.chunk_overlap)
        print_first(chunks)


if __name__ == "__main__":
    main()
