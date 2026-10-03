"""Evidencia de M4: indexa el corpus en un Chroma temporal y consulta Q1, Q2 y Q8.

Usa un directorio temporal (la ingesta a data/chroma es de M5). Imprime count, source_stats,
la metadata de la colección y el top-4 de cada pregunta consultada a través de Chroma.
Uso: python scripts/consultar_chroma.py [--top-k 4]
"""

import argparse
import tempfile

from similitud_preguntas import QUESTIONS  # mismo directorio scripts/

from rag.chunking import chunk_documents
from rag.config import get_settings
from rag.embeddings import get_embedder
from rag.loaders import load_directory
from rag.vectorstore import ChromaVectorStore

SELECTED = ("Q1", "Q2", "Q8")


def main() -> None:
    """Punto de entrada del script."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    chunks = chunk_documents(
        load_directory(settings.docs_dir), settings.chunk_size, settings.chunk_overlap
    )
    with tempfile.TemporaryDirectory() as tmp:
        store = ChromaVectorStore(
            tmp,
            settings.chroma_collection,
            get_embedder(settings),
            embedding_model=settings.embedding_model,
        )
        store.add_chunks(chunks)

        print(f"Colección: {settings.chroma_collection} (directorio temporal)")
        print(f"Metadata de la colección: {store.collection_metadata}")
        print(f"count(): {store.count()}")
        print(f"source_stats(): {store.source_stats()}\n")

        for qid, qtype, question in QUESTIONS:
            if qid not in SELECTED:
                continue
            print(f"{qid} [{qtype}] {question}")
            for pos, r in enumerate(store.query(question, top_k=args.top_k), start=1):
                excerpt = r.text[:100].replace("\n", " ⏎ ")
                print(
                    f"  {pos}. {r.score:.3f} | {r.metadata['source']} "
                    f"pág={r.metadata.get('page', '-')} idx={r.metadata['chunk_index']} "
                    f"| {r.chunk_id} | {excerpt}"
                )
            print()


if __name__ == "__main__":
    main()
