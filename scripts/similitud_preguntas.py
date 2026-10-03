"""Diagnóstico de recuperación (sin Chroma): top-3 chunks por similitud coseno para las 9
preguntas de evaluación de M10, con fuerza bruta en numpy.

No es un criterio de aceptación: sirve para anticipar la calibración del umbral (M7).
Uso: python scripts/similitud_preguntas.py [--top-k 3]
"""

import argparse
import time

import numpy as np

from rag.chunking import chunk_documents
from rag.config import get_settings
from rag.embeddings import get_embedder
from rag.loaders import load_directory

# Preguntas de docs/modulos/M10_evaluacion.md (el set definitivo se versiona en M10)
QUESTIONS = [
    (
        "Q1",
        "Contestable",
        "¿Cuántos días de vacaciones tengo por año trabajado y con cuánta "
        "anticipación debo solicitarlas?",
    ),
    (
        "Q2",
        "Contestable",
        "¿Cuál es el tope diario de alimentación en viajes nacionales e internacionales?",
    ),
    ("Q3", "Contestable", "¿Qué requisitos debe cumplir mi contraseña corporativa?"),
    (
        "Q4",
        "Parcial",
        "¿Cuántos días de permiso me dan por matrimonio y cómo se pagan las horas extra?",
    ),
    ("Q5", "Parcial", "¿Qué VPN debo usar y me prestan un celular corporativo?"),
    (
        "Q6",
        "Parcial",
        "¿Cuál es el tope de hospedaje en Medellín y cuánto reconocen por "
        "kilómetro con vehículo propio?",
    ),
    ("Q7", "No contestable", "¿Cuál es el salario promedio de un analista en Nexa?"),
    ("Q8", "No contestable", "¿Cuántos días a la semana puedo trabajar remoto?"),
    ("Q9", "No contestable", "¿Quién es el CEO de Nexa Logística?"),
]


def main() -> None:
    """Punto de entrada del script."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    chunks = chunk_documents(
        load_directory(settings.docs_dir), settings.chunk_size, settings.chunk_overlap
    )
    embedder = get_embedder(settings)

    start = time.perf_counter()
    _ = embedder.dimension  # fuerza la carga del modelo
    load_s = time.perf_counter() - start
    start = time.perf_counter()
    matrix = np.array(embedder.embed_documents([c.text for c in chunks]))
    corpus_s = time.perf_counter() - start

    print(
        f"Modelo: {settings.embedding_model} · CHUNK_SIZE={settings.chunk_size} · "
        f"CHUNK_OVERLAP={settings.chunk_overlap} · prefijos {settings.embedding_query_prefix!r}"
        f"/{settings.embedding_passage_prefix!r} · {len(chunks)} chunks"
    )
    print(f"Carga del modelo: {load_s:.1f} s · Embeddings del corpus: {corpus_s:.2f} s")
    print("Score = similitud coseno (vectores normalizados → producto punto)\n")

    for qid, qtype, question in QUESTIONS:
        scores = matrix @ np.array(embedder.embed_query(question))
        print(f"{qid} [{qtype}] {question}")
        for rank, idx in enumerate(np.argsort(-scores)[: args.top_k], start=1):
            c = chunks[idx]
            excerpt = c.text[:100].replace("\n", " ⏎ ")
            print(
                f"  {rank}. {scores[idx]:.3f} | {c.metadata['source']} "
                f"pág={c.metadata.get('page', '-')} | {excerpt}"
            )
        print()


if __name__ == "__main__":
    main()
