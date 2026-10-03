"""Compara variantes de modelo de embeddings / tamaño de chunk sobre las preguntas de M10.

Métricas (gold set: un chunk es relevante para un hecho si contiene su texto literal):
- Recall@4: % de hechos gold de Q1–Q6 cuyo chunk aparece en el top-4.
- Rango del primer chunk con "COP 120.000" para Q2.
- MRR de Q1–Q6 (primer chunk que contiene algún hecho gold de la pregunta).
- Margen = mín(top-1 de Q1–Q6) − máx(top-1 de Q7–Q9); positivo → un umbral las separa.
- Tiempo de carga (desde caché) y tamaño en disco del modelo.

Uso: python scripts/comparar_embeddings.py
"""

import time
from pathlib import Path

import numpy as np
from huggingface_hub.constants import HF_HUB_CACHE
from similitud_preguntas import QUESTIONS  # mismo directorio scripts/

from rag.chunking import chunk_documents
from rag.config import get_settings
from rag.embeddings import SentenceTransformerEmbedder
from rag.loaders import load_directory

TOP_K = 4
MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
E5 = "intfloat/multilingual-e5-small"

VARIANTS = [
    {"id": "A", "model": MINILM, "size": 500, "overlap": 80, "q": "", "p": ""},
    {"id": "B", "model": E5, "size": 500, "overlap": 80, "q": "query: ", "p": "passage: "},
    {"id": "C", "model": E5, "size": 800, "overlap": 120, "q": "query: ", "p": "passage: "},
]

GOLD = {
    "Q1": ["15 días hábiles", "15 días calendario"],
    "Q2": ["COP 120.000", "USD 90"],
    "Q3": ["12 caracteres"],
    "Q4": ["Matrimonio"],
    "Q5": ["FortiClient"],
    "Q6": ["COP 420.000"],
}
ANSWERABLE = list(GOLD)
UNANSWERABLE = ["Q7", "Q8", "Q9"]


def model_disk_mb(model: str) -> float:
    """Tamaño en disco (MB) del modelo en la caché de Hugging Face."""
    blobs = Path(HF_HUB_CACHE) / f"models--{model.replace('/', '--')}" / "blobs"
    return sum(f.stat().st_size for f in blobs.iterdir() if f.is_file()) / 1e6


def first_rank(order: list[int], chunks, text: str) -> int | None:
    """Posición (1-based) del primer chunk del ranking que contiene `text`."""
    for pos, idx in enumerate(order, start=1):
        if text in chunks[idx].text:
            return pos
    return None


def evaluate(variant: dict, docs) -> tuple[dict, list[str]]:
    """Calcula las métricas de una variante y el detalle top-k por pregunta."""
    chunks = chunk_documents(docs, variant["size"], variant["overlap"])
    emb = SentenceTransformerEmbedder(
        variant["model"], query_prefix=variant["q"], passage_prefix=variant["p"]
    )
    start = time.perf_counter()
    _ = emb.dimension  # fuerza la carga
    load_s = time.perf_counter() - start
    matrix = np.array(emb.embed_documents([c.text for c in chunks]))

    top1, ranks_any, hits, total = {}, [], 0, 0
    detail = [
        f"### Variante {variant['id']}: {variant['model']} · {variant['size']}/{variant['overlap']}"
        f" · prefijos {variant['q']!r}/{variant['p']!r} · {len(chunks)} chunks"
    ]
    rank_cop = None
    for qid, qtype, question in QUESTIONS:
        scores = matrix @ np.array(emb.embed_query(question))
        order = [int(i) for i in np.argsort(-scores)]
        top1[qid] = float(scores[order[0]])
        facts = GOLD.get(qid, [])
        for fact in facts:
            total += 1
            rank = first_rank(order, chunks, fact)
            hits += rank is not None and rank <= TOP_K
        if facts:
            ranks = [r for r in (first_rank(order, chunks, f) for f in facts) if r]
            ranks_any.append(min(ranks) if ranks else None)
        if qid == "Q2":
            rank_cop = first_rank(order, chunks, "COP 120.000")

        detail.append(f"{qid} [{qtype}] {question}")
        for pos, idx in enumerate(order[:TOP_K], start=1):
            c = chunks[idx]
            gold = [f for f in facts if f in c.text]
            mark = f"  ✓ {', '.join(gold)}" if gold else ""
            excerpt = c.text[:90].replace("\n", " ⏎ ")
            detail.append(
                f"  {pos}. {scores[idx]:.3f} | {c.metadata['source']} "
                f"pág={c.metadata.get('page', '-')} | {excerpt}{mark}"
            )
    detail.append("")

    metrics = {
        "id": variant["id"],
        "desc": f"{variant['model'].split('/')[-1]} {variant['size']}/{variant['overlap']}",
        "chunks": len(chunks),
        "recall": 100 * hits / total,
        "rank_cop": rank_cop,
        "mrr": float(np.mean([1 / r if r else 0 for r in ranks_any])),
        "min_ans": min(top1[q] for q in ANSWERABLE),
        "max_unans": max(top1[q] for q in UNANSWERABLE),
        "load_s": load_s,
        "disk_mb": model_disk_mb(variant["model"]),
        "top1": top1,
    }
    metrics["margin"] = metrics["min_ans"] - metrics["max_unans"]
    return metrics, detail


def main() -> None:
    """Punto de entrada del script."""
    docs = load_directory(get_settings().docs_dir)
    results, details = [], []
    for variant in VARIANTS:
        metrics, detail = evaluate(variant, docs)
        results.append(metrics)
        details.extend(detail)

    print("## Resumen\n")
    print(
        "| Var. | Modelo y chunk | Chunks | Recall@4 | Rango COP 120.000 (Q2) | MRR Q1–Q6 "
        "| mín top-1 Q1–Q6 | máx top-1 Q7–Q9 | Margen | Carga (caché) | Disco |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['id']} | {r['desc']} | {r['chunks']} | {r['recall']:.1f} % | {r['rank_cop']} "
            f"| {r['mrr']:.3f} | {r['min_ans']:.3f} | {r['max_unans']:.3f} | {r['margin']:+.3f} "
            f"| {r['load_s']:.1f} s | {r['disk_mb']:.0f} MB |"
        )

    print("\n## Top-1 por pregunta\n")
    ids = [q[0] for q in QUESTIONS]
    print("| Var. | " + " | ".join(ids) + " |")
    print("|---|" + "---|" * len(ids))
    for r in results:
        print(f"| {r['id']} | " + " | ".join(f"{r['top1'][q]:.3f}" for q in ids) + " |")

    print(f"\n## Detalle top-{TOP_K} por pregunta (✓ = contiene un hecho gold)\n")
    print("\n".join(details))


if __name__ == "__main__":
    main()
