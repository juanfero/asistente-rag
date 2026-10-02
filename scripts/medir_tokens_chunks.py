"""Mide cuántos tokens (tokenizer real del modelo, con tokens especiales) tienen los chunks.

Para cada configuración (chunk_size, chunk_overlap) reporta nº de chunks, tokens medio/p95/máx
y % de chunks que exceden `max_seq_length` (lo que exceda se trunca al embeber). Recomienda la
configuración más grande con ≤ 5 % de chunks truncados.

Uso: python scripts/medir_tokens_chunks.py [--configs 800:120 600:90 ...] [--dir data/docs]
"""

import argparse

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.chunking import chunk_documents
from rag.config import get_settings
from rag.loaders import load_directory

DEFAULT_CONFIGS = ["800:120", "600:90", "500:80", "400:60", "350:50"]
MAX_EXCEED_PCT = 5.0


def count_tokens(tokenizer, texts: list[str]) -> list[int]:
    """Nº de tokens de cada texto incluyendo los especiales (<s>, </s>), sin truncar."""
    return [
        len(tokenizer(t, add_special_tokens=True, truncation=False, verbose=False)["input_ids"])
        for t in texts
    ]


def measure(docs, tokenizer, max_len: int, size: int, overlap: int) -> dict:
    """Estadísticas de tokens de los chunks para una configuración."""
    chunks = chunk_documents(docs, size, overlap)
    tokens = np.array(count_tokens(tokenizer, [c.text for c in chunks]))
    return {
        "config": f"{size}/{overlap}",
        "size": size,
        "chunks": len(chunks),
        "mean": float(tokens.mean()),
        "p95": float(np.percentile(tokens, 95)),
        "max": int(tokens.max()),
        "exceed": int((tokens > max_len).sum()),
        "exceed_pct": float((tokens > max_len).mean() * 100),
    }


def main() -> None:
    """Punto de entrada del script."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--dir", default=str(settings.docs_dir))
    parser.add_argument("--model", default=settings.embedding_model)
    args = parser.parse_args()

    model = SentenceTransformer(args.model, device="cpu")
    tokenizer, max_len = model.tokenizer, model.max_seq_length
    special = len(tokenizer("", add_special_tokens=True)["input_ids"])
    docs = load_directory(args.dir)

    print(f"Modelo: {args.model}")
    print(f"max_seq_length: {max_len} tokens (incluye {special} tokens especiales por texto)")
    print(f"Corpus: {args.dir} ({len(docs)} documentos/páginas)\n")
    print("| Config | Chunks | Tokens medio | p95 | Máx | > max_seq_length | % excede |")
    print("|---|---|---|---|---|---|---|")
    rows = []
    for item in args.configs:
        size, overlap = (int(x) for x in item.split(":"))
        r = measure(docs, tokenizer, max_len, size, overlap)
        rows.append(r)
        print(
            f"| {r['config']} | {r['chunks']} | {r['mean']:.1f} | {r['p95']:.1f} | {r['max']} "
            f"| {r['exceed']} | {r['exceed_pct']:.1f} % |"
        )

    ok = [r for r in rows if r["exceed_pct"] <= MAX_EXCEED_PCT]
    print()
    if ok:
        best = max(ok, key=lambda r: r["size"])
        print(
            f"Recomendación: {best['config']} (la más grande con ≤ {MAX_EXCEED_PCT:.0f} % truncado)"
        )
    else:
        print(f"Ninguna configuración cumple ≤ {MAX_EXCEED_PCT:.0f} % truncado")


if __name__ == "__main__":
    main()
