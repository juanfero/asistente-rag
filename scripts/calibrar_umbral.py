"""Calibra MIN_SCORE como filtro de ruido (ADR-005) con el índice real.

Mide el top-1 (similitud coseno) de tres grupos de preguntas:
  a) legítimas (Q1–Q6 de M10), b) paráfrasis legítimas, c) fuera de dominio.
Regla: MIN_SCORE = min(top-1 de a+b) − 0,03. Si queda por encima de max(top-1 de c), el
umbral filtra fuera de dominio sin perder legítimas; si no, se usa igual (prioridad: no perder
legítimas) y el filtro de fuera de dominio no es efectivo. Q7–Q9 (en dominio sin respuesta) no
se usan: las rechaza el LLM.

Uso: python scripts/calibrar_umbral.py   (requiere `python -m rag.cli ingest data/docs`)
"""

import sys

from similitud_preguntas import QUESTIONS  # mismo directorio scripts/

from rag.config import get_settings
from rag.ingest import build_store

MARGIN = 0.03
LEGITIMATE = [(qid, q) for qid, _, q in QUESTIONS if qid in {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6"}]
PARAPHRASES = [
    ("P1", "¿Cuánto tiempo de descanso me corresponde al año?"),
    ("P2", "¿Cuánta plata me reconocen para comida cuando viajo dentro del país?"),
    ("P3", "¿Cada cuánto tengo que cambiar la clave del computador?"),
    ("P4", "¿Qué programa uso para conectarme desde la casa?"),
]
OUT_OF_DOMAIN = [
    ("F1", "¿Cuál es la capital de Francia?"),
    ("F2", "Dame una receta de arepas"),
    ("F3", "¿Quién ganó el último mundial de fútbol?"),
    ("F4", "Explícame la teoría de la relatividad"),
]
GROUPS = [
    ("a) Legítimas", LEGITIMATE),
    ("b) Paráfrasis legítimas", PARAPHRASES),
    ("c) Fuera de dominio", OUT_OF_DOMAIN),
]


def calibrate(top1: dict[str, list[float]]) -> tuple[float, bool]:
    """MIN_SCORE según la regla y si separa las legítimas de las fuera de dominio."""
    legit = top1["a) Legítimas"] + top1["b) Paráfrasis legítimas"]
    value = round(min(legit) - MARGIN, 3)
    return value, value > max(top1["c) Fuera de dominio"])


def main() -> int:
    """Punto de entrada del script."""
    store = build_store(get_settings())
    if store.count() == 0:
        print("ERROR: índice vacío. Ejecuta `python -m rag.cli ingest data/docs`.", file=sys.stderr)
        return 1

    top1: dict[str, list[float]] = {}
    print("| Grupo | Id | Pregunta | top-1 | Fuente top-1 |")
    print("|---|---|---|---|---|")
    for group, questions in GROUPS:
        top1[group] = []
        for qid, question in questions:
            best = store.query(question, top_k=1)[0]
            top1[group].append(best.score)
            print(
                f"| {group} | {qid} | {question} | {best.score:.3f} | {best.metadata['source']} |"
            )

    print("\n| Grupo | mín top-1 | máx top-1 |")
    print("|---|---|---|")
    for group, scores in top1.items():
        print(f"| {group} | {min(scores):.3f} | {max(scores):.3f} |")

    value, separates = calibrate(top1)
    legit_min = min(top1["a) Legítimas"] + top1["b) Paráfrasis legítimas"])
    print(
        f"\nmin(top-1 a+b) = {legit_min:.3f} → MIN_SCORE = {legit_min:.3f} − {MARGIN} = {value:.3f}"
    )
    print(f"max(top-1 c)   = {max(top1['c) Fuera de dominio']):.3f}")
    if separates:
        print("Resultado: el umbral SEPARA — filtra fuera de dominio sin tocar las legítimas.")
    else:
        print(
            "Resultado: el umbral NO separa fuera de dominio; se usa igual para no perder "
            "legítimas (el filtro de fuera de dominio no es efectivo)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
