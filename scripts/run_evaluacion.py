"""Evaluación M10 con el motor real (RAGEngine + Gemini) sobre el índice de data/docs.

Uso:
  python scripts/run_evaluacion.py --repeticiones 3          # corrida completa (consume créditos)
  python scripts/run_evaluacion.py --solo-reporte   # regenera el reporte SIN llamar a Gemini
Opciones: --pausa S (default 4 s entre llamadas), --sin-extra (omite el bloque de robustez).

Genera en evaluacion/: resultados.json, resultados.md y revision_manual.yaml (solo si no existe;
la completa el autor). La API key nunca se imprime.
"""

import argparse
import json
import sys
from pathlib import Path

from rag.config import effective_settings, get_settings, log_effective_settings
from rag.evaluation import (
    MANUAL_FILE,
    TokenTrackingLLM,
    corpus_sources,
    evaluate,
    load_questions,
    validate_dataset,
    write_manual_template,
    write_outputs,
)
from rag.ingest import build_store, ingest_paths
from rag.llm import GeminiClient
from rag.logging_conf import setup_logging
from rag.rag_engine import RAGEngine

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "evaluacion"


def ensure_corpus_only(store, settings) -> None:
    """Verifica que el índice tenga solo el corpus de data/docs; si no, re-ingiere con reset."""
    expected = corpus_sources(settings.docs_dir)
    indexed = set(store.source_stats())
    if indexed == expected:
        print(f"Índice OK: {store.count()} chunks, solo el corpus ({', '.join(sorted(expected))})")
        return
    print(
        f"El índice no coincide con el corpus (indexado: {sorted(indexed)}). "
        "Re-ingiriendo data/docs con reset…"
    )
    store.reset()
    report = ingest_paths([settings.docs_dir], store, settings)
    print(f"Re-ingesta: {report.total_in_store} chunks de {sorted(report.sources)}")


def solo_reporte() -> int:
    """Regenera resultados.md/json desde el último resultados.json + la revisión manual."""
    path = EVAL_DIR / "resultados.json"
    if not path.exists():
        print(
            "ERROR: no existe evaluacion/resultados.json; ejecuta primero la evaluación.",
            file=sys.stderr,
        )
        return 1
    results = json.loads(path.read_text(encoding="utf-8"))
    write_outputs(results, EVAL_DIR)
    print(f"Reporte regenerado sin llamar a Gemini (revisión manual: {MANUAL_FILE}).")
    print(json.dumps(results["resumen"], ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    """Punto de entrada del script."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeticiones", type=int, default=1)
    parser.add_argument("--pausa", type=float, default=4.0, help="segundos entre llamadas al LLM")
    parser.add_argument("--solo-reporte", action="store_true")
    parser.add_argument("--sin-extra", action="store_true")
    args = parser.parse_args()
    if args.solo_reporte:
        return solo_reporte()

    settings = get_settings()
    setup_logging(settings.log_level)
    log_effective_settings(settings)

    questions = load_questions(EVAL_DIR / "preguntas.yaml")
    extras = [] if args.sin_extra else load_questions(EVAL_DIR / "preguntas_extra.yaml")
    errors = validate_dataset(questions) + validate_dataset(extras, main=False)
    if errors:
        print("ERROR en los sets de preguntas:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 1

    store = build_store(settings)
    ensure_corpus_only(store, settings)
    tracker = TokenTrackingLLM(GeminiClient(settings))
    engine = RAGEngine(store, tracker, settings)

    config = effective_settings(settings)
    print(
        f"Evaluando {len(questions)} preguntas × {args.repeticiones} repeticiones + "
        f"{len(extras)} de robustez · pausa {args.pausa} s"
    )
    results = evaluate(
        engine,
        tracker,
        questions,
        extras,
        repetitions=args.repeticiones,
        pause_s=args.pausa,
        metadata={"config": config},
    )
    write_outputs(results, EVAL_DIR)
    created = write_manual_template(results, EVAL_DIR / MANUAL_FILE)
    print(
        f"\nPlantilla de revisión manual {'creada' if created else 'ya existía (no se tocó)'}: "
        f"evaluacion/{MANUAL_FILE}"
    )
    print(json.dumps(results["resumen"], ensure_ascii=False, indent=2))
    return 2 if results["metadata"].get("incompleto") else 0


if __name__ == "__main__":
    sys.exit(main())
