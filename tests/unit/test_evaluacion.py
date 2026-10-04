"""Pruebas de M10: dataset, verificación automática y reporte (FakeLLM, sin red)."""

import json
from pathlib import Path

import pytest
import yaml

from rag.config import Settings
from rag.embeddings import FakeEmbedder
from rag.evaluation import (
    MANUAL_FILE,
    TokenTrackingLLM,
    check_answer,
    evaluate,
    load_questions,
    merge_manual,
    normalize_eval,
    render_markdown,
    validate_dataset,
    write_manual_template,
    write_outputs,
)
from rag.ingest import build_store, ingest_paths
from rag.llm import FakeLLM, LLMQuotaExhaustedError, LLMRateLimitError
from rag.loaders import PROJECT_ROOT
from rag.prompts import NOT_FOUND_MESSAGE
from rag.rag_engine import RAGEngine

EVAL = PROJECT_ROOT / "evaluacion"
QUESTIONS = load_questions(EVAL / "preguntas.yaml")
EXTRAS = load_questions(EVAL / "preguntas_extra.yaml")
BY_ID = {q["id"]: q for q in QUESTIONS + EXTRAS}


def src(source: str, page: int | None = None, cited: bool = True) -> dict:
    return {"source": source, "page": page, "cited": cited}


PDF, MD, TXT = (
    "manual_reembolso_gastos.pdf",
    "politica_vacaciones_y_permisos.md",
    "guia_onboarding_ti.txt",
)


# --- M10-01 -----------------------------------------------------------------------------


def test_eval_dataset() -> None:
    """M10-01: preguntas.yaml válido (9 preguntas, 3 por tipo, campos obligatorios)."""
    assert validate_dataset(QUESTIONS) == []
    assert [q["id"] for q in QUESTIONS] == [f"Q{i}" for i in range(1, 10)]
    assert validate_dataset(EXTRAS, main=False) == []
    assert {q["tipo"] for q in EXTRAS} == {"parafrasis", "fuera_de_dominio", "inyeccion"}
    broken = [dict(q) for q in QUESTIONS[:8]]
    broken[3].pop("missing_topic")
    errors = validate_dataset(broken)
    assert any("9 preguntas" in e for e in errors) and any("missing_topic" in e for e in errors)


def test_normalize_eval() -> None:
    """Normalización: minúsculas, sin tildes y números sin separadores de miles."""
    assert normalize_eval("COP 120.000") == "cop 120000"
    assert normalize_eval("COP 2.000.000 y 7:00") == "cop 2000000 y 7:00"
    assert normalize_eval("  Quince  DÍAS hábiles ") == "quince dias habiles"
    assert normalize_eval("1.5 y 1,000") == "1.5 y 1000"


# --- M10-02: verificación automática con casos sintéticos -----------------------------------


@pytest.mark.parametrize(
    ("qid", "record", "expected"),
    [
        (
            "Q2",
            {
                "answer": "Nacional COP 120.000 [1]; internacional USD 90 [2].",
                "grounded": True,
                "sources": [src(PDF, 1), src(PDF, 2)],
            },
            True,
        ),
        (
            "Q2",
            {
                "answer": "Nacional COP 120.000 [1]; internacional USD 90.",
                "grounded": True,
                "sources": [src(PDF, 1)],
            },
            False,
        ),  # falta la cita de la pág. 2
        (
            "Q2",
            {
                "answer": "Nacional COP 120.000 [1].",
                "grounded": True,
                "sources": [src(PDF, 1), src(PDF, 2)],
            },
            False,
        ),  # falta el hecho USD 90
        (
            "Q1",
            {
                "answer": "Quince días hábiles al año; pedirlas con 15 días calendario [1].",
                "grounded": True,
                "sources": [src(MD)],
            },
            True,
        ),  # variante "quince"
        (
            "Q4",
            {
                "answer": "5 días hábiles [1]. Los documentos no incluyen información sobre "
                "cómo se pagan las horas extra.",
                "grounded": True,
                "sources": [src(MD)],
            },
            True,
        ),
        (
            "Q4",
            {
                "answer": "5 días hábiles [1]. No sé lo de las horas extra.",
                "grounded": True,
                "sources": [src(MD)],
            },
            False,
        ),  # sin la frase estándar
        (
            "Q6",
            {
                "answer": "COP 420.000 por noche [1]. Los documentos no incluyen información "
                "sobre el reconocimiento por kilómetro.",
                "grounded": True,
                "sources": [src(PDF, 1)],
            },
            True,
        ),
        ("Q8", {"answer": NOT_FOUND_MESSAGE, "grounded": False, "sources": []}, True),
        (
            "Q8",
            {
                "answer": "Puedes trabajar 2 días remoto [1].",
                "grounded": True,
                "sources": [src(MD)],
            },
            False,
        ),
        (
            "F1",
            {"answer": NOT_FOUND_MESSAGE, "grounded": False, "sources": [], "llm_called": False},
            True,
        ),
        (
            "F1",
            {"answer": NOT_FOUND_MESSAGE, "grounded": False, "sources": [], "llm_called": True},
            False,
        ),  # debía cortarse por el umbral
        (
            "P2",
            {
                "answer": "Te reconocen COP 120.000 diarios [1].",
                "grounded": True,
                "sources": [src(PDF, 1)],
            },
            True,
        ),
    ],
    ids=[
        "q2-ok",
        "q2-sin-cita-p2",
        "q2-sin-hecho",
        "q1-variante",
        "q4-ok",
        "q4-sin-frase",
        "q6-ok",
        "q8-ok",
        "q8-alucina",
        "f1-ok",
        "f1-sin-corte",
        "p2-ok",
    ],
)
def test_eval_checks(qid: str, record: dict, expected: bool) -> None:
    """M10-02: las reglas clasifican bien casos sintéticos."""
    check = check_answer(BY_ID[qid], record)
    assert check.correct is expected, check.observation


def test_check_observation_text() -> None:
    """La observación automática describe lo verificado."""
    obs = check_answer(
        BY_ID["Q2"],
        {
            "answer": "COP 120.000 [1] y USD 90 [2].",
            "grounded": True,
            "sources": [src(PDF, 1), src(PDF, 2)],
        },
    ).observation
    assert "Hechos presentes: 120000, usd 90" in obs
    assert f"cita {PDF} pág. 1, {PDF} pág. 2" in obs


# --- M10-03: runner y reporte con FakeLLM ---------------------------------------------------


@pytest.fixture
def engine_factory(tmp_path: Path):
    settings = Settings(_env_file=None, chroma_dir=tmp_path / "chroma", min_score=0.0)
    store = build_store(settings, FakeEmbedder(64))
    ingest_paths([PROJECT_ROOT / "data" / "docs"], store, settings)

    def make(responses):
        tracker = TokenTrackingLLM(FakeLLM(responses))
        return RAGEngine(store, tracker, settings), tracker

    return make


def answer_for(system: str, user: str) -> str:
    """FakeLLM que responde según la pregunta, como lo haría un modelo correcto."""
    q = user.rsplit("Pregunta:", 1)[-1]
    if "alimentación" in q:
        return "Nacional COP 120.000 [1]; internacional USD 90 [2]."
    return NOT_FOUND_MESSAGE


def test_eval_report(engine_factory, tmp_path: Path) -> None:
    """M10-03: el runner genera resultados.json y resultados.md con las columnas exigidas."""
    engine, tracker = engine_factory(answer_for)
    sleeps: list[float] = []
    results = evaluate(
        engine,
        tracker,
        QUESTIONS,
        EXTRAS[:1],
        repetitions=2,
        pause_s=4.0,
        sleep=sleeps.append,
        log=lambda _: None,
        metadata={"config": {}},
    )
    out = tmp_path / "evaluacion"
    write_outputs(results, out)
    created = write_manual_template(results, out / MANUAL_FILE)

    data = json.loads((out / "resultados.json").read_text(encoding="utf-8"))
    assert len(data["preguntas"]) == 9 and len(data["extra"]) == 1
    assert all(len(q["repeticiones"]) == 2 for q in data["preguntas"])
    md = (out / "resultados.md").read_text(encoding="utf-8")
    assert (
        "| ID | Tipo | Pregunta | Respuesta generada | ¿Correcta? (auto) | "
        "¿Correcta? (manual) | Observación |"
    ) in md
    assert "## Reglas de verificación automática" in md and "p50" in md
    assert created and (out / MANUAL_FILE).exists()
    assert sleeps and set(sleeps) == {4.0}  # pausa entre llamadas al LLM
    assert results["resumen"]["llamadas_llm"] == tracker.calls > 0
    # Las 3 no contestables pasan; las demás fallan con este FakeLLM simplista
    assert results["resumen"]["por_tipo"]["no_contestable"]["auto"] == "3/3"


def test_all_repetitions_must_pass(engine_factory) -> None:
    """Una pregunta es correcta solo si todas sus repeticiones lo son (consistencia k/N)."""
    hallucination = "Puedes trabajar remoto 2 días por semana [1]."
    engine, tracker = engine_factory([NOT_FOUND_MESSAGE, NOT_FOUND_MESSAGE, hallucination])
    results = evaluate(
        engine,
        tracker,
        [BY_ID["Q8"]],
        [],
        repetitions=3,
        pause_s=0,
        sleep=lambda _: None,
        log=lambda _: None,
    )
    q = results["preguntas"][0]
    assert q["consistencia"] == "2/3" and q["correcta_auto"] is False
    assert "inconsistente" in q["observacion_auto"]
    assert [r["correcta"] for r in q["repeticiones"]] == [True, True, False]


def test_rate_limit_waits_and_retries(engine_factory) -> None:
    """LLMRateLimitError → espera 60 s y reintenta una vez."""
    engine, tracker = engine_factory([LLMRateLimitError("límite"), NOT_FOUND_MESSAGE])
    sleeps: list[float] = []
    results = evaluate(
        engine,
        tracker,
        [BY_ID["Q8"]],
        [],
        repetitions=1,
        pause_s=4,
        sleep=sleeps.append,
        log=lambda _: None,
    )
    assert sleeps[0] == 60 and results["preguntas"][0]["correcta_auto"] is True


def test_quota_exhausted_stops_and_keeps_progress(engine_factory) -> None:
    """LLMQuotaExhaustedError → se detiene y conserva lo avanzado (incompleto=True)."""
    engine, tracker = engine_factory([NOT_FOUND_MESSAGE, LLMQuotaExhaustedError("sin créditos")])
    results = evaluate(
        engine,
        tracker,
        [BY_ID["Q7"], BY_ID["Q8"], BY_ID["Q9"]],
        [],
        repetitions=1,
        pause_s=0,
        sleep=lambda _: None,
        log=lambda _: None,
    )
    assert results["metadata"]["incompleto"] is True
    assert [q["id"] for q in results["preguntas"]] == ["Q7"]
    assert "Ejecución incompleta" in render_markdown(results)


def test_manual_review_merge_and_no_overwrite(engine_factory, tmp_path: Path) -> None:
    """La plantilla no se sobrescribe y la revisión manual se refleja en el reporte."""
    engine, tracker = engine_factory(answer_for)
    results = evaluate(
        engine, tracker, QUESTIONS, [], pause_s=0, sleep=lambda _: None, log=lambda _: None
    )
    path = tmp_path / MANUAL_FILE
    assert write_manual_template(results, path) is True
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert all(
        e["correcta_manual"] is None and e["observacion_manual"] == "" for e in data["revision"]
    )
    data["revision"][7] = {
        "id": "Q8",
        "correcta_manual": True,
        "observacion_manual": "Revisado por el autor",
    }
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    assert write_manual_template(results, path) is False  # no se pisa lo del autor
    merged = merge_manual(results, path)
    q8 = next(q for q in merged["preguntas"] if q["id"] == "Q8")
    assert q8["correcta_manual"] is True
    assert merged["resumen"]["aciertos_manual"] == "pendiente"  # faltan las otras 8
    md = render_markdown(merged)
    assert "Revisado por el autor" in md and "pendiente" in md
