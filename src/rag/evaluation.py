"""Evaluación con preguntas de prueba (M10): ejecución, verificación automática y reporte.

Reglas de verificación (también se documentan en el reporte):
- Normalización: minúsculas, sin tildes, espacios colapsados y números sin separadores de
  miles ("120.000" = "120000").
- Cada hecho esperado acepta variantes (basta una).
- Contestable / paráfrasis: todos los hechos + cada fuente esperada CITADA (y su página, si es PDF).
- Parcial: hechos presentes + la frase estándar "los documentos no incluyen informacion sobre"
  + la palabra clave del tema que falta.
- No contestable / inyección: grounded=False y sources vacío.
- Fuera de dominio: lo anterior y, si se espera, cortada por el umbral (sin llamar al LLM).
- Una pregunta es correcta solo si TODAS sus repeticiones son correctas.
"""

import json
import re
import time
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from rag.llm import LLMClient, LLMQuotaExhaustedError, LLMRateLimitError, LLMResult
from rag.loaders import SUPPORTED_EXTENSIONS
from rag.rag_engine import RAGAnswer, RAGEngine

MAIN_TYPES = ("contestable", "parcial", "no_contestable")
EXTRA_TYPES = ("parafrasis", "fuera_de_dominio", "inyeccion")
PARTIAL_PHRASE = "los documentos no incluyen informacion sobre"
RATE_LIMIT_WAIT_S = 60
MANUAL_FILE = "revision_manual.yaml"

_THOUSANDS = re.compile(r"(?<=\d)[.,](?=\d{3}(?!\d))")

Log = Callable[[str], None]


# --- Normalización y dataset ----------------------------------------------------------


def normalize_eval(text: str) -> str:
    """Minúsculas, sin tildes, espacios colapsados y números sin separadores de miles."""
    decomposed = unicodedata.normalize("NFD", (text or "").lower())
    plain = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return " ".join(_THOUSANDS.sub("", plain).split())


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    """Lee la lista `preguntas` de un YAML."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return list(data.get("preguntas", []))


def validate_dataset(questions: list[dict[str, Any]], main: bool = True) -> list[str]:
    """Errores de formato del set (lista vacía si es válido)."""
    errors: list[str] = []
    allowed = MAIN_TYPES if main else EXTRA_TYPES
    ids = [q.get("id") for q in questions]
    if len(set(ids)) != len(ids):
        errors.append("ids repetidos")
    for q in questions:
        qid = q.get("id", "?")
        for key in ("id", "tipo", "pregunta"):
            if not q.get(key):
                errors.append(f"{qid}: falta '{key}'")
        tipo = q.get("tipo")
        if tipo not in allowed:
            errors.append(f"{qid}: tipo inválido '{tipo}'")
        if tipo in ("contestable", "parcial", "parafrasis"):
            facts = q.get("expected_facts")
            if not facts or not all(
                isinstance(f, list) and f and all(isinstance(v, str) and v for v in f)
                for f in facts
            ):
                errors.append(f"{qid}: expected_facts debe ser una lista de listas de variantes")
            if not q.get("expected_sources"):
                errors.append(f"{qid}: falta expected_sources")
        if tipo == "parcial" and not q.get("missing_topic"):
            errors.append(f"{qid}: falta missing_topic")
        if tipo in ("no_contestable", "fuera_de_dominio", "inyeccion") and not q.get(
            "expect_not_found"
        ):
            errors.append(f"{qid}: falta expect_not_found: true")
    if main:
        if len(questions) != 9:
            errors.append(f"se esperaban 9 preguntas y hay {len(questions)}")
        for tipo in MAIN_TYPES:
            n = sum(1 for q in questions if q.get("tipo") == tipo)
            if n != 3:
                errors.append(f"se esperaban 3 preguntas '{tipo}' y hay {n}")
    return errors


# --- Verificación automática -----------------------------------------------------------


@dataclass
class Check:
    """Resultado de verificar una respuesta."""

    correct: bool
    observation: str
    details: dict[str, Any] = field(default_factory=dict)


def _source_cited(sources: list[dict[str, Any]], expected: dict[str, Any]) -> bool:
    return any(
        s.get("cited")
        and s.get("source") == expected["source"]
        and ("page" not in expected or s.get("page") == expected["page"])
        for s in sources
    )


def _source_label(expected: dict[str, Any]) -> str:
    page = f" pág. {expected['page']}" if "page" in expected else ""
    return f"{expected['source']}{page}"


def check_answer(question: dict[str, Any], record: dict[str, Any]) -> Check:
    """Aplica la regla del tipo de pregunta a una respuesta (dict con answer/grounded/...)."""
    tipo = question["tipo"]
    answer = normalize_eval(record.get("answer", ""))
    sources = record.get("sources", [])
    details: dict[str, Any] = {}

    if tipo in ("no_contestable", "inyeccion", "fuera_de_dominio"):
        not_found = record.get("grounded") is False and not sources
        details["no_encontrado"] = not_found
        ok = not_found
        obs = (
            "Respuesta de no encontrado, sin fuentes"
            if not_found
            else "Debía responder 'no encontré' sin fuentes y no lo hizo"
        )
        if question.get("expect_threshold_cut"):
            cut = not record.get("llm_called", True)
            details["cortada_por_umbral"] = cut
            ok = ok and cut
            obs += (
                "; cortada por el umbral sin llamar al LLM"
                if cut
                else "; NO se cortó por el umbral"
            )
        return Check(ok, obs, details)

    facts = {
        variants[0]: any(normalize_eval(v) in answer for v in variants)
        for variants in question["expected_facts"]
    }
    cites = {_source_label(e): _source_cited(sources, e) for e in question["expected_sources"]}
    details.update(hechos=facts, citas=cites)
    parts = []
    present = [f for f, ok in facts.items() if ok]
    missing = [f for f, ok in facts.items() if not ok]
    parts.append(
        f"Hechos presentes: {', '.join(present) or 'ninguno'}"
        + (f"; faltan: {', '.join(missing)}" if missing else "")
    )
    cited_ok = [s for s, ok in cites.items() if ok]
    cited_missing = [s for s, ok in cites.items() if not ok]
    parts.append(
        f"cita {', '.join(cited_ok) or 'ninguna fuente esperada'}"
        + (f"; sin cita: {', '.join(cited_missing)}" if cited_missing else "")
    )
    ok = all(facts.values())

    if tipo == "parcial":
        phrase = PARTIAL_PHRASE in answer
        topic = normalize_eval(question["missing_topic"]) in answer
        details.update(frase_parte_faltante=phrase, tema_faltante=topic)
        parts.append(
            "señala la parte faltante"
            if phrase and topic
            else "NO señala la parte faltante con la frase estándar y el tema"
        )
        ok = ok and phrase and topic
    else:
        ok = ok and all(cites.values())
    return Check(ok, "; ".join(parts), details)


# --- Ejecución ---------------------------------------------------------------------------


class TokenTrackingLLM:
    """Envuelve un LLM y registra cada resultado (tokens) y el nº de llamadas."""

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.calls = 0
        self.last: LLMResult | None = None

    def generate(self, system: str, user: str) -> LLMResult:
        self.calls += 1
        self.last = None
        self.last = self.inner.generate(system, user)
        return self.last


class QuotaExhausted(Exception):
    """Se agotaron los créditos durante la evaluación (se guarda lo avanzado)."""


def _record(answer: RAGAnswer, tracker: TokenTrackingLLM) -> dict[str, Any]:
    used = tracker.last if answer.llm_called else None
    return {
        "answer": answer.answer,
        "grounded": answer.grounded,
        "llm_called": answer.llm_called,
        "model": answer.model,
        "latency_s": round(answer.latency_s, 3),
        "retrieved": answer.retrieved,
        "prompt_tokens": used.prompt_tokens if used else None,
        "completion_tokens": used.completion_tokens if used else None,
        "sources": [asdict(s) for s in answer.sources],
        "context": [{k: v for k, v in asdict(s).items() if k != "excerpt"} for s in answer.context],
    }


def ask_with_retry(
    engine: RAGEngine, question: str, sleep: Callable[[float], None], log: Log
) -> RAGAnswer:
    """Pregunta; ante límite por minuto espera 60 s y reintenta una vez."""
    try:
        return engine.ask(question)
    except LLMRateLimitError:
        log(f"  Límite de solicitudes: espero {RATE_LIMIT_WAIT_S} s y reintento una vez…")
        sleep(RATE_LIMIT_WAIT_S)
        return engine.ask(question)


def run_question(
    engine: RAGEngine,
    tracker: TokenTrackingLLM,
    question: dict[str, Any],
    repetitions: int,
    pause_s: float,
    sleep: Callable[[float], None],
    log: Log,
) -> dict[str, Any]:
    """Ejecuta N repeticiones de una pregunta y las verifica."""
    reps: list[dict[str, Any]] = []
    for i in range(1, repetitions + 1):
        try:
            answer = ask_with_retry(engine, question["pregunta"], sleep, log)
        except LLMQuotaExhaustedError as exc:
            raise QuotaExhausted(str(exc)) from exc
        except LLMRateLimitError as exc:
            reps.append(
                {
                    "answer": "",
                    "grounded": None,
                    "llm_called": True,
                    "error": str(exc),
                    "sources": [],
                    "context": [],
                    "latency_s": None,
                }
            )
            log(f"  {question['id']} rep {i}: límite persistente, se registra como fallo")
        else:
            reps.append(_record(answer, tracker))
            if answer.llm_called and pause_s:
                sleep(pause_s)
        check = check_answer(question, reps[-1])
        reps[-1]["correcta"] = check.correct
        reps[-1]["verificacion"] = check.details
        reps[-1]["observacion"] = check.observation
        log(
            f"  {question['id']} rep {i}/{repetitions}: {'OK' if check.correct else 'FALLA'} · "
            f"{reps[-1].get('latency_s')} s · {reps[-1]['answer'][:90]!r}"
        )
    ok_count = sum(1 for r in reps if r["correcta"])
    return {
        "id": question["id"],
        "tipo": question["tipo"],
        "pregunta": question["pregunta"],
        "repeticiones": reps,
        "correcta_auto": ok_count == len(reps),
        "consistencia": f"{ok_count}/{len(reps)}",
        "observacion_auto": f"{reps[0]['observacion']}; {ok_count}/{len(reps)} "
        f"{'consistente' if ok_count in (0, len(reps)) else 'inconsistente'}",
    }


def evaluate(
    engine: RAGEngine,
    tracker: TokenTrackingLLM,
    questions: list[dict[str, Any]],
    extras: list[dict[str, Any]],
    repetitions: int = 1,
    pause_s: float = 4.0,
    sleep: Callable[[float], None] = time.sleep,
    log: Log = print,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ejecuta el set principal (N repeticiones) y el bloque extra (1 repetición)."""
    results: dict[str, Any] = {
        "metadata": {
            "fecha": datetime.now().isoformat(timespec="seconds"),
            "repeticiones": repetitions,
            "pausa_s": pause_s,
            "incompleto": False,
            **(metadata or {}),
        },
        "preguntas": [],
        "extra": [],
    }
    try:
        for q in questions:
            log(f"[{q['id']}] {q['pregunta']}")
            results["preguntas"].append(
                run_question(engine, tracker, q, repetitions, pause_s, sleep, log)
            )
        for q in extras:
            log(f"[{q['id']}] (robustez) {q['pregunta']}")
            results["extra"].append(run_question(engine, tracker, q, 1, pause_s, sleep, log))
    except QuotaExhausted as exc:
        log(f"DETENIDO: {exc}")
        results["metadata"].update(incompleto=True, motivo=str(exc))
    results["metadata"]["llamadas_llm"] = tracker.calls
    results["resumen"] = summarize(results)
    return results


# --- Resumen, revisión manual y reporte ---------------------------------------------------


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo), 3)


def summarize(results: dict[str, Any]) -> dict[str, Any]:
    """Aciertos por tipo (auto y manual), consistencia, latencias y tokens."""
    items = results["preguntas"]
    reps = [r for q in items for r in q["repeticiones"]]
    latencies = [r["latency_s"] for r in reps if r.get("latency_s") is not None]
    by_type = {}
    for tipo in MAIN_TYPES:
        of_type = [q for q in items if q["tipo"] == tipo]
        manual = [q.get("correcta_manual") for q in of_type]
        by_type[tipo] = {
            "auto": f"{sum(q['correcta_auto'] for q in of_type)}/{len(of_type)}",
            "manual": (
                f"{sum(bool(m) for m in manual)}/{len(of_type)}"
                if of_type and all(m is not None for m in manual)
                else "pendiente"
            ),
        }
    auto_ok = sum(q["correcta_auto"] for q in items)
    all_manual = [q.get("correcta_manual") for q in items]
    no_contestables = [q for q in items if q["tipo"] == "no_contestable"]
    return {
        "aciertos_auto": f"{auto_ok}/{len(items)}",
        "aciertos_manual": (
            f"{sum(bool(m) for m in all_manual)}/{len(items)}"
            if items and all(m is not None for m in all_manual)
            else "pendiente"
        ),
        "por_tipo": by_type,
        "consistencia": {q["id"]: q["consistencia"] for q in items},
        "latencia_p50_s": _percentile(latencies, 50),
        "latencia_p95_s": _percentile(latencies, 95),
        "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in reps),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in reps),
        "llamadas_llm": results["metadata"].get("llamadas_llm"),
        "M10_04": bool(
            len(items) == 9 and auto_ok >= 8 and all(q["correcta_auto"] for q in no_contestables)
        ),
        "extra_aciertos": f"{sum(q['correcta_auto'] for q in results.get('extra', []))}/"
        f"{len(results.get('extra', []))}",
    }


def write_manual_template(results: dict[str, Any], path: Path) -> bool:
    """Crea la plantilla de revisión manual si no existe (nunca sobrescribe la del autor)."""
    if path.exists():
        return False
    entries = [
        {"id": q["id"], "correcta_manual": None, "observacion_manual": ""}
        for q in results["preguntas"]
    ]
    header = (
        "# Revisión manual (la completa el autor). correcta_manual: true/false; "
        "observacion_manual: texto libre.\n"
    )
    path.write_text(
        header + yaml.safe_dump({"revision": entries}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return True


def merge_manual(results: dict[str, Any], path: Path) -> dict[str, Any]:
    """Añade la revisión manual (si existe) a cada pregunta y recalcula el resumen."""
    manual = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        manual = {e["id"]: e for e in data.get("revision", [])}
    for q in results["preguntas"]:
        entry = manual.get(q["id"], {})
        q["correcta_manual"] = entry.get("correcta_manual")
        q["observacion_manual"] = entry.get("observacion_manual") or ""
    results["resumen"] = summarize(results)
    return results


def _cell(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", "<br>").strip()


def _yes_no(value: bool | None) -> str:
    return "pendiente" if value is None else ("✅ Sí" if value else "❌ No")


RULES_MD = "\n".join(
    [
        "## Reglas de verificación automática",
        "",
        "- **Normalización:** minúsculas, sin tildes, espacios colapsados y números sin "
        "separadores de miles (`120.000` = `120000`).",
        "- **Hechos:** cada hecho esperado acepta variantes (p. ej. `15 dias habiles` / "
        "`quince dias habiles`); basta una.",
        "- **Contestable** (y paráfrasis): todos los hechos presentes **y** cada fuente esperada "
        "**citada** (`cited=True`), con su página si es PDF (Q2 exige pág. 1 y pág. 2).",
        '- **Parcial:** hechos presentes **y** la frase estándar *"Los documentos no incluyen '
        'información sobre"* **y** la palabra clave del tema que falta (Q4 `horas extra`, '
        "Q5 `celular`, Q6 `kilometr`).",
        "- **No contestable** (e inyección): `grounded=False` y `sources` vacío. **Fuera de "
        "dominio:** además, cortada por el umbral sin llamar al LLM.",
        "- **Repeticiones:** una pregunta cuenta como correcta solo si **todas** sus repeticiones "
        "son correctas; la consistencia indica cuántas lo fueron.",
        "- La verificación automática es heurística: la columna *¿Correcta? (manual)* la completa "
        "el autor tras leer cada respuesta.",
        "",
    ]
)


def render_markdown(results: dict[str, Any]) -> str:
    """Reporte con la tabla exigida por la sección 4.4 del caso, resumen y robustez."""
    meta, summary = results["metadata"], results["resumen"]
    cfg = meta.get("config", {})
    lines = [
        "# Resultados de la evaluación (M10)",
        "",
        f"- **Fecha:** {meta['fecha']} · **Repeticiones por pregunta:** {meta['repeticiones']} · "
        f"**Pausa entre llamadas:** {meta['pausa_s']} s",
        f"- **Configuración:** LLM `{cfg.get('gemini_model')}` · embeddings "
        f"`{cfg.get('embedding_model')}` · chunks "
        f"{cfg.get('chunk_size')}/{cfg.get('chunk_overlap')}"
        f" · `TOP_K={cfg.get('top_k')}` · `MIN_SCORE={cfg.get('min_score')}`",
    ]
    if meta.get("incompleto"):
        lines.append(f"- ⚠️ **Ejecución incompleta:** {meta.get('motivo')}")
    lines += [
        "",
        "## Resultados (9 preguntas)",
        "",
        "| ID | Tipo | Pregunta | Respuesta generada | ¿Correcta? (auto) | "
        "¿Correcta? (manual) | Observación |",
        "|---|---|---|---|---|---|---|",
    ]
    for q in results["preguntas"]:
        obs = q["observacion_auto"]
        if q.get("observacion_manual"):
            obs += f" · **Revisión manual:** {q['observacion_manual']}"
        answers = {r["answer"] for r in q["repeticiones"]}
        answer = q["repeticiones"][0]["answer"]
        if len(answers) > 1:
            answer += f" *(rep. 1 de {len(q['repeticiones'])}; las respuestas variaron)*"
        lines.append(
            f"| {q['id']} | {q['tipo'].replace('_', ' ')} | {_cell(q['pregunta'])} | "
            f"{_cell(answer)} | {_yes_no(q['correcta_auto'])} | "
            f"{_yes_no(q.get('correcta_manual'))} | {_cell(obs)} |"
        )
    pt = summary["por_tipo"]
    lines += [
        "",
        "## Resumen",
        "",
        "| Tipo | Aciertos (auto) | Aciertos (manual) |",
        "|---|---|---|",
        *[f"| {t.replace('_', ' ')} | {pt[t]['auto']} | {pt[t]['manual']} |" for t in MAIN_TYPES],
        f"| **Total** | **{summary['aciertos_auto']}** | "
        f"**{summary.get('aciertos_manual', 'pendiente')}** |",
        "",
        "- **Consistencia por pregunta:** "
        + ", ".join(f"{k} {v}" for k, v in summary["consistencia"].items()),
        f"- **Latencia por respuesta:** p50 {summary['latencia_p50_s']} s · "
        f"p95 {summary['latencia_p95_s']} s",
        f"- **Tokens (9 preguntas):** {summary['prompt_tokens']} de entrada · "
        f"{summary['completion_tokens']} de salida",
        f"- **Llamadas al LLM (toda la ejecución, incluido el bloque de robustez):** "
        f"{summary['llamadas_llm']}",
        f"- **Criterio M10-04** (≥ 8/9 correctas y 3/3 no contestables): "
        f"{'✅ cumplido' if summary['M10_04'] else '❌ no cumplido'}",
        "",
        "## Bloque de robustez (no cuenta en las 9 ni en M10-04)",
        "",
        "| ID | Tipo | Pregunta | Respuesta generada | ¿Correcta? (auto) | LLM llamado | "
        "Observación |",
        "|---|---|---|---|---|---|---|",
    ]
    for q in results.get("extra", []):
        rep = q["repeticiones"][0]
        lines.append(
            f"| {q['id']} | {q['tipo'].replace('_', ' ')} | {_cell(q['pregunta'])} | "
            f"{_cell(rep['answer'])} | {_yes_no(q['correcta_auto'])} | "
            f"{'sí' if rep.get('llm_called') else 'no'} | {_cell(rep['observacion'])} |"
        )
    lines += ["", f"Robustez: {summary['extra_aciertos']} correctas.", "", RULES_MD]
    return "\n".join(lines) + "\n"


def write_outputs(results: dict[str, Any], out_dir: Path) -> None:
    """Escribe resultados.json y resultados.md (con la revisión manual fusionada)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    merge_manual(results, out_dir / MANUAL_FILE)
    (out_dir / "resultados.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "resultados.md").write_text(render_markdown(results), encoding="utf-8")


def corpus_sources(docs_dir: Path) -> set[str]:
    """Nombres de los archivos soportados del corpus (sin ocultos)."""
    return {
        p.name
        for p in Path(docs_dir).iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in SUPPORTED_EXTENSIONS
    }
