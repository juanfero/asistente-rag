"""Pruebas de M7: RAGEngine con FakeEmbedder + FakeLLM (sin red)."""

from pathlib import Path

import pytest

from rag.config import Settings
from rag.embeddings import FakeEmbedder
from rag.ingest import build_store, ingest_paths
from rag.llm import FakeLLM, LLMQuotaExhaustedError
from rag.loaders import PROJECT_ROOT
from rag.prompts import NOT_FOUND_MESSAGE, SYSTEM_PROMPT
from rag.rag_engine import (
    EMPTY_INDEX_MESSAGE,
    EmptyIndexError,
    RAGAnswer,
    RAGEngine,
    is_not_found,
    normalize,
    parse_citations,
)

QUESTION = "¿Cuál es el tope diario de alimentación en viajes nacionales?"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # FakeEmbedder produce similitudes bajas: umbral 0 para estas pruebas (salvo M7-02)
    return Settings(_env_file=None, chroma_dir=tmp_path / "chroma", min_score=0.0, top_k=3)


@pytest.fixture
def store(settings: Settings):
    store = build_store(settings, FakeEmbedder(64))
    ingest_paths([PROJECT_ROOT / "data" / "docs"], store, settings)
    return store


def engine(store, settings: Settings, responses) -> tuple[RAGEngine, FakeLLM]:
    fake = FakeLLM(responses)
    return RAGEngine(store, fake, settings), fake


# --- M7-01 … M7-08 -----------------------------------------------------------------


@pytest.mark.parametrize("question", ["", "   ", "x" * 1001], ids=["vacia", "espacios", "larga"])
def test_validate_question(store, settings: Settings, question: str) -> None:
    """M7-01: pregunta vacía o > 1.000 caracteres → ValueError (sin llamar al LLM)."""
    rag, fake = engine(store, settings, ["no debe usarse"])
    with pytest.raises(ValueError):
        rag.ask(question)
    assert fake.calls == []
    assert rag.ask("x" * 1000).answer == "no debe usarse"  # el límite exacto es válido


def test_not_found_skips_llm(store, settings: Settings) -> None:
    """M7-02: sin chunks sobre el umbral → "no encontré", grounded=False y sin llamar al LLM."""
    rag, fake = engine(store, settings.model_copy(update={"min_score": 0.99}), ["no"])
    answer = rag.ask(QUESTION)
    assert answer.answer == NOT_FOUND_MESSAGE
    assert answer.grounded is False and answer.llm_called is False
    assert answer.sources == [] and answer.context == []
    assert answer.retrieved == 3 and answer.prompt is None
    assert fake.calls == []


def test_prompt_contents(store, settings: Settings) -> None:
    """M7-03: el prompt contiene la pregunta, los chunks numerados y su fuente/página."""
    rag, fake = engine(store, settings, ["Respuesta [1]."])
    answer = rag.ask(QUESTION)
    system, user = fake.calls[0]
    assert system == SYSTEM_PROMPT
    assert f"Pregunta: {QUESTION}" in user
    for ref in answer.context:
        page = f", pág. {ref.page}" if ref.page is not None else ""
        assert f"[{ref.index}] (fuente: {ref.source}{page})" in user
    assert answer.prompt == user


def test_citation_mapping(store, settings: Settings) -> None:
    """M7-04: las citas [1] y [3] se mapean a los SourceRef correctos con cited=True."""
    rag, _ = engine(store, settings, ["Dato A [1]. Dato B [3]."])
    answer = rag.ask(QUESTION)
    assert [ref.index for ref in answer.sources] == [1, 3]
    assert all(ref.cited for ref in answer.sources)
    assert [ref.cited for ref in answer.context] == [True, False, True]
    assert answer.sources[0].chunk_id == answer.context[0].chunk_id
    assert answer.grounded is True


def test_no_citations_fallback(store, settings: Settings) -> None:
    """M7-05: respuesta sin citas → sources = contexto recuperado con cited=False."""
    rag, _ = engine(store, settings, ["El tope es COP 120.000 por día."])
    answer = rag.ask(QUESTION)
    assert answer.sources == answer.context and len(answer.sources) == 3
    assert not any(ref.cited for ref in answer.sources)
    assert answer.grounded is True


def test_invalid_citation_index(store, settings: Settings) -> None:
    """M7-06: citas fuera de rango ([9], [0]) se ignoran sin error."""
    rag, _ = engine(store, settings, ["Dato [9] y otro [0] y uno válido [2]."])
    answer = rag.ask(QUESTION)
    assert [ref.index for ref in answer.sources] == [2]


def test_top_k_override(store, settings: Settings) -> None:
    """M7-07: top_k del argumento sobrescribe el de settings (3)."""
    rag, fake = engine(store, settings, ["a [1]", "b [1]"])
    assert len(rag.ask(QUESTION).context) == 3
    answer = rag.ask(QUESTION, top_k=2)
    assert len(answer.context) == 2 and answer.retrieved == 2
    assert "[3] (fuente:" not in fake.calls[1][1]


@pytest.mark.parametrize(
    "llm_text",
    [
        NOT_FOUND_MESSAGE,
        "no encontre informacion sobre eso en los documentos cargados",
        "  No   encontré   información   sobre ese tema en los documentos.  ",
    ],
    ids=["exacta", "sin-tildes", "parafraseada"],
)
def test_llm_not_found_phrase(store, settings: Settings, llm_text: str) -> None:
    """M7-08: el LLM devuelve la frase de no encontrado (normalizada) → grounded=False."""
    rag, _ = engine(store, settings, [llm_text])
    answer = rag.ask(QUESTION)
    assert answer.grounded is False and answer.llm_called is True
    assert answer.sources == []  # sin respuesta no se muestran fuentes
    assert len(answer.context) == 3 and not any(ref.cited for ref in answer.context)


# --- Reglas adicionales -------------------------------------------------------------


def test_not_found_with_citation_is_grounded(store, settings: Settings) -> None:
    """Si el texto menciona "no encontré información" pero cita fragmentos → grounded=True."""
    rag, _ = engine(
        store,
        settings,
        ["El tope es COP 120.000 [1]. No encontré información sobre viajes internacionales."],
    )
    assert rag.ask(QUESTION).grounded is True


def test_partial_answer_is_grounded(store, settings: Settings) -> None:
    """Respuesta parcial con la frase estándar de parte faltante → grounded=True."""
    rag, _ = engine(
        store,
        settings,
        ["Son 5 días hábiles [1]. Los documentos no incluyen información sobre horas extra."],
    )
    answer = rag.ask("¿Días por matrimonio y horas extra?")
    assert answer.grounded is True and answer.sources[0].cited


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[1]", {1}),
        ("a [1][3] b", {1, 3}),
        ("[1, 3]", {1, 3}),
        ("[1,3]", {1, 3}),
        ("[1-3]", {1, 2, 3}),
        ("[2–3]", {2, 3}),
        ("[9] [0] [3-1]", set()),
        ("sin citas", set()),
    ],
    ids=[
        "simple",
        "seguidas",
        "coma-espacio",
        "coma",
        "rango",
        "rango-guion-largo",
        "invalidas",
        "ninguna",
    ],
)
def test_parse_citations(text: str, expected: set[int]) -> None:
    """Formatos de cita reconocidos; índices fuera de 1..n se ignoran."""
    assert parse_citations(text, 4) == expected


def test_normalize_and_not_found_rule() -> None:
    """Normalización (minúsculas, sin tildes, espacios) y regla de no encontrado."""
    assert normalize("  No   ENCONTRÉ\nInformación ") == "no encontre informacion"
    assert is_not_found(NOT_FOUND_MESSAGE, set()) is True
    assert is_not_found(NOT_FOUND_MESSAGE, {1}) is False
    assert is_not_found("Son 15 días [1].", set()) is False


def test_empty_index(tmp_path: Path) -> None:
    """Índice vacío → EmptyIndexError (hija de ValueError) con mensaje claro."""
    settings = Settings(_env_file=None, chroma_dir=tmp_path / "vacio")
    rag = RAGEngine(build_store(settings, FakeEmbedder(64)), FakeLLM(["x"]), settings)
    with pytest.raises(EmptyIndexError) as exc:
        rag.ask(QUESTION)
    assert isinstance(exc.value, ValueError)
    assert str(exc.value) == EMPTY_INDEX_MESSAGE


def test_llm_errors_propagate(store, settings: Settings) -> None:
    """El motor no captura los errores del LLM (los traduce la CLI/API)."""
    rag, _ = engine(store, settings, [LLMQuotaExhaustedError("sin créditos")])
    with pytest.raises(LLMQuotaExhaustedError):
        rag.ask(QUESTION)


def test_answer_fields(store, settings: Settings) -> None:
    """RAGAnswer: pregunta, modelo, latencia, recuperados y extractos ≤ 300 caracteres."""
    rag, _ = engine(store, settings, ["Respuesta [2]."])
    answer = rag.ask(f"  {QUESTION}  ")
    assert isinstance(answer, RAGAnswer)
    assert answer.question == QUESTION
    assert answer.model == "fake" and answer.latency_s >= 0 and answer.retrieved == 3
    assert all(len(ref.excerpt) <= 300 for ref in answer.context)
    assert [ref.index for ref in answer.context] == [1, 2, 3]
