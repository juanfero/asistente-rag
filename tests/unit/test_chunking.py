"""Pruebas de M2: chunking recursivo con solapamiento e ids estables."""

import re

import pytest

from rag.chunking import (
    MIN_CHUNK_CHARS,
    _overlap_tail,
    chunk_documents,
    is_heading,
    make_chunk_id,
    split_text,
)
from rag.loaders import PROJECT_ROOT, load_directory
from rag.models import Chunk, Document

# Configuraciones a validar: la actual (800/120) y la probable tras M3 (500/80).
CORPUS_CONFIGS = [(800, 120), (500, 80)]

_WORDS = (
    "el colaborador debe registrar la solicitud en el portal con anticipación y el jefe "
    "inmediato revisa los soportes antes de aprobar el reembolso de los gastos del viaje"
).split()


def _synthetic_text(paragraphs: int = 12, sentences: int = 5) -> str:
    """Texto determinista con párrafos y oraciones de longitud variable."""
    out = []
    n = 0
    for _ in range(paragraphs):
        sents = []
        for _ in range(sentences):
            length = 6 + n % 9
            words = [_WORDS[(n + i) % len(_WORDS)] for i in range(length)]
            sents.append(" ".join(words).capitalize() + f" número {n}.")
            n += 1
        out.append(" ".join(sents))
    return "\n\n".join(out)


def _overlap_len(prev: str, nxt: str) -> int:
    """Longitud del mayor prefijo de `nxt` que es sufijo de `prev` (con al menos una letra)."""
    for k in range(min(len(prev), len(nxt)), 0, -1):
        if prev.endswith(nxt[:k]) and re.search(r"\w", nxt[:k]):
            return k
    return 0


@pytest.fixture(scope="module")
def corpus_docs() -> list[Document]:
    """Documents del corpus real (data/docs)."""
    return load_directory(PROJECT_ROOT / "data" / "docs")


def _consecutive_pairs(chunks: list[Chunk]):
    """Pares de chunks consecutivos del mismo Document (misma fuente y página)."""
    for a, b in zip(chunks, chunks[1:], strict=False):
        same_doc = (a.metadata["source"], a.metadata.get("page")) == (
            b.metadata["source"],
            b.metadata.get("page"),
        )
        if same_doc:
            yield a, b


# --- M2-01 ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "overlap"), [(40, 0), (60, 10), (100, 20), (250, 50), (500, 80), (800, 120)]
)
def test_max_size(size: int, overlap: int) -> None:
    """M2-01: ningún chunk supera chunk_size (tolerancia 0) en texto sintético."""
    chunks = split_text(_synthetic_text(), size, overlap)
    assert chunks
    assert max(len(c) for c in chunks) <= size


@pytest.mark.parametrize(("size", "overlap"), CORPUS_CONFIGS)
def test_max_size_corpus(corpus_docs: list[Document], size: int, overlap: int) -> None:
    """M2-01: con el corpus real tampoco se supera chunk_size."""
    chunks = chunk_documents(corpus_docs, size, overlap)
    assert all(c.metadata["char_count"] == len(c.text) <= size for c in chunks)


# --- M2-02 ---------------------------------------------------------------------------


@pytest.mark.parametrize(("size", "overlap"), CORPUS_CONFIGS + [(200, 40)])
def test_overlap(size: int, overlap: int) -> None:
    """M2-02: chunks consecutivos comparten solapamiento > 0 y ≤ chunk_overlap."""
    chunks = split_text(_synthetic_text(paragraphs=30), size, overlap)
    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert 0 < _overlap_len(prev, nxt) <= overlap


@pytest.mark.parametrize(("size", "overlap"), CORPUS_CONFIGS)
def test_overlap_corpus(corpus_docs: list[Document], size: int, overlap: int) -> None:
    """M2-02: en el corpus real, chunks consecutivos del mismo documento se solapan."""
    pairs = list(_consecutive_pairs(chunk_documents(corpus_docs, size, overlap)))
    assert pairs
    for a, b in pairs:
        assert 0 < _overlap_len(a.text, b.text) <= overlap, (a.chunk_id, b.chunk_id)


def test_no_overlap_when_zero() -> None:
    """Con overlap = 0 los chunks no repiten texto: concatenados reconstruyen las palabras."""
    text = _synthetic_text()
    chunks = split_text(text, 200, 0)
    assert " ".join(chunks).split() == text.split()


# --- M2-03 ---------------------------------------------------------------------------


@pytest.mark.parametrize(("size", "overlap"), [(40, 0), (100, 20), (500, 80), (800, 120)])
def test_no_content_loss(size: int, overlap: int) -> None:
    """M2-03: toda palabra del texto original aparece en algún chunk, en orden."""
    text = _synthetic_text()
    chunks = split_text(text, size, overlap)
    assert set(text.split()) <= set(" ".join(chunks).split())
    # Cada chunk es una subcadena contigua del original
    assert all(c in text for c in chunks)


@pytest.mark.parametrize(("size", "overlap"), CORPUS_CONFIGS)
def test_no_content_loss_corpus(corpus_docs: list[Document], size: int, overlap: int) -> None:
    """M2-03: en el corpus real no se pierde ninguna palabra (ni por descartes < 30)."""
    chunks = chunk_documents(corpus_docs, size, overlap)
    for doc in corpus_docs:
        own = [c.text for c in chunks if c.metadata == {**c.metadata, **doc.metadata}]
        assert set(doc.text.split()) <= set(" ".join(own).split()), doc.metadata


# --- M2-04 ---------------------------------------------------------------------------


def test_respects_paragraphs() -> None:
    """M2-04: 3 párrafos cortos → 1 chunk con size grande; cortes en \\n\\n con size pequeño."""
    paragraphs = [
        "Primer párrafo con una idea completa sobre vacaciones.",
        "Segundo párrafo que explica cómo pedir un permiso.",
        "Tercer párrafo con las reglas del día de cumpleaños.",
    ]
    text = "\n\n".join(paragraphs)
    assert split_text(text, 1000, 100) == [text]
    assert split_text(text, 60, 0) == paragraphs


def test_prefers_line_then_sentence() -> None:
    """Sin párrafos, se corta por línea; sin líneas, por oración ('. ')."""
    lines = ["Línea uno con texto suficiente.", "Línea dos con texto suficiente."]
    assert split_text("\n".join(lines), 40, 0) == lines
    sentence_text = "Primera oración con texto. Segunda oración con texto."
    assert split_text(sentence_text, 30, 0) == [
        "Primera oración con texto.",
        "Segunda oración con texto.",
    ]


# --- M2-05 ---------------------------------------------------------------------------


def test_long_token() -> None:
    """M2-05: una palabra más larga que chunk_size se corta por caracteres sin bucle infinito."""
    token = "x" * 950
    text = f"Inicio del texto {token} final del texto."
    chunks = split_text(text, 100, 20)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(c for c in chunks if set(c) == {"x"}).count("x") >= 900
    assert chunks[0].startswith("Inicio") and chunks[-1].endswith("final del texto.")


# --- M2-06 ---------------------------------------------------------------------------


def test_metadata() -> None:
    """M2-06: metadatos heredados + chunk_index secuencial por documento/página."""
    body = _synthetic_text(paragraphs=6)
    docs = [
        Document(body, {"source": "a.md", "doc_type": "md", "path": "data/docs/a.md"}),
        Document(body, {"source": "b.pdf", "doc_type": "pdf", "path": "x/b.pdf", "page": 1}),
        Document(body, {"source": "b.pdf", "doc_type": "pdf", "path": "x/b.pdf", "page": 2}),
    ]
    chunks = chunk_documents(docs, 200, 40)
    for doc in docs:
        own = [
            c
            for c in chunks
            if c.metadata["source"] == doc.metadata["source"]
            and c.metadata.get("page") == doc.metadata.get("page")
        ]
        assert len(own) > 1
        assert [c.metadata["chunk_index"] for c in own] == list(range(len(own)))
        for c in own:
            assert {k: c.metadata[k] for k in doc.metadata} == doc.metadata
            assert c.metadata["char_count"] == len(c.text)
    assert all("page" not in c.metadata for c in chunks if c.metadata["source"] == "a.md")


# --- M2-07 ---------------------------------------------------------------------------


def test_stable_ids(corpus_docs: list[Document]) -> None:
    """M2-07: chunk_id determinista y único en el corpus."""
    first = chunk_documents(corpus_docs, 800, 120)
    second = chunk_documents(corpus_docs, 800, 120)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)
    assert all(re.fullmatch(r"[0-9a-f]{16}", c.chunk_id) for c in first)

    base = make_chunk_id("a.md", "", 0, "texto")
    assert base == make_chunk_id("a.md", "", 0, "texto")
    assert base != make_chunk_id("a.md", "", 0, "texto distinto")
    assert base != make_chunk_id("a.md", 1, 0, "texto")
    assert base != make_chunk_id("a.md", "", 1, "texto")


# --- M2-08 ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t \n"])
def test_empty_and_tiny(text: str) -> None:
    """M2-08: texto vacío o solo espacios → 0 chunks."""
    assert split_text(text, 100, 10) == []
    assert chunk_documents([Document(text, {"source": "v.txt"})], 100, 10) == []


def test_tiny_chunks_discarded() -> None:
    """M2-08: fragmentos con < 30 caracteres no vacíos se descartan."""
    long_paragraph = "Este párrafo tiene suficiente contenido para ser un chunk válido."
    text = f"{long_paragraph}\n\nFin."
    assert split_text(text, 70, 0) == [long_paragraph, "Fin."]
    chunks = chunk_documents([Document(text, {"source": "t.txt"})], 70, 0)
    assert [c.text for c in chunks] == [long_paragraph]
    assert chunk_documents([Document("Muy corto.", {"source": "c.txt"})], 70, 0) == []
    assert MIN_CHUNK_CHARS == 30


def test_invalid_parameters() -> None:
    """chunk_size ≤ 0 u overlap fuera de [0, chunk_size) lanzan ValueError."""
    for size, overlap in [(0, 0), (100, 100), (100, -1)]:
        with pytest.raises(ValueError):
            split_text("texto", size, overlap)


# --- M2-09 y requisitos adicionales sobre el corpus real ---------------------------


@pytest.mark.parametrize(("size", "overlap"), CORPUS_CONFIGS)
def test_corpus_chunking(corpus_docs: list[Document], size: int, overlap: int) -> None:
    """M2-09: ≥ 1 chunk por documento y por página de PDF."""
    chunks = chunk_documents(corpus_docs, size, overlap)
    produced = {(c.metadata["source"], c.metadata.get("page")) for c in chunks}
    expected = {(d.metadata["source"], d.metadata.get("page")) for d in corpus_docs}
    assert produced == expected
    assert {s for s, _ in produced} == {
        "politica_vacaciones_y_permisos.md",
        "manual_reembolso_gastos.pdf",
        "guia_onboarding_ti.txt",
    }
    assert {p for s, p in produced if s == "manual_reembolso_gastos.pdf"} == {1, 2}


@pytest.mark.parametrize(("size", "overlap"), CORPUS_CONFIGS)
def test_no_cross_page_chunks(corpus_docs: list[Document], size: int, overlap: int) -> None:
    """Cada chunk es subcadena de su propio Document: nunca mezcla páginas del PDF."""
    by_key = {(d.metadata["source"], d.metadata.get("page")): d.text for d in corpus_docs}
    for c in chunk_documents(corpus_docs, size, overlap):
        assert c.text in by_key[(c.metadata["source"], c.metadata.get("page"))], c.chunk_id


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("## 5. Día de cumpleaños", True),
        ("# Política de Vacaciones y Permisos", True),
        ("6. SOPORTE TÉCNICO", True),
        ("4. Viajes internacionales", True),
        ("GUÍA DE ONBOARDING DE TECNOLOGÍA (TI)", True),
        ("1. La solicitud se registra en el portal **NexaPeople**.", False),
        ("- Alimentación: COP 120.000 por día.", False),
        ("Documento ficticio para prueba técnica.", False),
        ("| Permiso | Duración | Soporte |", False),
        ("", False),
    ],
)
def test_is_heading(line: str, expected: bool) -> None:
    """Detección de encabezados usada para no separarlos de su contenido."""
    assert is_heading(line) is expected


@pytest.mark.parametrize(("size", "overlap"), CORPUS_CONFIGS)
def test_no_heading_only_chunks(corpus_docs: list[Document], size: int, overlap: int) -> None:
    """Ningún chunk es solo encabezado ni termina en un encabezado sin su contenido."""
    chunks = chunk_documents(corpus_docs, size, overlap)
    for c in chunks:
        lines = [line for line in c.text.split("\n") if line.strip()]
        assert not all(is_heading(line) for line in lines), c.text
        assert not is_heading(lines[-1]), c.text

    # Cada encabezado del corpus comparte chunk con la primera línea de contenido que le sigue
    for doc in corpus_docs:
        lines = [line for line in doc.text.split("\n") if line.strip()]
        for i, line in enumerate(lines):
            if not is_heading(line):
                continue
            following = next((x for x in lines[i + 1 :] if not is_heading(x)), None)
            if following is None:
                continue
            assert any(
                line in c.text
                and following in c.text
                and c.text.index(line) < c.text.index(following)
                for c in chunks
            ), (line, following)

    # Ejemplos concretos pedidos
    texts = [c.text for c in chunks]
    assert any(
        "## 5. Día de cumpleaños" in t and "un día libre por su cumpleaños" in t for t in texts
    )
    assert any("6. SOPORTE TÉCNICO" in t and "soporte@nexalogistica.co" in t for t in texts)


@pytest.mark.parametrize(("size", "overlap"), CORPUS_CONFIGS)
def test_markdown_table_in_one_chunk(corpus_docs: list[Document], size: int, overlap: int) -> None:
    """La tabla de permisos (md §3) queda completa, con su encabezado, en un mismo chunk."""
    md = next(d for d in corpus_docs if d.metadata["doc_type"] == "md")
    table = "\n".join(line for line in md.text.split("\n") if line.startswith("|"))
    assert table.startswith("| Permiso | Duración | Soporte |")
    assert table.count("\n") == 4  # encabezado + separador + 3 filas
    chunks = chunk_documents([md], size, overlap)
    assert any(table in c.text for c in chunks)


# --- Ajuste M2: el solapamiento empieza en un límite natural -------------------------


def test_overlap_prefers_newline() -> None:
    """Con un salto de línea en la ventana, el solapamiento empieza en la línea siguiente."""
    text = "Una frase. Otra frase con texto. Y otra más\nLínea nueva con contenido"
    tail = _overlap_tail(text, 45)
    assert tail == "Línea nueva con contenido"


def test_overlap_newline_beats_sentence_end() -> None:
    """El salto de línea tiene prioridad aunque haya un fin de oración antes en la ventana."""
    text = "Inicio largo del texto previo. Frase uno. Frase dos\nTres cuatro cinco"
    tail = _overlap_tail(text, 40)
    assert ". Frase dos" in text[-40:]
    assert tail == "Tres cuatro cinco"


@pytest.mark.parametrize("sep", [". ", "? ", "! ", ": "])
def test_overlap_sentence_end(sep: str) -> None:
    """Sin salto de línea, el solapamiento empieza tras un fin de oración."""
    text = f"Uno dos tres cuatro cinco seis siete ocho{sep}Nueve diez once doce trece"
    tail = _overlap_tail(text, 40)
    assert tail == "Nueve diez once doce trece"


def test_overlap_word_fallback() -> None:
    """Sin salto de línea ni fin de oración, se corta en límite de palabra."""
    text = "alfa beta gama delta epsilon zeta eta theta iota kappa"
    tail = _overlap_tail(text, 20)
    assert len(tail) <= 20 and text.endswith(tail)
    assert text[len(text) - len(tail) - 1] == " "  # empieza en una palabra completa
    assert tail.split()[0] in text.split()


def test_overlap_edge_cases() -> None:
    """Presupuesto 0 → sin solapamiento; texto más corto que la ventana → texto completo."""
    assert _overlap_tail("algo de texto", 0) == ""
    assert _overlap_tail("corto", 50) == "corto"
    assert _overlap_tail("palabraextremadamentelargasinespacios", 10) == ""


@pytest.mark.parametrize(("size", "overlap"), CORPUS_CONFIGS)
def test_overlap_starts_at_natural_boundary_corpus(
    corpus_docs: list[Document], size: int, overlap: int
) -> None:
    """En el corpus, ningún chunk (salvo el primero de cada doc.) empieza a mitad de palabra."""
    chunks = chunk_documents(corpus_docs, size, overlap)
    by_key = {(d.metadata["source"], d.metadata.get("page")): d.text for d in corpus_docs}
    after_newline = 0
    for prev, nxt in _consecutive_pairs(chunks):
        doc_text = by_key[(nxt.metadata["source"], nxt.metadata.get("page"))]
        start = doc_text.index(nxt.text, doc_text.index(prev.text) + 1)
        before = doc_text[:start]
        assert before[-1].isspace() or before.endswith((". ", "? ", "! ", ": ")), nxt.text[:40]
        after_newline += before.endswith("\n")
    assert after_newline > 0
