"""Pruebas de M1: loaders y normalización de texto."""

import logging
from pathlib import Path

import pytest
from fpdf import FPDF

from rag.loaders import (
    PROJECT_ROOT,
    UnsupportedFileTypeError,
    load_directory,
    load_file,
    normalize_text,
    relative_path,
)
from rag.models import Document


def _make_pdf(path: Path, pages: list[str]) -> Path:
    """Crea un PDF de prueba; una página con texto vacío queda sin texto extraíble."""
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    for text in pages:
        pdf.add_page()
        if text:
            pdf.multi_cell(0, 8, text)
    pdf.output(str(path))
    return path


def test_load_text_and_markdown(tmp_path: Path) -> None:
    """M1-03: .md y .txt devuelven 1 Document con source y doc_type correctos."""
    md = tmp_path / "nota.md"
    md.write_text("# Título\n\nContenido en markdown.", encoding="utf-8")
    txt = tmp_path / "nota.txt"
    txt.write_text("Contenido en texto plano.", encoding="utf-8")

    for path, doc_type in [(md, "md"), (txt, "txt")]:
        docs = load_file(path, base_dir=tmp_path)
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].metadata["source"] == path.name
        assert docs[0].metadata["doc_type"] == doc_type
        assert "page" not in docs[0].metadata


@pytest.mark.parametrize("name", ["NOTA.MD", "nota.markdown", "Nota.TXT"])
def test_extensions_case_insensitive(tmp_path: Path, name: str) -> None:
    """Las extensiones se reconocen sin importar mayúsculas (incluye .markdown)."""
    path = tmp_path / name
    path.write_text("Texto de prueba suficiente.", encoding="utf-8")
    assert len(load_file(path)) == 1


def test_load_pdf_pages(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """M1-04: PDF → 1 Document por página con texto, `page` 1-based; vacías se omiten."""
    pdf = _make_pdf(tmp_path / "manual.pdf", ["Primera pagina", "", "Tercera pagina"])
    with caplog.at_level(logging.WARNING):
        docs = load_file(pdf, base_dir=tmp_path)

    assert [d.metadata["page"] for d in docs] == [1, 3]
    assert "Primera pagina" in docs[0].text
    assert all(d.metadata["doc_type"] == "pdf" for d in docs)
    assert all(d.metadata["source"] == "manual.pdf" for d in docs)
    assert "Página 2 sin texto" in caplog.text


def test_errors(tmp_path: Path) -> None:
    """M1-05: extensión no soportada → UnsupportedFileTypeError; inexistente → FileNotFoundError."""
    docx = tmp_path / "informe.docx"
    docx.write_bytes(b"PK")
    with pytest.raises(UnsupportedFileTypeError, match="informe.docx"):
        load_file(docx)
    with pytest.raises(FileNotFoundError):
        load_file(tmp_path / "no_existe.txt")
    with pytest.raises(FileNotFoundError):
        load_directory(tmp_path / "no_existe")


@pytest.mark.parametrize("content", ["", "   \n\n\t  "])
def test_empty_file(tmp_path: Path, caplog: pytest.LogCaptureFixture, content: str) -> None:
    """M1-06: archivo vacío (o solo espacios) → lista vacía + warning, sin excepción."""
    path = tmp_path / "vacio.txt"
    path.write_text(content, encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert load_file(path) == []
    assert "vacío" in caplog.text


def test_load_directory(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """M1-07: ignora .docx/ocultos y devuelve en orden determinista por nombre."""
    (tmp_path / "b_segundo.txt").write_text("Documento B", encoding="utf-8")
    (tmp_path / "a_primero.md").write_text("Documento A", encoding="utf-8")
    _make_pdf(tmp_path / "c_tercero.pdf", ["Documento C"])
    (tmp_path / "informe.docx").write_bytes(b"PK")
    (tmp_path / ".oculto.txt").write_text("No debe cargarse", encoding="utf-8")
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d_anidado.txt").write_text("Documento D", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        docs = load_directory(tmp_path, base_dir=tmp_path)
    assert [d.metadata["source"] for d in docs] == [
        "a_primero.md",
        "b_segundo.txt",
        "c_tercero.pdf",
    ]
    assert "informe.docx" in caplog.text
    assert ".oculto" not in caplog.text

    recursive = load_directory(tmp_path, recursive=True, base_dir=tmp_path)
    assert [d.metadata["source"] for d in recursive] == [
        "a_primero.md",
        "b_segundo.txt",
        "c_tercero.pdf",
        "d_anidado.txt",
    ]
    assert load_directory(tmp_path) == load_directory(tmp_path)


def test_path_metadata(tmp_path: Path) -> None:
    """`path` es relativa a la raíz del proyecto; fuera de ella, solo el nombre del archivo."""
    corpus_doc = PROJECT_ROOT / "data" / "docs" / "guia_onboarding_ti.txt"
    assert load_file(corpus_doc)[0].metadata["path"] == "data/docs/guia_onboarding_ti.txt"

    sub = tmp_path / "sub"
    sub.mkdir()
    inner = sub / "x.md"
    inner.write_text("Texto interno", encoding="utf-8")
    assert load_file(inner, base_dir=tmp_path)[0].metadata["path"] == "sub/x.md"

    # tmp_path está fuera del proyecto → solo el nombre
    assert load_file(inner)[0].metadata["path"] == "x.md"
    assert relative_path(inner, base_dir=PROJECT_ROOT) == "x.md"
    assert not Path(load_file(inner)[0].metadata["path"]).is_absolute()


def test_normalize_text() -> None:
    """M1-08: colapsa espacios, elimina controles y conserva párrafos."""
    raw = (
        "  Título   con    espacios  \r\n\r\n\r\n\r\n"
        "Párrafo\tdos\x00\x07 sigue.\nLínea  tres.  \n\n\n"
    )
    assert normalize_text(raw) == "Título con espacios\n\nPárrafo dos sigue.\nLínea tres."
    assert normalize_text("Uno.\n\nDos.") == "Uno.\n\nDos."
    assert normalize_text("   \n\t ") == ""


def test_normalize_text_nfc() -> None:
    """`normalize_text` aplica Unicode NFC (tildes descompuestas → compuestas)."""
    decomposed = "Contrasen\u0303a y camio\u0301n"
    result = normalize_text(decomposed)
    assert result == "Contraseña y camión"
    assert "\u0303" not in result and "\u0301" not in result


def test_normalize_text_nbsp() -> None:
    """`normalize_text` convierte el espacio no separable (\\xa0) en espacio normal."""
    assert normalize_text("COP\xa0120.000") == "COP 120.000"
    assert normalize_text("a\xa0\xa0 b") == "a b"


def test_latin1_fallback(tmp_path: Path) -> None:
    """M1-09: un archivo en latin-1 se carga sin error y con los acentos correctos."""
    path = tmp_path / "antiguo.txt"
    path.write_bytes("Contraseña de más de 12 caracteres.".encode("latin-1"))
    docs = load_file(path)
    assert docs[0].text == "Contraseña de más de 12 caracteres."
