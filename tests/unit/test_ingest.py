"""Pruebas de M5: pipeline de ingesta (FakeEmbedder + Chroma en tmp_path)."""

import shutil
from pathlib import Path

import pytest
from fpdf import FPDF

from rag.config import Settings
from rag.embeddings import FakeEmbedder
from rag.ingest import IngestReport, build_store, ingest_paths
from rag.loaders import PROJECT_ROOT

CORPUS = PROJECT_ROOT / "data" / "docs"
LONG_TEXT = "Este documento de prueba tiene contenido suficiente para generar un chunk válido."


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, chroma_dir=tmp_path / "chroma")


@pytest.fixture
def store(settings: Settings):
    return build_store(settings, FakeEmbedder(32))


@pytest.fixture
def corpus_copy(tmp_path: Path) -> Path:
    """Copia del corpus real (el original está congelado y no se modifica)."""
    target = tmp_path / "docs"
    shutil.copytree(CORPUS, target)
    return target


def test_ingest_corpus(store, settings: Settings) -> None:
    """M5-01: data/docs procesa los 3 archivos y ≥ 2 documentos (R5)."""
    report = ingest_paths([CORPUS], store, settings)
    assert isinstance(report, IngestReport)
    assert report.files_processed == 3
    assert report.files_skipped == 0  # .gitkeep es oculto: no cuenta
    assert report.documents >= 2  # md + txt + 2 páginas de PDF = 4
    assert report.chunks_added == report.total_in_store == store.count() > 0
    assert sorted(report.sources) == sorted(store.list_sources())
    assert report.duration_s >= 0


def test_reingest_idempotent(store, settings: Settings) -> None:
    """M5-02: re-ingerir el mismo directorio no cambia total_in_store."""
    first = ingest_paths([CORPUS], store, settings)
    second = ingest_paths([CORPUS], store, settings)
    assert second.total_in_store == first.total_in_store
    assert store.source_stats() == {
        "guia_onboarding_ti.txt": 6,
        "manual_reembolso_gastos.pdf": 4,
        "politica_vacaciones_y_permisos.md": 6,
    }


def test_replace_modified_file(store, settings: Settings, corpus_copy: Path) -> None:
    """M5-03: modificar un archivo y re-ingerir reemplaza sus chunks (no quedan los viejos)."""
    ingest_paths([corpus_copy], store, settings)
    target = corpus_copy / "guia_onboarding_ti.txt"
    old_ids = {
        r.chunk_id
        for r in store.query("FortiClient VPN", top_k=50)
        if r.metadata["source"] == target.name
    }
    others = {k: v for k, v in store.source_stats().items() if k != target.name}

    target.write_text(
        "GUÍA BREVE\n\nLa VPN corporativa ahora es WireGuard en todas las sedes.", encoding="utf-8"
    )
    ingest_paths([corpus_copy], store, settings)

    after = [r for r in store.query("VPN", top_k=50) if r.metadata["source"] == target.name]
    assert len(after) == 1 and "WireGuard" in after[0].text
    assert not old_ids & {r.chunk_id for r in after}
    assert not any("FortiClient" in r.text for r in after)
    assert {k: v for k, v in store.source_stats().items() if k != target.name} == others


def test_skips_unsupported(store, settings: Settings, tmp_path: Path) -> None:
    """M5-04: archivos no soportados se cuentan en files_skipped sin abortar."""
    folder = tmp_path / "mixtos"
    folder.mkdir()
    (folder / "informe.docx").write_bytes(b"PK")
    (folder / "datos.xlsx").write_bytes(b"PK")
    (folder / "nota.txt").write_text(LONG_TEXT, encoding="utf-8")
    report = ingest_paths([folder], store, settings)
    assert report.files_processed == 1
    assert report.files_skipped == 2
    assert any("informe.docx" in s and "no soportado" in s for s in report.skipped)


def test_skips_empty_and_textless_pdf(store, settings: Settings, tmp_path: Path) -> None:
    """Archivos vacíos y PDF sin texto extraíble cuentan como omitidos."""
    folder = tmp_path / "vacios"
    folder.mkdir()
    (folder / "vacio.md").write_text("", encoding="utf-8")
    pdf = FPDF()
    pdf.add_page()
    pdf.output(str(folder / "escaneado.pdf"))
    (folder / "ok.txt").write_text(LONG_TEXT, encoding="utf-8")
    report = ingest_paths([folder], store, settings)
    assert (report.files_processed, report.files_skipped) == (1, 2)
    assert all("sin texto" in s for s in report.skipped)


def test_duplicate_name_skipped(
    store, settings: Settings, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Dos archivos con el mismo nombre: se ingiere el primero y el segundo se omite con aviso."""
    for sub, text in [("a", LONG_TEXT), ("b", LONG_TEXT + " Versión distinta.")]:
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "manual.txt").write_text(text, encoding="utf-8")
    report = ingest_paths([tmp_path / "a", tmp_path / "b"], store, settings)
    assert (report.files_processed, report.files_skipped) == (1, 1)
    assert "nombre duplicado" in report.skipped[0]
    assert "Versión distinta" not in store.query("manual", top_k=5)[0].text
    assert "nombre duplicado" in caplog.text


def test_recursive_flag(store, settings: Settings, tmp_path: Path) -> None:
    """Por defecto no recorre subdirectorios; con recursive=True sí."""
    (tmp_path / "raiz").mkdir()
    (tmp_path / "raiz" / "uno.txt").write_text(LONG_TEXT, encoding="utf-8")
    (tmp_path / "raiz" / "sub").mkdir()
    (tmp_path / "raiz" / "sub" / "dos.txt").write_text(LONG_TEXT + " Dos.", encoding="utf-8")
    assert ingest_paths([tmp_path / "raiz"], store, settings).sources == ["uno.txt"]
    report = ingest_paths([tmp_path / "raiz"], store, settings, recursive=True)
    assert report.sources == ["dos.txt", "uno.txt"]  # orden por ruta: sub/… < uno


def test_single_file_and_missing_path(store, settings: Settings, tmp_path: Path) -> None:
    """Acepta archivos sueltos; una ruta inexistente lanza FileNotFoundError."""
    report = ingest_paths([CORPUS / "guia_onboarding_ti.txt"], store, settings)
    assert report.sources == ["guia_onboarding_ti.txt"]
    with pytest.raises(FileNotFoundError, match="No existe la ruta"):
        ingest_paths([tmp_path / "no_existe"], store, settings)
