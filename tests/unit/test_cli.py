"""Pruebas de M5: CLI `ingest`, `stats` y `reset` vía main(argv) con dependencias inyectadas."""

import re
from pathlib import Path

import pytest

from rag.cli import main
from rag.config import Settings
from rag.embeddings import FakeEmbedder
from rag.ingest import build_store
from rag.loaders import PROJECT_ROOT

CORPUS = PROJECT_ROOT / "data" / "docs"
TXT = CORPUS / "guia_onboarding_ti.txt"
MD = CORPUS / "politica_vacaciones_y_permisos.md"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, chroma_dir=tmp_path / "chroma")


def run(argv: list[str], settings: Settings, **kwargs) -> int:
    return main(argv, settings=settings, embedder=FakeEmbedder(32), **kwargs)


def test_cli_ingest_stats(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    """M5-06: `ingest` y `stats` devuelven 0 e imprimen las fuentes (y modelo/dimensión)."""
    assert run(["ingest", str(CORPUS)], settings) == 0
    out = capsys.readouterr().out
    assert "Archivos procesados" in out and "Chunks añadidos" in out
    for source in ("guia_onboarding_ti.txt", "manual_reembolso_gastos.pdf", MD.name):
        assert source in out

    assert run(["stats"], settings) == 0
    out = capsys.readouterr().out
    assert "Total de chunks: 16" in out
    assert "Modelo de embeddings: intfloat/multilingual-e5-small (dim 32)" in out
    assert re.search(r"manual_reembolso_gastos\.pdf\s+4 chunks", out)


def test_cli_reset(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    """M5-05: `ingest --reset` vacía el índice antes de ingerir."""
    assert run(["ingest", str(TXT)], settings) == 0
    assert run(["ingest", str(MD), "--reset"], settings) == 0
    store = build_store(settings, FakeEmbedder(32))
    assert store.list_sources() == [MD.name]
    assert "vaciado antes de ingerir" in capsys.readouterr().out


def test_cli_bad_path(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    """M5-07: ruta inexistente → mensaje claro y código ≠ 0, sin traceback."""
    code = run(["ingest", "ruta/que/no/existe"], settings)
    err = capsys.readouterr().err
    assert code != 0
    assert "No existe la ruta" in err and "Traceback" not in err


def test_cli_nothing_ingested(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Si no se ingiere ningún documento (solo no soportados) → código ≠ 0."""
    folder = tmp_path / "solo_docx"
    folder.mkdir()
    (folder / "a.docx").write_bytes(b"PK")
    assert run(["ingest", str(folder)], settings) != 0
    assert "no se ingirió ningún documento" in capsys.readouterr().err


def test_cli_reset_command(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    """`reset` pide confirmación (no → no cambia; --yes → vacía)."""
    run(["ingest", str(TXT)], settings)
    assert run(["reset"], settings, confirm=lambda _: "n") == 0
    assert build_store(settings, FakeEmbedder(32)).count() > 0
    assert run(["reset"], settings, confirm=lambda _: "s") == 0
    assert build_store(settings, FakeEmbedder(32)).count() == 0
    run(["ingest", str(TXT)], settings)
    assert (
        run(["reset", "--yes"], settings, confirm=lambda _: pytest.fail("no debe preguntar")) == 0
    )
    assert build_store(settings, FakeEmbedder(32)).count() == 0


def test_cli_model_mismatch(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    """Índice creado con otro modelo → mensaje claro con la solución, código ≠ 0, sin traceback;
    `reset --yes` (reset_collection) lo resuelve aunque el modelo no coincida."""
    run(["ingest", str(TXT)], settings)
    other = settings.model_copy(update={"embedding_model": "otro/modelo"})
    capsys.readouterr()
    assert run(["stats"], other) != 0
    err = capsys.readouterr().err
    assert "otro/modelo" in err and "reset --yes" in err and "Traceback" not in err

    assert run(["reset", "--yes"], other) == 0
    assert run(["ingest", str(TXT)], other) == 0
