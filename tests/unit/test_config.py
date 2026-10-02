"""Pruebas de M0: paquete, configuración y logging."""

import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.config import Settings, get_settings
from rag.logging_conf import LOG_FORMAT, setup_logging

pytestmark = pytest.mark.usefixtures("clean_env")

FAKE_KEY = "xai-test-NO-REAL-0123456789"


def test_import_package() -> None:
    """M0-01: el paquete se importa y expone su versión."""
    import rag

    assert rag.__version__


def test_defaults() -> None:
    """M0-02: sin .env ni variables se cargan los defaults documentados."""
    s = Settings(_env_file=None)
    assert s.xai_api_key is None
    assert s.xai_base_url == "https://api.x.ai/v1"
    assert s.xai_model == "grok-3-mini"
    assert s.llm_temperature == 0.1
    assert s.llm_max_tokens == 700
    assert s.embedding_model == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    assert s.docs_dir == Path("data/docs")
    assert s.chroma_dir == Path("data/chroma")
    assert s.chroma_collection == "documentos"
    assert s.chunk_size == 800
    assert s.chunk_overlap == 120
    assert s.top_k == 4
    assert s.min_score == 0.35
    assert s.api_url == "http://localhost:8000"
    assert s.log_level == "INFO"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """M0-03: las variables de entorno (en MAYÚSCULAS) sobreescriben los defaults."""
    monkeypatch.setenv("XAI_MODEL", "grok-otro")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "80")
    monkeypatch.setenv("TOP_K", "6")
    monkeypatch.setenv("MIN_SCORE", "0.5")
    monkeypatch.setenv("DOCS_DIR", "/tmp/otros_docs")
    s = Settings(_env_file=None)
    assert s.xai_model == "grok-otro"
    assert (s.chunk_size, s.chunk_overlap, s.top_k) == (500, 80, 6)
    assert s.min_score == 0.5
    assert s.docs_dir == Path("/tmp/otros_docs")


def test_env_file_is_read(tmp_path: Path) -> None:
    """Complemento M0-03: un archivo .env también sobreescribe los defaults."""
    env_file = tmp_path / ".env"
    env_file.write_text("XAI_MODEL=grok-desde-env\nTOP_K=2\n", encoding="utf-8")
    s = Settings(_env_file=env_file)
    assert s.xai_model == "grok-desde-env"
    assert s.top_k == 2


def test_get_settings_is_cached() -> None:
    """`get_settings` devuelve siempre la misma instancia."""
    assert get_settings() is get_settings()


@pytest.mark.parametrize(("size", "overlap"), [(800, 800), (500, 600)])
def test_invalid_overlap(size: int, overlap: int) -> None:
    """M0-04: CHUNK_OVERLAP >= CHUNK_SIZE lanza ValidationError."""
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        Settings(_env_file=None, chunk_size=size, chunk_overlap=overlap)


@pytest.mark.parametrize(
    "overrides",
    [{"min_score": -0.1}, {"min_score": 1.1}, {"top_k": 0}, {"top_k": -3}],
)
def test_invalid_ranges(overrides: dict) -> None:
    """M0-05: MIN_SCORE fuera de [0,1] o TOP_K < 1 lanza ValidationError."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **overrides)


@pytest.mark.parametrize("min_score", [0.0, 1.0])
def test_valid_range_limits(min_score: float) -> None:
    """Los extremos 0 y 1 de MIN_SCORE son válidos."""
    assert Settings(_env_file=None, min_score=min_score).min_score == min_score


def test_secret_not_exposed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """M0-06: la API key no aparece en repr/str/dump ni en logs."""
    monkeypatch.setenv("XAI_API_KEY", FAKE_KEY)
    s = Settings(_env_file=None)
    logger = logging.getLogger("rag.test")

    assert s.xai_api_key is not None
    assert s.xai_api_key.get_secret_value() == FAKE_KEY
    assert FAKE_KEY not in repr(s)
    assert FAKE_KEY not in str(s)
    assert FAKE_KEY not in s.model_dump_json()

    # Registros capturados por pytest
    with caplog.at_level(logging.DEBUG):
        logger.info("Configuración cargada: %s", s)
        logger.debug("Key: %s", s.xai_api_key)
    assert caplog.records
    assert FAKE_KEY not in caplog.text

    # Salida real formateada por el handler del proyecto
    setup_logging("DEBUG")
    logger.info("Configuración cargada: %s", s)
    logger.debug("Key: %s", s.xai_api_key)
    stderr = capsys.readouterr().err
    assert "Configuración cargada" in stderr
    assert FAKE_KEY not in stderr


def test_setup_logging() -> None:
    """`setup_logging` aplica nivel y formato del proyecto."""
    setup_logging("WARNING")
    root = logging.getLogger()
    assert root.level == logging.WARNING
    assert root.handlers
    assert root.handlers[0].formatter._fmt == LOG_FORMAT

    setup_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG
