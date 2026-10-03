"""Pruebas de M0: paquete, configuración y logging."""

import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.config import Settings, get_settings
from rag.logging_conf import LOG_FORMAT, setup_logging

pytestmark = pytest.mark.usefixtures("clean_env")

FAKE_KEY = "test-key-NO-REAL-0123456789"


def test_import_package() -> None:
    """M0-01: el paquete se importa y expone su versión."""
    import rag

    assert rag.__version__


def test_defaults() -> None:
    """M0-02: sin .env ni variables se cargan los defaults documentados."""
    s = Settings(_env_file=None)
    assert s.gemini_api_key is None
    assert s.gemini_base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert s.gemini_model == "gemini-3.1-flash-lite"  # ADR-010
    assert s.llm_temperature == 0.1
    assert s.llm_max_tokens == 700
    assert s.embedding_model == "intfloat/multilingual-e5-small"  # ADR-009
    assert s.embedding_query_prefix == "query: "
    assert s.embedding_passage_prefix == "passage: "
    assert s.docs_dir == Path("data/docs")
    assert s.chroma_dir == Path("data/chroma")
    assert s.chroma_collection == "documentos"
    assert s.chunk_size == 800  # ADR-004/ADR-009
    assert s.chunk_overlap == 120  # ADR-004/ADR-009
    assert s.top_k == 4
    assert s.min_score == 0.805  # calibrado en M7 (ADR-005)
    assert s.api_url == "http://localhost:8000"
    assert s.log_level == "INFO"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """M0-03: las variables de entorno (en MAYÚSCULAS) sobreescriben los defaults."""
    monkeypatch.setenv("GEMINI_MODEL", "gemini-otro")
    monkeypatch.setenv("CHUNK_SIZE", "700")
    monkeypatch.setenv("CHUNK_OVERLAP", "90")
    monkeypatch.setenv("TOP_K", "6")
    monkeypatch.setenv("MIN_SCORE", "0.5")
    monkeypatch.setenv("DOCS_DIR", "/tmp/otros_docs")
    s = Settings(_env_file=None)
    assert s.gemini_model == "gemini-otro"
    assert (s.chunk_size, s.chunk_overlap, s.top_k) == (700, 90, 6)
    assert s.min_score == 0.5
    assert s.docs_dir == Path("/tmp/otros_docs")


def test_env_file_is_read(tmp_path: Path) -> None:
    """Complemento M0-03: un archivo .env también sobreescribe los defaults."""
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_MODEL=gemini-desde-env\nTOP_K=2\n", encoding="utf-8")
    s = Settings(_env_file=env_file)
    assert s.gemini_model == "gemini-desde-env"
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
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    s = Settings(_env_file=None)
    logger = logging.getLogger("rag.test")

    assert s.gemini_api_key is not None
    assert s.gemini_api_key.get_secret_value() == FAKE_KEY
    assert FAKE_KEY not in repr(s)
    assert FAKE_KEY not in str(s)
    assert FAKE_KEY not in s.model_dump_json()

    # Registros capturados por pytest
    with caplog.at_level(logging.DEBUG):
        logger.info("Configuración cargada: %s", s)
        logger.debug("Key: %s", s.gemini_api_key)
    assert caplog.records
    assert FAKE_KEY not in caplog.text

    # Salida real formateada por el handler del proyecto
    setup_logging("DEBUG")
    logger.info("Configuración cargada: %s", s)
    logger.debug("Key: %s", s.gemini_api_key)
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
    # Las librerías ruidosas (peticiones HTTP de Hugging Face, etc.) quedan en WARNING
    for name in ("httpx", "httpx2", "openai", "huggingface_hub", "sentence_transformers"):
        assert logging.getLogger(name).level == logging.WARNING


def test_env_file_prefix_keeps_trailing_space(tmp_path: Path) -> None:
    """En .env, un prefijo entre comillas conserva el espacio final ("query: ")."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        'EMBEDDING_QUERY_PREFIX="query: "\nEMBEDDING_PASSAGE_PREFIX="passage: "\n',
        encoding="utf-8",
    )
    s = Settings(_env_file=env_file)
    assert s.embedding_query_prefix == "query: "
    assert s.embedding_passage_prefix == "passage: "


@pytest.mark.parametrize(
    ("environ", "has_key", "skipped"),
    [
        ({}, True, True),
        ({"RUN_LLM": "0"}, True, True),
        ({"RUN_LLM": "1"}, False, True),
        ({"RUN_LLM": "1"}, True, False),
    ],
    ids=["sin-opt-in", "run-llm-0", "sin-key", "habilitado"],
)
def test_llm_tests_are_opt_in(environ: dict, has_key: bool, skipped: bool) -> None:
    """Las pruebas `llm` solo corren con RUN_LLM=1 y GEMINI_API_KEY presentes."""
    from conftest import llm_skip_reason

    assert (llm_skip_reason(environ, has_key) is not None) is skipped


def test_log_effective_settings(caplog: pytest.LogCaptureFixture) -> None:
    """INFO con los valores efectivos (sin la key) y WARNING si un calibrado difiere."""
    from rag.config import calibrated_overrides, log_effective_settings

    s = Settings(_env_file=None, gemini_api_key=FAKE_KEY)
    with caplog.at_level(logging.INFO):
        log_effective_settings(s)
    assert "min_score=0.805" in caplog.text and "chunk_size=800" in caplog.text
    assert "gemini_api_key_configured=True" in caplog.text
    assert FAKE_KEY not in caplog.text
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
    assert calibrated_overrides(s) == {}

    caplog.clear()
    changed = Settings(_env_file=None, min_score=0.8, chunk_size=500, chunk_overlap=80, top_k=6)
    with caplog.at_level(logging.INFO):
        log_effective_settings(changed)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 3  # top_k no es calibrado de tokens/umbral: no avisa
    assert any("MIN_SCORE=0.8" in w and "0.805" in w for w in warnings)


def test_env_example_keeps_calibrated_defaults() -> None:
    """Con .env.example tal cual, los parámetros de ajuste quedan en los valores del código."""
    from rag.loaders import PROJECT_ROOT

    s = Settings(_env_file=PROJECT_ROOT / ".env.example")
    defaults = Settings(_env_file=None)
    for name in (
        "min_score",
        "chunk_size",
        "chunk_overlap",
        "top_k",
        "llm_temperature",
        "llm_max_tokens",
    ):
        assert getattr(s, name) == getattr(defaults, name), name
    assert s.uploads_dir == Path("data/uploads")


def test_openai_retries_logged_as_warning(capsys: pytest.CaptureFixture[str]) -> None:
    """Los reintentos del SDK openai aparecen como WARNING; su otro INFO queda oculto.

    Se lee la salida real (capsys): setup_logging(force=True) reemplaza el handler de caplog.
    """
    from rag.logging_conf import OPENAI_RETRY_LOGGER

    setup_logging("INFO")
    sdk = logging.getLogger(OPENAI_RETRY_LOGGER)
    sdk.info("Retrying request in 0.48 seconds (retry 1 of 2)")
    sdk.info("Request options: {...}")
    err = capsys.readouterr().err
    assert "WARNING openai._base_client - Retrying request in 0.48 seconds" in err
    assert "Request options" not in err
