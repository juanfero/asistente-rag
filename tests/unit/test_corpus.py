"""Pruebas de M1 sobre el corpus real de Nexa Logística (data/docs)."""

import hashlib
import importlib.util
import re
from pathlib import Path

import pytest
from pypdf import PdfReader

from rag.loaders import PROJECT_ROOT, load_file

DOCS_DIR = PROJECT_ROOT / "data" / "docs"
MD = "politica_vacaciones_y_permisos.md"
PDF = "manual_reembolso_gastos.pdf"
TXT = "guia_onboarding_ti.txt"

# Hechos clave: verdad de referencia para la evaluación (M10).
KEY_FACTS = {
    MD: [
        "2.1",
        "2026-01-15",
        "15 días hábiles",
        "máximo de 2 periodos",
        "NexaPeople",
        "15 días calendario",
        "3 días hábiles",
        "6 días hábiles",
        "Matrimonio",
        "5 días hábiles",
        "Calamidad doméstica",
        "Hasta 5 días",
        "Mudanza",
        "1 día por año",
        "paternidad",
        "2 semanas",
        "maternidad",
        "18 semanas",
        "ley colombiana",
        "cumpleaños",
        "mes del cumpleaños",
    ],
    PDF: [
        "10 días hábiles",
        "FR-021",
        "factura electrónica",
        "900.123.456-7",
        "COP 120.000",
        "COP 80.000",
        "COP 350.000",
        "Bogotá y Medellín",
        "COP 420.000",
        "USD 90",
        "70%",
        "jefe inmediato",
        "COP 2.000.000",
        "gerente de área",
        "siguiente quincena",
        "Bebidas alcohólicas",
        "Multas de tránsito",
        "minibar",
    ],
    TXT: [
        "Mesa de Ayuda",
        "piso 3",
        "correo personal",
        "12 caracteres",
        "90 días",
        "últimas 5",
        "Microsoft Authenticator",
        "FortiClient",
        "vpn.nexalogistica.co",
        "fuera de la oficina",
        "soporte@nexalogistica.co",
        "4040",
        "7:00 a 18:00",
        "4 horas hábiles",
        "1 hora",
        "NexaDesk",
        "software no autorizado",
    ],
}

# Temas que NO deben aparecer (base de las preguntas parciales/no contestables de M10).
EXCLUDED_TOPICS = {
    MD: ["horas extra", "remoto", "teletrabajo", "salario"],
    PDF: [
        "kilometraje",
        "kilómetro",
        "vehículo propio",
        "tarjeta corporativa",
        "tarjetas corporativas",
    ],
    TXT: ["celular", "préstamo", "para casa", "remoto"],
}


def _loaded_text(filename: str) -> str:
    """Texto completo cargado del documento, con espacios en blanco normalizados."""
    docs = load_file(DOCS_DIR / filename)
    return re.sub(r"\s+", " ", " ".join(d.text for d in docs))


def test_corpus_exists() -> None:
    """M1-01: existen los 3 documentos y el PDF tiene ≥2 páginas con texto extraíble."""
    for filename in (MD, PDF, TXT):
        assert (DOCS_DIR / filename).is_file(), filename

    reader = PdfReader(DOCS_DIR / PDF)
    pages_with_text = [p for p in reader.pages if (p.extract_text() or "").strip()]
    assert len(reader.pages) >= 2
    assert len(pages_with_text) >= 2


@pytest.mark.parametrize(
    ("filename", "fact"),
    [(filename, fact) for filename, facts in KEY_FACTS.items() for fact in facts],
)
def test_corpus_key_facts(filename: str, fact: str) -> None:
    """M1-02: cada hecho clave aparece literalmente en el texto cargado."""
    assert fact in _loaded_text(filename)


@pytest.mark.parametrize(
    ("filename", "topic"),
    [(filename, topic) for filename, topics in EXCLUDED_TOPICS.items() for topic in topics],
)
def test_corpus_excluded_topics(filename: str, topic: str) -> None:
    """Los temas fuera del corpus no aparecen (necesario para las preguntas de M10)."""
    assert topic.lower() not in _loaded_text(filename).lower()


def test_pdf_facts_by_page() -> None:
    """Los hechos del PDF quedan en la página documentada (permite citar la página)."""
    pages = {d.metadata["page"]: d.text for d in load_file(DOCS_DIR / PDF)}
    assert "FR-021" in pages[1] and "COP 120.000" in pages[1]
    assert "USD 90" in pages[2] and "minibar" in pages[2]


def test_pdf_is_reproducible(tmp_path: Path) -> None:
    """Regenerar el PDF con el script produce un archivo idéntico byte a byte."""
    script = PROJECT_ROOT / "scripts" / "generar_pdf_ejemplo.py"
    spec = importlib.util.spec_from_file_location("generar_pdf_ejemplo", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    regenerated = module.build_pdf(tmp_path / PDF)
    digest = hashlib.sha256(regenerated.read_bytes()).hexdigest()
    assert digest == hashlib.sha256((DOCS_DIR / PDF).read_bytes()).hexdigest()
