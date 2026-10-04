"""Pruebas de M11: el README cubre la sección 4.5 del caso y enlaza los entregables."""

import re

from rag.loaders import PROJECT_ROOT

README = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
HEADINGS = [
    line.lstrip("#").strip().lower() for line in README.splitlines() if line.startswith("## ")
]

# Secciones exigidas por la sección 4.5 del caso (texto que debe contener algún encabezado ##)
REQUIRED_SECTIONS = [
    "descripción de la solución",
    "dependencias",
    "instalación",
    "ejecución",
    "cómo cargar documentos",
    "cómo hacer preguntas",
    "limitaciones",
    "mejoras futuras",
]


def test_readme_sections() -> None:
    """M11-01: el README tiene un encabezado ## por cada sección exigida en 4.5."""
    missing = [s for s in REQUIRED_SECTIONS if not any(s in h for h in HEADINGS)]
    assert missing == []


def test_readme_links_deliverables() -> None:
    """M11-04/05: el README enlaza resultados, uso de IA, evidencias y la sección del video."""
    for target in (
        "evaluacion/resultados.md",
        "docs/04_USO_AI_ASSISTED.md",
        "docs/05_GUION_VIDEO.md",
        "evidencias/README.md",
    ):
        assert f"]({target}" in README, target
        assert (PROJECT_ROOT / target).exists(), target
    assert "video" in HEADINGS


def test_readme_local_links_exist() -> None:
    """Todos los enlaces relativos del README apuntan a archivos o carpetas existentes."""
    links = re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", README)
    local = [link for link in links if not link.startswith(("http", "mailto:", "VIDEO_URL"))]
    missing = [link for link in local if not (PROJECT_ROOT / link).exists()]
    assert local and missing == []
