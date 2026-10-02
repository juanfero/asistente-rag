"""Modelos de datos del pipeline RAG."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """Texto extraído de un archivo (o de una página de PDF) con sus metadatos.

    Metadatos: `source` (nombre de archivo), `doc_type` (`md|pdf|txt`), `path` (ruta relativa
    a la raíz del proyecto) y `page` (1-based, solo PDF; se omite en los demás tipos).
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
