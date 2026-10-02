"""Carga de documentos .txt, .md y .pdf como objetos `Document`."""

import logging
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from rag.models import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".txt": "txt",
    ".md": "md",
    ".markdown": "md",
    ".pdf": "pdf",
}

# Raíz del proyecto (layout src/): src/rag/loaders.py -> parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_HORIZONTAL_SPACES = re.compile(r"[ \t]+")
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")


class UnsupportedFileTypeError(ValueError):
    """La extensión del archivo no está soportada por los loaders."""


def normalize_text(text: str) -> str:
    """Limpia el texto conservando los párrafos.

    Aplica NFC, unifica saltos de línea, convierte espacios no separables en espacios,
    elimina caracteres de control, colapsa espacios por línea y deja como máximo una
    línea en blanco entre párrafos.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = text.replace("\t", " ")
    text = _CONTROL_CHARS.sub("", text)
    lines = [_HORIZONTAL_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return _EXTRA_BLANK_LINES.sub("\n\n", text).strip()


def relative_path(path: Path, base_dir: Path | None = None) -> str:
    """Ruta relativa a `base_dir` (por defecto la raíz del proyecto); si está fuera, el nombre."""
    base = (base_dir or PROJECT_ROOT).resolve()
    try:
        return path.resolve().relative_to(base).as_posix()
    except ValueError:
        return path.name


def _read_text_file(path: Path) -> str:
    """Lee un archivo de texto en UTF-8, con respaldo en latin-1."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.info("'%s' no es UTF-8 válido; se lee como latin-1", path.name)
        return path.read_text(encoding="latin-1")


def _load_text(path: Path, doc_type: str, base_metadata: dict) -> list[Document]:
    """Carga un .txt/.md como un único Document."""
    text = normalize_text(_read_text_file(path))
    if not text:
        logger.warning("Archivo vacío, se omite: %s", path.name)
        return []
    return [Document(text=text, metadata={**base_metadata, "doc_type": doc_type})]


def _load_pdf(path: Path, base_metadata: dict) -> list[Document]:
    """Carga un PDF como un Document por página con texto (page 1-based)."""
    reader = PdfReader(path)
    documents: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if not text:
            logger.warning("Página %d sin texto extraíble, se omite: %s", page_number, path.name)
            continue
        metadata = {**base_metadata, "doc_type": "pdf", "page": page_number}
        documents.append(Document(text=text, metadata=metadata))
    if not documents:
        logger.warning("PDF sin texto extraíble, se omite: %s", path.name)
    return documents


def load_file(path: str | Path, base_dir: Path | None = None) -> list[Document]:
    """Carga un archivo soportado y devuelve sus Documents (vacío si no tiene texto)."""
    path = Path(path)
    doc_type = SUPPORTED_EXTENSIONS.get(path.suffix.lower())
    if doc_type is None:
        raise UnsupportedFileTypeError(
            f"Tipo de archivo no soportado: '{path.name}'. "
            f"Extensiones válidas: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    base_metadata = {"source": path.name, "path": relative_path(path, base_dir)}
    if doc_type == "pdf":
        return _load_pdf(path, base_metadata)
    return _load_text(path, doc_type, base_metadata)


def _is_hidden(path: Path, root: Path) -> bool:
    """True si algún componente de la ruta (relativa a `root`) empieza por punto."""
    return any(part.startswith(".") for part in path.relative_to(root).parts)


def load_directory(
    directory: str | Path, recursive: bool = False, base_dir: Path | None = None
) -> list[Document]:
    """Carga todos los archivos soportados del directorio en orden determinista por nombre.

    Ignora archivos ocultos y avisa (warning) de los no soportados.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"No existe el directorio: {directory}")

    candidates = directory.rglob("*") if recursive else directory.iterdir()
    files = sorted(
        (p for p in candidates if p.is_file() and not _is_hidden(p, directory)),
        key=lambda p: p.relative_to(directory).as_posix(),
    )

    documents: list[Document] = []
    for path in files:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.warning("Archivo no soportado, se ignora: %s", path.name)
            continue
        documents.extend(load_file(path, base_dir=base_dir))
    return documents
