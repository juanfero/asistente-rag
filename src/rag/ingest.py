"""Pipeline de ingesta: cargar → fragmentar → embeber → guardar, con reporte."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from rag.chunking import chunk_documents
from rag.config import Settings
from rag.embeddings import Embedder, get_embedder
from rag.loaders import SUPPORTED_EXTENSIONS, load_file
from rag.vectorstore import ChromaVectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    """Resumen de una ingesta."""

    files_processed: int = 0
    files_skipped: int = 0
    documents: int = 0
    chunks_added: int = 0
    total_in_store: int = 0
    sources: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # "archivo: motivo"
    duration_s: float = 0.0


def build_store(settings: Settings, embedder: Embedder | None = None) -> ChromaVectorStore:
    """Vector store configurado (fábrica compartida por CLI, API y UI)."""
    return ChromaVectorStore(
        settings.chroma_dir,
        settings.chroma_collection,
        embedder or get_embedder(settings),
        embedding_model=settings.embedding_model,
    )


def _is_hidden(path: Path, root: Path) -> bool:
    """True si algún componente de la ruta (relativa a `root`) empieza por punto."""
    return any(part.startswith(".") for part in path.relative_to(root).parts)


def _expand(paths: list[Path], recursive: bool) -> list[Path]:
    """Archivos a procesar: los dados explícitamente y los de cada directorio (orden por nombre)."""
    files: list[Path] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"No existe la ruta: {path}")
        if path.is_file():
            files.append(path)
            continue
        candidates = path.rglob("*") if recursive else path.iterdir()
        files.extend(
            sorted(
                (p for p in candidates if p.is_file() and not _is_hidden(p, path)),
                key=lambda p: p.relative_to(path).as_posix(),
            )
        )
    return files


def _skip(report: IngestReport, path: Path, reason: str) -> None:
    """Cuenta un archivo omitido y lo avisa en el log."""
    report.files_skipped += 1
    report.skipped.append(f"{path.name}: {reason}")
    logger.warning("Se omite %s: %s", path, reason)


def ingest_paths(
    paths: list[Path],
    store: ChromaVectorStore,
    settings: Settings,
    replace: bool = True,
    recursive: bool = False,
) -> IngestReport:
    """Ingiere archivos y/o directorios en el vector store.

    Con `replace=True` borra los chunks previos de cada fuente antes de insertar (si el archivo
    cambió no quedan chunks viejos). Se omiten (y se cuentan en `files_skipped`): extensiones no
    soportadas, archivos vacíos o sin texto extraíble y archivos con el mismo nombre que otro ya
    ingerido en esta ejecución (`source` es el nombre del archivo).
    """
    start = time.perf_counter()
    report = IngestReport()
    seen: dict[str, Path] = {}

    for path in _expand([Path(p) for p in paths], recursive):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            _skip(report, path, "tipo de archivo no soportado")
            continue
        if path.name in seen:
            _skip(report, path, f"nombre duplicado (ya se ingirió {seen[path.name]})")
            continue
        documents = load_file(path)
        chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            _skip(report, path, "archivo vacío o sin texto extraíble")
            continue

        seen[path.name] = path
        if replace:
            store.delete_by_source(path.name)
        report.chunks_added += store.add_chunks(chunks)
        report.documents += len(documents)
        report.files_processed += 1
        report.sources.append(path.name)

    report.total_in_store = store.count()
    report.duration_s = time.perf_counter() - start
    return report
