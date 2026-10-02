"""Fragmentación (chunking) recursiva por separadores, con solapamiento e ids estables.

Algoritmo:
1. Partir el texto por el separador más "grueso" presente (párrafo → línea → oración →
   palabra); las partes que aún exceden `chunk_size` se parten con el siguiente nivel y,
   en último caso, por caracteres. Cada parte conserva su separador al final, de modo que
   concatenarlas reproduce el texto original.
2. Los encabezados (líneas Markdown `#`, títulos numerados cortos o líneas en MAYÚSCULAS)
   se pegan a la parte siguiente: un encabezado nunca queda separado de su contenido.
3. Fusionar partes consecutivas mientras quepan en `chunk_size`; cada chunk nuevo empieza
   con las últimas ~`chunk_overlap` letras del anterior, cortadas en límite de palabra.

Cada chunk es una subcadena contigua del Document de origen (nunca mezcla páginas).
"""

import hashlib
import re
from collections.abc import Iterable

from rag.models import Chunk, Document

SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ")
MIN_CHUNK_CHARS = 30  # chunks con menos caracteres no vacíos se descartan (ruido)

_NUMBERED_TITLE = re.compile(r"^\d+(\.\d+)*\.?\s+\S")
_MAX_TITLE_LEN = 80


def is_heading(line: str) -> bool:
    """True si la línea parece un encabezado (Markdown, título numerado o MAYÚSCULAS)."""
    line = line.strip()
    if not line or "\n" in line or len(line) > _MAX_TITLE_LEN:
        return False
    if line.startswith("#"):
        return True
    if line.endswith((".", ":", ";", ",")):
        return False
    return bool(_NUMBERED_TITLE.match(line)) or (line.isupper() and len(line) >= 4)


def _split_keep(text: str, sep: str) -> list[str]:
    """Parte por `sep` dejando el separador al final de cada parte (salvo la última)."""
    parts = text.split(sep)
    return [p + sep for p in parts[:-1]] + [parts[-1]]


def _glue_headings(parts: list[str]) -> list[str]:
    """Pega cada encabezado (y líneas vacías tras él) a la siguiente parte con contenido."""
    glued: list[str] = []
    pending = ""
    for part in parts:
        if is_heading(part) or (pending and not part.strip()):
            pending += part
            continue
        glued.append(pending + part)
        pending = ""
    if pending:
        glued.append(pending)
    return glued


def _split_pieces(text: str, chunk_size: int, separators: tuple[str, ...]) -> list[str]:
    """Divide recursivamente hasta que cada pieza mida como máximo `chunk_size`."""
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    sep, rest = separators[0], separators[1:]
    if sep not in text:
        return _split_pieces(text, chunk_size, rest)

    pieces: list[str] = []
    for part in _glue_headings(_split_keep(text, sep)):
        if len(part) <= chunk_size:
            pieces.append(part)
        else:
            pieces.extend(_split_pieces(part, chunk_size, rest))
    return pieces


def _overlap_tail(text: str, budget: int) -> str:
    """Sufijo de `text` de como máximo `budget` caracteres que empieza en límite de palabra."""
    if budget <= 0:
        return ""
    tail = text[-budget:]
    if len(text) > budget and not text[-budget - 1].isspace():
        cut = re.search(r"\s", tail)
        if cut is None:
            return ""
        tail = tail[cut.end() :]
    return tail if tail.strip() else ""


def _merge(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Fusiona piezas hasta `chunk_size` y aplica solapamiento entre chunks consecutivos."""
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if len((current + piece).strip()) <= chunk_size:
            current += piece
            continue
        if current.strip():
            chunks.append(current.strip())
        tail = _overlap_tail(current, min(chunk_overlap, chunk_size - len(piece.strip())))
        while tail and len((tail + piece).strip()) > chunk_size:
            tail = _overlap_tail(tail, len(tail) - 1)
        current = tail + piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _validate_sizes(chunk_size: int, chunk_overlap: int) -> None:
    """Valida los parámetros de fragmentación."""
    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser mayor que 0")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap debe estar en [0, chunk_size)")


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Divide un texto en fragmentos de como máximo `chunk_size` caracteres."""
    _validate_sizes(chunk_size, chunk_overlap)
    if not text.strip():
        return []
    return _merge(_split_pieces(text, chunk_size, SEPARATORS), chunk_size, chunk_overlap)


def make_chunk_id(source: str, page: int | str, chunk_index: int, text: str) -> str:
    """Id determinista: sha1(source|page|chunk_index|text)[:16]."""
    raw = f"{source}|{page}|{chunk_index}|{text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def chunk_documents(
    documents: Iterable[Document], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Fragmenta cada Document por separado (en PDF, cada página) y asigna ids estables.

    Descarta fragmentos con menos de `MIN_CHUNK_CHARS` caracteres no vacíos. `page` vale ""
    en el id cuando el Document no tiene página (md/txt).
    """
    chunks: list[Chunk] = []
    for doc in documents:
        texts = [
            t
            for t in split_text(doc.text, chunk_size, chunk_overlap)
            if len("".join(t.split())) >= MIN_CHUNK_CHARS
        ]
        for index, text in enumerate(texts):
            metadata = {**doc.metadata, "chunk_index": index, "char_count": len(text)}
            chunk_id = make_chunk_id(
                str(doc.metadata.get("source", "")), doc.metadata.get("page", ""), index, text
            )
            chunks.append(Chunk(text=text, metadata=metadata, chunk_id=chunk_id))
    return chunks
