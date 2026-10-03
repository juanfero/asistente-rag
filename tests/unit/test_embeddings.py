"""Pruebas de M3 (sin red): FakeEmbedder, carga perezosa y llamada al modelo (mock)."""

import json
import math
import os
import subprocess
import sys
import types

import numpy as np
import pytest

from rag.config import Settings
from rag.embeddings import FakeEmbedder, SentenceTransformerEmbedder, get_embedder

TEXTS = ["¿Cuántos días de vacaciones tengo?", "La VPN corporativa es FortiClient.", ""]


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_fake_embedder() -> None:
    """M3-01: FakeEmbedder determinista, dimensión correcta y vectores normalizados."""
    emb = FakeEmbedder(dimension=16)
    first = emb.embed_documents(TEXTS)
    assert first == FakeEmbedder(dimension=16).embed_documents(TEXTS)
    assert emb.dimension == 16
    assert all(len(v) == 16 for v in first)
    assert all(abs(_norm(v) - 1.0) < 1e-9 for v in first)
    assert emb.embed_query(TEXTS[0]) == first[0]


def test_fake_embedder_similarity() -> None:
    """Textos con palabras en común son más similares que textos sin relación."""
    emb = FakeEmbedder(dimension=64)
    q = emb.embed_query("días de vacaciones por año")
    near = emb.embed_query("tiene 15 días de vacaciones por cada año trabajado")
    far = emb.embed_query("servidor vpn fortinet soporte")
    assert _cos(q, near) > _cos(q, far)


_SUBPROCESS_CODE = (
    "import json; from rag.embeddings import FakeEmbedder; "
    f"print(json.dumps(FakeEmbedder(16).embed_documents({TEXTS!r})))"
)


def test_fake_embedder_deterministic_across_processes() -> None:
    """FakeEmbedder no depende de PYTHONHASHSEED: mismo resultado en procesos distintos."""
    results = []
    for seed in ("1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_CODE],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        results.append(json.loads(out.stdout))
    assert results[0] == results[1] == FakeEmbedder(16).embed_documents(TEXTS)


class _FakeSTModel:
    """Sustituto de SentenceTransformer que registra las llamadas."""

    instances = 0

    def __init__(self, name: str, device: str | None = None) -> None:
        type(self).instances += 1
        self.name, self.device = name, device
        self.max_seq_length = 128
        self.calls: list[dict] = []

    def get_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts, **kwargs):
        self.calls.append({"texts": list(texts), **kwargs})
        return np.array([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)


@pytest.fixture
def fake_st(monkeypatch: pytest.MonkeyPatch) -> type[_FakeSTModel]:
    """Reemplaza el módulo sentence_transformers por uno falso (sin descargar nada)."""
    _FakeSTModel.instances = 0
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeSTModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return _FakeSTModel


def test_empty_input(fake_st: type[_FakeSTModel]) -> None:
    """M3-02: lista vacía → lista vacía sin cargar el modelo."""
    emb = SentenceTransformerEmbedder("modelo-x")
    assert emb.embed_documents([]) == []
    assert fake_st.instances == 0
    assert FakeEmbedder().embed_documents([]) == []


def test_lazy_loading(fake_st: type[_FakeSTModel]) -> None:
    """M3-03: el modelo no se carga al instanciar; se carga una sola vez en el primer uso."""
    emb = SentenceTransformerEmbedder("modelo-x", batch_size=8)
    assert fake_st.instances == 0
    emb.embed_query("hola")
    emb.embed_documents(["a", "b"])
    assert fake_st.instances == 1


def test_encode_parameters(fake_st: type[_FakeSTModel]) -> None:
    """Se usa device='cpu', normalize_embeddings=True y el batch_size configurado."""
    emb = SentenceTransformerEmbedder("modelo-x", batch_size=8)
    vectors = emb.embed_documents(["a", "b"])
    model = emb.model
    assert model.name == "modelo-x" and model.device == "cpu"
    call = model.calls[0]
    assert call["normalize_embeddings"] is True
    assert call["batch_size"] == 8
    assert vectors == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert all(isinstance(x, float) for row in vectors for x in row)
    assert emb.dimension == 3 and emb.max_seq_length == 128


def test_same_encoding_when_prefixes_empty(fake_st: type[_FakeSTModel]) -> None:
    """Con prefijos vacíos explícitos, embed_query y embed_documents codifican igual."""
    emb = SentenceTransformerEmbedder("modelo-x", query_prefix="", passage_prefix="")
    assert (emb.query_prefix, emb.passage_prefix) == ("", "")
    emb.embed_query("pregunta")
    emb.embed_documents(["pregunta"])
    assert emb.model.calls[0]["texts"] == emb.model.calls[1]["texts"] == ["pregunta"]


def test_get_embedder() -> None:
    """La fábrica crea el embedder del modelo configurado sin cargarlo."""
    settings = Settings(_env_file=None, embedding_model="otro/modelo")
    emb = get_embedder(settings)
    assert isinstance(emb, SentenceTransformerEmbedder)
    assert emb.model_name == "otro/modelo" and emb._model is None


def test_empty_prefixes_keep_behavior(fake_st: type[_FakeSTModel]) -> None:
    """Con prefijos vacíos (default) encode recibe el texto sin cambios."""
    emb = SentenceTransformerEmbedder("modelo-x", query_prefix="", passage_prefix="")
    emb.embed_query("pregunta")
    emb.embed_documents(["fragmento"])
    assert [c["texts"] for c in emb.model.calls] == [["pregunta"], ["fragmento"]]


def test_prefixes_applied(fake_st: type[_FakeSTModel]) -> None:
    """Con prefijos, encode recibe 'query: ' en preguntas y 'passage: ' en documentos."""
    emb = SentenceTransformerEmbedder(
        "modelo-x", query_prefix="query: ", passage_prefix="passage: "
    )
    emb.embed_query("¿cuántos días?")
    emb.embed_documents(["texto a", "texto b"])
    assert emb.model.calls[0]["texts"] == ["query: ¿cuántos días?"]
    assert emb.model.calls[1]["texts"] == ["passage: texto a", "passage: texto b"]


def test_get_embedder_prefixes(monkeypatch: pytest.MonkeyPatch) -> None:
    """La fábrica toma los prefijos de settings (variables EMBEDDING_*_PREFIX)."""
    monkeypatch.setenv("EMBEDDING_QUERY_PREFIX", "query: ")
    monkeypatch.setenv("EMBEDDING_PASSAGE_PREFIX", "passage: ")
    emb = get_embedder(Settings(_env_file=None))
    assert (emb.query_prefix, emb.passage_prefix) == ("query: ", "passage: ")
    assert get_embedder(Settings(_env_file=None, embedding_query_prefix="")).query_prefix == ""


def test_prefix_only_at_encoding(fake_st: type[_FakeSTModel]) -> None:
    """El prefijo solo se aplica al codificar: no altera el texto ni el id de los chunks."""
    from rag.chunking import chunk_documents
    from rag.models import Document

    body = "Cada colaborador tiene derecho a 15 días hábiles de vacaciones por año trabajado."
    doc = Document(body, {"source": "v.md", "doc_type": "md", "path": "data/docs/v.md"})
    before = chunk_documents([doc], 800, 120)
    texts = [c.text for c in before]

    emb = SentenceTransformerEmbedder(
        "modelo-x", query_prefix="query: ", passage_prefix="passage: "
    )
    emb.embed_documents(texts)

    after = chunk_documents([doc], 800, 120)
    assert texts == [c.text for c in before] == [c.text for c in after]  # sin mutación
    assert [c.chunk_id for c in before] == [c.chunk_id for c in after]
    assert not any(c.text.startswith("passage: ") for c in after)
    assert emb.model.calls[0]["texts"] == ["passage: " + t for t in texts]
