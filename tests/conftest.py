import hashlib
import io
import json
import os
import re
import uuid
from typing import Dict, List

import numpy as np
import pytest
from PIL import Image

from sci_vizio_retrieval.stores.base import IndexRecord, SearchResult, VectorStore

POSTGRES_DSN_ENV = "SCI_TEST_POSTGRES_DSN"


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: texts sharing words are close."""

    def __init__(self, dimension: int = 64, model_name: str = "fake-hash"):
        self.dimension = dimension
        self.model_name = f"{model_name}-{dimension}"

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimension
            vec[bucket] += 1.0
        norm = np.linalg.norm(vec)
        if norm == 0:
            vec[0] = 1.0
            return vec
        return vec / norm

    def embed(self, texts) -> List[np.ndarray]:
        return [self._embed_one(t) for t in texts]


class InMemoryStore(VectorStore):
    """Minimal VectorStore used to test code that depends on the interface."""

    backend = "memory"

    def __init__(self, name: str = "memory:test", fail_ids=()):
        self._name = name
        self.records: Dict[str, IndexRecord] = {}
        self.fail_ids = set(fail_ids)
        self.embedder = FakeEmbedder()
        self.closed = False

    @property
    def name(self) -> str:
        return self._name

    def upsert(self, record: IndexRecord) -> None:
        if record.id in self.fail_ids:
            raise RuntimeError("simulated store failure")
        self.records[record.id] = record

    def search(self, query: str, n_results: int = 5) -> List[SearchResult]:
        (q,) = self.embedder.embed([query])
        scored = sorted(
            self.records.values(),
            key=lambda r: -float(np.dot(q, self.embedder.embed([r.document_text])[0])),
        )
        return [
            SearchResult(id=r.id, pdf_file=r.pdf_file, image_path=r.image_path,
                         analysis=r.analysis, image_bytes=r.image_bytes)
            for r in scored[:n_results]
        ]

    def count(self) -> int:
        return len(self.records)

    def close(self) -> None:
        self.closed = True


def make_png(color=(200, 30, 30, 255), size=(32, 24), mode="RGBA") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color[: len(mode)]).save(buf, format="PNG")
    return buf.getvalue()


def analysis_json(title: str, description: str, image_type: str = "diagram") -> str:
    return json.dumps({"image_type": image_type, "title": title, "description": description})


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def memory_store():
    return InMemoryStore()


@pytest.fixture
def png_bytes():
    return make_png()


@pytest.fixture
def image_file(tmp_path, png_bytes):
    path = tmp_path / "images" / "paper" / "page1_img1.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(png_bytes)
    return path


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Keep the developer's .env / shell overrides from leaking into tests."""
    for key in list(os.environ):
        if key.startswith("SCI_") and key != POSTGRES_DSN_ENV:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


# --- Real backend fixtures -------------------------------------------------

def _hash_embedding_function():
    chromadb = pytest.importorskip("chromadb")
    embedder = FakeEmbedder()

    class HashEmbeddingFunction(chromadb.EmbeddingFunction):
        def __init__(self):
            pass

        def __call__(self, input):
            return embedder.embed(input)

    return HashEmbeddingFunction()


@pytest.fixture
def chroma_store(tmp_path):
    from sci_vizio_retrieval.stores.chroma import ChromaVectorStore

    return ChromaVectorStore(
        tmp_path / "chroma",
        index_images=False,
        text_embedding_function=_hash_embedding_function(),
    )


@pytest.fixture
def postgres_dsn():
    dsn = os.environ.get(POSTGRES_DSN_ENV)
    if not dsn:
        pytest.skip(f"set {POSTGRES_DSN_ENV} to run PostgreSQL tests")
    pytest.importorskip("psycopg")
    pytest.importorskip("pgvector")
    return dsn


@pytest.fixture
def postgres_table(postgres_dsn):
    """A unique table name, dropped after the test."""
    import psycopg

    table = f"test_{uuid.uuid4().hex[:12]}"
    yield table
    with psycopg.connect(postgres_dsn, autocommit=True) as conn:
        conn.execute(f'DROP TABLE IF EXISTS "{table}"')


@pytest.fixture
def postgres_store(postgres_dsn, postgres_table, fake_embedder):
    from sci_vizio_retrieval.stores.postgres import PostgresVectorStore

    store = PostgresVectorStore(postgres_dsn, embedder=fake_embedder, table=postgres_table)
    yield store
    store.close()


@pytest.fixture(params=["chroma", pytest.param("postgres", marks=pytest.mark.postgres)])
def store(request):
    """Every real backend, so each contract test runs against both."""
    return request.getfixturevalue(f"{request.param}_store")
