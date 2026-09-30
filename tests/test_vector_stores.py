"""Behaviour every VectorStore backend must share, plus backend-specific checks."""

import numpy as np
import pytest

from sci_vizio_retrieval.stores import IndexRecord, create_vector_store
from tests.conftest import FakeEmbedder, make_png

NN = IndexRecord(
    id="1903.00780_page1_img1",
    pdf_file="1903.00780",
    image_path="/data/1903.00780/page1_img1.png",
    analysis={"image_type": "diagram", "title": "Neural network architecture",
              "description": "Encoder decoder neural network with attention layers"},
    image_bytes=make_png((10, 20, 30, 255)),
)
GDP = IndexRecord(
    id="paper_page2_img1",
    pdf_file="paper",
    image_path="/data/paper/page2_img1.png",
    analysis={"image_type": "bar chart", "title": "GDP by country",
              "description": "Bar chart comparing GDP of European countries in 2020"},
)


# --- Contract (runs for every backend) -------------------------------------

def test_empty_store(store):
    assert store.count() == 0
    assert store.search("anything", n_results=3) == []


def test_upsert_roundtrip(store):
    store.upsert(NN)

    assert store.count() == 1
    (hit,) = store.search("neural network", n_results=5)
    assert hit.id == NN.id
    assert hit.pdf_file == NN.pdf_file
    assert hit.image_path == NN.image_path
    assert hit.analysis == NN.analysis
    assert hit.image_bytes == NN.image_bytes
    assert hit.distance is not None


def test_record_without_image(store):
    store.upsert(GDP)
    (hit,) = store.search("GDP", n_results=1)
    assert hit.image_bytes is None


def test_upsert_is_idempotent_and_updates(store):
    store.upsert(NN)
    updated = IndexRecord(**{**NN.__dict__, "analysis": {**NN.analysis, "title": "Updated title"}})
    store.upsert(updated)

    assert store.count() == 1
    (hit,) = store.search("neural network", n_results=5)
    assert hit.analysis["title"] == "Updated title"


def test_search_ranks_relevant_first_and_limits(store):
    store.upsert(NN)
    store.upsert(GDP)

    hits = store.search("bar chart GDP european countries", n_results=2)
    assert [h.id for h in hits] == [GDP.id, NN.id]
    assert hits[0].distance <= hits[1].distance

    assert [h.id for h in store.search("encoder decoder attention", n_results=1)] == [NN.id]


def test_null_fields_from_model_are_stored_as_empty(store):
    record = IndexRecord(id="n", pdf_file="p", image_path="x.png",
                         analysis={"image_type": None, "title": None, "description": "d"})
    assert (record.image_type, record.title) == ("", "")
    store.upsert(record)
    (hit,) = store.search("d")
    assert hit.analysis["title"] is None  # the original analysis is preserved as-is


def test_name_identifies_backend(store):
    assert store.name.startswith(f"{store.backend}:")


# --- Chroma specifics -------------------------------------------------------

def test_chroma_indexes_images_when_enabled(tmp_path, image_file):
    chromadb = pytest.importorskip("chromadb")
    from sci_vizio_retrieval.stores.chroma import ChromaVectorStore
    from tests.conftest import _hash_embedding_function

    class MeanPixelEF(chromadb.EmbeddingFunction):
        def __init__(self):
            pass

        def __call__(self, input):
            return [np.asarray(img, dtype=np.float32).mean(axis=(0, 1)) for img in input]

    store = ChromaVectorStore(tmp_path / "db", index_images=True,
                              text_embedding_function=_hash_embedding_function(),
                              image_embedding_function=MeanPixelEF())
    store.upsert(IndexRecord(**{**NN.__dict__, "image_path": str(image_file)}))

    assert store.image_collection.count() == 1
    assert store.count() == 1


def test_chroma_reads_legacy_layout(tmp_path):
    """Indexes written by the pre-refactor indexer must stay searchable."""
    import base64
    import json

    from sci_vizio_retrieval.stores.chroma import DOC_COLLECTION, ChromaVectorStore
    from tests.conftest import _hash_embedding_function

    ef = _hash_embedding_function()
    import chromadb

    legacy = chromadb.PersistentClient(path=str(tmp_path / "db"))
    legacy.create_collection(DOC_COLLECTION, embedding_function=ef).add(
        ids=["old"],
        documents=[json.dumps(NN.analysis)],
        metadatas=[{"pdf_file": "p", "image_path": "x.png", "image_type": "diagram",
                    "title": "t", "image_data": base64.b64encode(b"img").decode()}],
    )

    store = ChromaVectorStore(tmp_path / "db", index_images=False, text_embedding_function=ef)
    (hit,) = store.search("neural network")
    assert hit.id == "old" and hit.image_bytes == b"img" and hit.analysis == NN.analysis


# --- PostgreSQL specifics ---------------------------------------------------

def test_postgres_rejects_invalid_table_name():
    pytest.importorskip("psycopg")
    from sci_vizio_retrieval.stores.postgres import PostgresVectorStore

    with pytest.raises(ValueError, match="Invalid PostgreSQL table name"):
        PostgresVectorStore("postgresql://unused", FakeEmbedder(), table='x"; DROP TABLE y; --')


@pytest.mark.postgres
def test_postgres_schema_and_name(postgres_store, postgres_dsn):
    import psycopg

    assert postgres_store.name.endswith(f"/{postgres_store.table}")
    assert "@" not in postgres_store.name  # never leak credentials

    with psycopg.connect(postgres_dsn) as conn:
        indexes = {row[0] for row in conn.execute(
            "SELECT indexdef FROM pg_indexes WHERE tablename = %s", (postgres_store.table,))}
    assert any("hnsw" in idx and "vector_cosine_ops" in idx for idx in indexes)


@pytest.mark.postgres
def test_postgres_reopen_existing_table(postgres_store, postgres_dsn, fake_embedder):
    from sci_vizio_retrieval.stores.postgres import PostgresVectorStore

    postgres_store.upsert(NN)
    with PostgresVectorStore(postgres_dsn, fake_embedder, table=postgres_store.table) as reopened:
        assert reopened.count() == 1


@pytest.mark.postgres
def test_postgres_refuses_dimension_mismatch(postgres_store, postgres_dsn):
    from sci_vizio_retrieval.stores.postgres import PostgresVectorStore

    with pytest.raises(ValueError, match="64-dimensional"):
        PostgresVectorStore(postgres_dsn, FakeEmbedder(dimension=32), table=postgres_store.table)


@pytest.mark.postgres
def test_postgres_refuses_mixing_models(postgres_store, postgres_dsn):
    from sci_vizio_retrieval.stores.postgres import PostgresVectorStore

    postgres_store.upsert(NN)
    with pytest.raises(ValueError, match="already contains embeddings"):
        PostgresVectorStore(postgres_dsn, FakeEmbedder(model_name="other"), table=postgres_store.table)


# --- Factory ----------------------------------------------------------------

def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown storage backend"):
        create_vector_store("mongo")


def test_factory_builds_chroma(tmp_path):
    pytest.importorskip("chromadb")
    from sci_vizio_retrieval.stores.chroma import ChromaVectorStore

    store = create_vector_store("chroma", chroma_path=str(tmp_path / "c"), index_images=False)
    assert isinstance(store, ChromaVectorStore)
    assert store.image_collection is None


@pytest.mark.postgres
def test_factory_builds_postgres(monkeypatch, postgres_dsn, postgres_table):
    import sci_vizio_retrieval.embeddings as embeddings
    from sci_vizio_retrieval.stores.postgres import PostgresVectorStore

    monkeypatch.setattr(embeddings, "SentenceTransformerEmbedder", lambda model_name: FakeEmbedder())
    with create_vector_store("POSTGRES", postgres_dsn=postgres_dsn, postgres_table=postgres_table) as store:
        assert isinstance(store, PostgresVectorStore)
        assert store.table == postgres_table
