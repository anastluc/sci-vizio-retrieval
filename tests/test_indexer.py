import sqlite3

import pytest

from sci_vizio_retrieval.config import CHROMA_PATH
from sci_vizio_retrieval.indexer import ImageAnalysisIndexer, extract_and_validate_json
from sci_vizio_retrieval.processor import ImageProcessor
from sci_vizio_retrieval.stores.chroma import chroma_store_name
from tests.conftest import InMemoryStore, analysis_json, make_png

VALID = analysis_json("Transformer", "Encoder decoder architecture")


@pytest.mark.parametrize("text", [
    VALID,
    f"Sure! Here is the JSON:\n```json\n{VALID}\n```",
])
def test_extract_valid_json(text):
    ok, obj, err = extract_and_validate_json(text)
    assert ok and err is None
    assert obj["title"] == "Transformer"


@pytest.mark.parametrize("text, message", [
    (None, "Empty response"),
    ("", "Empty response"),
    ("no json here", "No JSON object"),
    ("{not: valid}", "JSON parsing error"),
    ('{"title": "t"}', "Missing required fields: image_type, description"),
])
def test_extract_invalid_json(text, message):
    ok, obj, err = extract_and_validate_json(text)
    assert not ok and obj is None
    assert message in err


@pytest.fixture
def analyses_db(tmp_path):
    """A pipeline DB with 2 valid analyses, 1 unparsable one and 1 failed API call."""
    db = tmp_path / "state.db"
    ImageProcessor(db_path=str(db), images_dir=tmp_path, output_dir=tmp_path / "out",
                   vision_client=object(), embed_images=False)

    def add(pdf, name, success, response):
        path = tmp_path / "images" / pdf / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(make_png())
        with sqlite3.connect(db) as conn:
            conn.execute(
                "INSERT INTO image_processing (pdf_file, timestamp, image, image_path, success_status, response) "
                "VALUES (?, 'now', ?, ?, ?, ?)", (pdf, name, str(path), success, response))
        return path

    add("paperA", "page1_img1.png", True, VALID)
    add("paperA", "page2_img1.png", True, analysis_json("GDP", "Bar chart of GDP", "bar chart"))
    add("paperB", "page1_img1.png", True, "the model rambled instead of answering")
    add("paperB", "page2_img1.png", False, None)
    return db


def test_indexes_valid_analyses(analyses_db, memory_store):
    stats = ImageAnalysisIndexer(db_path=str(analyses_db), vector_store=memory_store).process_all_analyses()

    assert stats == {"total_processed": 3, "successful_validations": 2, "successful_indexing": 2, "failed": 1}
    assert set(memory_store.records) == {"paperA_page1_img1", "paperA_page2_img1"}
    record = memory_store.records["paperA_page1_img1"]
    assert record.analysis["title"] == "Transformer"
    assert record.image_bytes == make_png()


def test_second_run_skips_indexed(analyses_db, memory_store):
    indexer = ImageAnalysisIndexer(db_path=str(analyses_db), vector_store=memory_store)
    indexer.process_all_analyses()

    # Only the unparsable analysis is retried
    assert indexer.process_all_analyses()["total_processed"] == 1


def test_tracking_is_per_store(analyses_db):
    """Switching backend (e.g. chroma -> postgres) must index everything into the new store."""
    ImageAnalysisIndexer(db_path=str(analyses_db), vector_store=InMemoryStore("chroma:x")).process_all_analyses()

    other = InMemoryStore("postgres:localhost:5432/db/t")
    stats = ImageAnalysisIndexer(db_path=str(analyses_db), vector_store=other).process_all_analyses()
    assert stats["successful_indexing"] == 2
    assert other.count() == 2


def test_reindex_processes_everything(analyses_db, memory_store):
    indexer = ImageAnalysisIndexer(db_path=str(analyses_db), vector_store=memory_store)
    indexer.process_all_analyses()
    assert indexer.process_all_analyses(reindex=True)["total_processed"] == 3


def test_pdf_filter(analyses_db, memory_store):
    stats = ImageAnalysisIndexer(db_path=str(analyses_db), vector_store=memory_store) \
        .process_all_analyses(pdf_names=["paperB"])
    assert stats["total_processed"] == 1
    assert memory_store.count() == 0


def test_store_failure_is_recorded_and_retried(analyses_db):
    failing = InMemoryStore(fail_ids={"paperA_page1_img1"})
    indexer = ImageAnalysisIndexer(db_path=str(analyses_db), vector_store=failing)

    stats = indexer.process_all_analyses()
    assert stats["successful_indexing"] == 1 and stats["failed"] == 2

    with sqlite3.connect(analyses_db) as conn:
        (error,) = conn.execute(
            "SELECT error_message FROM vector_indexing WHERE image_path LIKE '%paperA%page1_img1.png'").fetchone()
    assert error == "Failed to index in memory"

    failing.fail_ids.clear()
    assert indexer.process_all_analyses()["successful_indexing"] == 1


def test_missing_image_file_fails_gracefully(analyses_db, memory_store, tmp_path):
    (tmp_path / "images" / "paperA" / "page1_img1.png").unlink()
    stats = ImageAnalysisIndexer(db_path=str(analyses_db), vector_store=memory_store).process_all_analyses()
    assert stats["successful_indexing"] == 1


def test_no_analyses_table(tmp_path, memory_store):
    stats = ImageAnalysisIndexer(db_path=str(tmp_path / "empty.db"), vector_store=memory_store) \
        .process_all_analyses()
    assert stats["total_processed"] == 0


def _add_legacy_tracking(db):
    with sqlite3.connect(db) as conn:
        conn.execute("""CREATE TABLE json_indexing (id INTEGER PRIMARY KEY, pdf_file TEXT, image_path TEXT,
                        index_status BOOLEAN, timestamp TEXT, error_message TEXT, UNIQUE(pdf_file, image_path))""")
        conn.execute("""INSERT INTO json_indexing (pdf_file, image_path, index_status, timestamp)
                        SELECT pdf_file, image_path, TRUE, 'then' FROM image_processing
                        WHERE success_status = TRUE AND pdf_file = 'paperA'""")


def test_legacy_chroma_tracking_is_migrated(analyses_db):
    _add_legacy_tracking(analyses_db)
    store = InMemoryStore(chroma_store_name(CHROMA_PATH))

    stats = ImageAnalysisIndexer(db_path=str(analyses_db), vector_store=store).process_all_analyses()
    assert stats["total_processed"] == 1  # only paperB's unindexed analysis


def test_legacy_tracking_not_applied_to_other_stores(analyses_db):
    _add_legacy_tracking(analyses_db)
    stats = ImageAnalysisIndexer(db_path=str(analyses_db), vector_store=InMemoryStore("postgres:x")) \
        .process_all_analyses()
    assert stats["total_processed"] == 3


def test_end_to_end_with_real_backend(analyses_db, store):
    """Indexer + each real backend: what gets indexed is searchable."""
    stats = ImageAnalysisIndexer(db_path=str(analyses_db), vector_store=store).process_all_analyses()
    assert stats["successful_indexing"] == 2

    (hit,) = store.search("bar chart GDP", n_results=1)
    assert hit.id == "paperA_page2_img1"
    assert hit.image_bytes == make_png()
