import pytest

from sci_vizio_retrieval.stores import IndexRecord, SearchResult
from sci_vizio_retrieval.ui import (
    SearchService,
    create_result_html,
    load_result_image,
    pdf_link,
    query_and_display,
)
from tests.conftest import InMemoryStore, make_png


@pytest.fixture
def service():
    store = InMemoryStore()
    store.upsert(IndexRecord(id="a", pdf_file="1912.08777v2", image_path="/nope.png",
                             analysis={"image_type": "graph", "title": "Loss curve",
                                       "description": "training loss <script>alert(1)</script>"},
                             image_bytes=make_png()))
    store.upsert(IndexRecord(id="b", pdf_file="my-paper", image_path="/nope.png",
                             analysis={"image_type": "table", "title": "Results", "description": "accuracy"}))
    return SearchService(store)


@pytest.mark.parametrize("name, link", [
    ("1903.00780", "https://arxiv.org/abs/1903.00780"),
    ("1912.08777v2", "https://arxiv.org/abs/1912.08777v2"),
    ("my-paper", None),
])
def test_pdf_link(name, link):
    assert pdf_link(name) == link


def test_query_database_formats_results(service):
    first, second = service.query_database("training loss curve", 2)
    assert first["id"] == "a"
    assert first["pdf_link"] == "https://arxiv.org/abs/1912.08777v2"
    assert first["image"].size == (32, 24)
    assert '"title": "Loss curve"' in first["json_content"]
    assert second["image"] is None


def test_result_html_escapes_model_output(service):
    (result,) = service.query_database("training loss", 1)
    html = create_result_html(result)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert 'href="https://arxiv.org/abs/1912.08777v2"' in html


def test_result_html_without_link():
    html = create_result_html({"pdf_file": "x<y", "pdf_link": None, "json_content": "{}"})
    assert "x&lt;y" in html and "<a " not in html


def test_query_and_display(service):
    images, html = query_and_display(service, "accuracy results", 2)
    assert len(images) == 1  # only one hit has an image
    assert html.count("<h3>") == 2
    assert query_and_display(service, "   ", 2) == ([], "")


def test_load_result_image_falls_back_to_disk(image_file):
    hit = SearchResult(id="x", pdf_file="p", image_path=str(image_file), analysis={})
    assert load_result_image(hit) is not None
    assert load_result_image(SearchResult(id="x", pdf_file="p", image_path="", analysis={},
                                          image_bytes=b"garbage")) is None


def test_create_demo_builds(memory_store):
    pytest.importorskip("gradio")
    from sci_vizio_retrieval.ui import create_demo

    assert create_demo(memory_store) is not None
