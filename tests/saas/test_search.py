import pytest

from tests.saas.conftest import ALICE, BLUE, BOB, GREEN, RED, YELLOW, drain, make_pdf, upload


@pytest.fixture
def corpus(client, session_factory, services):
    """Alice: 'Thesis' collection with a chart+diagram+logo paper, and an unfiled scatter paper."""
    thesis = client.post("/api/collections", json={"name": "Thesis"}, headers=ALICE).json()
    first = upload(client, make_pdf([RED, BLUE, GREEN]), filename="first.pdf", collection_id=thesis["id"])
    second = upload(client, make_pdf([YELLOW], text_title=""), filename="second.pdf")
    upload(client, make_pdf([RED]), headers=BOB, filename="bobs.pdf")
    drain(session_factory, services)
    return {"thesis": thesis, "first": first, "second": second}


def search(client, **body):
    resp = client.post("/api/search", json=body, headers=ALICE)
    assert resp.status_code == 200, resp.text
    return resp.json()


def titles(result):
    return [r["title"] for r in result["results"]]


def test_semantic_ranking(client, corpus):
    result = search(client, query="transformer attention encoder decoder")
    assert titles(result)[0] == "Transformer architecture"
    assert result["results"][0]["score"] > result["results"][-1]["score"]

    assert titles(search(client, query="accuracy bar chart"))[0] == "Accuracy by model"


def test_decorative_hidden_by_default(client, corpus):
    assert "Company logo" not in titles(search(client))
    assert search(client)["total"] == 3
    assert "Company logo" in titles(search(client, include_decorative=True))
    assert titles(search(client, figure_types=["Decorative"])) == ["Company logo"]


def test_only_own_figures(client, corpus):
    result = search(client, query="accuracy", include_decorative=True)
    assert {r["document"]["filename"] for r in result["results"]} == {"first.pdf", "second.pdf"}


def test_filters(client, corpus):
    assert titles(search(client, collections=["unfiled"])) == ["Loss versus epochs"]
    assert set(titles(search(client, collections=[corpus["thesis"]["id"]]))) == \
        {"Accuracy by model", "Transformer architecture"}
    assert titles(search(client, documents=[corpus["second"]["id"]])) == ["Loss versus epochs"]
    assert titles(search(client, figure_types=["Bar chart", "Scatter plot"], query="loss")) == \
        ["Loss versus epochs", "Accuracy by model"]


def test_facets_ignore_their_own_filter(client, corpus):
    facets = search(client, figure_types=["Bar chart"])["facets"]
    # Type counts reflect other filters only, so other types stay selectable.
    assert {f["value"]: f["count"] for f in facets["figure_types"]} == \
        {"Bar chart": 1, "Diagram": 1, "Decorative": 1, "Scatter plot": 1}
    # Collection counts do honour the type filter.
    assert {f["label"]: f["count"] for f in facets["collections"]} == {"Thesis": 1}
    assert {f["label"] for f in search(client)["facets"]["collections"]} == {"Thesis", "Unfiled"}
    assert {f["label"] for f in search(client)["facets"]["documents"]} == {"Test paper", "second.pdf"}


def test_browse_and_pagination(client, corpus):
    page1 = search(client, limit=2)
    page2 = search(client, limit=2, offset=2)
    assert page1["total"] == 3 and len(page1["results"]) == 2 and len(page2["results"]) == 1
    assert page1["results"][0]["score"] is None
    assert not set(titles(page1)) & set(titles(page2))


def test_invalid_filters(client, corpus):
    assert client.post("/api/search", json={"collections": ["nope"]}, headers=ALICE).status_code == 422
    assert client.post("/api/search", json={"limit": 1000}, headers=ALICE).status_code == 422
