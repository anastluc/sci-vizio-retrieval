from sqlalchemy import select, update

from sci_vizio_saas.models import Document, Figure, Job
from tests.saas.conftest import (
    ALICE,
    BLUE,
    BOB,
    GREEN,
    RED,
    YELLOW,
    drain,
    make_pdf,
    put_signed,
    start_upload,
    upload,
)


def test_requires_authentication(client):
    assert client.get("/api/documents").status_code == 401
    assert client.get("/api/documents", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_upload_process_and_list(client, session_factory, services, fakes):
    document = upload(client, make_pdf([RED, BLUE, GREEN]))
    assert document["status"] == "queued"

    drain(session_factory, services)

    doc = client.get(f"/api/documents/{document['id']}", headers=ALICE).json()
    assert doc["status"] == "ready"
    assert doc["page_count"] == 3 and doc["figure_count"] == 3 and doc["title"] == "Test paper"

    figures = client.get(f"/api/documents/{document['id']}/figures", headers=ALICE).json()
    assert [f["figure_type"] for f in figures] == ["Bar chart", "Diagram", "Decorative"]
    image = client.get(figures[0]["image_url"].replace("http://testserver", ""))
    assert image.status_code == 200 and image.content.startswith(b"\x89PNG")

    listing = client.get("/api/documents", headers=ALICE).json()
    assert listing["total"] == 1 and listing["items"][0]["id"] == document["id"]


def test_figures_are_deduplicated_and_capped(client, session_factory, services, fakes):
    # Plan allows 3 figures per document; the repeated red image is analysed once.
    document = upload(client, make_pdf([RED, RED, BLUE, GREEN, YELLOW]))
    drain(session_factory, services)

    doc = client.get(f"/api/documents/{document['id']}", headers=ALICE).json()
    assert doc["figure_count"] == 3 and doc["skipped_figure_count"] == 1
    assert fakes["vision"].calls == 3


def test_small_images_are_ignored(client, session_factory, services, fakes):
    document = upload(client, make_pdf([RED], size=(20, 20)))
    drain(session_factory, services)
    doc = client.get(f"/api/documents/{document['id']}", headers=ALICE).json()
    assert doc["status"] == "ready" and doc["figure_count"] == 0 and fakes["vision"].calls == 0


def test_free_plan_document_limit(client):
    upload(client, make_pdf([RED]))
    upload(client, make_pdf([BLUE]))
    resp = start_upload(client, make_pdf([GREEN]))
    assert resp.status_code == 402
    assert resp.json()["detail"]["code"] == "document_limit"


def test_file_size_limit_at_start(client):
    resp = client.post("/api/documents/uploads", headers=ALICE,
                       json={"filename": "big.pdf", "size_bytes": 2 * 1024 * 1024})
    assert resp.status_code == 413 and resp.json()["detail"]["code"] == "file_too_large"


def test_file_size_enforced_after_upload(client, db):
    """The signed URL can't cap the size, so a lying client is caught at completion."""
    small = make_pdf([RED])
    body = start_upload(client, small).json()
    put_signed(client, body["upload"]["url"], small + b"\0" * (2 * 1024 * 1024))

    resp = client.post(f"/api/documents/{body['document']['id']}/complete", headers=ALICE)
    assert resp.status_code == 413
    assert db.scalar(select(Document)) is None


def test_complete_before_upload(client):
    body = start_upload(client, make_pdf([RED])).json()
    resp = client.post(f"/api/documents/{body['document']['id']}/complete", headers=ALICE)
    assert resp.status_code == 409


def test_duplicate_upload_rejected(client):
    pdf = make_pdf([RED])
    upload(client, pdf)
    resp = start_upload(client, pdf)
    assert resp.status_code == 409 and resp.json()["detail"]["code"] == "duplicate"


def test_non_pdf_is_rejected_without_retry(client, session_factory, services, db):
    data = b"hello, not a pdf"
    body = client.post("/api/documents/uploads", headers=ALICE,
                       json={"filename": "x.pdf", "size_bytes": len(data)}).json()
    put_signed(client, body["upload"]["url"], data)
    client.post(f"/api/documents/{body['document']['id']}/complete", headers=ALICE)

    drain(session_factory, services)

    doc = client.get(f"/api/documents/{body['document']['id']}", headers=ALICE).json()
    assert doc["status"] == "failed" and doc["error"] == "The file is not a PDF."
    assert db.scalar(select(Job)).attempts == 1


def test_checksum_mismatch_rejected(client, session_factory, services):
    body = start_upload(client, make_pdf([RED])).json()
    put_signed(client, body["upload"]["url"], make_pdf([BLUE]))
    client.post(f"/api/documents/{body['document']['id']}/complete", headers=ALICE)
    drain(session_factory, services)
    doc = client.get(f"/api/documents/{body['document']['id']}", headers=ALICE).json()
    assert doc["status"] == "failed" and "checksum" in doc["error"]


def test_vision_outage_retries_then_fails(client, session_factory, services, fakes, db):
    fakes["vision"].fail = True
    document = upload(client, make_pdf([RED]))

    for attempt in range(3):
        db.execute(update(Job).values(run_after=Job.created_at))  # skip the backoff wait
        db.commit()
        drain(session_factory, services)
        doc = client.get(f"/api/documents/{document['id']}", headers=ALICE).json()
        assert doc["status"] == ("queued" if attempt < 2 else "failed")

    # The user can retry once the outage is over.
    fakes["vision"].fail = False
    assert client.post(f"/api/documents/{document['id']}/retry", headers=ALICE).status_code == 202
    drain(session_factory, services)
    assert client.get(f"/api/documents/{document['id']}", headers=ALICE).json()["status"] == "ready"


def test_reprocessing_replaces_figures(client, session_factory, services, db):
    document = upload(client, make_pdf([RED, BLUE]))
    drain(session_factory, services)

    from sci_vizio_saas.processing import process_document

    process_document(db, services, document["id"])
    assert len(db.scalars(select(Figure)).all()) == 2


def test_delete_removes_files(client, session_factory, services, settings):
    from pathlib import Path

    document = upload(client, make_pdf([RED]))
    drain(session_factory, services)
    storage_root = Path(settings.local_storage_dir)
    assert any(storage_root.rglob("*.png"))

    assert client.delete(f"/api/documents/{document['id']}", headers=ALICE).status_code == 204
    assert client.get(f"/api/documents/{document['id']}", headers=ALICE).status_code == 404
    assert not any(p.is_file() for p in storage_root.rglob("*"))


def test_download_url(client):
    pdf = make_pdf([RED])
    document = upload(client, pdf, filename="my paper.pdf")
    url = client.get(f"/api/documents/{document['id']}/download", headers=ALICE).json()["url"]
    resp = client.get(url.replace("http://testserver", ""))
    assert resp.content == pdf
    assert "my%20paper.pdf" in resp.headers["content-disposition"]


def test_other_users_cannot_access(client):
    document = upload(client, make_pdf([RED]))
    for method, path in [("get", ""), ("delete", ""), ("get", "/figures"), ("get", "/download")]:
        assert getattr(client, method)(f"/api/documents/{document['id']}{path}", headers=BOB).status_code == 404
    assert client.get("/api/documents", headers=BOB).json()["total"] == 0


def test_collections_and_filters(client):
    thesis = client.post("/api/collections", json={"name": "Thesis"}, headers=ALICE).json()
    in_thesis = upload(client, make_pdf([RED]), filename="alpha.pdf", collection_id=thesis["id"])
    unfiled = upload(client, make_pdf([BLUE]), filename="beta.pdf")

    def ids(**params):
        return [d["id"] for d in client.get("/api/documents", params=params, headers=ALICE).json()["items"]]

    assert ids(collection=thesis["id"]) == [in_thesis["id"]]
    assert ids(collection="unfiled") == [unfiled["id"]]
    assert ids(q="bet") == [unfiled["id"]]
    assert ids(status="queued") and not ids(status="ready")

    # Move, then delete the collection: its documents become unfiled.
    client.patch(f"/api/documents/{unfiled['id']}", json={"collection_id": thesis["id"]}, headers=ALICE)
    collections = client.get("/api/collections", headers=ALICE).json()
    assert collections[0]["document_count"] == 2
    assert client.delete(f"/api/collections/{thesis['id']}", headers=ALICE).status_code == 204
    assert len(ids(collection="unfiled")) == 2


def test_cannot_file_into_someone_elses_collection(client):
    bobs = client.post("/api/collections", json={"name": "Bob's"}, headers=BOB).json()
    resp = start_upload(client, make_pdf([RED]), collection_id=bobs["id"])
    assert resp.status_code == 404
