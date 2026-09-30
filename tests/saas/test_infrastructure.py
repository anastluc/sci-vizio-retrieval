"""Job queue, storage backends, settings, migrations and processing helpers."""

import io
import time
from datetime import datetime, timedelta, timezone

import pytest
from PIL import Image
from sqlalchemy import update

from sci_vizio_retrieval.extractor import ExtractedImage
from sci_vizio_saas import jobs
from sci_vizio_saas.models import Document, Job, JobStatus, User
from sci_vizio_saas.processing import to_web_image
from sci_vizio_saas.settings import Settings
from sci_vizio_saas.storage import LocalStorage, S3Storage
from sci_vizio_saas.worker import run_maintenance

# --- Job queue ---------------------------------------------------------------


def _document(db, **kwargs):
    user = User(firebase_uid=f"u{time.monotonic_ns()}")
    db.add(user)
    db.flush()
    doc = Document(user_id=user.id, filename="f.pdf", storage_key=f"k{time.monotonic_ns()}", size_bytes=1,
                   **kwargs)
    db.add(doc)
    db.commit()
    return doc


def test_jobs_claimed_once_in_order(session_factory, db):
    first, second = _document(db), _document(db)
    jobs.enqueue(db, first.id)
    jobs.enqueue(db, second.id)
    db.commit()

    with session_factory() as a, session_factory() as b:
        job_a = jobs.claim_next(a)
        job_b = jobs.claim_next(b)
        assert job_a.document_id == first.id and job_b.document_id == second.id
        assert jobs.claim_next(a) is None
        assert job_a.status == JobStatus.RUNNING and job_a.attempts == 1


def test_skip_locked_under_concurrency(session_factory, db):
    """A job locked by one worker's open transaction is invisible to others."""
    from sqlalchemy import select

    jobs.enqueue(db, _document(db).id)
    db.commit()
    with session_factory() as a, session_factory() as b:
        a.scalars(select(Job).with_for_update()).first()  # hold the row lock
        assert jobs.claim_next(b) is None


def test_retry_backoff_and_final_failure(session_factory, db):
    jobs.enqueue(db, _document(db).id, max_attempts=2)
    db.commit()

    job = jobs.claim_next(db)
    assert jobs.mark_failed(db, job, "boom", retryable=True) is True
    assert job.run_after > datetime.now(timezone.utc) + timedelta(seconds=20)
    assert jobs.claim_next(db) is None  # still backing off

    db.execute(update(Job).values(run_after=datetime.now(timezone.utc)))
    db.commit()
    job = jobs.claim_next(db)
    assert jobs.mark_failed(db, job, "boom", retryable=True) is False
    assert job.status == JobStatus.FAILED and job.last_error == "boom"


def test_maintenance_requeues_stale_and_removes_abandoned(session_factory, services, db):
    jobs.enqueue(db, _document(db, status="queued").id)
    db.commit()
    job = jobs.claim_next(db)
    job.locked_at = datetime.now(timezone.utc) - timedelta(hours=1)
    abandoned = _document(db, status="uploading")
    abandoned.created_at = datetime.now(timezone.utc) - timedelta(hours=3)
    abandoned_id, abandoned_key = abandoned.id, abandoned.storage_key
    services.storage.put(abandoned_key, b"partial", "application/pdf")
    db.commit()

    run_maintenance(session_factory, services)

    db.expire_all()
    assert db.get(Job, job.id).status == JobStatus.QUEUED
    assert db.get(Document, abandoned_id) is None
    assert services.storage.size(abandoned_key) is None


# --- Storage -----------------------------------------------------------------


def test_local_storage_signatures(tmp_path):
    storage = LocalStorage(tmp_path, "http://api", "secret")
    url = storage.presign_download("users/u/x.pdf", expires=60)
    assert url.startswith("http://api/api/files/users/u/x.pdf?")
    expires = int(time.time()) + 60
    sig = storage.sign("GET", "k", expires)
    assert storage.verify("GET", "k", expires, sig)
    assert not storage.verify("PUT", "k", expires, sig)           # method-bound
    assert not storage.verify("GET", "other", expires, sig)       # key-bound
    assert not storage.verify("GET", "k", int(time.time()) - 1, storage.sign("GET", "k", int(time.time()) - 1))
    with pytest.raises(ValueError):
        storage.path_for("../../etc/passwd")


def test_signed_url_route_rejects_tampering(client, services):
    url = services.storage.presign_upload("users/x/doc.pdf", "application/pdf").url.replace("http://testserver", "")
    assert client.put(url.replace("doc.pdf", "evil.pdf"), content=b"x").status_code == 403
    assert client.put(url, content=b"%PDF-").status_code == 200
    assert client.get(url).status_code == 403  # PUT signature can't be used to read


def test_s3_storage_with_moto():
    moto = pytest.importorskip("moto")
    import boto3
    from botocore.config import Config

    with moto.mock_aws():
        client = boto3.client("s3", region_name="us-east-1", config=Config(signature_version="s3v4"))
        client.create_bucket(Bucket="figs")
        storage = S3Storage("figs", client=client)

        storage.put("users/a/doc/1.png", b"png", "image/png")
        storage.put("users/a/doc/2.png", b"png2", "image/png")
        storage.put("users/b/doc/1.png", b"other", "image/png")
        assert storage.size("users/a/doc/1.png") == 3 and storage.size("missing") is None
        assert storage.get("users/a/doc/2.png") == b"png2"

        upload = storage.presign_upload("users/a/new.pdf", "application/pdf")
        assert upload.method == "PUT" and "X-Amz-Signature" in upload.url
        assert "response-content-disposition" in storage.presign_download("users/a/doc/1.png", filename="a b.png")

        storage.delete_prefix("users/a/")
        assert storage.size("users/a/doc/1.png") is None and storage.size("users/b/doc/1.png") == 5


# --- Settings ----------------------------------------------------------------


def test_heroku_database_url_is_normalised():
    assert Settings(_env_file=None, database_url="postgres://u:p@h:5432/d").database_url == \
        "postgresql+psycopg://u:p@h:5432/d"


def test_production_settings_are_validated(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="FIREBASE_PROJECT_ID.*STORAGE_BACKEND=local.*OPENROUTER_API_KEY"):
        Settings(_env_file=None, environment="production")
    ok = Settings(_env_file=None, environment="production", firebase_project_id="p", storage_backend="s3",
                  s3_bucket="b", openrouter_api_key="k", cors_origins="https://a.web.app, https://b.com")
    assert ok.cors_origin_list == ["https://a.web.app", "https://b.com"]
    assert not ok.billing_enabled


# --- Migrations --------------------------------------------------------------


def test_migrations_match_models(engine):
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from sci_vizio_saas.db import Base

    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == []


# --- Processing helpers ------------------------------------------------------


def _image(ext, data, mode="RGB"):
    return ExtractedImage(page=1, index=1, ext=ext, data=data, width=10, height=10)


def test_to_web_image():
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), "red").save(buf, format="TIFF")
    data, ext, mime = to_web_image(_image("tiff", buf.getvalue()))
    assert (ext, mime) == ("png", "image/png") and data.startswith(b"\x89PNG")

    assert to_web_image(_image("jpeg", b"raw"))[1:] == ("jpeg", "image/jpeg")
    assert to_web_image(_image("jb2", b"not decodable")) is None
