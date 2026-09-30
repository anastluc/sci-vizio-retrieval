"""
Fixtures for the SaaS backend tests.

They need PostgreSQL with pgvector (SCI_TEST_POSTGRES_DSN): each test session
creates a throwaway database, migrates it with Alembic, and drops it at the end.
"""

import io
import os
import uuid
from urllib.parse import urlsplit, urlunsplit

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("psycopg")

import psycopg  # noqa: E402
import pymupdf  # noqa: E402
from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402
from sqlalchemy import text  # noqa: E402

from sci_vizio_retrieval.client import VisionResponse  # noqa: E402
from sci_vizio_saas.auth import InvalidToken  # noqa: E402
from sci_vizio_saas.db import make_engine, make_session_factory  # noqa: E402
from sci_vizio_saas.main import create_app  # noqa: E402
from sci_vizio_saas.services import Services  # noqa: E402
from sci_vizio_saas.settings import Settings  # noqa: E402
from sci_vizio_saas.storage import LocalStorage  # noqa: E402
from tests.conftest import POSTGRES_DSN_ENV, FakeEmbedder, analysis_json  # noqa: E402

ALEMBIC_INI = os.path.join(os.path.dirname(__file__), "..", "..", "sci_vizio_saas", "alembic.ini")

# Figure colour -> what the fake vision model "sees".
COLOR_ANALYSES = {
    (220, 30, 30): ("bar chart", "Accuracy by model", "Bar chart comparing accuracy of neural models"),
    (30, 30, 220): ("architecture diagram", "Transformer architecture",
                    "Encoder decoder transformer with attention layers"),
    (30, 180, 30): ("logo", "Company logo", "A green logo"),
    (240, 200, 0): ("scatter plot", "Loss versus epochs", "Scatter plot of training loss over epochs"),
}
RED, BLUE, GREEN, YELLOW = COLOR_ANALYSES


def make_pdf(colors=(RED, BLUE), text_title="Test paper", size=(300, 200)) -> bytes:
    """A one-page-per-figure PDF whose images are solid colours."""
    doc = pymupdf.open()
    doc.set_metadata({"title": text_title})
    for color in colors:
        page = doc.new_page()
        page.insert_text((72, 72), f"Figure in {color}")
        buf = io.BytesIO()
        Image.new("RGB", size, color).save(buf, format="PNG")
        page.insert_image(pymupdf.Rect(72, 100, 372, 300), stream=buf.getvalue())
    data = doc.tobytes()
    doc.close()
    return data


class FakeIdentity:
    """Accepts tokens of the form ``token:<uid>[:<email>]``."""

    def __init__(self):
        self.deleted = []

    def verify(self, id_token):
        parts = id_token.split(":")
        if parts[0] != "token" or len(parts) < 2:
            raise InvalidToken("bad token")
        return {"uid": parts[1], "email": parts[2] if len(parts) > 2 else f"{parts[1]}@example.com"}

    def delete_user(self, uid):
        self.deleted.append(uid)


class FakeVision:
    def __init__(self):
        self.calls = 0
        self.fail = False

    def analyze_image_bytes(self, data, mime_type, prompt):
        self.calls += 1
        if self.fail:
            raise RuntimeError("503 upstream overloaded")
        with Image.open(io.BytesIO(data)) as img:
            color = img.convert("RGB").getpixel((0, 0))
        kind, title, description = COLOR_ANALYSES.get(color, ("photo", "Unknown", "Something else"))
        return VisionResponse(text=analysis_json(title, description, kind))


class FakeBilling:
    def __init__(self):
        self.customers = []
        self.cancelled = []
        self.subscriptions = {}
        self.valid_signature = "sig-ok"

    def create_customer(self, email, user_id):
        self.customers.append(user_id)
        return f"cus_{len(self.customers)}"

    def create_checkout_url(self, customer_id, user_id, success_url, cancel_url):
        return f"https://checkout.test/{customer_id}?success={success_url}"

    def create_portal_url(self, customer_id, return_url):
        return f"https://portal.test/{customer_id}"

    def parse_webhook(self, payload, signature):
        import json

        if signature != self.valid_signature:
            raise ValueError("bad signature")
        return json.loads(payload)

    def get_subscription(self, subscription_id):
        return self.subscriptions[subscription_id]

    def cancel_subscription(self, subscription_id):
        self.cancelled.append(subscription_id)


def _with_database(dsn: str, name: str) -> str:
    parts = urlsplit(dsn)
    return urlunsplit(parts._replace(path=f"/{name}"))


@pytest.fixture(scope="session")
def database_url():
    dsn = os.environ.get(POSTGRES_DSN_ENV)
    if not dsn:
        pytest.skip(f"set {POSTGRES_DSN_ENV} to run the SaaS backend tests")
    name = f"saas_test_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    url = _with_database(dsn, name)

    config = Config(ALEMBIC_INI)
    config.set_main_option("sqlalchemy.url", Settings(database_url=url).database_url)
    command.upgrade(config, "head")

    yield url

    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture(scope="session")
def engine(database_url):
    engine = make_engine(Settings(database_url=database_url).database_url)
    yield engine
    engine.dispose()


@pytest.fixture
def session_factory(engine):
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE users, stripe_events, jobs RESTART IDENTITY CASCADE"))
    return make_session_factory(engine)


@pytest.fixture
def db(session_factory):
    with session_factory() as session:
        yield session


@pytest.fixture
def settings(database_url, tmp_path):
    return Settings(
        _env_file=None,
        environment="test",
        database_url=database_url,
        storage_backend="local",
        local_storage_dir=str(tmp_path / "storage"),
        public_api_url="http://testserver",
        signing_secret="test-secret",
        free_max_documents=2,
        free_max_file_mb=1,
        free_max_figures_per_document=3,
        premium_max_documents=5,
        premium_max_file_mb=5,
        premium_max_figures_per_document=10,
        figure_concurrency=2,
        min_figure_side=50,
    )


@pytest.fixture
def fakes():
    return {"identity": FakeIdentity(), "vision": FakeVision(), "billing": FakeBilling(),
            "embedder": FakeEmbedder(dimension=384)}


@pytest.fixture
def services(settings, fakes):
    storage = LocalStorage(settings.local_storage_dir, settings.public_api_url, settings.signing_secret)
    return Services(settings, storage=storage, **fakes)


@pytest.fixture
def client(settings, services, session_factory):
    with TestClient(create_app(settings, services, session_factory)) as c:
        yield c


def auth(uid="alice", email=None):
    token = f"token:{uid}" + (f":{email}" if email else "")
    return {"Authorization": f"Bearer {token}"}


ALICE = auth("alice")
BOB = auth("bob")


def start_upload(client, pdf: bytes, headers=ALICE, filename="paper.pdf", **extra):
    import hashlib

    body = {"filename": filename, "size_bytes": len(pdf), "sha256": hashlib.sha256(pdf).hexdigest(), **extra}
    return client.post("/api/documents/uploads", json=body, headers=headers)


def put_signed(client, url: str, data: bytes, headers=None):
    path = url.replace("http://testserver", "")
    return client.put(path, content=data, headers=headers or {})


def upload(client, pdf: bytes, headers=ALICE, filename="paper.pdf", **extra) -> dict:
    """Full browser flow: reserve, PUT to the signed URL, complete. Returns the document."""
    resp = start_upload(client, pdf, headers, filename, **extra)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert put_signed(client, body["upload"]["url"], pdf, body["upload"]["headers"]).status_code == 200
    done = client.post(f"/api/documents/{body['document']['id']}/complete", headers=headers)
    assert done.status_code == 202, done.text
    return done.json()


def drain(session_factory, services, limit=20):
    """Run the worker until the queue is empty."""
    from sci_vizio_saas.worker import run_once

    for _ in range(limit):
        if not run_once(session_factory, services):
            return
