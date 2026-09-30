import json
from pathlib import Path

from sci_vizio_saas.models import User
from tests.saas.conftest import ALICE, BLUE, GREEN, RED, auth, drain, make_pdf, start_upload, upload


def webhook(client, event, signature="sig-ok"):
    return client.post("/api/billing/webhook", content=json.dumps(event),
                       headers={"Stripe-Signature": signature, "Content-Type": "application/json"})


def subscription(status="active", sub_id="sub_1", customer="cus_1"):
    # Newer Stripe API versions put the billing period on the items.
    return {"id": sub_id, "customer": customer, "status": status, "metadata": {},
            "items": {"data": [{"current_period_end": 1_900_000_000}]}}


def test_plans_are_public(client):
    plans = client.get("/api/plans").json()
    assert [(p["id"], p["max_documents"]) for p in plans] == [("free", 2), ("premium", 5)]
    assert plans[1]["price_monthly_usd"] == 9.0


def test_me_creates_user_and_reports_usage(client, session_factory, services):
    me = client.get("/api/me", headers=auth("carol", "carol@uni.edu")).json()
    assert me["email"] == "carol@uni.edu" and me["plan"]["id"] == "free"
    assert me["usage"] == {"documents": 0, "figures": 0, "storage_bytes": 0}

    pdf = make_pdf([RED, BLUE])
    upload(client, pdf, headers=auth("carol"))
    drain(session_factory, services)
    me = client.get("/api/me", headers=auth("carol")).json()
    assert me["usage"] == {"documents": 1, "figures": 2, "storage_bytes": len(pdf)}
    assert me["email"] == "carol@example.com"  # email changes in Firebase are picked up


def test_update_profile(client):
    assert client.patch("/api/me", json={"display_name": "  Alice A  "}, headers=ALICE).json()["display_name"] \
        == "Alice A"


def test_delete_account(client, session_factory, services, fakes, db, settings):
    upload(client, make_pdf([RED]))
    drain(session_factory, services)
    user = db.query(User).one()
    user.stripe_subscription_id, user.subscription_status = "sub_9", "active"
    db.commit()

    assert client.delete("/api/me", headers=ALICE).status_code == 204

    assert fakes["billing"].cancelled == ["sub_9"]
    assert fakes["identity"].deleted == ["alice"]
    db.expire_all()
    assert db.query(User).count() == 0
    assert not any(p.is_file() for p in Path(settings.local_storage_dir).rglob("*"))


def test_checkout_and_portal(client, fakes):
    url = client.post("/api/billing/checkout", headers=ALICE).json()["url"]
    assert url.startswith("https://checkout.test/cus_1")
    assert "app/account?checkout=success" in url
    # The Stripe customer is reused.
    client.post("/api/billing/checkout", headers=ALICE)
    assert len(fakes["billing"].customers) == 1
    assert client.post("/api/billing/portal", headers=ALICE).json()["url"] == "https://portal.test/cus_1"


def test_billing_not_configured(settings, session_factory, fakes):
    from fastapi.testclient import TestClient

    from sci_vizio_saas.main import create_app
    from sci_vizio_saas.services import Services

    services = Services(settings, identity=fakes["identity"], billing=None)
    with TestClient(create_app(settings, services, session_factory)) as c:
        assert c.post("/api/billing/checkout", headers=ALICE).status_code == 503
        assert c.get("/api/me", headers=ALICE).json()["billing_enabled"] is False


def test_webhook_rejects_bad_signature(client):
    assert webhook(client, {"id": "evt"}, signature="forged").status_code == 400


def test_subscription_lifecycle(client, fakes, db):
    client.post("/api/billing/checkout", headers=ALICE)  # creates customer cus_1
    fakes["billing"].subscriptions["sub_1"] = subscription()

    completed = {"id": "evt_1", "type": "checkout.session.completed",
                 "data": {"object": {"id": "cs_1", "mode": "subscription", "customer": "cus_1",
                                     "subscription": "sub_1", "client_reference_id": "ignored"}}}
    assert webhook(client, completed).json() == {"received": True, "duplicate": False}
    me = client.get("/api/me", headers=ALICE).json()
    assert me["plan"]["id"] == "premium" and me["subscription_status"] == "active"
    assert me["current_period_end"].startswith("2030")

    # Redelivery is ignored.
    assert webhook(client, completed).json()["duplicate"] is True

    # Payment trouble keeps access while Stripe retries...
    webhook(client, {"id": "evt_2", "type": "customer.subscription.updated",
                     "data": {"object": subscription("past_due")}})
    assert client.get("/api/me", headers=ALICE).json()["plan"]["id"] == "premium"

    # ...and cancellation downgrades.
    webhook(client, {"id": "evt_3", "type": "customer.subscription.deleted",
                     "data": {"object": subscription("canceled")}})
    assert client.get("/api/me", headers=ALICE).json()["plan"]["id"] == "free"


def test_upgrade_raises_limits(client, db):
    upload(client, make_pdf([RED]))
    upload(client, make_pdf([BLUE]))
    assert start_upload(client, make_pdf([GREEN])).status_code == 402

    user = db.query(User).one()
    user.plan = "premium"
    db.commit()
    assert start_upload(client, make_pdf([GREEN])).status_code == 201

    assert client.post("/api/billing/checkout", headers=ALICE).status_code == 409
