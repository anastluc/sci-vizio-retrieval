"""Stripe subscriptions: Checkout for upgrades, the Customer Portal for management, webhooks for state."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Protocol

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from sci_vizio_saas.models import StripeEvent, User
from sci_vizio_saas.plans import FREE, PREMIUM, PREMIUM_STATUSES

logger = logging.getLogger(__name__)


class BillingGateway(Protocol):
    def create_customer(self, email: Optional[str], user_id: str) -> str: ...
    def create_checkout_url(self, customer_id: str, user_id: str, success_url: str, cancel_url: str) -> str: ...
    def create_portal_url(self, customer_id: str, return_url: str) -> str: ...
    def parse_webhook(self, payload: bytes, signature: str) -> dict: ...
    def get_subscription(self, subscription_id: str) -> dict: ...
    def cancel_subscription(self, subscription_id: str) -> None: ...


class StripeGateway:
    def __init__(self, secret_key: str, price_id: str, webhook_secret: Optional[str]):
        import stripe

        self._stripe = stripe
        self._key = secret_key
        self.price_id = price_id
        self.webhook_secret = webhook_secret

    @staticmethod
    def _as_dict(obj) -> dict:
        return obj.to_dict() if hasattr(obj, "to_dict") else dict(obj)

    def create_customer(self, email, user_id):
        customer = self._stripe.Customer.create(api_key=self._key, email=email, metadata={"user_id": user_id})
        return customer["id"]

    def create_checkout_url(self, customer_id, user_id, success_url, cancel_url):
        session = self._stripe.checkout.Session.create(
            api_key=self._key,
            mode="subscription",
            customer=customer_id,
            client_reference_id=user_id,
            line_items=[{"price": self.price_id, "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
            allow_promotion_codes=True,
            subscription_data={"metadata": {"user_id": user_id}},
        )
        return session["url"]

    def create_portal_url(self, customer_id, return_url):
        session = self._stripe.billing_portal.Session.create(
            api_key=self._key, customer=customer_id, return_url=return_url
        )
        return session["url"]

    def parse_webhook(self, payload, signature):
        if not self.webhook_secret:
            raise ValueError("STRIPE_WEBHOOK_SECRET is not configured")
        event = self._stripe.Webhook.construct_event(payload, signature, self.webhook_secret)
        return self._as_dict(event)

    def get_subscription(self, subscription_id):
        return self._as_dict(self._stripe.Subscription.retrieve(subscription_id, api_key=self._key))

    def cancel_subscription(self, subscription_id):
        self._stripe.Subscription.cancel(subscription_id, api_key=self._key)


def ensure_customer(db: Session, gateway: BillingGateway, user: User) -> str:
    if not user.stripe_customer_id:
        user.stripe_customer_id = gateway.create_customer(user.email, str(user.id))
        db.commit()
    return user.stripe_customer_id


def _period_end(subscription: dict) -> Optional[datetime]:
    # Newer Stripe API versions report the period on each subscription item.
    ts = subscription.get("current_period_end")
    if ts is None:
        items = (subscription.get("items") or {}).get("data") or []
        ts = items[0].get("current_period_end") if items else None
    return datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None


def apply_subscription(user: User, subscription: dict) -> None:
    """Copy a Stripe subscription's state onto the user and derive their plan."""
    status = subscription.get("status")
    user.stripe_subscription_id = subscription.get("id")
    user.subscription_status = status
    user.current_period_end = _period_end(subscription)
    user.plan = PREMIUM if status in PREMIUM_STATUSES else FREE


def _find_user(db: Session, customer_id: Optional[str], user_id: Optional[str]) -> Optional[User]:
    if customer_id:
        user = db.scalar(select(User).where(User.stripe_customer_id == customer_id))
        if user:
            return user
    if user_id:
        try:
            return db.get(User, uuid.UUID(str(user_id)))
        except ValueError:  # malformed id
            return None
    return None


def handle_event(db: Session, gateway: BillingGateway, event: dict) -> bool:
    """
    Apply a verified Stripe event. Returns False if it was already processed.

    Relevant events: checkout.session.completed and customer.subscription.*.
    """
    db.add(StripeEvent(id=event["id"], type=event["type"]))
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        return False

    obj = event["data"]["object"]
    event_type = event["type"]

    if event_type == "checkout.session.completed" and obj.get("mode") == "subscription":
        user = _find_user(db, obj.get("customer"), obj.get("client_reference_id"))
        if user is None:
            logger.warning(f"Checkout {obj.get('id')} for unknown user")
        else:
            user.stripe_customer_id = obj.get("customer")
            if obj.get("subscription"):
                apply_subscription(user, gateway.get_subscription(obj["subscription"]))

    elif event_type.startswith("customer.subscription."):
        user = _find_user(db, obj.get("customer"), (obj.get("metadata") or {}).get("user_id"))
        if user is None:
            logger.warning(f"Subscription {obj.get('id')} for unknown customer {obj.get('customer')}")
        elif event_type == "customer.subscription.deleted":
            apply_subscription(user, {**obj, "status": "canceled"})
        elif not user.stripe_subscription_id or user.stripe_subscription_id == obj.get("id") \
                or obj.get("status") in PREMIUM_STATUSES:
            apply_subscription(user, obj)

    db.commit()
    return True
