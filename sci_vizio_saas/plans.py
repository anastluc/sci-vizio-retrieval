"""Freemium plans and quota checks. Prices and limits are indicative and come from settings."""

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from sci_vizio_saas.models import Document, User

FREE = "free"
PREMIUM = "premium"

# Stripe subscription statuses that grant premium. past_due keeps access while
# Stripe retries the payment; after its retries Stripe cancels the subscription.
PREMIUM_STATUSES = {"active", "trialing", "past_due"}


@dataclass(frozen=True)
class Plan:
    id: str
    name: str
    price_monthly_usd: float
    max_documents: int
    max_file_mb: int
    max_figures_per_document: int

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024


class QuotaExceeded(Exception):
    """Raised when an action would exceed the user's plan limits."""

    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


def all_plans(settings) -> dict:
    return {
        FREE: Plan(FREE, "Free", 0.0, settings.free_max_documents, settings.free_max_file_mb,
                   settings.free_max_figures_per_document),
        PREMIUM: Plan(PREMIUM, "Premium", settings.premium_price_monthly_usd, settings.premium_max_documents,
                      settings.premium_max_file_mb, settings.premium_max_figures_per_document),
    }


def plan_for(user: User, settings) -> Plan:
    return all_plans(settings)[PREMIUM if user.plan == PREMIUM else FREE]


def document_count(db: Session, user: User) -> int:
    """Documents counting towards the quota, including uploads in progress."""
    return db.scalar(select(func.count()).select_from(Document).where(Document.user_id == user.id))


def check_can_upload(db: Session, user: User, size_bytes: int, settings) -> Plan:
    """
    Raise QuotaExceeded if the user may not add a document of this size.

    Locks the user's row so concurrent uploads can't both pass the check.
    """
    db.execute(select(User.id).where(User.id == user.id).with_for_update())
    plan = plan_for(user, settings)

    if size_bytes > plan.max_file_bytes:
        raise QuotaExceeded(
            f"Files on the {plan.name} plan can be at most {plan.max_file_mb} MB.", "file_too_large"
        )
    if document_count(db, user) >= plan.max_documents:
        hint = " Upgrade to Premium for more." if plan.id == FREE else ""
        raise QuotaExceeded(
            f"The {plan.name} plan allows {plan.max_documents} PDFs.{hint}", "document_limit"
        )
    return plan
