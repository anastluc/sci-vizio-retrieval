import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from sci_vizio_saas.deps import get_current_user, get_db, get_services
from sci_vizio_saas.models import Document, Figure, User
from sci_vizio_saas.plans import all_plans, plan_for
from sci_vizio_saas.schemas import MeOut, MeUpdate, PlanOut, UsageOut
from sci_vizio_saas.services import Services

logger = logging.getLogger(__name__)
router = APIRouter(tags=["account"])


def _plan_out(plan) -> PlanOut:
    return PlanOut(id=plan.id, name=plan.name, price_monthly_usd=plan.price_monthly_usd,
                   max_documents=plan.max_documents, max_file_mb=plan.max_file_mb,
                   max_figures_per_document=plan.max_figures_per_document)


def _me(db: Session, user: User, services: Services) -> MeOut:
    documents, storage_bytes = db.execute(
        select(func.count(Document.id), func.coalesce(func.sum(Document.size_bytes), 0))
        .where(Document.user_id == user.id)
    ).one()
    figures = db.scalar(select(func.count()).select_from(Figure).where(Figure.user_id == user.id))
    return MeOut(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        plan=_plan_out(plan_for(user, services.settings)),
        usage=UsageOut(documents=documents, figures=figures, storage_bytes=storage_bytes),
        subscription_status=user.subscription_status,
        current_period_end=user.current_period_end,
        billing_enabled=services.billing is not None,
        created_at=user.created_at,
    )


@router.get("/plans", response_model=List[PlanOut])
def list_plans(services: Services = Depends(get_services)):
    """Public: plans and their (indicative) prices and limits."""
    return [_plan_out(p) for p in all_plans(services.settings).values()]


@router.get("/me", response_model=MeOut)
def get_me(user: User = Depends(get_current_user), db: Session = Depends(get_db),
           services: Services = Depends(get_services)):
    return _me(db, user, services)


@router.patch("/me", response_model=MeOut)
def update_me(body: MeUpdate, user: User = Depends(get_current_user), db: Session = Depends(get_db),
              services: Services = Depends(get_services)):
    if "display_name" in body.model_fields_set:
        user.display_name = (body.display_name or "").strip() or None
    db.commit()
    return _me(db, user, services)


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
def delete_me(user: User = Depends(get_current_user), db: Session = Depends(get_db),
              services: Services = Depends(get_services)):
    """Delete the account: subscription, stored files, database rows and the Firebase user."""
    if user.stripe_subscription_id and user.subscription_status not in (None, "canceled", "incomplete_expired"):
        if services.billing is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Billing is unavailable; try again later.")
        try:
            services.billing.cancel_subscription(user.stripe_subscription_id)
        except Exception:
            logger.exception(f"Could not cancel subscription for user {user.id}")
            raise HTTPException(status.HTTP_502_BAD_GATEWAY,
                                "Could not cancel your subscription; account not deleted. Try again later.")

    firebase_uid = user.firebase_uid
    services.storage.delete_prefix(f"users/{user.id}/")
    db.delete(user)
    db.commit()

    try:
        services.identity.delete_user(firebase_uid)
    except Exception:
        # Needs service-account credentials; the client also deletes its own Firebase user.
        logger.warning(f"Could not delete Firebase user {firebase_uid}", exc_info=True)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
