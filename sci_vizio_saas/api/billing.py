import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from sci_vizio_saas.billing import ensure_customer, handle_event
from sci_vizio_saas.deps import get_current_user, get_db, get_services
from sci_vizio_saas.models import User
from sci_vizio_saas.plans import PREMIUM
from sci_vizio_saas.schemas import UrlOut
from sci_vizio_saas.services import Services

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/billing", tags=["billing"])


def _gateway(services: Services):
    if services.billing is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Billing is not configured")
    return services.billing


@router.post("/checkout", response_model=UrlOut)
def create_checkout(user: User = Depends(get_current_user), db: Session = Depends(get_db),
                    services: Services = Depends(get_services)):
    """Start a Stripe Checkout session for the Premium subscription."""
    gateway = _gateway(services)
    if user.plan == PREMIUM:
        raise HTTPException(status.HTTP_409_CONFLICT, "You are already on Premium")
    customer_id = ensure_customer(db, gateway, user)
    base = services.settings.frontend_url.rstrip("/")
    url = gateway.create_checkout_url(customer_id, str(user.id),
                                      success_url=f"{base}/app/account?checkout=success",
                                      cancel_url=f"{base}/app/account?checkout=cancelled")
    return UrlOut(url=url)


@router.post("/portal", response_model=UrlOut)
def create_portal(user: User = Depends(get_current_user), db: Session = Depends(get_db),
                  services: Services = Depends(get_services)):
    """Open the Stripe Customer Portal (update card, cancel, invoices)."""
    gateway = _gateway(services)
    if not user.stripe_customer_id:
        raise HTTPException(status.HTTP_409_CONFLICT, "No billing account yet")
    url = gateway.create_portal_url(user.stripe_customer_id,
                                    return_url=f"{services.settings.frontend_url.rstrip('/')}/app/account")
    return UrlOut(url=url)


@router.post("/webhook", include_in_schema=False)
async def stripe_webhook(request: Request, stripe_signature: str = Header(default=""),
                         db: Session = Depends(get_db), services: Services = Depends(get_services)):
    gateway = _gateway(services)
    payload = await request.body()
    try:
        event = gateway.parse_webhook(payload, stripe_signature)
    except Exception as e:
        logger.warning(f"Rejected Stripe webhook: {e}")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid signature")
    processed = handle_event(db, gateway, event)
    return {"received": True, "duplicate": not processed}
