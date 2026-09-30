"""FastAPI dependencies."""

from typing import Iterator, Optional

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from sci_vizio_saas.auth import InvalidToken, get_or_create_user
from sci_vizio_saas.models import User
from sci_vizio_saas.services import Services


def get_services(request: Request) -> Services:
    return request.app.state.services


def get_db(request: Request) -> Iterator[Session]:
    with request.app.state.session_factory() as session:
        yield session


def get_current_user(
    authorization: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
    services: Services = Depends(get_services),
) -> User:
    """The signed-in user, from a ``Authorization: Bearer <Firebase ID token>`` header."""
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token",
                            headers={"WWW-Authenticate": "Bearer"})
    try:
        claims = services.identity.verify(token)
    except InvalidToken:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token",
                            headers={"WWW-Authenticate": "Bearer"})
    return get_or_create_user(db, claims)
