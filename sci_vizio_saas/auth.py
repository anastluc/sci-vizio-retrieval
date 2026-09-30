"""Firebase ID-token verification and mapping Firebase users to local users."""

import json
import logging
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from sci_vizio_saas.models import User

logger = logging.getLogger(__name__)


class InvalidToken(Exception):
    pass


class IdentityProvider(Protocol):
    def verify(self, id_token: str) -> dict:
        """Return the token's claims (at least ``uid``) or raise InvalidToken."""

    def delete_user(self, uid: str) -> None:
        """Delete the identity-provider account."""


class FirebaseIdentityProvider:
    """
    Verifies Firebase ID tokens with firebase-admin.

    Verification only needs the project id (Google's public keys are fetched
    and cached). Deleting accounts needs service-account credentials. Set
    FIREBASE_AUTH_EMULATOR_HOST to use the local Auth emulator.
    """

    def __init__(self, project_id: str = None, credentials_json: str = None, require_verified_email: bool = False):
        import firebase_admin
        from firebase_admin import credentials

        options = {"projectId": project_id} if project_id else None
        cred = credentials.Certificate(json.loads(credentials_json)) if credentials_json else None
        try:
            self._app = firebase_admin.get_app("sci-vizio")
        except ValueError:
            self._app = firebase_admin.initialize_app(cred, options, name="sci-vizio")
        self.require_verified_email = require_verified_email

    def verify(self, id_token: str) -> dict:
        from firebase_admin import auth

        try:
            claims = auth.verify_id_token(id_token, app=self._app, clock_skew_seconds=10)
        except Exception as e:
            raise InvalidToken(str(e)) from e
        if self.require_verified_email and not claims.get("email_verified"):
            raise InvalidToken("Email address not verified")
        return claims

    def delete_user(self, uid: str) -> None:
        from firebase_admin import auth

        try:
            auth.delete_user(uid, app=self._app)
        except auth.UserNotFoundError:
            pass


def get_or_create_user(db: Session, claims: dict) -> User:
    """Local user for the token's Firebase uid, created on first sign-in."""
    uid = claims["uid"]
    user = db.scalar(select(User).where(User.firebase_uid == uid))
    if user is None:
        user = User(firebase_uid=uid, email=claims.get("email"), display_name=claims.get("name"))
        db.add(user)
        try:
            db.commit()
        except IntegrityError:  # created concurrently by another request
            db.rollback()
            user = db.scalar(select(User).where(User.firebase_uid == uid))
    elif claims.get("email") and claims["email"] != user.email:
        user.email = claims["email"]
        db.commit()
    return user
