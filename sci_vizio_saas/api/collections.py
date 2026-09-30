import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from sci_vizio_saas.deps import get_current_user, get_db
from sci_vizio_saas.models import Collection, Document, User
from sci_vizio_saas.schemas import CollectionIn, CollectionOut, CollectionUpdate

router = APIRouter(prefix="/collections", tags=["collections"])

MAX_COLLECTIONS = 200


def get_owned_collection(db: Session, user: User, collection_id: uuid.UUID) -> Collection:
    collection = db.get(Collection, collection_id)
    if collection is None or collection.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Collection not found")
    return collection


def _out(collection: Collection, document_count: int) -> CollectionOut:
    out = CollectionOut.model_validate(collection)
    out.document_count = document_count
    return out


def _commit_unique_name(db: Session):
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "You already have a collection with that name")


@router.get("", response_model=List[CollectionOut])
def list_collections(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.execute(
        select(Collection, func.count(Document.id))
        .outerjoin(Document, Document.collection_id == Collection.id)
        .where(Collection.user_id == user.id)
        .group_by(Collection.id)
        .order_by(func.lower(Collection.name))
    ).all()
    return [_out(c, n) for c, n in rows]


@router.post("", response_model=CollectionOut, status_code=status.HTTP_201_CREATED)
def create_collection(body: CollectionIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    count = db.scalar(select(func.count()).select_from(Collection).where(Collection.user_id == user.id))
    if count >= MAX_COLLECTIONS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"You can have at most {MAX_COLLECTIONS} collections")
    collection = Collection(user_id=user.id, name=body.name, description=body.description)
    db.add(collection)
    _commit_unique_name(db)
    return _out(collection, 0)


@router.patch("/{collection_id}", response_model=CollectionOut)
def update_collection(collection_id: uuid.UUID, body: CollectionUpdate,
                      user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    collection = get_owned_collection(db, user, collection_id)
    if body.name is not None:
        name = body.name.strip()
        if not name:
            raise HTTPException(422, "Name must not be blank")
        collection.name = name
    if "description" in body.model_fields_set:
        collection.description = body.description
    _commit_unique_name(db)
    count = db.scalar(select(func.count()).select_from(Document).where(Document.collection_id == collection.id))
    return _out(collection, count)


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_collection(collection_id: uuid.UUID, user: User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    """Delete a collection. Its documents are kept and become unfiled."""
    db.delete(get_owned_collection(db, user, collection_id))
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
