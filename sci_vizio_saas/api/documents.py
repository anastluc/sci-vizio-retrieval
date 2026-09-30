import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from sci_vizio_saas import jobs
from sci_vizio_saas.api.collections import get_owned_collection
from sci_vizio_saas.deps import get_current_user, get_db, get_services
from sci_vizio_saas.models import Document, DocumentStatus, Figure, User
from sci_vizio_saas.plans import QuotaExceeded, check_can_upload, plan_for
from sci_vizio_saas.schemas import (
    UNFILED,
    DocumentOut,
    DocumentPage,
    DocumentUpdate,
    FigureOut,
    PresignedUploadOut,
    UploadRequest,
    UploadResponse,
    UrlOut,
)
from sci_vizio_saas.services import Services

router = APIRouter(prefix="/documents", tags=["documents"])


def document_prefix(document: Document) -> str:
    return f"users/{document.user_id}/documents/{document.id}/"


def figure_out(figure: Figure, services: Services) -> dict:
    return dict(
        id=figure.id, document_id=figure.document_id, page=figure.page, figure_type=figure.figure_type,
        image_type=figure.image_type, title=figure.title, description=figure.description,
        analysis=figure.analysis, width=figure.width, height=figure.height,
        image_url=services.storage.presign_download(figure.storage_key),
    )


def get_owned_document(db: Session, user: User, document_id: uuid.UUID) -> Document:
    document = db.get(Document, document_id)
    if document is None or document.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return document


@router.get("", response_model=DocumentPage)
def list_documents(
    collection: Optional[str] = Query(default=None, description="Collection id, or 'unfiled'"),
    status_filter: Optional[str] = Query(default=None, alias="status"),
    q: Optional[str] = Query(default=None, max_length=200, description="Filename/title contains"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conds = [Document.user_id == user.id]
    if collection == UNFILED:
        conds.append(Document.collection_id.is_(None))
    elif collection:
        try:
            conds.append(Document.collection_id == uuid.UUID(collection))
        except ValueError:
            raise HTTPException(422, "Invalid collection")
    if status_filter:
        if status_filter not in DocumentStatus.ALL:
            raise HTTPException(422, "Invalid status")
        conds.append(Document.status == status_filter)
    if q:
        escaped = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"
        conds.append(or_(Document.filename.ilike(pattern), Document.title.ilike(pattern)))

    total = db.scalar(select(func.count()).select_from(Document).where(*conds))
    items = db.scalars(
        select(Document).where(*conds).order_by(Document.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return DocumentPage(items=items, total=total)


@router.post("/uploads", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
def start_upload(body: UploadRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db),
                 services: Services = Depends(get_services)):
    """Reserve a document and return a signed URL the browser PUTs the PDF to."""
    if body.collection_id:
        get_owned_collection(db, user, body.collection_id)
    try:
        check_can_upload(db, user, body.size_bytes, services.settings)
    except QuotaExceeded as e:
        code = 413 if e.code == "file_too_large" \
            else status.HTTP_402_PAYMENT_REQUIRED
        raise HTTPException(code, detail={"code": e.code, "message": str(e)})

    document_id = uuid.uuid4()
    document = Document(
        id=document_id,
        user_id=user.id,
        collection_id=body.collection_id,
        filename=body.filename,
        storage_key=f"users/{user.id}/documents/{document_id}/original.pdf",
        size_bytes=body.size_bytes,
        sha256=body.sha256,
        status=DocumentStatus.UPLOADING,
    )
    db.add(document)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT,
                            detail={"code": "duplicate", "message": "You have already uploaded this PDF."})

    upload = services.storage.presign_upload(document.storage_key, body.content_type)
    return UploadResponse(document=document, upload=PresignedUploadOut(**upload.__dict__))


@router.post("/{document_id}/complete", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED)
def complete_upload(document_id: uuid.UUID, user: User = Depends(get_current_user),
                    db: Session = Depends(get_db), services: Services = Depends(get_services)):
    """Confirm the browser finished uploading; queues the document for processing."""
    document = get_owned_document(db, user, document_id)
    if document.status != DocumentStatus.UPLOADING:
        raise HTTPException(status.HTTP_409_CONFLICT, "Upload already completed")

    size = services.storage.size(document.storage_key)
    if size is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "The file has not been uploaded yet")
    plan = plan_for(user, services.settings)
    if size > plan.max_file_bytes:
        # Signed PUT URLs can't cap the size, so enforce it here.
        services.storage.delete_prefix(document_prefix(document))
        db.delete(document)
        db.commit()
        raise HTTPException(413,
                            detail={"code": "file_too_large",
                                    "message": f"Files can be at most {plan.max_file_mb} MB."})

    document.size_bytes = size
    document.status = DocumentStatus.QUEUED
    jobs.enqueue(db, document.id, max_attempts=services.settings.job_max_attempts)
    db.commit()
    return document


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: uuid.UUID, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return get_owned_document(db, user, document_id)


@router.patch("/{document_id}", response_model=DocumentOut)
def update_document(document_id: uuid.UUID, body: DocumentUpdate, user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    document = get_owned_document(db, user, document_id)
    if "title" in body.model_fields_set:
        document.title = (body.title or "").strip() or None
    if "collection_id" in body.model_fields_set:
        if body.collection_id is not None:
            get_owned_collection(db, user, body.collection_id)
        document.collection_id = body.collection_id
    db.commit()
    return document


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: uuid.UUID, user: User = Depends(get_current_user),
                    db: Session = Depends(get_db), services: Services = Depends(get_services)):
    document = get_owned_document(db, user, document_id)
    services.storage.delete_prefix(document_prefix(document))
    db.delete(document)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{document_id}/download", response_model=UrlOut)
def download_document(document_id: uuid.UUID, user: User = Depends(get_current_user),
                      db: Session = Depends(get_db), services: Services = Depends(get_services)):
    document = get_owned_document(db, user, document_id)
    if document.status == DocumentStatus.UPLOADING:
        raise HTTPException(status.HTTP_409_CONFLICT, "The upload has not completed")
    return UrlOut(url=services.storage.presign_download(document.storage_key, expires=600,
                                                        filename=document.filename))


@router.post("/{document_id}/retry", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED)
def retry_document(document_id: uuid.UUID, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db), services: Services = Depends(get_services)):
    document = get_owned_document(db, user, document_id)
    if document.status != DocumentStatus.FAILED:
        raise HTTPException(status.HTTP_409_CONFLICT, "Only failed documents can be retried")
    document.status = DocumentStatus.QUEUED
    document.error = None
    jobs.enqueue(db, document.id, max_attempts=services.settings.job_max_attempts)
    db.commit()
    return document


@router.get("/{document_id}/figures", response_model=List[FigureOut])
def list_figures(document_id: uuid.UUID, user: User = Depends(get_current_user), db: Session = Depends(get_db),
                 services: Services = Depends(get_services)):
    document = get_owned_document(db, user, document_id)
    figures = db.scalars(
        select(Figure).where(Figure.document_id == document.id).order_by(Figure.page, Figure.image_index)
    ).all()
    return [figure_out(f, services) for f in figures]
