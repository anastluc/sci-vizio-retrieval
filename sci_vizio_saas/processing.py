"""Turn an uploaded PDF into searchable figures: extract, describe, embed, store."""

import hashlib
import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import pymupdf
from PIL import Image
from sqlalchemy import delete
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from sci_vizio_retrieval.extractor import ExtractedImage, iter_pdf_images
from sci_vizio_retrieval.figure_types import normalize_figure_type
from sci_vizio_retrieval.indexer import extract_and_validate_json
from sci_vizio_retrieval.processor import ImageProcessor
from sci_vizio_saas.models import Document, DocumentStatus, Figure, User
from sci_vizio_saas.plans import plan_for
from sci_vizio_saas.services import Services

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"
WEB_FORMATS = {"png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg"}


class DocumentRejected(Exception):
    """The document can never be processed (not a PDF, corrupt, duplicate...). Not retried."""


def figure_key(document: Document, image: ExtractedImage, ext: str) -> str:
    return f"users/{document.user_id}/documents/{document.id}/figures/page{image.page}_img{image.index}.{ext}"


def to_web_image(image: ExtractedImage) -> Optional[Tuple[bytes, str, str]]:
    """(bytes, ext, mime) in a format browsers and vision APIs accept, converting e.g. JPX/TIFF to PNG."""
    ext = image.ext.lower()
    if ext in WEB_FORMATS:
        return image.data, ext, WEB_FORMATS[ext]
    try:
        with Image.open(io.BytesIO(image.data)) as img:
            out = io.BytesIO()
            img.convert("RGB").save(out, format="PNG", optimize=True)
        return out.getvalue(), "png", "image/png"
    except Exception as e:
        logger.info(f"Skipping unconvertible {ext} image on page {image.page}: {e}")
        return None


def select_images(pdf: pymupdf.Document, min_side: int, limit: int) -> Tuple[List[ExtractedImage], int]:
    """Figures to analyse (deduplicated, capped at ``limit``) and how many were skipped."""
    selected, seen, skipped = [], set(), 0
    for image in iter_pdf_images(pdf, min_side=min_side):
        digest = hashlib.sha1(image.data).digest()
        if digest in seen:  # logos/headers repeated on every page
            continue
        seen.add(digest)
        if len(selected) >= limit:
            skipped += 1
            continue
        selected.append(image)
    return selected, skipped


def _describe(services: Services, image: ExtractedImage):
    """Run the vision model on one figure. Returns (image, web image, analysis or None, error or None)."""
    web = to_web_image(image)
    if web is None:
        return image, None, None, "unsupported image format"
    data, _, mime = web
    try:
        response = services.vision.analyze_image_bytes(data, mime, ImageProcessor.USER_PROMPT)
    except Exception as e:
        return image, web, None, f"vision API: {e}"
    ok, analysis, error = extract_and_validate_json(response.text)
    return image, web, analysis if ok else None, error


def process_document(db: Session, services: Services, document_id) -> Optional[Document]:
    """
    Process one document end to end. Safe to re-run: previous figures are replaced.

    Raises DocumentRejected for permanent problems; other exceptions are
    treated as transient by the job runner and retried.
    """
    document = db.get(Document, document_id)
    if document is None:
        return None
    user = db.get(User, document.user_id)
    plan = plan_for(user, services.settings)
    storage = services.storage

    document.status = DocumentStatus.PROCESSING
    document.error = None
    db.commit()

    data = storage.get(document.storage_key)
    if not data.startswith(PDF_MAGIC):
        raise DocumentRejected("The file is not a PDF.")
    if len(data) > plan.max_file_bytes:
        raise DocumentRejected(f"The file is larger than {plan.max_file_mb} MB.")

    digest = hashlib.sha256(data).hexdigest()
    if document.sha256 and document.sha256 != digest:
        raise DocumentRejected("The uploaded file doesn't match the checksum sent before upload.")
    document.sha256 = digest
    document.size_bytes = len(data)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise DocumentRejected("You have already uploaded this PDF.")

    try:
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise DocumentRejected(f"The PDF could not be read: {e}")

    with pdf:
        document.page_count = pdf.page_count
        metadata_title = (pdf.metadata or {}).get("title", "").strip()
        if metadata_title and not document.title:
            document.title = metadata_title[:1000]
        images, skipped = select_images(pdf, services.settings.min_figure_side, plan.max_figures_per_document)

    # Replace figures from any earlier attempt.
    old_keys = [f.storage_key for f in document.figures]
    db.execute(delete(Figure).where(Figure.document_id == document.id))
    db.commit()
    storage.delete(old_keys)

    with ThreadPoolExecutor(max_workers=max(1, services.settings.figure_concurrency)) as pool:
        described = list(pool.map(lambda img: _describe(services, img), images))

    successes = [(img, web, analysis) for img, web, analysis, _ in described if analysis is not None]
    errors = [err for _, _, analysis, err in described if analysis is None]
    if images and not successes and any(e.startswith("vision API") for e in errors):
        # Nothing worked and the API was failing: likely an outage, so retry later.
        raise RuntimeError(f"All {len(images)} figures failed: {errors[0]}")

    embeddings = services.embedder.embed([_embedding_text(analysis) for _, _, analysis in successes]) \
        if successes else []

    for (image, (web_data, ext, mime), analysis), vector in zip(successes, embeddings, strict=True):
        key = figure_key(document, image, ext)
        storage.put(key, web_data, mime)
        db.add(Figure(
            document_id=document.id,
            user_id=document.user_id,
            page=image.page,
            image_index=image.index,
            storage_key=key,
            content_type=mime,
            width=image.width,
            height=image.height,
            figure_type=normalize_figure_type(analysis.get("image_type")),
            image_type=_text(analysis.get("image_type"), 200),
            title=_text(analysis.get("title")),
            description=_text(analysis.get("description")),
            analysis=analysis,
            embedding=vector,
            embedding_model=services.embedder.model_name,
        ))

    document.figure_count = len(successes)
    document.skipped_figure_count = skipped + len(errors)
    document.status = DocumentStatus.READY
    document.processed_at = datetime.now(timezone.utc)
    if errors:
        document.error = f"{len(errors)} of {len(images)} figures could not be analysed."
    db.commit()
    return document


def _embedding_text(analysis: dict) -> str:
    # Same text the CLI indexes (IndexRecord.document_text), so vectors are comparable.
    return json.dumps(analysis)


def _text(value, limit: int = None) -> Optional[str]:
    if value is None:
        return None
    value = value if isinstance(value, str) else str(value)
    return value[:limit] if limit else value
