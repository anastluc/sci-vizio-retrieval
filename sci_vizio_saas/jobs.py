"""A small job queue on PostgreSQL (``SELECT ... FOR UPDATE SKIP LOCKED``), so no Redis is needed."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from sci_vizio_saas.models import Document, DocumentStatus, Job, JobStatus

logger = logging.getLogger(__name__)

PROCESS_DOCUMENT = "process_document"
RETRY_BASE_DELAY = timedelta(seconds=30)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def enqueue(db: Session, document_id, max_attempts: int = 3) -> Job:
    job = Job(kind=PROCESS_DOCUMENT, document_id=document_id, max_attempts=max_attempts)
    db.add(job)
    return job


def claim_next(db: Session) -> Optional[Job]:
    """Atomically take the oldest runnable job, or None. Safe with many workers."""
    job = db.scalars(
        select(Job)
        .where(Job.status == JobStatus.QUEUED, Job.run_after <= _now())
        .order_by(Job.id)
        .limit(1)
        .with_for_update(skip_locked=True)
    ).first()
    if job is None:
        db.rollback()
        return None
    job.status = JobStatus.RUNNING
    job.attempts += 1
    job.locked_at = _now()
    db.commit()
    return job


def mark_done(db: Session, job: Job) -> None:
    job.status = JobStatus.DONE
    job.finished_at = _now()
    job.last_error = None
    db.commit()


def mark_failed(db: Session, job: Job, error: str, retryable: bool) -> bool:
    """Record a failure. Returns True if the job will be retried."""
    job.last_error = error[:5000]
    if retryable and job.attempts < job.max_attempts:
        job.status = JobStatus.QUEUED
        job.run_after = _now() + RETRY_BASE_DELAY * (2 ** (job.attempts - 1))
        job.locked_at = None
        db.commit()
        return True
    job.status = JobStatus.FAILED
    job.finished_at = _now()
    db.commit()
    return False


def requeue_stale(db: Session, older_than: timedelta = timedelta(minutes=30)) -> int:
    """Requeue jobs whose worker died mid-run (e.g. a dyno restart)."""
    result = db.execute(
        update(Job)
        .where(Job.status == JobStatus.RUNNING, Job.locked_at < _now() - older_than)
        .values(status=JobStatus.QUEUED, locked_at=None)
    )
    db.commit()
    return result.rowcount


def abandoned_uploads(db: Session, older_than: timedelta = timedelta(hours=2)):
    """Documents whose upload was started but never completed."""
    return db.scalars(
        select(Document).where(
            Document.status == DocumentStatus.UPLOADING, Document.created_at < _now() - older_than
        )
    ).all()
