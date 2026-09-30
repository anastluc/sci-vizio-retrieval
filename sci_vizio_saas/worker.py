"""
Background worker: processes uploaded PDFs.

Run with ``python -m sci_vizio_saas.worker`` (the Heroku ``worker`` process).
Several workers can run at once; jobs are claimed with row locks.
"""

import logging
import signal
import time

from sqlalchemy.orm import sessionmaker

from sci_vizio_saas import jobs
from sci_vizio_saas.db import make_engine, make_session_factory
from sci_vizio_saas.models import Document, DocumentStatus
from sci_vizio_saas.processing import DocumentRejected, process_document
from sci_vizio_saas.services import Services
from sci_vizio_saas.settings import get_settings

logger = logging.getLogger("sci_vizio_saas.worker")

MAINTENANCE_INTERVAL = 60.0


def run_job(session_factory: sessionmaker, services: Services, job) -> None:
    with session_factory() as db:
        job = db.merge(job)
        try:
            process_document(db, services, job.document_id)
        except DocumentRejected as e:
            db.rollback()
            jobs.mark_failed(db, job, str(e), retryable=False)
            _fail_document(db, job.document_id, str(e))
            logger.info(f"Job {job.id}: document rejected: {e}")
        except Exception as e:
            db.rollback()
            logger.exception(f"Job {job.id} failed (attempt {job.attempts}/{job.max_attempts})")
            if jobs.mark_failed(db, job, repr(e), retryable=True):
                _set_status(db, job.document_id, DocumentStatus.QUEUED, "Processing failed, retrying shortly.")
            else:
                _fail_document(db, job.document_id, "Processing failed. Please try again later.")
        else:
            jobs.mark_done(db, job)


def _set_status(db, document_id, status, error):
    document = db.get(Document, document_id)
    if document is not None:
        document.status = status
        document.error = error
        db.commit()


def _fail_document(db, document_id, error):
    _set_status(db, document_id, DocumentStatus.FAILED, error)


def run_once(session_factory: sessionmaker, services: Services) -> bool:
    """Process at most one job. Returns True if a job was found."""
    with session_factory() as db:
        job = jobs.claim_next(db)
    if job is None:
        return False
    run_job(session_factory, services, job)
    return True


def run_maintenance(session_factory: sessionmaker, services: Services) -> None:
    with session_factory() as db:
        if requeued := jobs.requeue_stale(db):
            logger.warning(f"Requeued {requeued} stale jobs")
        for document in jobs.abandoned_uploads(db):
            services.storage.delete([document.storage_key])
            db.delete(document)
        db.commit()


class Worker:
    def __init__(self, session_factory: sessionmaker, services: Services):
        self.session_factory = session_factory
        self.services = services
        self.running = True

    def stop(self, *_):
        logger.info("Shutdown requested; finishing current job")
        self.running = False

    def run(self):
        poll = self.services.settings.worker_poll_seconds
        last_maintenance = 0.0
        while self.running:
            if time.monotonic() - last_maintenance > MAINTENANCE_INTERVAL:
                run_maintenance(self.session_factory, self.services)
                last_maintenance = time.monotonic()
            try:
                found = run_once(self.session_factory, self.services)
            except Exception:
                logger.exception("Worker loop error")
                found = False
            if not found:
                time.sleep(poll)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    settings = get_settings()
    session_factory = make_session_factory(make_engine(settings.database_url, pool_size=2))
    services = Services(settings)
    services.embedder  # load the model before taking jobs  # noqa: B018

    worker = Worker(session_factory, services)
    signal.signal(signal.SIGTERM, worker.stop)  # Heroku sends SIGTERM on restarts/deploys
    signal.signal(signal.SIGINT, worker.stop)
    logger.info("Worker started")
    worker.run()


if __name__ == "__main__":
    main()
