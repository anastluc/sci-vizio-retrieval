"""
FastAPI application.

Run with ``uvicorn sci_vizio_saas.main:create_app --factory`` (add ``--reload``
locally); the Heroku ``web`` process runs the same command.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from sci_vizio_saas.api import account, billing, collections, documents, files, search
from sci_vizio_saas.db import make_engine, make_session_factory
from sci_vizio_saas.deps import get_db
from sci_vizio_saas.services import Services
from sci_vizio_saas.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def create_app(settings: Settings = None, services: Services = None,
               session_factory: sessionmaker = None) -> FastAPI:
    if settings is None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        settings = get_settings()
    services = services or Services(settings)
    if session_factory is None:
        session_factory = make_session_factory(make_engine(settings.database_url, settings.db_pool_size))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.environment == "production":
            services.embedder  # load the model before serving traffic  # noqa: B018
        yield

    app = FastAPI(
        title="Sci-Vizio API",
        version="0.2.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings
    app.state.services = services
    app.state.session_factory = session_factory

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
        max_age=3600,
    )

    api = APIRouter(prefix="/api")

    @api.get("/health", tags=["meta"])
    def health(db: Session = Depends(get_db)):
        db.execute(text("SELECT 1"))
        return {"status": "ok"}

    for module in (account, collections, documents, search, billing):
        api.include_router(module.router)
    if settings.storage_backend == "local":
        api.include_router(files.router)

    app.include_router(api)
    return app
