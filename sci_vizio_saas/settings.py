"""Runtime configuration, read from environment variables (and a local ``.env``)."""

from functools import lru_cache
from typing import List, Literal, Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: Literal["development", "test", "production"] = "development"

    # --- Database (Heroku sets DATABASE_URL) ---
    database_url: str = "postgresql+psycopg://sci:sci@localhost:5432/sci_vizio"
    db_pool_size: int = 5

    # --- Web ---
    frontend_url: str = "http://localhost:5173"
    # Comma-separated list of origins allowed to call the API.
    cors_origins: str = "http://localhost:5173"
    # Public base URL of this API; used to build local-storage file URLs.
    public_api_url: str = "http://localhost:8000"

    # --- Firebase Authentication ---
    firebase_project_id: Optional[str] = None
    # Service-account JSON (the file's contents). Only needed for admin actions
    # such as deleting a user's Firebase account; token verification works without it.
    firebase_credentials_json: Optional[str] = None
    require_verified_email: bool = False

    # --- Object storage ---
    storage_backend: Literal["local", "s3"] = "local"
    local_storage_dir: str = "var/storage"
    # HMAC key for signed local-storage URLs.
    signing_secret: str = "dev-insecure-signing-secret"
    s3_bucket: Optional[str] = None
    s3_endpoint_url: Optional[str] = None  # e.g. https://<account>.r2.cloudflarestorage.com
    s3_region: str = "auto"
    s3_access_key_id: Optional[str] = None
    s3_secret_access_key: Optional[str] = None

    # --- Processing ---
    openrouter_api_key: Optional[str] = None
    # Any OpenAI-compatible endpoint works (e.g. a proxy, or a stub in end-to-end tests).
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    vision_model: str = "qwen/qwen3.5-flash-02-23"
    vision_max_tokens: int = 2048
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_cache_dir: Optional[str] = None
    # Images smaller than this (px, either side) are skipped: icons, logos, rules.
    min_figure_side: int = 100
    figure_concurrency: int = 4
    worker_poll_seconds: float = 2.0
    job_max_attempts: int = 3

    # --- Plans (indicative values) ---
    free_max_documents: int = 10
    free_max_file_mb: int = 20
    free_max_figures_per_document: int = 40
    premium_max_documents: int = 500
    premium_max_file_mb: int = 100
    premium_max_figures_per_document: int = 200
    premium_price_monthly_usd: float = 9.0

    # --- Stripe ---
    stripe_secret_key: Optional[str] = None
    stripe_webhook_secret: Optional[str] = None
    stripe_price_id: Optional[str] = None

    @field_validator("database_url")
    @classmethod
    def _use_psycopg_driver(cls, url: str) -> str:
        # Heroku provides postgres://...; SQLAlchemy needs an explicit driver.
        for prefix in ("postgres://", "postgresql://"):
            if url.startswith(prefix):
                return "postgresql+psycopg://" + url[len(prefix):]
        return url

    @model_validator(mode="after")
    def _check_production(self):
        if self.environment != "production":
            return self
        problems = []
        if not self.firebase_project_id:
            problems.append("FIREBASE_PROJECT_ID is required")
        if self.storage_backend == "local":
            problems.append("STORAGE_BACKEND=local is not suitable for production (dyno disks are ephemeral)")
        if self.storage_backend == "s3" and not self.s3_bucket:
            problems.append("S3_BUCKET is required")
        if not self.openrouter_api_key:
            problems.append("OPENROUTER_API_KEY is required")
        if problems:
            raise ValueError("Invalid production configuration: " + "; ".join(problems))
        return self

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def billing_enabled(self) -> bool:
        return bool(self.stripe_secret_key and self.stripe_price_id)


@lru_cache
def get_settings() -> Settings:
    return Settings()
