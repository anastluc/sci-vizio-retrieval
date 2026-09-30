"""Central configuration: ``config.ini`` values with environment-variable overrides."""

import configparser
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# Default config content to write if config.ini doesn't exist
DEFAULT_CONFIG_CONTENT = """# Sci-Vizio-Retrieval Configuration
# -----------------------------------
# This file controls pipeline settings. Edit the values below as needed.
# Any value can be overridden with an environment variable named
# SCI_{SECTION}_{KEY} or {SECTION}_{KEY}, e.g. SCI_VISION_MODEL / VISION_MODEL.

[vision]
# OpenRouter model slug for image analysis.
# Browse available models at: https://openrouter.ai/models
# Examples:
#   qwen/qwen3.5-flash-02-23        (fast, cheap)
#   google/gemini-2.0-flash-001      (fast, good quality)
#   openai/gpt-4o                    (high quality)
#   anthropic/claude-sonnet-4        (high quality)
model = qwen/qwen3.5-flash-02-23

# Seconds to wait between API calls to avoid rate limits.
api_delay = 2.0

# Number of retry attempts on transient failures.
max_retries = 3

# Maximum tokens for the model response.
max_tokens = 2048

[pipeline]
# Default input directory for PDFs.
pdf_input_dir = pdfs

# Default output directory for extracted content.
output_dir = output

# Directory for extracted images (relative to output_dir).
images_subdir = extracted_images

# Directory for image analysis results (relative to output_dir).
analysis_subdir = image_process

# SQLite database path for tracking processing state.
db_path = pdf_processing.db

[storage]
# Vector store backend used for indexing and search: chroma | postgres
backend = chroma

# ChromaDB persistent vector database directory (backend = chroma).
chroma_path = chroma_db

# PostgreSQL connection string (backend = postgres). Requires the pgvector
# extension. Prefer setting SCI_STORAGE_POSTGRES_DSN in .env over storing
# credentials here.
postgres_dsn = postgresql://sci:sci@localhost:5432/sci_vizio

# Table that holds the indexed image analyses (backend = postgres).
postgres_table = image_analysis

# Sentence-transformers model used to embed descriptions (backend = postgres).
embedding_model = sentence-transformers/all-MiniLM-L6-v2
"""

config = configparser.ConfigParser()
config_file = PROJECT_ROOT / "config.ini"

if not config_file.exists():
    try:
        config_file.write_text(DEFAULT_CONFIG_CONTENT, encoding="utf-8")
    except OSError as e:  # e.g. read-only install location
        logger.warning(f"Could not write default config to {config_file}: {e}")

config.read(config_file)


def get_config_value(section: str, key: str, default: str) -> str:
    """
    Get config value with support for environment variable overrides.
    Checks:
      1. Prefixed env var: SCI_{SECTION}_{KEY} (e.g. SCI_VISION_MODEL)
      2. Short env var: {SECTION}_{KEY} (e.g. VISION_MODEL)
      3. config.ini value
      4. Default fallback
    """
    env_short = f"{section.upper()}_{key.upper()}"
    for env_name in (f"SCI_{env_short}", env_short):
        if env_name in os.environ:
            return os.environ[env_name]

    return config.get(section, key, fallback=default)


def get_config_float(section: str, key: str, default: float) -> float:
    val = get_config_value(section, key, str(default))
    try:
        return float(val)
    except ValueError:
        return default


def get_config_int(section: str, key: str, default: int) -> int:
    val = get_config_value(section, key, str(default))
    try:
        return int(val)
    except ValueError:
        return default


def resolve_path(path_str: str | Path) -> Path:
    """Resolve path relative to project root if it is relative."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


# --- Vision Settings ---
VISION_MODEL = get_config_value("vision", "model", "qwen/qwen3.5-flash-02-23")
VISION_API_DELAY = get_config_float("vision", "api_delay", 2.0)
VISION_MAX_RETRIES = get_config_int("vision", "max_retries", 3)
VISION_MAX_TOKENS = get_config_int("vision", "max_tokens", 2048)

# --- Pipeline Settings ---
PDF_INPUT_DIR = str(resolve_path(get_config_value("pipeline", "pdf_input_dir", "pdfs")))
OUTPUT_DIR = str(resolve_path(get_config_value("pipeline", "output_dir", "output")))
IMAGES_SUBDIR = get_config_value("pipeline", "images_subdir", "extracted_images")
ANALYSIS_SUBDIR = get_config_value("pipeline", "analysis_subdir", "image_process")
DB_PATH = str(resolve_path(get_config_value("pipeline", "db_path", "pdf_processing.db")))

# --- Storage Settings ---
STORAGE_BACKENDS = ("chroma", "postgres")
STORAGE_BACKEND = get_config_value("storage", "backend", "chroma").strip().lower()
# chroma_path used to live in [pipeline]; keep honouring it there as a fallback.
CHROMA_PATH = str(resolve_path(get_config_value(
    "storage", "chroma_path", get_config_value("pipeline", "chroma_path", "chroma_db")
)))
POSTGRES_DSN = get_config_value(
    "storage", "postgres_dsn", "postgresql://sci:sci@localhost:5432/sci_vizio"
)
POSTGRES_TABLE = get_config_value("storage", "postgres_table", "image_analysis")
EMBEDDING_MODEL = get_config_value(
    "storage", "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
)


def get_images_dir(base_output: str = None) -> Path:
    """Get the path to the directory where PDF images are extracted."""
    base = resolve_path(base_output or OUTPUT_DIR)
    return base / IMAGES_SUBDIR


def get_analysis_dir(base_output: str = None) -> Path:
    """Get the path to the directory where LLM image analyses are stored."""
    base = resolve_path(base_output or OUTPUT_DIR)
    return base / ANALYSIS_SUBDIR
