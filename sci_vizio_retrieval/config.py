import configparser
import os
from pathlib import Path
from dotenv import load_dotenv

# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# Default config content to write if config.ini doesn't exist
DEFAULT_CONFIG_CONTENT = """# Sci-Vizio-Retrieval Configuration
# -----------------------------------
# This file controls pipeline settings. Edit the values below as needed.

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

# ChromaDB persistent vector database directory.
chroma_path = chroma_db
"""

config = configparser.ConfigParser()
config_file = PROJECT_ROOT / "config.ini"

if not config_file.exists():
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(DEFAULT_CONFIG_CONTENT)

config.read(config_file)

def get_config_value(section: str, key: str, default: str) -> str:
    """
    Get config value with support for environment variable overrides.
    Checks:
      1. Prefixed env var: SCI_{SECTION}_{KEY} (e.g. SCI_VISION_MODEL)
      2. Key-only env var: {KEY} (e.g. VISION_MODEL)
      3. config.ini value
      4. Default fallback
    """
    # 1. SCI_{SECTION}_{KEY}
    env_prefixed = f"SCI_{section.upper()}_{key.upper()}"
    if env_prefixed in os.environ:
        return os.environ[env_prefixed]
        
    # 2. {KEY}
    env_key = f"{key.upper()}"
    if env_key in os.environ:
        return os.environ[env_key]
        
    # 3. config.ini
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

def resolve_path(path_str: str) -> Path:
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
CHROMA_PATH = str(resolve_path(get_config_value("pipeline", "chroma_path", "chroma_db")))

def get_images_dir(base_output: str = None) -> Path:
    """Get the path to the directory where PDF images are extracted."""
    base = resolve_path(base_output or OUTPUT_DIR)
    return base / IMAGES_SUBDIR

def get_analysis_dir(base_output: str = None) -> Path:
    """Get the path to the directory where LLM image analyses are stored."""
    base = resolve_path(base_output or OUTPUT_DIR)
    return base / ANALYSIS_SUBDIR
