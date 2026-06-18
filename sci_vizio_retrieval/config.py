import configparser
from pathlib import Path

# Load config if exists, otherwise use defaults
config = configparser.ConfigParser()
config_file = Path("config.ini")

if config_file.exists():
    config.read(config_file)

# --- Vision Settings ---
VISION_MODEL = config.get("vision", "model", fallback="qwen/qwen3.5-flash-02-23")
VISION_API_DELAY = config.getfloat("vision", "api_delay", fallback=2.0)
VISION_MAX_RETRIES = config.getint("vision", "max_retries", fallback=3)
VISION_MAX_TOKENS = config.getint("vision", "max_tokens", fallback=2048)

# --- Pipeline Settings ---
PDF_INPUT_DIR = config.get("pipeline", "pdf_input_dir", fallback="pdfs")
OUTPUT_DIR = config.get("pipeline", "output_dir", fallback="output")
IMAGES_SUBDIR = config.get("pipeline", "images_subdir", fallback="extracted_images")
ANALYSIS_SUBDIR = config.get("pipeline", "analysis_subdir", fallback="image_process")
DB_PATH = config.get("pipeline", "db_path", fallback="pdf_processing.db")
CHROMA_PATH = config.get("pipeline", "chroma_path", fallback="chroma_db")

def get_images_dir(base_output: str = None) -> Path:
    """Get the path to the directory where PDF images are extracted."""
    base = Path(base_output or OUTPUT_DIR)
    return base / IMAGES_SUBDIR

def get_analysis_dir(base_output: str = None) -> Path:
    """Get the path to the directory where LLM image analyses are stored."""
    base = Path(base_output or OUTPUT_DIR)
    return base / ANALYSIS_SUBDIR
