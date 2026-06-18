# Sci-Vizio-Retrieval

A scientific paper image analysis and semantic search pipeline. Extract images from research PDFs, describe them with Vision LLMs, index the descriptions in a vector database, and search across all analysed images using natural language.

![Gradio Search UI](docs/gradio_ui_screenshot.png)

## Overview

**Sci-Vizio-Retrieval** automates the process of extracting, understanding, and retrieving visual content from scientific literature. It features:

- **PDF image extraction** using PyMuPDF.
- **Vision LLM analysis** via [OpenRouter](https://openrouter.ai) — access 200+ models (Gemini, GPT-4o, Claude, Llama, etc.) through a single API key.
- **Vector indexing** via ChromaDB for semantic search.
- **Gradio web UI** for interactive natural language search.
- **Robust configuration system** supporting both `config.ini` and environment variable overrides.

---

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌───────────┐     ┌────────────┐
│  PDF Files   │────▶│  Image Extractor  │────▶│  Vision    │────▶│  ChromaDB  │
│  (pdfs/)     │     │  (PyMuPDF)        │     │  LLM API   │     │  Indexer   │
└─────────────┘     └──────────────────┘     └───────────┘     └────────────┘
                            │                       │                  │
                      output/extracted_images   output/image_process    │
                                                                        │
                                                               ┌────────▼────────┐
                                                               │   Gradio Web UI  │
                                                               │  (Semantic Search)│
                                                               └─────────────────┘
```

---

## Configuration (`config.ini` & Environment Variables)

The pipeline is controlled by a central `config.ini` file. If this file is missing, the application automatically generates a default template on the first run.

### Configuration Options

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `[vision]` | `model` | `qwen/qwen3.5-flash-02-23` | OpenRouter model identifier slug |
| `[vision]` | `api_delay` | `2.0` | Seconds to wait between API calls to avoid rate limits |
| `[vision]` | `max_retries` | `3` | Retry attempts on transient API failures |
| `[vision]` | `max_tokens` | `2048` | Maximum output token size |
| `[pipeline]` | `pdf_input_dir` | `pdfs` | Input directory containing research papers |
| `[pipeline]` | `output_dir` | `output` | Base output directory for pipeline runs |
| `[pipeline]` | `images_subdir` | `extracted_images` | Subfolder inside output_dir for extracted images |
| `[pipeline]` | `analysis_subdir`| `image_process` | Subfolder inside output_dir for JSON descriptions |
| `[pipeline]` | `db_path` | `pdf_processing.db` | Path to SQLite database tracking file state |
| `[pipeline]` | `chroma_path` | `chroma_db` | Persistent ChromaDB vector database directory |

### Environment Overrides

Any configuration option can be overridden using environment variables. The configuration loader checks in the following order:
1. **Prefixed Environment Variable:** `SCI_{SECTION}_{KEY}` (e.g., `SCI_VISION_MODEL=openai/gpt-4o`)
2. **Short Environment Variable:** `{KEY}` (e.g., `VISION_MODEL=openai/gpt-4o`)
3. **INI Value:** Reads from `config.ini`
4. **Fallback:** Application default values

Additionally, paths specified in configuration are resolved relative to the project root directory, making executions safe from any working directory.

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone <repository-url>
cd sci-vizio-retrieval
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure secrets

Copy the template secrets file:

```bash
cp .env-dist .env
```

Edit the generated `.env` file and add your OpenRouter API key:

```
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxx
```

Get an API key at [openrouter.ai/keys](https://openrouter.ai/keys).

### 4. Place PDF files

Put your scientific paper PDFs into the input folder (defaults to `pdfs/`, created automatically on run).

---

## CLI Usage (`main.py`)

The root-level CLI script `main.py` provides commands to run either the entire pipeline at once or execute individual stages.

### Run the overall pipeline in one go

Execute the complete extraction, vision description, and ChromaDB indexing sequential flow:

```bash
python3 main.py run
```

*Note: Scoped execution is enabled by default during the `run` command—it will only process images and index metadata corresponding to the PDFs found in the input folder.*

### Run individual pipeline stages

**1. PDF Extraction**
```bash
python3 main.py extract --input-dir pdfs/
```
Extracts text and embedded images from PDFs to `output/extracted_text/` and `output/extracted_images/`.

**2. Vision LLM Processing**
```bash
python3 main.py process --model qwen/qwen3.5-flash-02-23
```
Sends extracted images to the vision model and saves JSON descriptions to `output/image_process/`. Add `--retry` to reprocess only failed entries.

**3. Vector Database Indexing**
```bash
python3 main.py index
```
Validates the JSON results and builds vector indices in ChromaDB (supporting both CLIP-based visual searches and text descriptions).

**4. Start gradu search UI**
```bash
python3 main.py serve --port 7860
```
Starts the interactive Gradio query web application.

---

## Project Structure

```
sci-vizio-retrieval/
├── main.py                       # Unified CLI pipeline entrypoint
├── config.ini                    # Central pipeline configuration (auto-generated)
├── requirements.txt              # Project dependencies
├── .env-dist                     # Template secrets file
│
├── sci_vizio_retrieval/          # Core Python Package
│   ├── __init__.py               # Package exports
│   ├── config.py                 # Config loader with env overrides & root resolution
│   ├── client.py                 # OpenRouter API Vision Client
│   ├── extractor.py              # PDFProcessor class
│   ├── processor.py              # ImageProcessor and ImageProcessorRetry classes
│   ├── indexer.py                # ImageAnalysisIndexer class
│   └── ui.py                     # Gradio app & ChromaDBQuerier
│
├── gradio_app.py                 # Backward-compatibility wrapper (UI Search)
├── pdf_image_extractor.py        # Backward-compatibility wrapper (Extraction)
├── image_processor.py            # Backward-compatibility wrapper (Vision processing)
├── image_processor_retry.py      # Backward-compatibility wrapper (Retries)
├── indexer.py                    # Backward-compatibility wrapper (Indexing)
└── openrouter_client.py          # Backward-compatibility wrapper (Client API)
```

### Data Directories (gitignored)

| Directory | Purpose |
|-----------|---------|
| `pdfs/` | Input PDF files |
| `output/extracted_images/` | Extracted images per PDF |
| `output/extracted_text/` | Extracted text per PDF |
| `output/image_process/` | Vision LLM JSON responses |
| `chroma_db/` | ChromaDB vector database |
| `logs/` | Processing logs |

---

## License

This project is for research and educational purposes.
