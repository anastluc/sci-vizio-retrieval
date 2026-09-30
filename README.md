# Sci-Vizio-Retrieval

A scientific paper image analysis and semantic search pipeline. Extract images from research PDFs, describe them with Vision LLMs, index the descriptions in a vector database (**ChromaDB** or **PostgreSQL + pgvector**), and search across all analysed images using natural language.

![Gradio Search UI](docs/gradio_ui_screenshot.png)

## Overview

**Sci-Vizio-Retrieval** automates the process of extracting, understanding, and retrieving visual content from scientific literature. It features:

- **PDF image extraction** using PyMuPDF.
- **Vision LLM analysis** via [OpenRouter](https://openrouter.ai) — access 200+ models (Gemini, GPT-4o, Claude, Llama, etc.) through a single API key.
- **Pluggable vector storage** for semantic search:
  - **ChromaDB** (default) — embedded, zero-setup, file-based.
  - **PostgreSQL + pgvector** — a real database server: concurrent access, backups, SQL over results, HNSW index.
- **Gradio web UI** for interactive natural language search.
- **Robust configuration system** supporting both `config.ini` and environment variable overrides.

For how this compares with current academic and industry systems, see [docs/STATE_OF_THE_ART.md](docs/STATE_OF_THE_ART.md).

The project ships in two forms:

* **Command-line pipeline** (this README): single-user, runs on your machine against a local PDF folder.
* **Web app** (`sci_vizio_saas/` + `frontend/`): a multi-user service where people sign in with Firebase, upload PDFs into collections, and search their figures with filters. It has a free plan and a paid Premium plan (Stripe), keeps all data in PostgreSQL, and stores files in object storage. It deploys the frontend to Firebase Hosting and the API, worker and Postgres to Heroku. See **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** for architecture, setup and deployment.

---

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌────────────┐    ┌──────────────────────────┐
│  PDF Files  │───▶│  Image Extractor │───▶│  Vision    │───▶│  Indexer                 │
│  (pdfs/)    │    │  (PyMuPDF)       │    │  LLM API   │    │   └─ VectorStore         │
└─────────────┘    └──────────────────┘    └────────────┘    │       ├─ ChromaDB        │
                           │                     │           │       └─ Postgres+pgvector│
             output/extracted_images   output/image_process  └────────────┬─────────────┘
                                                                          │
                   pdf_processing.db (SQLite): pipeline state      ┌──────▼──────────┐
                   for every stage                                 │  Gradio Web UI  │
                                                                   │ (semantic search)│
                                                                   └─────────────────┘
```

Pipeline state (which PDFs were extracted, which images were analysed, which analyses are indexed *in which store*) is kept in a local SQLite database regardless of the vector backend. Because indexing status is tracked per store, switching from Chroma to Postgres (or to another Chroma directory / Postgres table) indexes everything into the new store instead of silently skipping it.

### Storage backends

| | ChromaDB (`chroma`) | PostgreSQL (`postgres`) |
|---|---|---|
| Setup | None (files in `chroma_db/`) | PostgreSQL ≥ 13 with the [pgvector](https://github.com/pgvector/pgvector) extension (`docker compose up -d`) |
| Text embedding | Chroma default (`all-MiniLM-L6-v2`, ONNX) | `sentence-transformers` (`all-MiniLM-L6-v2` by default, configurable) |
| Image (CLIP) embeddings | Yes, second collection (OpenCLIP) | No — the raw image is stored in a `BYTEA` column |
| Index | HNSW (Chroma built-in) | HNSW, cosine distance |
| Schema | Two collections, JSON in documents/metadata | One table: `analysis JSONB`, `image_data BYTEA`, `embedding vector(n)`, `embedding_model` |
| Good for | Single-user, local exploration | Shared/multi-user deployments, larger corpora, joining with other data |

Both backends embed the same text (the vision model's JSON output), so their results are comparable. The Postgres store refuses to mix embeddings from different models or dimensions in one table — use a different `postgres_table` (or drop it and `--reindex`) when you change `embedding_model`.

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
| `[pipeline]` | `db_path` | `pdf_processing.db` | Path to SQLite database tracking pipeline state |
| `[storage]` | `backend` | `chroma` | Vector store: `chroma` or `postgres` |
| `[storage]` | `chroma_path` | `chroma_db` | Persistent ChromaDB directory (also read from `[pipeline]` for older configs) |
| `[storage]` | `postgres_dsn` | `postgresql://sci:sci@localhost:5432/sci_vizio` | PostgreSQL connection string |
| `[storage]` | `postgres_table` | `image_analysis` | Table holding indexed analyses |
| `[storage]` | `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | Text embedding model for the Postgres backend |

### Environment Overrides

Any configuration option can be overridden using environment variables (also read from `.env`). The configuration loader checks in the following order:
1. **Prefixed Environment Variable:** `SCI_{SECTION}_{KEY}` (e.g., `SCI_VISION_MODEL=openai/gpt-4o`, `SCI_STORAGE_BACKEND=postgres`)
2. **Short Environment Variable:** `{SECTION}_{KEY}` (e.g., `VISION_MODEL=openai/gpt-4o`, `STORAGE_POSTGRES_DSN=...`)
3. **INI Value:** Reads from `config.ini`
4. **Fallback:** Application default values

Keep database passwords out of `config.ini`: set `SCI_STORAGE_POSTGRES_DSN` in `.env` instead.

Paths specified in configuration are resolved relative to the project root directory, making executions safe from any working directory.

---

## Setup

### 1. Clone and create virtual environment

Requires Python 3.10+.

```bash
git clone <repository-url>
cd sci-vizio-retrieval
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

Everything (both backends, UI, ML models):

```bash
pip install -r requirements.txt
```

Or install only what you need via extras (this also provides a `sci-vizio` command equivalent to `python main.py`):

```bash
pip install -e ".[ml,chroma,ui]"       # original ChromaDB setup
pip install -e ".[ml,postgres,ui]"     # PostgreSQL setup
pip install -e ".[all,dev]"            # everything + test tools
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

### 4. (Optional) Use PostgreSQL instead of ChromaDB

Start a local PostgreSQL with pgvector:

```bash
docker compose up -d
```

and select the backend in `.env` (or in `config.ini` under `[storage]`):

```
SCI_STORAGE_BACKEND=postgres
SCI_STORAGE_POSTGRES_DSN=postgresql://sci:sci@localhost:5432/sci_vizio
```

Any PostgreSQL server with pgvector ≥ 0.5 works (managed services such as AWS RDS, Cloud SQL, Azure, Supabase and Neon all offer it). The extension, table and indexes are created automatically on first use; the database user needs permission to run `CREATE EXTENSION vector` unless an administrator has already enabled it.

To move an existing Chroma index to Postgres, just run `python main.py index --backend postgres` — analyses are re-indexed from the SQLite state database, no vision-LLM calls are repeated.

### 5. Place PDF files

Put your scientific paper PDFs into the input folder (defaults to `pdfs/`, created automatically on run).

---

## CLI Usage (`main.py`)

The root-level CLI script `main.py` provides commands to run either the entire pipeline at once or execute individual stages. Run `python main.py <command> --help` for all options.

### Run the overall pipeline in one go

Execute the complete extraction, vision description, and indexing sequential flow:

```bash
python main.py run
python main.py run --backend postgres
```

*Note: Scoped execution is enabled by default during the `run` command—it will only process images and index metadata corresponding to the PDFs found in the input folder.*

### Run individual pipeline stages

**1. PDF Extraction**
```bash
python main.py extract --input-dir pdfs/
```
Extracts text and embedded images from PDFs to `output/extracted_text/` and `output/extracted_images/`. PDFs whose content duplicates an already-processed PDF are skipped; pass `--delete-duplicates` to also delete them from the input folder.

**2. Vision LLM Processing**
```bash
python main.py process --model qwen/qwen3.5-flash-02-23
```
Sends extracted images to the vision model and saves JSON descriptions to `output/image_process/`. Add `--retry` to reprocess only failed entries.

**3. Vector Database Indexing**
```bash
python main.py index                                  # configured backend (chroma by default)
python main.py index --backend postgres \
    --postgres-dsn postgresql://sci:sci@localhost:5432/sci_vizio
```
Validates the JSON results and indexes them into the selected vector store. With ChromaDB this builds both a text-description collection and a CLIP image-embedding collection. Only analyses not yet indexed *in that store* are processed; add `--reindex` to rebuild everything.

**4. Start the search UI**
```bash
python main.py serve --port 7860
python main.py serve --backend postgres
```
Starts the interactive Gradio query web application. Papers named after arXiv identifiers (e.g. `1903.00780.pdf`) link to their arXiv page.

---

## Development

```bash
pip install -r requirements-dev.txt     # or: pip install -e ".[all,dev]"
pytest                                  # unit + ChromaDB tests
ruff check sci_vizio_retrieval tests
```

The tests use deterministic fake embedders and a fake vision client, so they need no API key, GPU or model downloads. PostgreSQL tests run when a pgvector-enabled database is available:

```bash
docker compose up -d
SCI_TEST_POSTGRES_DSN=postgresql://sci:sci@localhost:5432/sci_vizio pytest
```

Each test uses its own uniquely named table and drops it afterwards. CI (`.github/workflows/tests.yml`) runs the full suite, including PostgreSQL, on every push and pull request.

### Adding a storage backend

Implement `sci_vizio_retrieval.stores.base.VectorStore` (`name`, `upsert`, `search`, `count`, optionally `close`), register it in `create_vector_store`, and add it to the `store` fixture in `tests/conftest.py` — the shared contract tests in `tests/test_vector_stores.py` then run against it.

---

## Project Structure

```
sci-vizio-retrieval/
├── main.py                       # CLI entrypoint (wraps sci_vizio_retrieval.cli)
├── config.ini                    # Central pipeline configuration (auto-generated)
├── pyproject.toml                # Packaging, optional extras, pytest & ruff config
├── requirements.txt              # Full runtime dependencies
├── requirements-dev.txt          # + test/lint tools
├── docker-compose.yml            # Local PostgreSQL + pgvector
├── .env-dist                     # Template secrets file
│
├── sci_vizio_retrieval/          # Core Python Package
│   ├── __init__.py               # Package exports
│   ├── cli.py                    # Argument parsing & pipeline commands
│   ├── config.py                 # Config loader with env overrides & root resolution
│   ├── client.py                 # OpenRouter API Vision Client
│   ├── extractor.py              # PDFProcessor class
│   ├── processor.py              # ImageProcessor and ImageProcessorRetry classes
│   ├── indexer.py                # ImageAnalysisIndexer (backend-agnostic)
│   ├── embeddings.py             # Text embedders (sentence-transformers)
│   ├── stores/                   # Vector storage backends
│   │   ├── base.py               # VectorStore interface, IndexRecord, SearchResult
│   │   ├── chroma.py             # ChromaVectorStore
│   │   └── postgres.py           # PostgresVectorStore (pgvector)
│   └── ui.py                     # Gradio app & SearchService
│
├── sci_vizio_saas/               # Web app backend (FastAPI API + worker)
│   ├── main.py                   # App factory (uvicorn sci_vizio_saas.main:create_app --factory)
│   ├── worker.py                 # Background PDF processing (Postgres job queue)
│   ├── api/                      # Routes: account, collections, documents, search, billing
│   ├── models.py, migrations/    # SQLAlchemy schema + Alembic migrations
│   ├── auth.py, billing.py       # Firebase token verification, Stripe
│   ├── storage.py                # S3-compatible / local object storage with signed URLs
│   └── plans.py, search.py, processing.py
├── frontend/                     # React + TypeScript web UI (Vite), deployed to Firebase Hosting
├── Dockerfile, heroku.yml        # Heroku container deploy (web, worker, release migrations)
├── firebase.json                 # Hosting + Auth emulator config
├── requirements-saas.txt         # Web app dependencies (no torch / chromadb)
│
├── tests/                        # pytest suite (tests/saas: web backend)
├── docs/
│   ├── DEPLOYMENT.md             # Web app architecture, setup & deployment
│   └── STATE_OF_THE_ART.md       # Landscape & relevance report
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

## Changes in 0.2.0

- New multi-user web app with Firebase authentication, collections, faceted figure search, and free and Premium plans; see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

- New PostgreSQL + pgvector storage backend (`--backend postgres` / `[storage] backend`).
- Indexing status is tracked per vector store; `--reindex` forces a rebuild. Existing Chroma indexing state is migrated automatically.
- Indexing uses upserts, so re-running after a partial failure no longer leaves duplicate or conflicting entries.
- Duplicate input PDFs are no longer deleted unless `--delete-duplicates` is passed.
- Short environment overrides are now `{SECTION}_{KEY}` (e.g. `VISION_MODEL`), as documented; previously the code read the bare key (`MODEL`).
- Transparent images are flattened onto white instead of black when extracted.
- PNG images are sent to the vision model with the correct MIME type.
- The real vision-API error is stored instead of a generic "status 500" message.
- The search UI escapes model output and links arXiv papers correctly.
- torch/torchvision, ChromaDB, psycopg and Gradio are imported only when used.
- The fixed 1 s / 5 s sleeps between images were removed; use `[vision] api_delay` to throttle.

---

## License

This project is for research and educational purposes.
