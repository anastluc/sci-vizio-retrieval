"""
Sci-Vizio-Retrieval Unified CLI

Provides a single entrypoint to run the entire extraction, processing,
and indexing pipeline, as well as running individual stages or launching
the Gradio query interface.
"""

import argparse
import logging
import sys
from pathlib import Path

from sci_vizio_retrieval.config import (
    CHROMA_PATH,
    DB_PATH,
    OUTPUT_DIR,
    PDF_INPUT_DIR,
    POSTGRES_TABLE,
    STORAGE_BACKEND,
    STORAGE_BACKENDS,
    VISION_MODEL,
    get_analysis_dir,
    get_images_dir,
    resolve_path,
)


def setup_logging(verbose=False):
    """Setup application-wide logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logs_dir = resolve_path("logs")
    logs_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / "pipeline.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def make_store(args, index_images=True):
    """Build the vector store selected by the storage CLI arguments."""
    from sci_vizio_retrieval.stores import create_vector_store

    return create_vector_store(
        args.backend,
        chroma_path=args.chroma_path,
        postgres_dsn=args.postgres_dsn,
        postgres_table=args.postgres_table,
        index_images=index_images,
    )


def run_indexing(args, pdf_names=None):
    from sci_vizio_retrieval.indexer import ImageAnalysisIndexer

    with make_store(args) as store:
        indexer = ImageAnalysisIndexer(db_path=args.db_path, vector_store=store)
        return indexer.process_all_analyses(pdf_names=pdf_names, reindex=args.reindex)


def cmd_run(args):
    """Execute the full pipeline sequentially."""
    from sci_vizio_retrieval.extractor import PDFProcessor
    from sci_vizio_retrieval.processor import ImageProcessor

    logging.info("Starting overall pipeline execution...")

    # Scan input directory for PDFs to restrict processing scope
    input_path = Path(args.input_dir)
    pdf_files = list(input_path.rglob("*.pdf"))
    pdf_names = [f.stem for f in pdf_files]
    logging.info(f"Found {len(pdf_names)} PDFs in input directory: {pdf_names}")

    # 1. PDF Extraction
    logging.info("--- Stage 1: PDF Extraction ---")
    extractor = PDFProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        db_path=args.db_path,
        delete_duplicates=args.delete_duplicates,
    )
    extract_stats = extractor.process_directory()
    logging.info(f"Extraction stats: {extract_stats}")

    # 2. Vision LLM Processing
    logging.info("--- Stage 2: Vision LLM Processing ---")
    processor = ImageProcessor(
        model=args.model,
        images_dir=extractor.images_dir,
        output_dir=get_analysis_dir(args.output_dir),
        db_path=args.db_path
    )
    proc_stats = processor.process_directory(pdf_names=pdf_names)
    logging.info(f"Processing stats: {proc_stats}")

    # 3. Vector Indexing
    logging.info(f"--- Stage 3: Vector Indexing ({args.backend}) ---")
    idx_stats = run_indexing(args, pdf_names=pdf_names)
    logging.info(f"Indexing stats: {idx_stats}")

    logging.info("Overall pipeline completed successfully!")


def cmd_extract(args):
    """Execute PDF extraction stage only."""
    from sci_vizio_retrieval.extractor import PDFProcessor

    logging.info("Running PDF extraction stage...")
    extractor = PDFProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        db_path=args.db_path,
        delete_duplicates=args.delete_duplicates,
    )
    stats = extractor.process_directory()
    logging.info(f"Extraction summary: {stats}")


def cmd_process(args):
    """Execute image processing stage only."""
    from sci_vizio_retrieval.processor import ImageProcessor, ImageProcessorRetry

    # Resolve directories
    images_dir = args.images_dir or get_images_dir(args.output_dir)
    analysis_dir = args.analysis_dir or get_analysis_dir(args.output_dir)

    if args.retry:
        logging.info("Running image processing retry stage...")
        processor = ImageProcessorRetry(
            model=args.model,
            images_dir=images_dir,
            output_dir=analysis_dir,
            db_path=args.db_path
        )
        stats = processor.process_failed_entries()
        logging.info(f"Reprocessing summary: {stats}")
    else:
        logging.info("Running image processing stage...")
        processor = ImageProcessor(
            model=args.model,
            images_dir=images_dir,
            output_dir=analysis_dir,
            db_path=args.db_path
        )
        stats = processor.process_directory()
        logging.info(f"Processing summary: {stats}")


def cmd_index(args):
    """Execute indexing stage only."""
    logging.info(f"Running vector indexing stage ({args.backend})...")
    stats = run_indexing(args)
    logging.info(f"Indexing summary: {stats}")


def cmd_serve(args):
    """Start Gradio UI service."""
    from sci_vizio_retrieval.ui import launch_ui

    logging.info(f"Starting Gradio search UI ({args.backend})...")
    launch_ui(
        store=make_store(args, index_images=False),
        share=args.share,
        server_port=args.port
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sci-Vizio-Retrieval: Scientific paper image extraction, analysis & search pipeline."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Pipeline command to execute")

    # Common arguments definition
    def add_common_pipeline_args(sub_parser):
        sub_parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Base directory for output files")
        sub_parser.add_argument("--db-path", default=DB_PATH, help="Path to SQLite pipeline-state database")

    def add_storage_args(sub_parser):
        group = sub_parser.add_argument_group("vector storage")
        group.add_argument("--backend", choices=STORAGE_BACKENDS, default=STORAGE_BACKEND,
                           help=f"Vector store backend (default: {STORAGE_BACKEND})")
        group.add_argument("--chroma-path", default=CHROMA_PATH, help="Path to ChromaDB persistent storage")
        group.add_argument("--postgres-dsn", default=None,
                           help="PostgreSQL connection string (default: from config / SCI_STORAGE_POSTGRES_DSN)")
        group.add_argument("--postgres-table", default=POSTGRES_TABLE, help="PostgreSQL table name")

    def add_indexing_args(sub_parser):
        sub_parser.add_argument("--reindex", action="store_true",
                                help="Re-index all analyses, even ones already indexed in this store")

    def add_extraction_args(sub_parser):
        sub_parser.add_argument("--input-dir", default=PDF_INPUT_DIR, help="Directory containing input PDFs")
        sub_parser.add_argument("--delete-duplicates", action="store_true",
                                help="Delete input PDFs whose content duplicates an already-processed PDF")

    # Command: run (overall pipeline)
    parser_run = subparsers.add_parser("run", help="Run the entire pipeline in one go (extract, process, index)")
    add_extraction_args(parser_run)
    parser_run.add_argument("--model", default=VISION_MODEL, help="OpenRouter model slug")
    add_common_pipeline_args(parser_run)
    add_storage_args(parser_run)
    add_indexing_args(parser_run)
    parser_run.set_defaults(func=cmd_run)

    # Command: extract
    parser_extract = subparsers.add_parser("extract", help="Extract text and images from PDFs")
    add_extraction_args(parser_extract)
    add_common_pipeline_args(parser_extract)
    parser_extract.set_defaults(func=cmd_extract)

    # Command: process
    parser_process = subparsers.add_parser("process", help="Analyze extracted images with Vision LLM")
    parser_process.add_argument("--images-dir", help="Directory containing extracted images (defaults to config path)")
    parser_process.add_argument("--analysis-dir", help="Directory to save JSON analyses (defaults to config path)")
    parser_process.add_argument("--model", default=VISION_MODEL, help="OpenRouter model slug")
    parser_process.add_argument("--retry", action="store_true", help="Retry previously failed runs instead of new ones")
    add_common_pipeline_args(parser_process)
    parser_process.set_defaults(func=cmd_process)

    # Command: index
    parser_index = subparsers.add_parser("index", help="Index image analyses into the vector store")
    add_common_pipeline_args(parser_index)
    add_storage_args(parser_index)
    add_indexing_args(parser_index)
    parser_index.set_defaults(func=cmd_index)

    # Command: serve
    parser_serve = subparsers.add_parser("serve", help="Start the Gradio web search interface")
    add_storage_args(parser_serve)
    parser_serve.add_argument("--share", action="store_true", help="Generate a public Gradio share link")
    parser_serve.add_argument("--port", type=int, help="Gradio server port")
    parser_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)

    try:
        args.func(args)
    except Exception:
        logging.exception(f"Pipeline command '{args.command}' failed:")
        sys.exit(1)


if __name__ == "__main__":
    main()
