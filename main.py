#!/usr/bin/env python3
"""
Sci-Vizio-Retrieval Unified CLI

Provides a single entrypoint to run the entire extraction, processing,
and indexing pipeline, as well as running individual stages or launching
the Gradio query interface.
"""

import argparse
import sys
import logging
from pathlib import Path

from sci_vizio_retrieval.config import (
    PDF_INPUT_DIR,
    OUTPUT_DIR,
    DB_PATH,
    CHROMA_PATH,
    VISION_MODEL,
    get_images_dir,
    get_analysis_dir,
)
from sci_vizio_retrieval.extractor import PDFProcessor
from sci_vizio_retrieval.processor import ImageProcessor, ImageProcessorRetry
from sci_vizio_retrieval.indexer import ImageAnalysisIndexer
from sci_vizio_retrieval.ui import launch_ui

def setup_logging(verbose=False):
    """Setup application-wide logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("logs/pipeline.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def cmd_run(args):
    """Execute the full pipeline sequentially."""
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
        db_path=args.db_path
    )
    extract_stats = extractor.process_directory()
    logging.info(f"Extraction stats: {extract_stats}")
    
    # 2. Vision LLM Processing
    logging.info("--- Stage 2: Vision LLM Processing ---")
    # Resolve extracted images directory path
    images_dir = extractor.images_dir
    analysis_dir = Path(args.output_dir) / "image_process"
    
    processor = ImageProcessor(
        model=args.model,
        images_dir=images_dir,
        output_dir=analysis_dir,
        db_path=args.db_path
    )
    proc_stats = processor.process_directory(pdf_names=pdf_names)
    logging.info(f"Processing stats: {proc_stats}")
    
    # 3. ChromaDB Indexing
    logging.info("--- Stage 3: ChromaDB Indexing ---")
    indexer = ImageAnalysisIndexer(
        db_path=args.db_path,
        chroma_path=args.chroma_path
    )
    idx_stats = indexer.process_all_analyses(pdf_names=pdf_names)
    logging.info(f"Indexing stats: {idx_stats}")
    
    logging.info("Overall pipeline completed successfully!")

def cmd_extract(args):
    """Execute PDF extraction stage only."""
    logging.info("Running PDF extraction stage...")
    extractor = PDFProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        db_path=args.db_path
    )
    stats = extractor.process_directory()
    logging.info(f"Extraction summary: {stats}")

def cmd_process(args):
    """Execute image processing stage only."""
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
    logging.info("Running ChromaDB indexing stage...")
    indexer = ImageAnalysisIndexer(
        db_path=args.db_path,
        chroma_path=args.chroma_path
    )
    stats = indexer.process_all_analyses()
    logging.info(f"Indexing summary: {stats}")

def cmd_serve(args):
    """Start Gradio UI service."""
    logging.info("Starting Gradio search UI...")
    launch_ui(
        chroma_path=args.chroma_path,
        share=args.share,
        server_port=args.port
    )

def main():
    parser = argparse.ArgumentParser(
        description="Sci-Vizio-Retrieval: Scientific paper image extraction, analysis & search pipeline."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Pipeline command to execute")
    
    # Common arguments definition
    def add_common_pipeline_args(sub_parser):
        sub_parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Base directory for output files")
        sub_parser.add_argument("--db-path", default=DB_PATH, help="Path to SQLite database")

    # Command: run (overall pipeline)
    parser_run = subparsers.add_parser("run", help="Run the entire pipeline in one go (extract, process, index)")
    parser_run.add_argument("--input-dir", default=PDF_INPUT_DIR, help="Directory containing input PDFs")
    parser_run.add_argument("--model", default=VISION_MODEL, help="OpenRouter model slug")
    parser_run.add_argument("--chroma-path", default=CHROMA_PATH, help="Path to ChromaDB persistent storage")
    add_common_pipeline_args(parser_run)
    parser_run.set_defaults(func=cmd_run)
    
    # Command: extract
    parser_extract = subparsers.add_parser("extract", help="Extract text and images from PDFs")
    parser_extract.add_argument("--input-dir", default=PDF_INPUT_DIR, help="Directory containing input PDFs")
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
    parser_index = subparsers.add_parser("index", help="Index metadata and CLIP embeddings into ChromaDB")
    parser_index.add_argument("--chroma-path", default=CHROMA_PATH, help="Path to ChromaDB persistent storage")
    add_common_pipeline_args(parser_index)
    parser_index.set_defaults(func=cmd_index)
    
    # Command: serve
    parser_serve = subparsers.add_parser("serve", help="Start the Gradio web search interface")
    parser_serve.add_argument("--chroma-path", default=CHROMA_PATH, help="Path to ChromaDB persistent storage")
    parser_serve.add_argument("--share", action="store_true", help="Generate a public Gradio share link")
    parser_serve.add_argument("--port", type=int, help="Gradio server port")
    parser_serve.set_defaults(func=cmd_serve)
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    try:
        args.func(args)
    except Exception as e:
        logging.exception(f"Pipeline command '{args.command}' failed:")
        sys.exit(1)

if __name__ == "__main__":
    main()
