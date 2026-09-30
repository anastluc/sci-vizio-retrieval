"""Extract, describe, index and search images from scientific PDFs."""

from sci_vizio_retrieval.client import OpenRouterVision
from sci_vizio_retrieval.extractor import PDFProcessor
from sci_vizio_retrieval.indexer import ImageAnalysisIndexer
from sci_vizio_retrieval.processor import ImageProcessor, ImageProcessorRetry
from sci_vizio_retrieval.stores import IndexRecord, SearchResult, VectorStore, create_vector_store
from sci_vizio_retrieval.ui import ChromaDBQuerier, SearchService, launch_ui

__version__ = "0.2.0"

__all__ = [
    "OpenRouterVision",
    "PDFProcessor",
    "ImageProcessor",
    "ImageProcessorRetry",
    "ImageAnalysisIndexer",
    "IndexRecord",
    "SearchResult",
    "VectorStore",
    "create_vector_store",
    "ChromaDBQuerier",
    "SearchService",
    "launch_ui",
]
