import hashlib
import io
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, List

import pymupdf
from PIL import Image

from sci_vizio_retrieval.config import (
    DB_PATH,
    IMAGES_SUBDIR,
    OUTPUT_DIR,
    PDF_INPUT_DIR,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractedImage:
    """An image embedded in a PDF page."""

    page: int   # 1-based page number
    index: int  # 1-based position among the page's images
    ext: str
    data: bytes = field(repr=False)
    width: int
    height: int

    @property
    def filename(self) -> str:
        return f"page{self.page}_img{self.index}.{self.ext}"


def optimize_image_bytes(data: bytes) -> bytes:
    """Re-encode an image compressed, flattening transparency onto white.

    Returns the input unchanged if Pillow can't re-encode it (e.g. JPX, JBIG2).
    """
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
        fmt = img.format
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, 'white')
            background.paste(img, mask=img.getchannel('A'))
            img = background
        out = io.BytesIO()
        img.save(out, format=fmt, optimize=True, quality=85)
        return out.getvalue()
    except Exception as e:
        logger.warning(f"Could not optimize image: {e}")
        return data


def iter_pdf_images(pdf_document, min_side: int = 0, optimize: bool = True) -> Iterator[ExtractedImage]:
    """
    Yield the raster images embedded in an open PyMuPDF document.

    Args:
        pdf_document: An open ``pymupdf.Document``.
        min_side: Skip images whose width or height is below this many pixels
            (icons, logos, rule lines).
        optimize: Re-encode images with :func:`optimize_image_bytes`.
    """
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        for img_index, img_info in enumerate(page.get_images(full=True)):
            base_image = pdf_document.extract_image(img_info[0])
            if not base_image:
                continue
            width, height = base_image.get("width", 0), base_image.get("height", 0)
            if min(width, height) < min_side:
                continue
            data = base_image["image"]
            yield ExtractedImage(
                page=page_num + 1,
                index=img_index + 1,
                ext=base_image["ext"],
                data=optimize_image_bytes(data) if optimize else data,
                width=width,
                height=height,
            )


def extract_pdf_images(pdf: bytes | str | Path, min_side: int = 0, optimize: bool = True) -> List[ExtractedImage]:
    """Extract the raster images from a PDF given as a path or in-memory bytes."""
    if isinstance(pdf, (bytes, bytearray)):
        document = pymupdf.open(stream=bytes(pdf), filetype="pdf")
    else:
        document = pymupdf.open(str(pdf))
    with document:
        return list(iter_pdf_images(document, min_side=min_side, optimize=optimize))


class PDFProcessor:
    def __init__(self, input_dir=None, output_dir=None, db_path=None, delete_duplicates=False):
        """
        Initialize the PDF processor with directories and database connection.
        
        Args:
            input_dir (str/Path): Directory containing PDF files
            output_dir (str/Path): Base directory for extracted content
            db_path (str): Path to SQLite database
            delete_duplicates (bool): Delete input PDFs whose content duplicates
                an already-processed PDF. Duplicates are always skipped.
        """
        self.input_dir = Path(input_dir or PDF_INPUT_DIR)
        self.output_dir = Path(output_dir or OUTPUT_DIR)
        self.images_dir = self.output_dir / IMAGES_SUBDIR
        self.text_dir = self.output_dir / "extracted_text"
        self.db_path = db_path or DB_PATH
        self.delete_duplicates = delete_duplicates
        
        # Create necessary directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database and create necessary tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            
            # Create main processing table with hash column
            cur.execute('''
                CREATE TABLE IF NOT EXISTS pdf_processing (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pdf_path TEXT UNIQUE,
                    pdf_hash TEXT,
                    process_timestamp TEXT,
                    text_extracted BOOLEAN,
                    images_extracted BOOLEAN,
                    image_info TEXT,
                    error_message TEXT,
                    is_duplicate BOOLEAN DEFAULT FALSE,
                    original_path TEXT
                )
            ''')
            
            # Create index on pdf_hash for faster duplicate checking
            cur.execute('''
                CREATE INDEX IF NOT EXISTS idx_pdf_hash 
                ON pdf_processing(pdf_hash)
            ''')
            
            conn.commit()

    def calculate_hash(self, file_path):
        """
        Calculate SHA-256 hash of a file.
        
        Args:
            file_path (Path): Path to the file
        
        Returns:
            str: Hexadecimal hash string
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def check_duplicate(self, pdf_path, pdf_hash):
        """
        Check if a PDF with the same hash exists in the database.
        
        Args:
            pdf_path (Path): Path to the PDF file
            pdf_hash (str): SHA-256 hash of the PDF
        
        Returns:
            tuple: (is_duplicate, original_path)
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT pdf_path FROM pdf_processing 
                WHERE pdf_hash = ? AND pdf_path != ? 
                AND is_duplicate = FALSE
                LIMIT 1
                """,
                (pdf_hash, str(pdf_path))
            )
            result = cur.fetchone()
            return (True, result[0]) if result else (False, None)

    def handle_duplicate(self, pdf_path, original_path):
        """
        Record a duplicate PDF in the database and, if enabled, delete it.
        
        Args:
            pdf_path (Path): Path to the duplicate PDF
            original_path (str): Path to the original PDF
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO pdf_processing 
                (pdf_path, pdf_hash, process_timestamp, text_extracted, 
                images_extracted, is_duplicate, original_path)
                VALUES (?, '', ?, FALSE, FALSE, TRUE, ?)
                """,
                (str(pdf_path), datetime.now().isoformat(), original_path)
            )
            conn.commit()

        if not self.delete_duplicates:
            return

        try:
            pdf_path.unlink()
            logger.info(f"Deleted duplicate file: {pdf_path}")
        except Exception as e:
            logger.warning(f"Could not delete duplicate file {pdf_path}: {e}")

    def is_processed(self, pdf_path):
        """
        Check if a PDF has already been processed.
        
        Args:
            pdf_path (Path): Path to the PDF file
        
        Returns:
            tuple: (bool, dict) - (is_processed, processing_details)
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM pdf_processing WHERE pdf_path = ?",
                (str(pdf_path),)
            )
            result = cur.fetchone()
            
            if result:
                return True, {
                    'id': result[0],
                    'pdf_path': result[1],
                    'pdf_hash': result[2],
                    'timestamp': result[3],
                    'text_extracted': bool(result[4]),
                    'images_extracted': bool(result[5]),
                    'image_info': json.loads(result[6]) if result[6] else None,
                    'error_message': result[7],
                    'is_duplicate': bool(result[8]),
                    'original_path': result[9]
                }
            return False, None

    def extract_text(self, pdf_path):
        """
        Extract text from PDF and save to file.
        
        Args:
            pdf_path (Path): Path to the PDF file
        
        Returns:
            tuple: (success, output_path or error_message)
        """
        try:
            text_content = []
            with pymupdf.open(str(pdf_path)) as pdf_document:
                for page_num in range(pdf_document.page_count):
                    page = pdf_document[page_num]
                    text_content.append(f"--- Page {page_num + 1} ---\n")
                    text_content.append(page.get_text())
                    text_content.append("\n\n")
            
            relative_path = pdf_path.stem
            output_path = self.text_dir / f"{relative_path}.txt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            output_path.write_text('\n'.join(text_content), encoding='utf-8')
            return True, output_path
            
        except Exception as e:
            return False, str(e)

    @staticmethod
    def _optimize_image(image_path):
        """Re-save an extracted image compressed, flattening transparency onto white."""
        image_path.write_bytes(optimize_image_bytes(image_path.read_bytes()))

    def extract_images(self, pdf_path):
        """
        Extract images from PDF and save to directory.
        
        Args:
            pdf_path (Path): Path to the PDF file
        
        Returns:
            tuple: (success, list of image paths or error message)
        """
        try:
            relative_path = pdf_path.stem
            pdf_output_dir = self.images_dir / relative_path
            pdf_output_dir.mkdir(parents=True, exist_ok=True)
            
            extracted_images = []
            for image in extract_pdf_images(pdf_path):
                image_path = pdf_output_dir / image.filename
                image_path.write_bytes(image.data)
                extracted_images.append({
                    'filename': image.filename,
                    'path': str(image_path.relative_to(self.output_dir))
                })

            return True, extracted_images
            
        except Exception as e:
            return False, str(e)

    def process_pdf(self, pdf_path):
        """
        Process a single PDF file, extracting text and images.
        
        Args:
            pdf_path (Path): Path to the PDF file
        
        Returns:
            dict: Processing results and statistics
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            return {
                'pdf_path': str(pdf_path),
                'error_message': 'File no longer exists'
            }
        
        is_processed, details = self.is_processed(pdf_path)
        if is_processed:
            logger.info(f"PDF already processed on {details['timestamp']}")
            return details
        
        try:
            pdf_hash = self.calculate_hash(pdf_path)
            is_duplicate, original_path = self.check_duplicate(pdf_path, pdf_hash)
            
            if is_duplicate:
                logger.info(f"Duplicate PDF found: {pdf_path}")
                logger.info(f"Original file: {original_path}")
                self.handle_duplicate(pdf_path, original_path)
                return {
                    'pdf_path': str(pdf_path),
                    'pdf_hash': pdf_hash,
                    'is_duplicate': True,
                    'original_path': original_path
                }
        except Exception as e:
            return {
                'pdf_path': str(pdf_path),
                'error_message': f"Error calculating hash: {str(e)}"
            }
        
        results = {
            'pdf_path': str(pdf_path),
            'pdf_hash': pdf_hash,
            'timestamp': datetime.now().isoformat(),
            'text_extracted': False,
            'images_extracted': False,
            'image_info': None,
            'error_message': None,
            'is_duplicate': False,
            'original_path': None
        }
        
        try:
            text_success, text_result = self.extract_text(pdf_path)
            results['text_extracted'] = text_success
            if not text_success:
                results['error_message'] = f"Text extraction failed: {text_result}"
            
            images_success, images_result = self.extract_images(pdf_path)
            results['images_extracted'] = images_success
            if images_success:
                results['image_info'] = images_result
            else:
                image_error = f"Image extraction failed: {images_result}"
                results['error_message'] = "; ".join(filter(None, [results['error_message'], image_error]))
            
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute('''
                    INSERT INTO pdf_processing 
                    (pdf_path, pdf_hash, process_timestamp, text_extracted, 
                    images_extracted, image_info, error_message, is_duplicate, original_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    str(pdf_path),
                    results['pdf_hash'],
                    results['timestamp'],
                    results['text_extracted'],
                    results['images_extracted'],
                    json.dumps(results['image_info']) if results['image_info'] else None,
                    results['error_message'],
                    results['is_duplicate'],
                    results['original_path']
                ))
                conn.commit()
            
        except Exception as e:
            results['error_message'] = str(e)
            
        return results

    def process_directory(self):
        """
        Process all PDF files in the input directory and its subdirectories.
        
        Returns:
            dict: Processing statistics
        """
        stats = {
            'total_pdfs': 0,
            'processed_pdfs': 0,
            'skipped_pdfs': 0,
            'failed_pdfs': 0,
            'duplicate_pdfs': 0,
            'total_images': 0,
            'total_text_files': 0
        }
        
        for pdf_path in self.input_dir.rglob('*.pdf'):
            stats['total_pdfs'] += 1
            is_processed, details = self.is_processed(pdf_path)
            
            if is_processed:
                if details.get('is_duplicate', False):
                    logger.info(f"Skipping [{stats['total_pdfs']}] {pdf_path.name} (duplicate of {details['original_path']})")
                    stats['duplicate_pdfs'] += 1
                else:
                    logger.info(f"Skipping [{stats['total_pdfs']}] {pdf_path.name} (already processed)")
                    stats['skipped_pdfs'] += 1
                continue
            
            logger.info(f"Processing [{stats['total_pdfs']}] {pdf_path.name}")
            results = self.process_pdf(pdf_path)
            
            if results.get('is_duplicate', False):
                stats['duplicate_pdfs'] += 1
                continue
            
            if results.get('text_extracted') and results.get('images_extracted'):
                stats['processed_pdfs'] += 1
                stats['total_images'] += len(results['image_info']) if results['image_info'] else 0
                stats['total_text_files'] += 1
            else:
                stats['failed_pdfs'] += 1
            
            logger.info(f"Text extracted: {results.get('text_extracted', False)}")
            logger.info(f"Images extracted: {results.get('images_extracted', False)}")
            if results.get('error_message'):
                logger.error(f"Errors: {results['error_message']}")
        
        return stats
