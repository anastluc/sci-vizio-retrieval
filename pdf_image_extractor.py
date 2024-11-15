import fitz  # PyMuPDF
import os
from PIL import Image
import io
from pathlib import Path
import sqlite3
from datetime import datetime
import json
import hashlib
import shutil

class PDFProcessor:
    def __init__(self, input_dir, output_dir, db_path="pdf_processing.db"):
        """
        Initialize the PDF processor with directories and database connection.
        
        Args:
            input_dir (str/Path): Directory containing PDF files
            output_dir (str/Path): Base directory for extracted content
            db_path (str): Path to SQLite database
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "extracted_images"
        self.text_dir = self.output_dir / "extracted_text"
        self.db_path = db_path
        
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
            # Read the file in chunks to handle large files
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
        Handle duplicate PDF by recording it in database and optionally deleting it.
        
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
        
        # Delete the duplicate file
        try:
            pdf_path.unlink()
            print(f"Deleted duplicate file: {pdf_path}")
        except Exception as e:
            print(f"Warning: Could not delete duplicate file {pdf_path}: {e}")

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
            pdf_document = fitz.open(str(pdf_path))
            text_content = []
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text_content.append(f"--- Page {page_num + 1} ---\n")
                text_content.append(page.get_text())
                text_content.append("\n\n")
            
            # Create output path preserving directory structure
            relative_path = pdf_path.stem
            output_path = self.text_dir / f"{relative_path}.txt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save text content
            output_path.write_text('\n'.join(text_content), encoding='utf-8')
            
            pdf_document.close()
            return True, output_path
            
        except Exception as e:
            return False, str(e)

    def extract_images(self, pdf_path):
        """
        Extract images from PDF and save to directory.
        
        Args:
            pdf_path (Path): Path to the PDF file
        
        Returns:
            tuple: (success, list of image paths or error message)
        """
        try:
            # Create output directory for this PDF
            relative_path = pdf_path.stem
            pdf_output_dir = self.images_dir / relative_path
            pdf_output_dir.mkdir(parents=True, exist_ok=True)
            
            pdf_document = fitz.open(str(pdf_path))
            extracted_images = []
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                image_list = page.get_images(full=True)
                
                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]
                    base_image = pdf_document.extract_image(xref)
                    
                    if base_image:
                        image_filename = f"page{page_num + 1}_img{img_index + 1}.{base_image['ext']}"
                        image_path = pdf_output_dir / image_filename
                        
                        # Save the image
                        image_path.write_bytes(base_image["image"])
                        
                        # Optimize image
                        try:
                            with Image.open(image_path) as img:
                                if img.mode == 'RGBA':
                                    img = img.convert('RGB')
                                img.save(image_path, optimize=True, quality=85)
                        except Exception as e:
                            print(f"Warning: Could not optimize {image_filename}: {str(e)}")
                        
                        extracted_images.append({
                            'filename': image_filename,
                            'path': str(image_path.relative_to(self.output_dir))
                        })
            
            pdf_document.close()
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
        
        # Check if file still exists (might have been deleted as duplicate)
        if not pdf_path.exists():
            return {
                'pdf_path': str(pdf_path),
                'error_message': 'File no longer exists'
            }
        
        # Check if already processed
        is_processed, details = self.is_processed(pdf_path)
        if is_processed:
            print(f"PDF already processed on {details['timestamp']}")
            return details
        
        # Calculate hash and check for duplicates
        try:
            pdf_hash = self.calculate_hash(pdf_path)
            is_duplicate, original_path = self.check_duplicate(pdf_path, pdf_hash)
            
            if is_duplicate:
                print(f"Duplicate PDF found: {pdf_path}")
                print(f"Original file: {original_path}")
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
            # Extract text
            text_success, text_result = self.extract_text(pdf_path)
            results['text_extracted'] = text_success
            if not text_success:
                results['error_message'] = f"Text extraction failed: {text_result}"
            
            # Extract images
            images_success, images_result = self.extract_images(pdf_path)
            results['images_extracted'] = images_success
            if images_success:
                results['image_info'] = images_result
            else:
                results['error_message'] = f"{results['error_message']}; Image extraction failed: {images_result}"
            
            # Store results in database
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
                    print(f"\nSkipping [{stats['total_pdfs']}] {pdf_path.name} (duplicate of {details['original_path']})")
                    stats['duplicate_pdfs'] += 1
                else:
                    print(f"\nSkipping [{stats['total_pdfs']}] {pdf_path.name} (already processed)")
                    stats['skipped_pdfs'] += 1
                continue
            
            print(f"\nProcessing [{stats['total_pdfs']}] {pdf_path.name}")
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
            
            print(f"Text extracted: {results.get('text_extracted', False)}")
            print(f"Images extracted: {results.get('images_extracted', False)}")
            if results.get('error_message'):
                print(f"Errors: {results['error_message']}")
        
        return stats

def main():
    # Get directory paths from user input or use defaults
    input_directory = input("Enter the directory containing PDFs (or press Enter for 'pdfs'): ").strip() or "pdfs"
    output_directory = input("Enter the output directory (or press Enter for 'output'): ").strip() or "output"
    
    print(f"\nProcessing PDFs from: {input_directory}")
    print(f"Saving content to: {output_directory}")
    
    try:
        # Initialize processor and process all PDFs
        processor = PDFProcessor(input_directory, output_directory)
        stats = processor.process_directory()
        
        # Print summary
        print("\nProcessing Complete!")
        print(f"Total PDFs found: {stats['total_pdfs']}")
        print(f"Successfully processed: {stats['processed_pdfs']}")
        print(f"Skipped (already processed): {stats['skipped_pdfs']}")
        print(f"Duplicates found and removed: {stats['duplicate_pdfs']}")
        print(f"Failed to process: {stats['failed_pdfs']}")
        print(f"Total images extracted: {stats['total_images']}")
        print(f"Total text files created: {stats['total_text_files']}")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())