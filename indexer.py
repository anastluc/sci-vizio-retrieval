import sqlite3
import json
import chromadb
from chromadb.utils import embedding_functions
import re
from datetime import datetime
import logging
from typing import Dict, Optional, Tuple
import time
from pathlib import Path
import base64

class ImageAnalysisIndexer:
    def __init__(self, db_path: str = "pdf_processing.db"):
        """
        Initialize the indexer.
        
        Args:
            db_path (str): Path to SQLite database
        """
        self.db_path = db_path
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/json_indexing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize ChromaDB with default embedding function
        print("11")
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        print("12")
        self.default_ef = embedding_functions.DefaultEmbeddingFunction()
        print("13")
        
        # Initialize database
        self._init_database()
        
        # Initialize or get collection
        self._init_collection()

    def _init_database(self):
        """Initialize SQLite database with indexing tracking table."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            
            cur.execute('''
                CREATE TABLE IF NOT EXISTS json_indexing (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pdf_file TEXT,
                    image_path TEXT,
                    index_status BOOLEAN,
                    timestamp TEXT,
                    error_message TEXT,
                    UNIQUE(pdf_file, image_path)
                )
            ''')
            
            conn.commit()

    def _init_collection(self):
        """Initialize or get ChromaDB collection."""
        try:
            # First try to get the collection
            self.collection = self.chroma_client.create_collection(
                name="image_analysis",
                metadata={"description": "Image analysis results with image embeddings"}
            )
            self.logger.info("Created new collection: image_analysis")
        except Exception as e:
            self.logger.info("Collection exists, getting existing collection")
            self.collection = self.chroma_client.get_collection(name="image_analysis")

    def extract_and_validate_json(self, response_text: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Extract and validate JSON from response text."""
        try:
            # Find the first { and last }
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx == -1 or end_idx == -1:
                return False, None, "No JSON object found in response"
            
            # Extract JSON string
            json_str = response_text[start_idx:end_idx + 1]
            
            # Try to parse JSON
            json_obj = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['image_type', 'title', 'description']
            missing_fields = [field for field in required_fields if field not in json_obj]
            
            if missing_fields:
                return False, None, f"Missing required fields: {', '.join(missing_fields)}"
            
            return True, json_obj, None
            
        except json.JSONDecodeError as e:
            return False, None, f"JSON parsing error: {str(e)}"
        except Exception as e:
            return False, None, f"Validation error: {str(e)}"

    def get_image_embedding(self, image_path: str) -> Optional[str]:
        """Get base64 encoded image for embedding."""
        try:
            with open(image_path, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error encoding image {image_path}: {str(e)}")
            return None

    def store_indexing_result(self, pdf_file: str, image_path: str, 
                            success: bool, error_message: Optional[str] = None):
        """Store indexing result in database."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            
            cur.execute('''
                INSERT OR REPLACE INTO json_indexing 
                (pdf_file, image_path, index_status, timestamp, error_message)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                pdf_file,
                image_path,
                success,
                datetime.now().isoformat(),
                error_message
            ))
            
            conn.commit()

    def index_document(self, pdf_file: str, image_path: str, 
                      json_obj: Dict, document_id: str) -> bool:
        """Index document and image in ChromaDB."""
        try:
            # Get image embedding
            image_b64 = self.get_image_embedding(image_path)
            if not image_b64:
                return False

            # Convert JSON to string for metadata
            json_str = json.dumps(json_obj)
            
            # Create document for indexing with both text and image
            self.collection.add(
                documents=[json_str],
                metadatas=[{
                    "pdf_file": pdf_file,
                    "image_path": image_path,
                    "image_type": json_obj.get("image_type", ""),
                    "title": json_obj.get("title", ""),
                    "image_data": image_b64  # Store base64 image in metadata
                }],
                ids=[document_id]
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error indexing document {document_id}: {str(e)}")
            return False

    def process_all_analyses(self):
        print("2")
        """Process all image analyses from the database."""
        stats = {
            'total_processed': 0,
            'successful_validations': 0,
            'successful_indexing': 0,
            'failed': 0
        }
        
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            
            # Get all rows from image_processing that haven't been indexed
            cur.execute('''
                SELECT ip.pdf_file, ip.image_path, ip.response
                FROM image_processing ip
                LEFT JOIN json_indexing ji 
                ON ip.pdf_file = ji.pdf_file AND ip.image_path = ji.image_path
                WHERE ip.success_status = TRUE 
                AND (ji.index_status IS NULL OR ji.index_status = FALSE)
            ''')
            
            for pdf_file, image_path, response in cur.fetchall():
                print("3")
                stats['total_processed'] += 1
                self.logger.info(f"Processing [{stats['total_processed']}] {image_path}")
                
                # Validate JSON
                success, json_obj, error_message = self.extract_and_validate_json(response)
                
                if success:
                    stats['successful_validations'] += 1
                    
                    # Create unique document ID
                    document_id = f"{pdf_file}_{Path(image_path).stem}"
                    
                    # Index in ChromaDB
                    if self.index_document(pdf_file, image_path, json_obj, document_id):
                        stats['successful_indexing'] += 1
                        self.store_indexing_result(pdf_file, image_path, True)
                    else:
                        stats['failed'] += 1
                        self.store_indexing_result(
                            pdf_file, image_path, False, 
                            "Failed to index in ChromaDB"
                        )
                else:
                    stats['failed'] += 1
                    self.store_indexing_result(
                        pdf_file, image_path, False, 
                        error_message
                    )
                
                # Add small delay to avoid overwhelming the system
                time.sleep(0.1)
        
        return stats

def main():
    print("1")
    try:
        # Initialize indexer and process all analyses
        indexer = ImageAnalysisIndexer()
        stats = indexer.process_all_analyses()
        
        # Print summary
        print("\nProcessing Complete!")
        print(f"Total items processed: {stats['total_processed']}")
        print(f"Successful JSON validations: {stats['successful_validations']}")
        print(f"Successfully indexed: {stats['successful_indexing']}")
        print(f"Failed: {stats['failed']}")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())