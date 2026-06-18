import sqlite3
import json
import chromadb
from chromadb.utils import embedding_functions
import logging
from typing import Dict, Optional, Tuple, List
from datetime import datetime
from pathlib import Path
import base64
import numpy as np
from PIL import Image

from sci_vizio_retrieval.config import (
    DB_PATH,
    CHROMA_PATH,
)

logger = logging.getLogger(__name__)

class ImageAnalysisIndexer:
    def __init__(self, db_path: str = None, chroma_path: str = None):
        """
        Initialize the indexer.
        
        Args:
            db_path (str): Path to SQLite database
            chroma_path (str): Path to ChromaDB persistent storage
        """
        self.db_path = db_path or DB_PATH
        self.chroma_path = chroma_path or CHROMA_PATH
        
        # chroma settings
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        self.embedding_function = embedding_functions.OpenCLIPEmbeddingFunction()
        
        # Initialize database
        self._init_database()
        
        # Initialize or get collections
        self._init_collections()

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

    def _init_collections(self):
        """Initialize or get ChromaDB collections."""
        try:
            self.image_collection = self.chroma_client.create_collection(
                name="image_analysis_image_embeddings",
                metadata={"description": "Image analysis results with image embeddings"},
                embedding_function=self.embedding_function
            )
            logger.info("Created new collection: image_analysis_image_embeddings")
        except Exception:
            logger.info("Collection exists, getting existing collection: image_analysis_image_embeddings")
            self.image_collection = self.chroma_client.get_collection(
                name="image_analysis_image_embeddings",
                embedding_function=self.embedding_function
            )
        
        try:
            self.doc_collection = self.chroma_client.create_collection(
                name="image_analysis_description_documents",
                metadata={"description": "Image analysis results with vision analysis documents"},
            )
            logger.info("Created new collection: image_analysis_description_documents")
        except Exception:
            logger.info("Collection exists, getting existing collection: image_analysis_description_documents")            
            self.doc_collection = self.chroma_client.get_collection(
                name="image_analysis_description_documents"
            )

    def extract_and_validate_json(self, response_text: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Extract and validate JSON from response text."""
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx == -1 or end_idx == -1:
                return False, None, "No JSON object found in response"
            
            json_str = response_text[start_idx:end_idx + 1]
            json_obj = json.loads(json_str)
            
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
        """Get base64 encoded image for embedding/metadata inclusion."""
        try:
            with open(image_path, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {str(e)}")
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
            image_b64 = self.get_image_embedding(image_path)
            if not image_b64:
                return False

            json_str = json.dumps(json_obj)
            
            self.doc_collection.add(
                documents=[json_str],
                metadatas=[{
                    "pdf_file": pdf_file,
                    "image_path": image_path,
                    "image_type": json_obj.get("image_type", ""),
                    "title": json_obj.get("title", ""),
                    "image_data": image_b64
                }],
                ids=[document_id]
            )

            img = Image.open(image_path)
            img_array = np.array(img)
            
            self.image_collection.add(
                metadatas=[{
                    "pdf_file": pdf_file,
                    "image": image_path,
                    "image_path": image_path
                }],
                images=[img_array],
                ids=[document_id]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error indexing document {document_id}: {str(e)}")
            return False

    def process_all_analyses(self, pdf_names: List[str] = None) -> Dict:
        """Process all image analyses from the database."""
        stats = {
            'total_processed': 0,
            'successful_validations': 0,
            'successful_indexing': 0,
            'failed': 0
        }
        
        query = '''
            SELECT ip.pdf_file, ip.image_path, ip.response
            FROM image_processing ip
            LEFT JOIN json_indexing ji 
            ON ip.pdf_file = ji.pdf_file AND ip.image_path = ji.image_path
            WHERE ip.success_status = TRUE 
            AND (ji.index_status IS NULL OR ji.index_status = FALSE)
        '''
        params = []
        if pdf_names is not None:
            placeholders = ', '.join(['?'] * len(pdf_names))
            query += f" AND ip.pdf_file IN ({placeholders})"
            params.extend(pdf_names)
            
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(query, params)
            
            for pdf_file, image_path, response in cur.fetchall():
                stats['total_processed'] += 1
                logger.info(f"Processing [{stats['total_processed']}] {image_path}")
                
                success, json_obj, error_message = self.extract_and_validate_json(response)
                
                if success:
                    stats['successful_validations'] += 1
                    document_id = f"{pdf_file}_{Path(image_path).stem}"
                    
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
        
        return stats
