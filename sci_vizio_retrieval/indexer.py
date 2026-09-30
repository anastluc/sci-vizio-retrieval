import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sci_vizio_retrieval.config import CHROMA_PATH, DB_PATH
from sci_vizio_retrieval.stores import IndexRecord, VectorStore, create_vector_store
from sci_vizio_retrieval.stores.chroma import chroma_store_name

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ("image_type", "title", "description")


def extract_and_validate_json(response_text: Optional[str]) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """
    Extract the outermost JSON object from a vision-LLM response and validate it.

    Returns:
        (success, parsed_object, error_message)
    """
    if not response_text:
        return False, None, "Empty response"

    try:
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')

        if start_idx == -1 or end_idx == -1:
            return False, None, "No JSON object found in response"

        json_obj = json.loads(response_text[start_idx:end_idx + 1])

        missing_fields = [f for f in REQUIRED_FIELDS if f not in json_obj]
        if missing_fields:
            return False, None, f"Missing required fields: {', '.join(missing_fields)}"

        return True, json_obj, None

    except json.JSONDecodeError as e:
        return False, None, f"JSON parsing error: {str(e)}"


class ImageAnalysisIndexer:
    def __init__(self, db_path: str = None, chroma_path: str = None, vector_store: VectorStore = None):
        """
        Initialize the indexer.

        Args:
            db_path (str): Path to SQLite database holding pipeline state
            chroma_path (str): ChromaDB path, used only when no vector_store is given
                and the configured backend is chroma
            vector_store (VectorStore): Destination store. Defaults to the backend
                configured in config.ini.
        """
        self.db_path = db_path or DB_PATH
        self.store = vector_store or create_vector_store(chroma_path=chroma_path)
        self._init_database()

    def _init_database(self):
        """Create the per-store indexing tracking table, migrating legacy rows once."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute('''
                CREATE TABLE IF NOT EXISTS vector_indexing (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    store TEXT NOT NULL,
                    pdf_file TEXT,
                    image_path TEXT,
                    index_status BOOLEAN,
                    timestamp TEXT,
                    error_message TEXT,
                    UNIQUE(store, pdf_file, image_path)
                )
            ''')

            # Earlier versions tracked Chroma indexing (at the default path) in
            # `json_indexing`; carry that over so existing indexes aren't rebuilt.
            legacy_exists = cur.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'json_indexing'"
            ).fetchone()
            store_has_rows = cur.execute(
                "SELECT 1 FROM vector_indexing WHERE store = ? LIMIT 1", (self.store.name,)
            ).fetchone()
            if legacy_exists and not store_has_rows and self.store.name == chroma_store_name(CHROMA_PATH):
                cur.execute('''
                    INSERT OR IGNORE INTO vector_indexing
                    (store, pdf_file, image_path, index_status, timestamp, error_message)
                    SELECT ?, pdf_file, image_path, index_status, timestamp, error_message
                    FROM json_indexing
                ''', (self.store.name,))
                logger.info(f"Migrated {cur.rowcount} legacy indexing records to {self.store.name}")
            conn.commit()

    # Kept as a method for backwards compatibility.
    def extract_and_validate_json(self, response_text: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Extract and validate JSON from response text."""
        return extract_and_validate_json(response_text)

    def store_indexing_result(self, pdf_file: str, image_path: str,
                              success: bool, error_message: Optional[str] = None):
        """Store indexing result in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO vector_indexing
                (store, pdf_file, image_path, index_status, timestamp, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                self.store.name,
                pdf_file,
                image_path,
                success,
                datetime.now().isoformat(),
                error_message
            ))
            conn.commit()

    def index_document(self, pdf_file: str, image_path: str,
                       json_obj: Dict, document_id: str) -> bool:
        """Index one analysed image in the vector store."""
        try:
            image_bytes = Path(image_path).read_bytes()
            self.store.upsert(IndexRecord(
                id=document_id,
                pdf_file=pdf_file,
                image_path=image_path,
                analysis=json_obj,
                image_bytes=image_bytes,
            ))
            return True

        except Exception as e:
            logger.error(f"Error indexing document {document_id}: {str(e)}")
            return False

    def process_all_analyses(self, pdf_names: List[str] = None, reindex: bool = False) -> Dict:
        """
        Index every successful image analysis not yet indexed in this store.

        Args:
            pdf_names: Only index images belonging to these PDFs (by stem).
            reindex: Re-index everything, ignoring previous indexing status.
        """
        stats = {
            'total_processed': 0,
            'successful_validations': 0,
            'successful_indexing': 0,
            'failed': 0
        }

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            has_analyses = cur.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'image_processing'"
            ).fetchone()
            if not has_analyses:
                logger.warning("No image analyses found; run the 'process' stage first.")
                return stats

            query = '''
                SELECT ip.pdf_file, ip.image_path, ip.response
                FROM image_processing ip
                LEFT JOIN vector_indexing vi
                ON vi.store = ? AND ip.pdf_file = vi.pdf_file AND ip.image_path = vi.image_path
                WHERE ip.success_status = TRUE
            '''
            params = [self.store.name]
            if not reindex:
                query += " AND (vi.index_status IS NULL OR vi.index_status = FALSE)"
            if pdf_names is not None:
                placeholders = ', '.join(['?'] * len(pdf_names))
                query += f" AND ip.pdf_file IN ({placeholders})"
                params.extend(pdf_names)

            rows = cur.execute(query, params).fetchall()

        logger.info(f"Indexing {len(rows)} analyses into {self.store.name}")
        for pdf_file, image_path, response in rows:
            stats['total_processed'] += 1
            logger.info(f"Processing [{stats['total_processed']}] {image_path}")

            success, json_obj, error_message = extract_and_validate_json(response)

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
                        f"Failed to index in {self.store.backend}"
                    )
            else:
                stats['failed'] += 1
                self.store_indexing_result(pdf_file, image_path, False, error_message)

        return stats
