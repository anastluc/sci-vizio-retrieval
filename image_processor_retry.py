import base64
import requests
import io
from PIL import Image
import os
from pathlib import Path
import sqlite3
from datetime import datetime
import json
import logging
import time
from typing import Dict, Optional, List
from dotenv import load_dotenv
import os

class ImageProcessorRetry:
    load_dotenv()
    GROQ_API_URL = os.getenv("GROQ_API_URL")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    def __init__(self,  images_dir: str, output_dir: str, db_path: str = "pdf_processing.db"):
        """
        Initialize the image processor for retrying failed entries.
        
        Args:
            db_path (str): Path to SQLite database
        """
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.db_path = db_path
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('image_processing_retry.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

         # Initialize database
        self._init_database()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _init_database(self):
        """Initialize SQLite database with image processing table."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            
            cur.execute('''
                CREATE TABLE IF NOT EXISTS image_processing (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pdf_file TEXT,
                    timestamp TEXT,
                    image TEXT,
                    image_path TEXT,
                    success_status BOOLEAN,
                    response_status_code INTEGER,
                    response TEXT,
                    error_message TEXT
                )
            ''')
            
            conn.commit()

    def get_failed_entries(self) -> List[Dict]:
        """
        Get all entries with success_status = 0 from the database.
        
        Returns:
            list: List of dictionaries containing failed entries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            
            cur.execute('''
                SELECT id, pdf_file, image_path 
                FROM image_processing 
                WHERE success_status = 0
                ORDER BY id ASC 
                LIMIT 1200, 200
            ''')
            
            return [dict(row) for row in cur.fetchall()]

    def process_image(self, image_path: str) -> Dict:
        """
        Process a single image using the Groq API.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Processing results
        """
        result = {
            'success_status': False,
            'response': None,
            'error_message': None
        }
        
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Read and encode image
            with open(image_path, 'rb') as img:
                image_bytes = img.read()
                encoded = base64.b64encode(image_bytes).decode('utf-8')

                # Validate image format
                img = Image.open(io.BytesIO(image_bytes))
                img.verify()

            # Prepare API request
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                        {"type": "text", "text": self.USER_PROMPT}            
                    ]
                }
            ]

            # Make API request
            response = self.make_api_request("llama-3.2-90b-vision-preview", messages)

            result["response_status_code"] = response.status_code

            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                result['success_status'] = True
                result['response'] = content
                
                
            else:
                result['error_message'] = f"API request failed with status {response.status_code}"
                
        except Exception as e:
            result['error_message'] = str(e)
            self.logger.error(f"Error processing {image_path}: {str(e)}")
        
        return result

    def make_api_request(self, model: str, messages: list) -> requests.Response:
        """Make request to Groq API."""
        return requests.post(
            self.GROQ_API_URL,
            json={
                "model": model,
                "messages": messages,
                "max_tokens": 1000
            },
            headers={
                "Authorization": f"Bearer {self.GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=30
        )

    def update_entry(self, entry_id: int, result: Dict):
        """Update database entry with new processing result."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute('''
                UPDATE image_processing 
                SET success_status = ?,
                    response = ?,
                    error_message = ?,
                    timestamp = ?,
                    response_status_code = ?
                WHERE id = ?
            ''', (
                result['success_status'],
                result['response'],
                result['error_message'],
                datetime.now().isoformat(),
                result['response_status_code'],
                entry_id
            ))
            conn.commit()

    def process_failed_entries(self):
        """Process all failed entries."""
        failed_entries = self.get_failed_entries()
        
        if not failed_entries:
            self.logger.info("No failed entries found to retry.")
            return {
                'total_entries': 0,
                'successful': 0,
                'failed': 0
            }
        
        stats = {
            'total_entries': len(failed_entries),
            'successful': 0,
            'failed': 0
        }
        
        self.logger.info(f"Found {stats['total_entries']} failed entries to retry")
        
        for idx, entry in enumerate(failed_entries, 1):
            self.logger.info(f"Processing [{idx}/{stats['total_entries']}] {entry['image_path']}")
            
            try:
                result = self.process_image(entry['image_path'])
                self.update_entry(entry['id'], result)
                
                if result['success_status']:
                    stats['successful'] += 1
                    self.logger.info(f"Successfully processed {entry['image_path']}")
                else:
                    stats['failed'] += 1
                    self.logger.error(f"Failed to process {entry['image_path']}: {result['error_message']}")
                
                # Add delay to avoid API rate limits
                time.sleep(5)
                
            except Exception as e:
                stats['failed'] += 1
                self.logger.error(f"Error processing {entry['image_path']}: {str(e)}")
        
        return stats

    USER_PROMPT = """You will be provided with an image. 
    Your response should contain as much information as possible from this diagram. 
    It should contain a description of what type of image is (e.g. diagram, graph, flowchart, etc.) 
    and the data that comprises it.
    The response should be a valid JSON object as a string. 
    The JSON should necessarily have the following attributes:
    image_type, title, description
    But also if applicable the following:
    time_period, x-axis, y-axis, sources, sections, labels, ticks, key patterns
    Begin immediately with outputting the JSON object, do NOT prefix with any extra text, start straight with the json object, i.e. the first character should be {
    Do NOT suffix with any extra text, finish with the json object, i.e. the last character should be }
    """

def main():

    # Get directory paths
    images_dir = input("Enter the directory containing extracted images (or press Enter for 'output/extracted_images'): ").strip() \
        or "output/extracted_images"
    output_dir = input("Enter the output directory for analysis (or press Enter for 'output/image_process'): ").strip() \
        or "output/image_process"
    
    print(f"\nProcessing images from: {images_dir}")
    print(f"Saving analysis to: {output_dir}")


    try:
        # Initialize processor and process failed entries
        processor = ImageProcessorRetry(images_dir, output_dir)
        stats = processor.process_failed_entries()
        
        # Print summary
        print("\nProcessing Complete!")
        print(f"Total failed entries found: {stats['total_entries']}")
        print(f"Successfully reprocessed: {stats['successful']}")
        print(f"Failed again: {stats['failed']}")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())