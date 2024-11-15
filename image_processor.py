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
from typing import Dict, Optional

class ImageProcessor:
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_API_KEY = "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"
    
    def __init__(self, images_dir: str, output_dir: str, db_path: str = "pdf_processing.db"):
        """
        Initialize the image processor.
        
        Args:
            images_dir (str): Directory containing extracted images
            output_dir (str): Directory to store processing results
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
                logging.FileHandler('image_processing.log'),
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

    def process_image(self, image_path: Path, pdf_file: str) -> Dict:
        """
        Process a single image using the Groq API.
        
        Args:
            image_path (Path): Path to the image file
            pdf_file (str): Name of the PDF file the image came from
            
        Returns:
            dict: Processing results
        """
        result = {
            'pdf_file': pdf_file,
            'timestamp': datetime.now().isoformat(),
            'image': image_path.name,
            'image_path': str(image_path),
            'success_status': False,
            'response_status_code': None,
            'response': None,
            'error_message': None
        }
        
        try:
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
                
                # Save response to file
                output_file = self.output_dir / f"{image_path.stem}_analysis.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                result['success_status'] = True
                result['response'] = content
                
            else:
                result['error_message'] = f"API request failed with status {response.status_code}"
                
        except Exception as e:
            result['error_message'] = str(e)
            self.logger.error(f"Error processing {image_path}: {str(e)}")
        
        
        
        # Store result in database
        self.store_result(result)
        
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

    def store_result(self, result: Dict):
        """Store processing result in database."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute('''
                INSERT INTO image_processing 
                (pdf_file, timestamp, image, image_path, success_status, response_status_code, response, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['pdf_file'],
                result['timestamp'],
                result['image'],
                result['image_path'],
                result['success_status'],
                result['response_status_code'],
                result['response'],
                result['error_message']
            ))
            conn.commit()

    def process_directory(self):
        """Process all images in the images directory."""
        stats = {
            'total_images': 0,
            'successful': 0,
            'failed': 0
        }
        
        # Walk through all subdirectories in extracted_images
        for pdf_dir in self.images_dir.iterdir():
            if not pdf_dir.is_dir():
                continue
                
            pdf_name = pdf_dir.name
            self.logger.info(f"Processing images from PDF: {pdf_name}")
            
            # Process each image in the PDF's directory
            for img_path in pdf_dir.glob('*.[jp][pn][g]'):  # Match jpg, jpeg, png
                stats['total_images'] += 1
                self.logger.info(f"Processing image {stats['total_images']}: {img_path.name}")
                
                try:
                    result = self.process_image(img_path, pdf_name)
                    if result['success_status']:
                        stats['successful'] += 1
                    else:
                        stats['failed'] += 1
                        
                    # Add delay to avoid API rate limits
                    time.sleep(1)
                    
                except Exception as e:
                    stats['failed'] += 1
                    self.logger.error(f"Failed to process {img_path}: {str(e)}")
        
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
        # Initialize processor and process all images
        processor = ImageProcessor(images_dir, output_dir)
        stats = processor.process_directory()
        
        # Print summary
        print("\nProcessing Complete!")
        print(f"Total images processed: {stats['total_images']}")
        print(f"Successfully processed: {stats['successful']}")
        print(f"Failed to process: {stats['failed']}")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())