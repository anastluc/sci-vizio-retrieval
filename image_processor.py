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
from dotenv import load_dotenv
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np

from vision_models.vision_API import VisionAPI
from vision_models.factory import create_vision_api
import re

def sanitize_filename(filename: str) -> str:
    """Convert filename to filesystem-friendly version"""
    # Replace invalid characters with underscore
    return re.sub(r'[<>:"/\\|?*]', '_', filename)
class ImageProcessor:
    
    def __init__(self, model:str, images_dir: str, output_dir: str, db_path: str = "pdf_processing.db"):
        """
        Initialize the image processor.
        
        Args:
            images_dir (str): Directory containing extracted images
            output_dir (str): Directory to store processing results
            db_path (str): Path to SQLite database
        """
        self.vision_api: VisionAPI = create_vision_api(model)

        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.db_path = db_path

        # Initialize image model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/image_processing.log'),
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
                    error_message TEXT,
                    embedding BLOB
                )
            ''')
            
            conn.commit()

    def get_image_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Generate embedding for an image."""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(image_tensor)
                embedding = embedding.squeeze().cpu().numpy()
                
            return embedding / np.linalg.norm(embedding)
            
        except Exception as e:
            self.logger.error(f"Error generating embedding for {image_path}: {str(e)}")
            return None

    def _check_image_processed(self, image_path: str, pdf_file: str) -> Optional[Dict]:
        """
        Check if image has already been processed by looking up in the database.
        
        Args:
            image_path (str): Path to the image file
            pdf_file (str): Name of the PDF file the image came from
            
        Returns:
            Optional[Dict]: Previous processing result if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute('''
                SELECT pdf_file, timestamp, image, image_path, success_status, 
                       response_status_code, response, error_message, embedding
                FROM image_processing
                WHERE image_path = ? AND pdf_file = ?
            ''', (str(image_path), pdf_file))
            
            row = cur.fetchone()
            
            if row:
                return {
                    'pdf_file': row[0],
                    'timestamp': row[1],
                    'image': row[2],
                    'image_path': row[3],
                    'success_status': row[4],
                    'response_status_code': row[5],
                    'response': row[6],
                    'error_message': row[7],
                    'embedding': row[8]
                }
            return None
    
    def process_image(self, image_path: Path, pdf_file: str) -> Dict:
        """
        Process a single image using the Groq API.
        
        Args:
            image_path (Path): Path to the image file
            pdf_file (str): Name of the PDF file the image came from
            
        Returns:
            dict: Processing results
        """

        # Check if image has already been processed
        cached_result = self._check_image_processed(str(image_path), pdf_file)
        if cached_result:
            self.logger.info(f"Using cached result for {image_path}")
            
            # If the result was successful, ensure the output file exists
            if cached_result['success_status']:
                pdf_subdir = sanitize_filename(pdf_file)
                output_subdir = self.output_dir / pdf_subdir
                output_subdir.mkdir(exist_ok=True)
                output_file = output_subdir / f"{image_path.stem}_analysis.json"
                
                # Recreate the output file if it doesn't exist
                if not output_file.exists() and cached_result['response']:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(cached_result['response'])
            
            return cached_result
    

        result = {
            'pdf_file': pdf_file,
            'timestamp': datetime.now().isoformat(),
            'image': image_path.name,
            'image_path': str(image_path),
            'success_status': False,
            'response_status_code': None,
            'response': None,
            'error_message': None,
            'embedding': None
        }
        
        print(f"Processing image: {image_path} of PDF: {pdf_file}")

        try:
            # Generate embedding
            embedding = self.get_image_embedding(image_path)
            if embedding is not None:
                result['embedding'] = embedding.tobytes()

            try:
                response = self.vision_api.analyze_image(image_path, self.USER_PROMPT)
                status = 200
            except Exception as e:
                print(f"Failed to get response: {str(e)}")
                status = 500

            result["response_status_code"] = status
            print(response.text)
            
            if status == 200:
                content = response.text#json()
                
                # # Save response to file
                # output_file = self.output_dir / f"{image_path.stem}_analysis.json"
                pdf_subdir = sanitize_filename(pdf_file)
                output_subdir = self.output_dir / pdf_subdir
                output_subdir.mkdir(exist_ok=True)

                # Save response to file
                output_file = output_subdir / f"{image_path.stem}_analysis.json"
                
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
                (pdf_file, timestamp, image, image_path, success_status, response_status_code, response, error_message, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['pdf_file'],
                result['timestamp'],
                result['image'],
                result['image_path'],
                result['success_status'],
                result['response_status_code'],
                result['response'],
                result['error_message'],
                result['embedding']
            ))
            conn.commit()

    def process_directory(self):
        """Process all images in the images directory."""
        stats = {
            'total_images': 0,
            'successful': 0,
            'failed': 0,
            'cached': 0
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
                    if '_check_image_processed' in str(result.get('timestamp', '')):
                        stats['cached'] += 1
                    elif result['success_status']:
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
        processor = ImageProcessor(model="gemini-2.0-flash-exp", images_dir=images_dir, output_dir=output_dir)
        stats = processor.process_directory()
        
        # Print summary
        print("\nProcessing Complete!")
        print(f"Total images processed: {stats['total_images']}")
        print(f"Successfully processed: {stats['successful']}")
        print(f"Failed to process: {stats['failed']}")
        print(f"Cached results: {stats.get('cached', 0)}")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())