import os
import time
import sqlite3
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

from sci_vizio_retrieval.client import OpenRouterVision
from sci_vizio_retrieval.config import (
    VISION_MODEL,
    DB_PATH,
    get_images_dir,
    get_analysis_dir,
)

logger = logging.getLogger(__name__)

def sanitize_filename(filename: str) -> str:
    """Convert filename to filesystem-friendly version"""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


class ImageProcessor:
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

    def __init__(self, model: str = None, images_dir: str = None, output_dir: str = None, db_path: str = None):
        """
        Initialize the image processor.

        Args:
            model (str): OpenRouter model slug. If None, loaded from config.ini.
            images_dir (str): Directory containing extracted images. If None, loaded from config.ini.
            output_dir (str): Directory to store processing results. If None, loaded from config.ini.
            db_path (str): Path to SQLite database. If None, loaded from config.ini.
        """
        self.images_dir = Path(images_dir if images_dir is not None else get_images_dir())
        self.output_dir = Path(output_dir if output_dir is not None else get_analysis_dir())
        self.db_path = db_path or DB_PATH

        self.vision_api = OpenRouterVision(model=model)

        # Initialize image embedding model
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

    def get_image_embedding(self, image_path: str | Path) -> Optional[np.ndarray]:
        """Generate embedding for an image."""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(image_tensor)
                embedding = embedding.squeeze().cpu().numpy()
                
            return embedding / np.linalg.norm(embedding)
            
        except Exception as e:
            logger.error(f"Error generating embedding for {image_path}: {str(e)}")
            return None

    def _check_image_processed(self, image_path: str, pdf_file: str) -> Optional[Dict]:
        """Check if image has already been processed by looking up in the database."""
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
        """Process a single image using the OpenRouter Vision API."""
        cached_result = self._check_image_processed(str(image_path), pdf_file)
        if cached_result:
            logger.info(f"Using cached result for {image_path}")
            
            if cached_result['success_status']:
                pdf_subdir = sanitize_filename(pdf_file)
                output_subdir = self.output_dir / pdf_subdir
                output_subdir.mkdir(exist_ok=True)
                output_file = output_subdir / f"{image_path.stem}_analysis.json"
                
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
        
        logger.info(f"Processing image: {image_path} of PDF: {pdf_file}")

        try:
            # Generate embedding
            embedding = self.get_image_embedding(image_path)
            if embedding is not None:
                result['embedding'] = embedding.tobytes()

            response = None
            try:
                response = self.vision_api.analyze_image(image_path, self.USER_PROMPT)
                status = 200
            except Exception as e:
                logger.error(f"Failed to get response: {str(e)}")
                status = 500
                result['error_message'] = f"Vision API call failed: {str(e)}"

            result["response_status_code"] = status

            if status == 200 and response is not None:
                content = response.text
                pdf_subdir = sanitize_filename(pdf_file)
                output_subdir = self.output_dir / pdf_subdir
                output_subdir.mkdir(exist_ok=True)

                output_file = output_subdir / f"{image_path.stem}_analysis.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)

                result['success_status'] = True
                result['response'] = content
            elif status != 200:
                result['error_message'] = f"API request failed with status {status}"

        except Exception as e:
            result['error_message'] = str(e)
            logger.error(f"Error processing {image_path}: {str(e)}")

        self.store_result(result)
        return result

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

    def process_directory(self, pdf_names: List[str] = None) -> Dict:
        """Process images in the images directory."""
        stats = {
            'total_images': 0,
            'successful': 0,
            'failed': 0,
            'cached': 0
        }
        
        if not self.images_dir.exists():
            logger.warning(f"Images directory {self.images_dir} does not exist.")
            return stats

        for pdf_dir in self.images_dir.iterdir():
            if not pdf_dir.is_dir():
                continue
                
            pdf_name = pdf_dir.name
            
            # If pdf_names filter is provided, skip directory if not in the list
            if pdf_names is not None and pdf_name not in pdf_names:
                continue

            logger.info(f"Processing images from PDF: {pdf_name}")
            
            img_paths = [p for p in pdf_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
            for img_path in img_paths:
                stats['total_images'] += 1
                logger.info(f"Processing image {stats['total_images']}: {img_path.name}")
                
                try:
                    cached = self._check_image_processed(str(img_path), pdf_name)
                    result = self.process_image(img_path, pdf_name)
                    if cached is not None:
                        stats['cached'] += 1
                    else:
                        if result['success_status']:
                            stats['successful'] += 1
                        else:
                            stats['failed'] += 1
                        # Add delay to avoid API rate limits
                        time.sleep(1)
                    
                except Exception as e:
                    stats['failed'] += 1
                    logger.error(f"Failed to process {img_path}: {str(e)}")
        
        return stats


class ImageProcessorRetry:
    USER_PROMPT = ImageProcessor.USER_PROMPT

    def __init__(self, model: str = None, images_dir: str = None, output_dir: str = None, db_path: str = None):
        """Initialize the image processor for retrying failed entries."""
        self.images_dir = Path(images_dir if images_dir is not None else get_images_dir())
        self.output_dir = Path(output_dir if output_dir is not None else get_analysis_dir())
        self.db_path = db_path or DB_PATH       
        
        self.vision_api = OpenRouterVision(model=model)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_failed_entries(self) -> List[Dict]:
        """Get all entries with non-200 status or failed status from database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute('''
                SELECT id, pdf_file, image_path 
                FROM image_processing 
                WHERE success_status = FALSE OR response_status_code != 200
                ORDER BY id ASC                
            ''')
            return [dict(row) for row in cur.fetchall()]

    def process_image(self, image_path: str, pdf_file: str) -> Dict:
        """Process a single image using the Vision API."""
        result = {
            'pdf_file': pdf_file,
            'timestamp': datetime.now().isoformat(),
            'image': Path(image_path).name,
            'image_path': str(image_path),
            'success_status': False,
            'response_status_code': None,
            'response': None,
            'error_message': None,
            'embedding': None
        }

        try:
            img_path = Path(image_path)
            if not img_path.exists():
                raise FileNotFoundError(f"Image file not found: {img_path}")

            response = None
            try:
                response = self.vision_api.analyze_image(img_path, self.USER_PROMPT)
            except Exception as e:
                logger.error(f"Failed to get response: {str(e)}")
                result['error_message'] = f"Vision API call failed: {str(e)}"

            if response is not None:
                content = response.text
                result['success_status'] = True
                result['response'] = content
                result['response_status_code'] = 200
                
                # Write to disk
                pdf_subdir = sanitize_filename(pdf_file)
                output_subdir = self.output_dir / pdf_subdir
                output_subdir.mkdir(exist_ok=True)
                output_file = output_subdir / f"{img_path.stem}_analysis.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
            else:
                result['response_status_code'] = 500
                if not result['error_message']:
                    result['error_message'] = "No response received from Vision API"

        except Exception as e:
            result['error_message'] = str(e)
            logger.error(f"Error processing {image_path}: {str(e)}")

        return result

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

    def process_failed_entries(self) -> Dict:
        """Process all failed entries."""
        failed_entries = self.get_failed_entries()
        
        if not failed_entries:
            logger.info("No failed entries found to retry.")
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
        
        logger.info(f"Found {stats['total_entries']} failed entries to retry")
        
        for idx, entry in tqdm(enumerate(failed_entries, 1)):
            logger.info(f"Processing [{idx}/{stats['total_entries']}] {entry['image_path']}")
            
            try:
                if not Path(entry['image_path']).exists():
                    logger.warning(f"File {entry['image_path']} does not exist, skipping.")
                    stats['failed'] += 1
                    continue

                image_file_size = os.path.getsize(entry['image_path'])
                if image_file_size > 4 * 1024 * 1024:
                    logger.warning(f"Image {entry['image_path']} is too large ({image_file_size} bytes), skipping.")
                    stats['failed'] += 1
                    continue
                
                result = self.process_image(entry['image_path'], entry['pdf_file'])
                self.update_entry(entry['id'], result)
                
                if result['success_status']:
                    stats['successful'] += 1
                    logger.info(f"Successfully processed {entry['image_path']}")
                else:
                    stats['failed'] += 1
                    logger.error(f"Failed to process {entry['image_path']}: {result['error_message']}")
                
                # Add delay to avoid API rate limits
                time.sleep(5)
                
            except Exception as e:
                stats['failed'] += 1
                logger.error(f"Error processing {entry['image_path']}: {str(e)}")
        
        return stats
