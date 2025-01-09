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
from typing import Dict, Literal, Optional, List
from dotenv import load_dotenv
import os
import torch
# from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
import numpy as np
# import chromadb
# from chromadb.utils import embedding_functions
from tqdm import tqdm
from vision_models.vision_API import VisionAPI
from vision_models.claude_vision import ClaudeVision
from vision_models.gemini_vision import GeminiVision
from vision_models.groq_vision import GroqVision
from vision_models.openai_vision import OpenAIVision
from vision_models.xai_vision import XAI_Vision

MLLM_Provider = Literal[
    "openai",
    "anthropic", 
    "google", 
    "groq-vision",
    "xai"]

class ImageProcessorRetry:
    
    def __init__(self,  model:str , images_dir: str, output_dir: str, db_path: str = "pdf_processing.db"):
        """
        Initialize the image processor for retrying failed entries.
        
        Args:
            db_path (str): Path to SQLite database
        """
        self.vision_api: VisionAPI = self.create_vision_api(model)

        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.db_path = db_path       
        
        

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/image_processing_retry.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)


        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_vision_api(self,specific_model:str) -> VisionAPI:
        if specific_model == "openai":
            return OpenAIVision(specific_model)
        elif specific_model == "anthropic":
            return ClaudeVision(specific_model)
        elif specific_model in ["gemini-2.0-flash-exp"]:
            return GeminiVision(specific_model)
        elif specific_model == "groq-vision":
            return GroqVision(specific_model)
        elif specific_model == "xai":
            return XAI_Vision(specific_model)
        else:
            raise ValueError(f"Unsupported provider: {specific_model}")
        
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
                WHERE response_status_code = 429
                ORDER BY id ASC                
            ''')
            
            return [dict(row) for row in cur.fetchall()]


    def get_image_embedding(self, image_path):
        
        img = Image.open(image_path)
        img_array = np.array(img)
        
        emb = self.embedding_function(img_array)
        return emb

    def process_image(self, image_path: str,pdf_file: str) -> Dict:
        """
        Process a single image using the Groq API.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Processing results
        """
        result = {
            'pdf_file': pdf_file,
            'timestamp': datetime.now().isoformat(),
            'image': image_path,
            'image_path': str(image_path),
            'success_status': False,
            'response_status_code': None,
            'response': None,
            'error_message': None,
            'embedding': None
        }
        
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Generate embedding
            # print(image_path)
            # embedding = self.get_image_embedding(image_path)
            embedding = None # do not calculate embedding
            # print("\n\n---------\n- EMBEDDING -\n---------")
            # print(embedding)
            # print("---------")
            
                

            # Read and encode image
            with open(image_path, 'rb') as img:
                image_bytes = img.read()
                encoded = base64.b64encode(image_bytes).decode('utf-8')

                # Validate image format
                img = Image.open(io.BytesIO(image_bytes))
                img.verify()

            try:
                response = self.vision_api.analyze_image(image_path, self.USER_PROMPT)
            except Exception as e:
                print(f"Failed to get response: {str(e)}")


            # result["response_status_code"] = response.status_code
            print(response.text)

            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                result['success_status'] = True
                result['response'] = content

            if embedding is not None:
                result['embedding'] = embedding#.tolist()
                # print(result['embedding'])
                
            
            else:
                result['error_message'] = f"API request failed with status {response.status_code}"
                
        except Exception as e:
            result['error_message'] = str(e)
            self.logger.error(f"Error processing {image_path}: {str(e)}")
        
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
                    response_status_code = ?,
                    embedding = ?
                WHERE id = ?
            ''', (
                result['success_status'],
                result['response'],
                result['error_message'],
                datetime.now().isoformat(),
                result['response_status_code'],
                json.dumps(result['embedding']),
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
        
        for idx, entry in tqdm(enumerate(failed_entries, 1)):
            self.logger.info(f"Processing [{idx}/{stats['total_entries']}] {entry['image_path']}")

            image_file_size = os.path.getsize(entry['image_path'])
            if image_file_size > 4 * 1024 * 1024:
                print(f"Image {entry['image_path']} is {image_file_size} - larger than 4 Mb")
                continue
            
            try:
                result = self.process_image(entry['image_path'],entry['pdf_file'])
                
                self.update_entry(entry['id'], result)
                
                if result['success_status']:
                    stats['successful'] += 1
                    self.logger.info(f"Successfully processed {entry['image_path']}")
                else:
                    stats['failed'] += 1
                    self.logger.error(f"Failed to process {entry['image_path']}: {result['error_message']}")
                
                # Add delay to avoid API rate limits
                time.sleep(10)
                
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
        processor = ImageProcessorRetry(model="gemini-2.0-flash-exp", images_dir = images_dir, output_dir=output_dir)
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