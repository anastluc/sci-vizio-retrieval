import os
from pathlib import Path
import sqlite3
from datetime import datetime
import json
import logging
import time
from typing import Dict, Optional, List
from tqdm import tqdm

from openrouter_client import OpenRouterVision

class ImageProcessorRetry:
    
    def __init__(self, model: str = None, images_dir: str = None, output_dir: str = None, db_path: str = None):
        """
        Initialize the image processor for retrying failed entries.

        Args:
            model (str): OpenRouter model slug (e.g. "google/gemini-2.0-flash-001"). If None, loaded from config.ini.
            images_dir (str): Directory containing extracted images. If None, loaded from config.ini.
            output_dir (str): Directory to store processing results. If None, loaded from config.ini.
            db_path (str): Path to SQLite database. If None, loaded from config.ini.
        """
        import configparser
        config = configparser.ConfigParser()
        config_file = Path("config.ini")
        
        # Default fallback values
        config_model = None
        config_images_dir = "output/extracted_images"
        config_output_dir = "output/image_process"
        config_db_path = "pdf_processing.db"
        
        if config_file.exists():
            config.read(config_file)
            config_model = config.get("vision", "model", fallback=config_model)
            
            # Read pipeline config
            pipeline_output = config.get("pipeline", "output_dir", fallback="output")
            pipeline_images = config.get("pipeline", "images_subdir", fallback="extracted_images")
            pipeline_analysis = config.get("pipeline", "analysis_subdir", fallback="image_process")
            
            config_images_dir = str(Path(pipeline_output) / pipeline_images)
            config_output_dir = str(Path(pipeline_output) / pipeline_analysis)
            config_db_path = config.get("pipeline", "db_path", fallback=config_db_path)

        resolved_model = model if model is not None else config_model
        self.vision_api = OpenRouterVision(model=resolved_model)

        self.images_dir = Path(images_dir if images_dir is not None else config_images_dir)
        self.output_dir = Path(output_dir if output_dir is not None else config_output_dir)
        self.db_path = db_path if db_path is not None else config_db_path
        
        

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




    def process_image(self, image_path: str, pdf_file: str) -> Dict:
        """
        Process a single image using the Vision API.

        Args:
            image_path (str): Path to the image file
            pdf_file (str): Name of the source PDF

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

            response = None
            try:
                response = self.vision_api.analyze_image(image_path, self.USER_PROMPT)
            except Exception as e:
                self.logger.error(f"Failed to get response: {str(e)}")
                result['error_message'] = f"Vision API call failed: {str(e)}"

            if response is not None:
                content = response.text
                result['success_status'] = True
                result['response'] = content
                result['response_status_code'] = 200
            else:
                result['response_status_code'] = 500
                if not result['error_message']:
                    result['error_message'] = "No response received from Vision API"

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
    # Load config defaults to show as prompts
    import configparser
    config = configparser.ConfigParser()
    config_file = Path("config.ini")
    
    default_images_dir = "output/extracted_images"
    default_output_dir = "output/image_process"
    default_model = "qwen/qwen3.5-flash-02-23"
    
    if config_file.exists():
        config.read(config_file)
        default_model = config.get("vision", "model", fallback=default_model)
        pipeline_output = config.get("pipeline", "output_dir", fallback="output")
        pipeline_images = config.get("pipeline", "images_subdir", fallback="extracted_images")
        pipeline_analysis = config.get("pipeline", "analysis_subdir", fallback="image_process")
        
        default_images_dir = str(Path(pipeline_output) / pipeline_images)
        default_output_dir = str(Path(pipeline_output) / pipeline_analysis)

    # Get directory paths
    images_dir = input(f"Enter the directory containing extracted images (or press Enter for '{default_images_dir}'): ").strip() or None
    output_dir = input(f"Enter the output directory for analysis (or press Enter for '{default_output_dir}'): ").strip() or None
    
    # Resolve display names for logging
    display_images_dir = images_dir or default_images_dir
    display_output_dir = output_dir or default_output_dir
    
    print(f"\nUsing Vision Model: {default_model}")
    print(f"Processing images from: {display_images_dir}")
    print(f"Saving analysis to: {display_output_dir}")
    
    try:
        # Initialize processor and process failed entries
        processor = ImageProcessorRetry(model=None, images_dir=images_dir, output_dir=output_dir)
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