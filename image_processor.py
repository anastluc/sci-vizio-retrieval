"""
Legacy wrapper for backwards compatibility.
"""
import sys
import logging
from sci_vizio_retrieval.processor import ImageProcessor
from sci_vizio_retrieval.config import (
    VISION_MODEL,
    get_images_dir,
    get_analysis_dir,
)

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Recreate original user prompting behavior but with config defaults
    default_images_dir = str(get_images_dir())
    default_output_dir = str(get_analysis_dir())
    
    images_dir = input(f"Enter the directory containing extracted images (or press Enter for '{default_images_dir}'): ").strip() or None
    output_dir = input(f"Enter the output directory for analysis (or press Enter for '{default_output_dir}'): ").strip() or None
    
    display_images_dir = images_dir or default_images_dir
    display_output_dir = output_dir or default_output_dir
    
    print(f"\nUsing Vision Model: {VISION_MODEL}")
    print(f"Processing images from: {display_images_dir}")
    print(f"Saving analysis to: {display_output_dir}")
    
    try:
        processor = ImageProcessor(model=None, images_dir=images_dir, output_dir=output_dir)
        stats = processor.process_directory()
        
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