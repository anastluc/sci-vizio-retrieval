"""
Legacy wrapper for backwards compatibility.
"""
import sys
import logging
from sci_vizio_retrieval.processor import ImageProcessorRetry
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
        processor = ImageProcessorRetry(model=None, images_dir=images_dir, output_dir=output_dir)
        stats = processor.process_failed_entries()
        
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