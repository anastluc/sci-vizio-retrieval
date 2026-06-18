"""
Legacy wrapper for backwards compatibility.
"""
import sys
import logging
from sci_vizio_retrieval.indexer import ImageAnalysisIndexer

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        indexer = ImageAnalysisIndexer()
        stats = indexer.process_all_analyses()
        
        print("\nProcessing Complete!")
        print(f"Total items processed: {stats['total_processed']}")
        print(f"Successful JSON validations: {stats['successful_validations']}")
        print(f"Successfully indexed: {stats['successful_indexing']}")
        print(f"Failed: {stats['failed']}")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())