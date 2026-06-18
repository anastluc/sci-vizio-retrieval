"""
Legacy wrapper for backwards compatibility.
"""
import sys
import logging
from sci_vizio_retrieval.extractor import PDFProcessor

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Keep original user-prompting behavior
    input_directory = input("Enter the directory containing PDFs (or press Enter for 'pdfs'): ").strip() or "pdfs"
    output_directory = input("Enter the output directory (or press Enter for 'output'): ").strip() or "output"
    
    print(f"\nProcessing PDFs from: {input_directory}")
    print(f"Saving content to: {output_directory}")
    
    try:
        processor = PDFProcessor(input_dir=input_directory, output_dir=output_directory)
        stats = processor.process_directory()
        
        print("\nProcessing Complete!")
        print(f"Total PDFs found: {stats['total_pdfs']}")
        print(f"Successfully processed: {stats['processed_pdfs']}")
        print(f"Skipped (already processed): {stats['skipped_pdfs']}")
        print(f"Duplicates found and removed: {stats['duplicate_pdfs']}")
        print(f"Failed to process: {stats['failed_pdfs']}")
        print(f"Total images extracted: {stats['total_images']}")
        print(f"Total text files created: {stats['total_text_files']}")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())