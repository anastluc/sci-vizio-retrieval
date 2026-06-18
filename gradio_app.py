"""
Legacy wrapper for backwards compatibility.
"""
import logging
from sci_vizio_retrieval.ui import launch_ui

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    launch_ui(share=False)

if __name__ == "__main__":
    main()