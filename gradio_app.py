import gradio as gr
import chromadb
from chromadb.utils import embedding_functions
import base64
import json
from PIL import Image
import io
from pathlib import Path
import os

class ChromaDBQuerier:
    def __init__(self, chroma_path="./chroma_db"):
        """Initialize ChromaDB connection."""
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection(name="image_analysis_description_documents")
        
    def query_database(self, query_text: str, n_results: int = 5):
        """
        Query ChromaDB and format results.
        
        Args:
            query_text (str): Text to search for
            n_results (int): Number of results to return
            
        Returns:
            list: List of dictionaries containing formatted results
        """
        # Query the collection
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        formatted_results = []
        
        for idx in range(len(results['ids'][0])):
            try:
                # Get document data
                doc = json.loads(results['documents'][0][idx])
                metadata = results['metadatas'][0][idx]
                
                # Convert base64 image to PIL Image
                image_data = base64.b64decode(metadata['image_data'])
                image = Image.open(io.BytesIO(image_data))
                
                # Format JSON string with indentation
                formatted_json = json.dumps(doc, indent=2)
                
                # Create PDF link
                pdf_file = metadata['pdf_file']
                pdf_path = f"arxiv-papers/{pdf_file}.pdf"  # Adjust path as needed
                
                result = {
                    'id': results['ids'][0][idx],
                    'pdf_file': pdf_file,
                    'pdf_link': pdf_path,
                    'image': image,
                    'json_content': formatted_json
                }
                
                formatted_results.append(result)
                
            except Exception as e:
                print(f"Error formatting result {idx}: {str(e)}")
                continue
        
        return formatted_results

def create_result_html(result):
    """Create HTML for displaying a single result."""
    return f"""
    <div style="margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px;">
        <h3>PDF: <a href="{result['pdf_link']}" target="_blank">{result['pdf_file']}</a></h3>
        <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto;">
{result['json_content']}
        </pre>
    </div>
    """

def query_and_display(query_text: str, num_results: int = 5):
    """
    Query ChromaDB and format results for Gradio display.
    
    Args:
        query_text (str): Text to search for
        num_results (int): Number of results to return
        
    Returns:
        tuple: (list of images, html output)
    """
    querier = ChromaDBQuerier()
    results = querier.query_database(query_text, num_results)
    
    # Prepare output
    images = [result['image'] for result in results]
    html_output = "".join([create_result_html(result) for result in results])
    
    return images, html_output

# Create Gradio interface
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("""
    # Image Analysis Query Interface
    Search through analyzed images from research papers. Enter a text query to find relevant images and their descriptions.
    """)
    
    with gr.Row():
        query_input = gr.Textbox(
            label="Enter your query", 
            placeholder="Example: flowchart showing neural network architecture"
        )
        num_results = gr.Slider(
            minimum=1,
            maximum=10,
            value=5,
            step=1,
            label="Number of results"
        )
    
    search_button = gr.Button("Search", variant="primary")
    
    with gr.Row():
        gallery = gr.Gallery(
            label="Retrieved Images",
            show_label=True,
            elem_id="gallery",
            columns=[2],
            rows=[2],
            height="auto"
        )
    
    results_html = gr.HTML(label="Results")
    
    search_button.click(
        fn=query_and_display,
        inputs=[query_input, num_results],
        outputs=[gallery, results_html]
    )

if __name__ == "__main__":
    demo.launch(share=True)