import html
import io
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from sci_vizio_retrieval.stores import SearchResult, VectorStore, create_vector_store

logger = logging.getLogger(__name__)

# New-style arXiv identifiers, e.g. 1903.00780 or 1912.08777v2
_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


def pdf_link(pdf_file: str) -> Optional[str]:
    """Link to the paper's arXiv page when the PDF is named after an arXiv id."""
    if _ARXIV_ID_RE.match(pdf_file):
        return f"https://arxiv.org/abs/{pdf_file}"
    return None


def load_result_image(result: SearchResult) -> Optional[Image.Image]:
    """Image stored in the index, falling back to the extracted file on disk."""
    try:
        if result.image_bytes:
            return Image.open(io.BytesIO(result.image_bytes))
        if result.image_path and Path(result.image_path).exists():
            return Image.open(result.image_path)
    except Exception as e:
        logger.warning(f"Could not load image for {result.id}: {e}")
    return None


class SearchService:
    """Runs text queries against a vector store and formats hits for display."""

    def __init__(self, store: VectorStore):
        self.store = store

    def query_database(self, query_text: str, n_results: int = 5) -> List[Dict]:
        """
        Query the vector store and format results.

        Args:
            query_text (str): Text to search for
            n_results (int): Number of results to return

        Returns:
            list: List of dictionaries containing formatted results
        """
        formatted_results = []
        for hit in self.store.search(query_text, n_results=int(n_results)):
            formatted_results.append({
                'id': hit.id,
                'pdf_file': hit.pdf_file,
                'pdf_link': pdf_link(hit.pdf_file),
                'image': load_result_image(hit),
                'json_content': json.dumps(hit.analysis, indent=2),
                'distance': hit.distance,
            })
        return formatted_results


class ChromaDBQuerier(SearchService):
    """Deprecated: use ``SearchService(create_vector_store(...))``."""

    def __init__(self, chroma_path: str = None):
        super().__init__(create_vector_store("chroma", chroma_path=chroma_path, index_images=False))


def create_result_html(result: Dict) -> str:
    """Create HTML for displaying a single result. All model output is escaped."""
    name = html.escape(result['pdf_file'])
    if result.get('pdf_link'):
        heading = f'<a href="{html.escape(result["pdf_link"])}" target="_blank" rel="noopener">{name}</a>'
    else:
        heading = name
    return f"""
    <div style="margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px;">
        <h3>PDF: {heading}</h3>
        <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; white-space: pre-wrap;">
{html.escape(result['json_content'])}
        </pre>
    </div>
    """


def query_and_display(service: SearchService, query_text: str, num_results: int = 5):
    """
    Query the store and format results for Gradio display.

    Returns:
        tuple: (list of images, html output)
    """
    if not query_text or not query_text.strip():
        return [], ""
    results = service.query_database(query_text, num_results)

    images = [result['image'] for result in results if result['image'] is not None]
    html_output = "".join(create_result_html(result) for result in results)

    return images, html_output


def create_demo(store: VectorStore):
    """Create the Gradio interface Block."""
    import gradio as gr

    service = SearchService(store)

    with gr.Blocks(title="Sci-Vizio-Retrieval") as demo:
        gr.Markdown(f"""
        # Image Analysis Query Interface
        Search through analyzed images from research papers. Enter a text query to find relevant images and their descriptions.
        *Backend: {store.backend} ({store.count()} indexed images)*
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

        def on_search(q, n):
            return query_and_display(service, q, n)

        search_button.click(fn=on_search, inputs=[query_input, num_results], outputs=[gallery, results_html])
        query_input.submit(fn=on_search, inputs=[query_input, num_results], outputs=[gallery, results_html])
    return demo


def launch_ui(
    chroma_path: str = None,
    share: bool = False,
    server_port: int = None,
    backend: str = None,
    postgres_dsn: str = None,
    postgres_table: str = None,
    store: VectorStore = None,
):
    """Launch the Gradio web application against the configured (or given) vector store."""
    if store is None:
        store = create_vector_store(
            backend,
            chroma_path=chroma_path,
            postgres_dsn=postgres_dsn,
            postgres_table=postgres_table,
            index_images=False,
        )
    demo = create_demo(store)
    launch_kwargs = {"share": share}
    if server_port is not None:
        launch_kwargs["server_port"] = server_port
    demo.launch(**launch_kwargs)
