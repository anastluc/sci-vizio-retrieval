"""ChromaDB-backed vector store (the original storage backend)."""

import base64
import json
import logging
from pathlib import Path
from typing import List, Optional

from sci_vizio_retrieval.stores.base import IndexRecord, SearchResult, VectorStore

logger = logging.getLogger(__name__)

DOC_COLLECTION = "image_analysis_description_documents"
IMAGE_COLLECTION = "image_analysis_image_embeddings"


def chroma_store_name(path: str | Path) -> str:
    return f"chroma:{Path(path).resolve()}"


class ChromaVectorStore(VectorStore):
    """
    Stores descriptions in a text collection (Chroma's default embedding model)
    and, optionally, the images themselves in a CLIP-embedded image collection.

    Collection names and metadata layout are unchanged from earlier versions,
    so existing ``chroma_db`` directories keep working.

    Args:
        path: Directory of the persistent ChromaDB database.
        index_images: Also write CLIP image embeddings. Loads an OpenCLIP model,
            so leave it off for search-only use.
        text_embedding_function: Override Chroma's default text embedder.
        image_embedding_function: Override the OpenCLIP image embedder.
    """

    backend = "chroma"

    def __init__(
        self,
        path: str | Path,
        index_images: bool = True,
        text_embedding_function=None,
        image_embedding_function=None,
    ):
        import chromadb

        self.path = str(path)
        self._client = chromadb.PersistentClient(path=self.path)

        doc_kwargs = {"metadata": {"description": "Image analysis results with vision analysis documents"}}
        if text_embedding_function is not None:
            doc_kwargs["embedding_function"] = text_embedding_function
        self.doc_collection = self._client.get_or_create_collection(name=DOC_COLLECTION, **doc_kwargs)

        self.image_collection = None
        if index_images:
            if image_embedding_function is None:
                from chromadb.utils import embedding_functions
                image_embedding_function = embedding_functions.OpenCLIPEmbeddingFunction()
            self.image_collection = self._client.get_or_create_collection(
                name=IMAGE_COLLECTION,
                metadata={"description": "Image analysis results with image embeddings"},
                embedding_function=image_embedding_function,
            )

    @property
    def name(self) -> str:
        return chroma_store_name(self.path)

    def upsert(self, record: IndexRecord) -> None:
        metadata = {
            "pdf_file": record.pdf_file,
            "image_path": record.image_path,
            "image_type": record.image_type,
            "title": record.title,
        }
        if record.image_bytes is not None:
            metadata["image_data"] = base64.b64encode(record.image_bytes).decode("utf-8")

        self.doc_collection.upsert(
            documents=[record.document_text],
            metadatas=[metadata],
            ids=[record.id],
        )

        if self.image_collection is not None:
            import numpy as np
            from PIL import Image

            with Image.open(record.image_path) as img:
                img_array = np.array(img.convert("RGB"))
            self.image_collection.upsert(
                metadatas=[{
                    "pdf_file": record.pdf_file,
                    "image": record.image_path,
                    "image_path": record.image_path,
                }],
                images=[img_array],
                ids=[record.id],
            )

    def search(self, query: str, n_results: int = 5) -> List[SearchResult]:
        results = self.doc_collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for idx, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][idx] or {}
            image_data: Optional[str] = metadata.get("image_data")
            hits.append(SearchResult(
                id=doc_id,
                pdf_file=metadata.get("pdf_file", ""),
                image_path=metadata.get("image_path", ""),
                analysis=json.loads(results["documents"][0][idx]),
                image_bytes=base64.b64decode(image_data) if image_data else None,
                distance=results["distances"][0][idx],
            ))
        return hits

    def count(self) -> int:
        return self.doc_collection.count()
