"""Pluggable vector storage backends (ChromaDB or PostgreSQL/pgvector)."""

from sci_vizio_retrieval import config
from sci_vizio_retrieval.stores.base import IndexRecord, SearchResult, VectorStore

__all__ = ["IndexRecord", "SearchResult", "VectorStore", "create_vector_store"]


def create_vector_store(
    backend: str = None,
    *,
    chroma_path: str = None,
    postgres_dsn: str = None,
    postgres_table: str = None,
    embedding_model: str = None,
    index_images: bool = True,
) -> VectorStore:
    """
    Build the configured vector store. Unset arguments fall back to ``config.ini``.

    Backend libraries are imported lazily, so only the chosen backend's
    dependencies need to be installed.

    Args:
        backend: ``"chroma"`` or ``"postgres"``.
        chroma_path: ChromaDB directory (chroma only).
        postgres_dsn: Connection string (postgres only).
        postgres_table: Table name (postgres only).
        embedding_model: sentence-transformers model name (postgres only).
        index_images: Also store CLIP image embeddings (chroma only). Pass
            False for search-only use to avoid loading the CLIP model.
    """
    backend = (backend or config.STORAGE_BACKEND).strip().lower()

    if backend == "chroma":
        from sci_vizio_retrieval.stores.chroma import ChromaVectorStore

        return ChromaVectorStore(
            path=config.resolve_path(chroma_path or config.CHROMA_PATH),
            index_images=index_images,
        )

    if backend == "postgres":
        from sci_vizio_retrieval.embeddings import SentenceTransformerEmbedder
        from sci_vizio_retrieval.stores.postgres import PostgresVectorStore

        return PostgresVectorStore(
            dsn=postgres_dsn or config.POSTGRES_DSN,
            embedder=SentenceTransformerEmbedder(embedding_model or config.EMBEDDING_MODEL),
            table=postgres_table or config.POSTGRES_TABLE,
        )

    raise ValueError(
        f"Unknown storage backend {backend!r}; expected one of {', '.join(config.STORAGE_BACKENDS)}"
    )
