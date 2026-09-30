"""Text embedders used by vector stores that don't embed data themselves (e.g. PostgreSQL)."""

from typing import List, Protocol, Sequence

import numpy as np


class TextEmbedder(Protocol):
    """Anything that turns texts into fixed-size vectors."""

    #: Identifier of the underlying model, stored alongside vectors to detect mismatches.
    model_name: str
    #: Length of every vector returned by :meth:`embed`.
    dimension: int

    def embed(self, texts: Sequence[str]) -> List[np.ndarray]:
        ...


class SentenceTransformerEmbedder:
    """
    Embeds text with a sentence-transformers model.

    The default, ``all-MiniLM-L6-v2``, is the same model ChromaDB uses by
    default, so search quality is comparable across backends.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        # Renamed in sentence-transformers 5.x; keep supporting older releases.
        get_dim = getattr(self._model, "get_embedding_dimension", None) \
            or self._model.get_sentence_embedding_dimension
        self.dimension = int(get_dim())

    def embed(self, texts: Sequence[str]) -> List[np.ndarray]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return [np.asarray(v, dtype=np.float32) for v in vectors]


class FastEmbedEmbedder:
    """
    Embeds text with fastembed (ONNX Runtime, no torch): a much smaller install
    and memory footprint, suited to small server dynos.

    For ``all-MiniLM-L6-v2`` its vectors match :class:`SentenceTransformerEmbedder`,
    so both can share a table.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache_dir: str = None):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name, cache_dir=cache_dir)
        (probe,) = self.embed(["dimension probe"])
        self.dimension = int(probe.shape[0])

    def embed(self, texts: Sequence[str]) -> List[np.ndarray]:
        vectors = []
        for v in self._model.embed(list(texts)):
            v = np.asarray(v, dtype=np.float32)
            norm = np.linalg.norm(v)
            vectors.append(v / norm if norm else v)
        return vectors
