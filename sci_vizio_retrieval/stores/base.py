"""Backend-agnostic vector store interface and record types."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class IndexRecord:
    """One analysed image, ready to be written to a vector store."""

    id: str
    pdf_file: str
    image_path: str
    analysis: Dict
    image_bytes: Optional[bytes] = field(default=None, repr=False)

    @property
    def image_type(self) -> str:
        return str(self.analysis.get("image_type") or "")

    @property
    def title(self) -> str:
        return str(self.analysis.get("title") or "")

    @property
    def document_text(self) -> str:
        """Text that gets embedded. Identical across backends so results are comparable."""
        return json.dumps(self.analysis)


@dataclass
class SearchResult:
    """One hit returned by :meth:`VectorStore.search`."""

    id: str
    pdf_file: str
    image_path: str
    analysis: Dict
    image_bytes: Optional[bytes] = field(default=None, repr=False)
    # Backend-specific distance (lower is closer); not comparable across backends.
    distance: Optional[float] = None


class VectorStore(ABC):
    """Semantic index over image analyses."""

    #: Short backend identifier, e.g. ``"chroma"`` or ``"postgres"``.
    backend: str = ""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier of this store instance (backend + location).

        Used to track which images have been indexed *into this store*, so
        switching backends or locations re-indexes instead of silently skipping.
        Must not contain credentials.
        """

    @abstractmethod
    def upsert(self, record: IndexRecord) -> None:
        """Insert or replace a record. Must be idempotent for the same ``record.id``."""

    @abstractmethod
    def search(self, query: str, n_results: int = 5) -> List[SearchResult]:
        """Return up to ``n_results`` records closest to ``query``, best first."""

    @abstractmethod
    def count(self) -> int:
        """Number of records in the store."""

    def close(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Release any held resources."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
