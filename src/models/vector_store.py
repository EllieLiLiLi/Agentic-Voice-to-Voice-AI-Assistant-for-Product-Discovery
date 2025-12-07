"""Vector store abstraction used by both RAG and MCP tools."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple


class VectorStore:
    """Minimal vector store facade to hide backend implementation.

    TODO: Replace with a concrete implementation (e.g., FAISS, Chroma) once
    embedding strategy and dependency footprint are finalized. The interface
    is intentionally small to keep callers decoupled.
    """

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.index_path = index_dir / "products.index"

    def build(self, docs: Iterable[dict]) -> None:
        """Build the index from an iterable of documents.

        Args:
            docs: Iterable of dictionaries with ``id`` and ``text`` keys.
        """

        # TODO: Generate embeddings and persist to disk.
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.touch()

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search the index and return (doc_id, score) tuples.

        TODO: Replace with actual vector similarity search.
        """

        return [("example", 0.0)] * k
