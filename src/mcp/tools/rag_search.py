"""Local RAG search tool implementation for MCP."""
from __future__ import annotations

from typing import Dict, List

from src.models.vector_store import VectorStore
from src.models.schemas import Product, SearchResult


# TODO: Inject via dependency management instead of global instantiation.
VECTOR_STORE = VectorStore(index_dir=Path("data/indexes"))


from pathlib import Path  # noqa: E402


def rag_search(query: str, k: int = 5) -> Dict:
    """Run similarity search over the local vector index.

    Returns a dictionary-friendly payload suited for MCP JSON responses.
    """

    hits = VECTOR_STORE.search(query=query, k=k)
    results: List[SearchResult] = []
    for doc_id, score in hits:
        # TODO: Fetch product metadata based on doc_id.
        product = Product(id=doc_id, title="TODO: lookup title")
        results.append(SearchResult(product=product, score=score, source="local"))

    return {"results": [r.dict() for r in results]}
