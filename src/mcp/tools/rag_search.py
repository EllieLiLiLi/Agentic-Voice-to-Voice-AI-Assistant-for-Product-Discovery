"""RAG search MCP tool using the persisted Chroma index."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import chromadb

from src.data.embedding import get_openai_client

logger = logging.getLogger(__name__)


def _flatten_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize Chroma's nested query output into a flat list of records."""

    metadatas = results.get("metadatas", [[]]) or [[]]
    documents = results.get("documents", [[]]) or [[]]
    distances = results.get("distances", [[]]) or [[]]

    flat_metadatas = metadatas[0] if metadatas else []
    flat_documents = documents[0] if documents else []
    flat_distances = distances[0] if distances else []

    normalized: List[Dict[str, Any]] = []
    for metadata, document, distance in zip(flat_metadatas, flat_documents, flat_distances):
        normalized.append(
            {
                "title": metadata.get("title") if metadata else None,
                "brand": metadata.get("brand") if metadata else None,
                "category": metadata.get("category") if metadata else None,
                "document": document,
                "score": distance,
            }
        )

    return normalized


def rag_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Run semantic search over the existing Chroma index.

    Args:
        query: Natural language query to embed and search.
        top_k: Number of documents to return (default: 5).

    Returns:
        A dictionary suitable for MCP JSON responses containing the query and results.
    """

    if not query:
        return {"query": query, "results": []}

    chroma_client = chromadb.PersistentClient(path="data/processed/chroma_index")
    collection = chroma_client.get_or_create_collection(name="products")

    openai_client = get_openai_client()
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[query],
    )
    query_embedding = response.data[0].embedding

    raw_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )

    normalized_results = _flatten_results(raw_results)
    logger.info("rag.search returned %d results for query '%s'", len(normalized_results), query)

    return {"query": query, "results": normalized_results}
