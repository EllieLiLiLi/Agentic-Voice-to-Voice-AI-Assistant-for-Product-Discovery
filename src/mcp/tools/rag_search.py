"""RAG search MCP tool using the persisted Chroma index."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import chromadb

from src.data.embedding import get_openai_client

logger = logging.getLogger(__name__)


from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


def _flatten_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Normalize Chroma query results into a list of product dicts.

    Expected input format from chroma_collection.query():
        {
            "ids": [[...]],
            "documents": [[...]],
            "metadatas": [[{"product_id": ..., "title": ..., "price": ..., "url": ...}, ...]],
            "distances": [[...]],
        }
    """
    flat: List[Dict[str, Any]] = []

    ids_batches = results.get("ids") or []
    docs_batches = results.get("documents") or []
    metas_batches = results.get("metadatas") or []
    dists_batches = results.get("distances") or []

    if not ids_batches:
        return flat

    for batch_idx in range(len(ids_batches)):
        ids = ids_batches[batch_idx] or []
        docs = docs_batches[batch_idx] or []
        metas = metas_batches[batch_idx] or []
        dists = dists_batches[batch_idx] or []

        for i in range(len(ids)):
            meta = metas[i] if i < len(metas) else {}
            distance = dists[i] if i < len(dists) else None

            # ⭐ 调试：只打印前几条的 metadata key
            if i == 0:
                logger.info("[DEBUG] RAG meta keys: %s", list(meta.keys()))
                logger.info("[DEBUG] RAG raw meta: %s", meta)

            product_id = meta.get("product_id") or ids[i]
            title = meta.get("title") or (docs[i] if i < len(docs) else None)
            price = meta.get("price")
            url = meta.get("url")

            # 距离越小越相似，这里简单转成一个 0~1 的 score，防止除以 0
            score = None
            if distance is not None:
                score = max(0.0, 1.0 - float(distance))

            flat.append(
                {
                    "id": product_id,
                    "title": title,
                    "price": price,
                    "url": url,
                    "score": score,
                    "source": "rag",
                }
            )

    logger.info("Normalized %d RAG results", len(flat))
    return flat



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
