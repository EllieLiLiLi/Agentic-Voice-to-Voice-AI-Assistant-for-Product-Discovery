"""Web search tool implementation for MCP."""
from __future__ import annotations

import os
from typing import Dict, List

import httpx

from src.models.schemas import Product, SearchResult


API_URL = os.getenv("WEB_SEARCH_URL", "https://example-search-api.local")
API_KEY = os.getenv("WEB_SEARCH_API_KEY")


def web_search(query: str, k: int = 5) -> Dict:
    """Call the external web search API and normalize results.

    TODO: Replace the placeholder API contract with the chosen provider's
    response schema. Ensure networking is resilient and retries are applied
    before production use.
    """

    if API_KEY is None:
        # TODO: Decide whether to raise or return an informative error payload.
        return {"results": [], "error": "WEB_SEARCH_API_KEY not set"}

    client = httpx.Client(timeout=10)
    response = client.get(
        API_URL,
        params={"q": query, "k": k},
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    response.raise_for_status()

    payload = response.json()
    # TODO: Map provider-specific schema to Product/SearchResult.
    results: List[SearchResult] = []
    for item in payload.get("results", [])[:k]:
        product = Product(
            id=item.get("id", "unknown"),
            title=item.get("title", ""),
            description=item.get("snippet"),
            url=item.get("url"),
        )
        results.append(SearchResult(product=product, score=item.get("score", 0.0), source="web"))

    return {"results": [r.dict() for r in results]}
