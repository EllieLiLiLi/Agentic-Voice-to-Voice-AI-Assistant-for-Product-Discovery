"""Web search MCP tool implementation.

Provides a thin wrapper around an HTTP search provider (configured via env
vars) and normalizes results for MCP clients. It is intentionally defensive:
missing API keys or provider errors result in empty result sets with an error
message rather than raising.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import httpx

logger = logging.getLogger(__name__)

# Environment configuration
WEB_SEARCH_URL = os.getenv("WEB_SEARCH_URL", "https://example-search-api.local")
WEB_SEARCH_API_KEY = os.getenv("WEB_SEARCH_API_KEY")
DEFAULT_TOP_K = 5


def _normalize_result(item: Dict[str, Any]) -> Dict[str, Any]:
    """Map provider-specific fields into a consistent shape."""

    title = item.get("title") or item.get("name") or ""
    url = item.get("url") or item.get("link")
    snippet = item.get("snippet") or item.get("description") or ""
    source = item.get("source") or item.get("domain") or "web"

    score = item.get("score")
    try:
        score = float(score) if score is not None else None
    except (TypeError, ValueError):
        score = None

    return {
        "title": title,
        "url": url,
        "snippet": snippet,
        "source": source,
        "score": score,
    }


def web_search(query: str, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
    """Call the configured web search provider and normalize results.

    Args:
        query: Natural language query string.
        top_k: Maximum number of results to return.

    Returns:
        A JSON-serializable dictionary with the query, results, and optional
        error message.
    """

    if not query:
        logger.warning("web.search called with empty query")
        return {"query": query, "results": [], "error": "empty query"}

    if not WEB_SEARCH_API_KEY:
        logger.warning("WEB_SEARCH_API_KEY not set; returning empty results")
        return {"query": query, "results": [], "error": "WEB_SEARCH_API_KEY not set"}

    headers = {"Authorization": f"Bearer {WEB_SEARCH_API_KEY}"}
    params = {"q": query, "k": top_k}

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(WEB_SEARCH_URL, params=params, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("web.search request failed: %s", exc)
        return {"query": query, "results": [], "error": str(exc)}

    raw_results: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        raw_results = payload.get("results", []) or []
    elif isinstance(payload, list):
        raw_results = payload

    normalized = [_normalize_result(item) for item in raw_results[:top_k]]
    logger.info("web.search returned %d results for query '%s'", len(normalized), query)

    return {"query": query, "results": normalized}


__all__ = ["web_search"]
