"""Web search MCP tool implemented with Tavily.

This module provides a defensive wrapper around the Tavily Search API and
normalizes results for MCP clients. Missing API keys or provider errors return
empty result sets with an error field instead of raising.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

try:
    from tavily import TavilyClient
except ImportError:  # pragma: no cover - optional dependency
    TavilyClient = None

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5


def _normalize_result(item: Dict[str, Any]) -> Dict[str, Any]:
    """Map Tavily fields into a consistent MCP response shape."""

    title = item.get("title") or ""
    url = item.get("url")
    snippet = item.get("content") or item.get("snippet") or ""

    score = item.get("score")
    try:
        score = float(score) if score is not None else None
    except (TypeError, ValueError):
        score = None

    return {
        "title": title,
        "url": url,
        "snippet": snippet,
        "score": score,
    }


def web_search(query: str, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
    """Call Tavily search and normalize results for MCP clients.

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

    api_key = os.getenv("WEB_SEARCH_API_KEY")
    if not api_key:
        logger.warning("WEB_SEARCH_API_KEY not set; returning empty results")
        return {"query": query, "results": [], "error": "WEB_SEARCH_API_KEY not set"}

    if TavilyClient is None:
        logger.warning("tavily-python dependency not installed; returning empty results")
        return {
            "query": query,
            "results": [],
            "error": "tavily-python not installed",
        }

    try:
        client = TavilyClient(api_key=api_key)
    except Exception as exc:  # pragma: no cover - init errors
        logger.warning("Failed to initialize Tavily client: %s", exc)
        return {"query": query, "results": [], "error": str(exc)}

    try:
        response = client.search(query=query, max_results=top_k)
    except Exception as exc:  # pragma: no cover - network/API errors
        logger.warning("web.search request failed: %s", exc)
        return {"query": query, "results": [], "error": str(exc)}

    raw_results: List[Dict[str, Any]] = []
    if isinstance(response, dict):
        raw_results = response.get("results", []) or []

    normalized = [_normalize_result(item) for item in raw_results[:top_k]]
    logger.info("web.search returned %d results for query '%s'", len(normalized), query)

    return {"query": query, "results": normalized}


__all__ = ["web_search"]
