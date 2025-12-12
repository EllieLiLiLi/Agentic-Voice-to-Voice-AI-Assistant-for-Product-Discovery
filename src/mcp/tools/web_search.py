"""Web search MCP tool implementation (Tavily).

Goal:
- Provide a thin wrapper around Tavily web search.
- Normalize results to a stable JSON shape for MCP clients.
- Best-effort extract `price` from snippets so downstream can enforce budgets.

Env:
- TAVILY_API_KEY (preferred) or WEB_SEARCH_API_KEY (backward compatible)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5

# Backward compatible env name (your earlier version used WEB_SEARCH_API_KEY)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or os.getenv("WEB_SEARCH_API_KEY")


_PRICE_PATTERNS = [
    # $12.99, $ 12, $12
    re.compile(r"\$\s*(\d+(?:\.\d{1,2})?)", re.IGNORECASE),
    # USD 12.99 / USD12.99
    re.compile(r"\bUSD\s*(\d+(?:\.\d{1,2})?)\b", re.IGNORECASE),
    # 12.99 USD
    re.compile(r"\b(\d+(?:\.\d{1,2})?)\s*USD\b", re.IGNORECASE),
    # 12 dollars
    re.compile(r"\b(\d+(?:\.\d{1,2})?)\s*(?:dollars|usd)\b", re.IGNORECASE),
]


def _extract_price(text: str) -> Optional[float]:
    """Best-effort extract a price from free text."""
    if not text:
        return None
    for pat in _PRICE_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                val = float(m.group(1))
                # avoid absurd values from random numbers
                if 0 < val < 10000:
                    return val
            except Exception:
                continue
    return None


def _normalize_tavily_results(raw: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    items = raw.get("results") or []
    normalized: List[Dict[str, Any]] = []
    for it in items[:top_k]:
        title = (it.get("title") or "").strip()
        url = (it.get("url") or "").strip()
        snippet = (it.get("content") or it.get("snippet") or "").strip()
        score = it.get("score")

        price = _extract_price(title) or _extract_price(snippet)

        normalized.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "price": price,  # float or None
                "score": score,  # tavily similarity score if provided
            }
        )
    return normalized


def web_search(query: str, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
    """Call Tavily web search and normalize results.

    Returns:
        {
          "query": str,
          "results": [
            {"title": str, "url": str, "snippet": str, "price": float|None, "score": float|None},
            ...
          ],
          "error": str|None
        }
    """
    if not query:
        return {"query": query, "results": [], "error": None}

    if not TAVILY_API_KEY:
        msg = "TAVILY_API_KEY (or WEB_SEARCH_API_KEY) not set; returning empty results"
        logger.warning(msg)
        return {"query": query, "results": [], "error": "WEB_SEARCH_API_KEY not set"}

    try:
        # Import lazily so pytest collection doesn't require it.
        from tavily import TavilyClient  # type: ignore
    except Exception as e:
        logger.exception("tavily-python not installed or failed to import")
        return {
            "query": query,
            "results": [],
            "error": f"tavily import error: {e}",
        }

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)

        # Tavily returns: {"query": ..., "results":[{title,url,content,score,...}], ...}
        raw = client.search(
            query=query,
            max_results=top_k,
            include_answer=False,
            include_raw_content=False,
        )

        results = _normalize_tavily_results(raw, top_k=top_k)
        logger.info("web.search returned %d results for query '%s'", len(results), query)
        return {"query": query, "results": results, "error": None}

    except Exception as e:
        logger.exception("web.search request failed")
        return {"query": query, "results": [], "error": str(e)}
