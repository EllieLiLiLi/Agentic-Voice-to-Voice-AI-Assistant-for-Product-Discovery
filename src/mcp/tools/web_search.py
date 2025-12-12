"""
Web search MCP tool implementation (Tavily + minimal SerpAPI fallback).

Goal:
- Primary: Use Tavily Web Search and normalize output.
- If Tavily results have no price (common for listing pages),
  fallback to SerpAPI Google Shopping to extract a product price.
- Maintain existing JSON shape for MCP clients.
"""

from __future__ import annotations

import logging
import os
import re
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
ALLOWED_DOMAINS = [
    "amazon.com",
    "walmart.com",
    "target.com",
]

# Backward compatible env name (your earlier version used WEB_SEARCH_API_KEY)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or os.getenv("WEB_SEARCH_API_KEY")

# SerpAPI Shopping fallback
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


# -----------------------
# Price Extraction Patterns
# -----------------------
_PRICE_PATTERNS = [
    re.compile(r"\$\s*(\d+(?:\.\d{1,2})?)", re.IGNORECASE),
    re.compile(r"\bUSD\s*(\d+(?:\.\d{1,2})?)\b", re.IGNORECASE),
    re.compile(r"\b(\d+(?:\.\d{1,2})?)\s*USD\b", re.IGNORECASE),
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
                if 0 < val < 10000:
                    return val
            except Exception:
                continue
    return None


# -------------------------
# SerpAPI Fallback for Price
# -------------------------
def _serpapi_get_price(query: str) -> Optional[float]:
    """Minimal SerpAPI Google Shopping fallback: return first product price."""
    if not SERPAPI_API_KEY:
        return None

    try:
        params = {
            "engine": "google_shopping",
            "q": query,
            "api_key": SERPAPI_API_KEY,
        }
        resp = requests.get("https://serpapi.com/search", params=params, timeout=8)
        data = resp.json()
        products = data.get("shopping_results", [])
        if not products:
            return None

        price_str = products[0].get("price")
        if not price_str:
            return None

        cleaned = price_str.replace("$", "").replace(",", "").strip()
        return float(cleaned)
    except Exception:
        return None


# -------------------------
# Normalize Tavily Output
# -------------------------
def _normalize_tavily_results(raw: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    items = raw.get("results") or []
    normalized: List[Dict[str, Any]] = []

    for it in items[:top_k]:
        title = (it.get("title") or "").strip()
        url = (it.get("url") or "").strip()
        snippet = (it.get("content") or it.get("snippet") or "").strip()
        score = it.get("score")

        domain = urlparse(url).netloc.lower()
        if not any(domain.endswith(allowed) for allowed in ALLOWED_DOMAINS):
            logger.debug("Skipping non-retail domain: %s", domain)
            continue

        # First try to extract price from tavily snippet/title
        price = _extract_price(title) or _extract_price(snippet)

        # Fallback: if no price found in Tavily result, try SerpAPI Shopping
        if price is None:
            fallback_query = title or snippet
            price = _serpapi_get_price(fallback_query)

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


# -------------------------
# Main Web Search Function
# -------------------------
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
        from tavily import TavilyClient  # lazy import
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
            include_domains=ALLOWED_DOMAINS,
            include_answer=False,
            include_raw_content=False,
        )

        results = _normalize_tavily_results(raw, top_k=top_k)
        logger.info("web.search returned %d results for '%s'", len(results), query)
        return {"query": query, "results": results, "error": None}

    except Exception as e:
        logger.exception("web.search request failed")
        return {"query": query, "results": [], "error": str(e)}
