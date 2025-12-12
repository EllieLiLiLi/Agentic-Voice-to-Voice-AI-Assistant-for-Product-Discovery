"""
Web search MCP tool implementation (Tavily + Rainforest Amazon fallback).

Goal:
- Primary: Use Tavily Web Search and normalize output.
- If Tavily results have no price (common for listing pages),
  fallback to Rainforest API for Amazon product price + title.
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
ALLOWED_DOMAINS = ["amazon.com", "walmart.com", "target.com"]

_PRODUCT_PATTERNS = {
    "amazon.com": [
        re.compile(r"/dp/[A-Z0-9]{10}", re.IGNORECASE),
        re.compile(r"/gp/product/[A-Z0-9]{10}", re.IGNORECASE),
        re.compile(r"/gp/aw/d/[A-Z0-9]{10}", re.IGNORECASE),
        re.compile(r"/gp/offer-listing/[A-Z0-9]{10}", re.IGNORECASE),
    ],
    "walmart.com": [
        re.compile(r"/ip/", re.IGNORECASE),
        re.compile(r"/checkout/", re.IGNORECASE),
    ],
    "target.com": [
        re.compile(r"/p/", re.IGNORECASE),
        re.compile(r"/-/A-\d+", re.IGNORECASE),
    ],
}

# Backward compatible env name
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or os.getenv("WEB_SEARCH_API_KEY")

# ✅ Rainforest API (Amazon authoritative source)
RAINFOREST_API_KEY = os.getenv("RAINFOREST_API_KEY")


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


def _matched_allowed_domain(url: str) -> Optional[str]:
    hostname = urlparse(url).hostname or ""
    hostname = hostname.lower()
    for allowed in ALLOWED_DOMAINS:
        if hostname == allowed or hostname.endswith(f".{allowed}"):
            return allowed
    return None


def _is_allowed_domain(url: str) -> bool:
    return _matched_allowed_domain(url) is not None


def _is_product_page(url: str) -> bool:
    allowed = _matched_allowed_domain(url)
    if not allowed:
        return False
    path = urlparse(url).path or ""
    patterns = _PRODUCT_PATTERNS.get(allowed, [])
    return any(pat.search(path) for pat in patterns)


# -------------------------
# Rainforest helpers (NEW)
# -------------------------
def _extract_asin(url: str) -> Optional[str]:
    for pat in _PRODUCT_PATTERNS["amazon.com"]:
        m = pat.search(url)
        if m:
            return m.group(0).split("/")[-1]
    return None


def _rainforest_get_amazon_product(asin: str) -> Optional[Dict[str, Any]]:
    if not RAINFOREST_API_KEY or not asin:
        return None

    try:
        params = {
            "api_key": RAINFOREST_API_KEY,
            "type": "product",
            "amazon_domain": "amazon.com",
            "asin": asin,
        }
        resp = requests.get(
            "https://api.rainforestapi.com/request",
            params=params,
            timeout=10,
        )
        data = resp.json()
        product = data.get("product")
        if not product:
            return None

        price = None
        price_obj = product.get("price")
        if isinstance(price_obj, dict):
            price = price_obj.get("value")

        return {
            "title": product.get("title"),
            "price": price,
        }
    except Exception:
        logger.exception("Rainforest API failed for ASIN %s", asin)
        return None


# -------------------------
# Normalize Tavily Output
# -------------------------
def _normalize_tavily_results(raw: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    items = raw.get("results") or []
    normalized: List[Dict[str, Any]] = []
    allowed_candidates: List[Dict[str, Any]] = []

    for it in items[:top_k]:
        title = (it.get("title") or "").strip()
        url = (it.get("url") or "").strip()
        snippet = (it.get("content") or it.get("snippet") or "").strip()
        score = it.get("score")

        if not _is_allowed_domain(url):
            continue

        allowed_candidates.append({"title": title, "url": url, "snippet": snippet, "score": score})

        if not _is_product_page(url):
            continue

        normalized.append(
            _normalize_single_result(title=title, url=url, snippet=snippet, score=score)
        )

    if not normalized:
        for it in allowed_candidates:
            normalized.append(
                _normalize_single_result(
                    title=it["title"], url=it["url"], snippet=it["snippet"], score=it["score"]
                )
            )

    return normalized


def _normalize_single_result(*, title: str, url: str, snippet: str, score: Any) -> Dict[str, Any]:
    price = _extract_price(title) or _extract_price(snippet)
    final_title = title

    domain = _matched_allowed_domain(url)

    # ✅ Amazon authoritative price + title
    if domain == "amazon.com":
        asin = _extract_asin(url)
        rf = _rainforest_get_amazon_product(asin)
        if rf:
            final_title = rf.get("title") or final_title
            price = rf.get("price")

    return {
        "title": final_title,
        "url": url,
        "snippet": snippet,
        "price": price,
        "score": score,
    }


# -------------------------
# Main Web Search Function
# -------------------------
def web_search(query: str, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
    if not query:
        return {"query": query, "results": [], "error": None}

    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not set")
        return {"query": query, "results": [], "error": "WEB_SEARCH_API_KEY not set"}

    try:
        from tavily import TavilyClient
    except Exception as e:
        return {"query": query, "results": [], "error": f"tavily import error: {e}"}

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
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
        logger.exception("web.search failed")
        return {"query": query, "results": [], "error": str(e)}
