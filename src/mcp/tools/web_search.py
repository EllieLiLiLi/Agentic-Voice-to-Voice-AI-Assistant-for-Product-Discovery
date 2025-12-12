"""
web.search MCP tool implementation (Tavily-backed).

Policy:
- Only keep results from: amazon.com, walmart.com, target.com
- Normalize to a stable schema for downstream agent nodes:
  { query: str, results: [{title, url, snippet, score, price}], error: Optional[str] }
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from tavily import TavilyClient
from tavily.errors import InvalidAPIKeyError

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = int(os.getenv("WEB_SEARCH_TOP_K", "5"))
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

ALLOWED_DOMAINS = {"amazon.com", "walmart.com", "target.com"}

QUERY_SUFFIX = (
    " buy price product page "
    "site:amazon.com OR site:walmart.com OR site:target.com"
)

_PRICE_RE = re.compile(
    r"(?:USD\s*)?\$?\s*(\d{1,4}(?:[.,]\d{1,2})?)",
    flags=re.IGNORECASE,
)

def _domain_ok(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)

def _extract_price(candidate: Any) -> Optional[float]:
    """
    Try to parse a price from Tavily's fields or from title/snippet.
    Returns float if found, else None.
    """
    if candidate is None:
        return None

    # If Tavily provides numeric price sometimes
    if isinstance(candidate, (int, float)):
        try:
            return float(candidate)
        except Exception:
            return None

    # If it's a string that contains a price
    if isinstance(candidate, str):
        m = _PRICE_RE.search(candidate)
        if not m:
            return None
        raw = m.group(1).replace(",", ".").strip()
        try:
            return float(raw)
        except Exception:
            return None

    return None

def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)

def web_search(query: str, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
    """
    Call Tavily search and normalize results.

    Returns:
      {
        "query": str,
        "results": [
          {"title": str, "url": str, "snippet": str, "score": float, "price": Optional[float]}
        ],
        "error": Optional[str]
      }
    """
    if not query or not query.strip():
        return {"query": query, "results": [], "error": None}

    if not TAVILY_API_KEY:
        msg = "TAVILY_API_KEY not set"
        logger.warning(msg)
        return {"query": query, "results": [], "error": msg}

    rewritten_query = f"{query.strip()} {QUERY_SUFFIX}"

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)

        raw = client.search(
            query=rewritten_query,
            max_results=max(10, int(top_k) * 3), 
            include_answer=False,
            include_raw_content=False,
            include_images=False,
        )
    except InvalidAPIKeyError as e:
        msg = "Unauthorized: missing or invalid API key"
        logger.exception("web.search request failed: %s", e)
        return {"query": query, "results": [], "error": msg}
    except Exception as e:
        logger.exception("web.search request failed")
        return {"query": query, "results": [], "error": str(e)}

    raw_results = raw.get("results", []) if isinstance(raw, dict) else []
    normalized: List[Dict[str, Any]] = []

    for r in raw_results:
        if not isinstance(r, dict):
            continue

        url = _safe_str(r.get("url")).strip()
        if not url or not _domain_ok(url):
            continue 

        title = _safe_str(r.get("title")).strip()
        snippet = _safe_str(r.get("content") or r.get("snippet") or r.get("description")).strip()
        score = r.get("score", 0.0)
        try:
            score_f = float(score) if score is not None else 0.0
        except Exception:
            score_f = 0.0

        price = _extract_price(r.get("price"))
        if price is None:
            price = _extract_price(title) or _extract_price(snippet)

        normalized.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "score": score_f,
                "price": price,
            }
        )

        if len(normalized) >= int(top_k):
            break

    logger.info("web.search returned %d filtered results for query '%s'", len(normalized), query)
    return {"query": query, "results": normalized, "error": None}
