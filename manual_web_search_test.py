"""Manual runner for the web.search MCP tool.

Usage:
    python manual_web_search_test.py

This script executes a sample query against Tavily via the web_search tool and
prints the normalized results.
"""
from __future__ import annotations

from pprint import pprint


if __name__ == "__main__":
    # Import inside main to avoid pulling optional dependencies during test collection
    from src.mcp.tools.web_search import web_search

    query = "latest toys trend for 3 year old girls"
    result = web_search(query=query)
    pprint(result)
