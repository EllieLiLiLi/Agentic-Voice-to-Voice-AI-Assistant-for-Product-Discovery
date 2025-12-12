"""Manual runner for the web.search MCP tool.

Run from repo root:
    python -m tests.manual.manual_web_search_test
"""

from __future__ import annotations

from pprint import pprint


def main() -> None:
    from src.mcp.tools.web_search import web_search

    query = "educational toy for 3 year old under $25"
    result = web_search(query=query, top_k=5)

    print("\n=== WEB SEARCH RESULT (normalized) ===")
    pprint(result)


if __name__ == "__main__":
    main()
