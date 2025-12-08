"""Minimal MCP server skeleton exposing RAG and web search tools."""
from __future__ import annotations

from typing import Dict

from fastmcp import FastMCP

from src.mcp.tools.rag_search import rag_search
from src.mcp.tools.web_search import web_search


app = FastMCP(
    "product-discovery-assistant",
    summary="Agentic assistant MCP server with RAG and web search tools",
)


@app.tool(name="rag.search")
def rag_search_tool(query: str, top_k: int = 5) -> Dict:
    """Bridge decorator to expose local RAG search via MCP."""

    return rag_search(query=query, top_k=top_k)


@app.tool(name="web.search")
def web_search_tool(query: str, top_k: int = 5) -> Dict:
    """Bridge decorator to expose live web search via MCP."""

    return web_search(query=query, top_k=top_k)


if __name__ == "__main__":  # pragma: no cover
    # TODO: Load config and pass shared clients into tool functions if needed.
    app.run()
