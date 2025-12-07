"""LangGraph node definitions for the agentic workflow."""
from __future__ import annotations

from typing import Any, Dict


def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Route user intent to planner or direct retrieval.

    TODO: Integrate LLM prompt templating for intent classification and
    return structured decisions for downstream nodes.
    """

    return {**state, "route": "plan"}


def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a plan (tool calls) given the conversation state."""

    # TODO: Implement LangGraph planning with LLM outputs.
    return {**state, "plan": ["rag.search"], "route": "retrieve"}


def retriever_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute retrieval steps (local + web) based on the plan."""

    # TODO: Invoke MCP tools and merge results.
    return {**state, "retrieval_results": []}


def answerer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Synthesize the final answer and critique it."""

    # TODO: Use LLM to combine retrieved evidence with guardrails.
    return {**state, "final_answer": "TODO: synthesized answer"}
