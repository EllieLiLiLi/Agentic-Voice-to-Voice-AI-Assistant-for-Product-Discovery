"""LangGraph assembly for the agentic product discovery assistant."""
from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, StateGraph

from src.graph.nodes import answerer_node, planner_node, retriever_node, router_node


class ConversationState(Dict[str, Any]):
    """Simple state container for LangGraph execution."""

    # TODO: Replace with TypedDict or pydantic model once schema stabilizes.
    pass


def build_graph() -> StateGraph:
    """Construct and return the LangGraph execution graph."""

    graph = StateGraph(ConversationState)
    graph.add_node("router", router_node)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("answerer", answerer_node)

    graph.set_entry_point("router")
    graph.add_edge("router", "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "answerer")
    graph.add_edge("answerer", END)

    return graph
