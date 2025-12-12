"""LangGraph assembly for the agentic product discovery assistant."""
from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, StateGraph

from src.graph.nodes import answerer_node, planner_node, retriever_node, router_node


class ConversationState(Dict[str, Any]):
    """Simple state container for LangGraph execution.
    
    State fields:
        user_query: str - Original user query
        intent: Optional[Dict] - {type: str, confidence: float, reasoning: str}
        constraints: Optional[Dict] - Extracted constraints (price, age, etc.)
        safety_flags: Optional[List[str]] - Safety concerns
        plan: Optional[List[str]] - Tool execution plan ["rag.search", "web.search"]
        search_strategy: Optional[str] - "rag_only" | "web_only" | "hybrid"
        search_params: Optional[Dict] - {top_k: int, filters: Dict}
        rag_results: Optional[List[Dict]] - Results from RAG search
        web_results: Optional[List[Dict]] - Results from web search
        reconciled_results: Optional[List[Dict]] - Merged and deduplicated results
        final_answer: Optional[Dict] - {spoken_summary, detailed_analysis, citations, ...}
        citations: Optional[List[Dict]] - Citation references
        timestamp: Optional[str] - ISO format timestamp
        node_logs: Optional[List[str]] - Execution trace logs
    """
    pass


def should_continue_after_router(state: ConversationState) -> str:
    """Conditional edge after Router node.
    
    Routes to END if the query is out_of_scope, otherwise continues to Planner.
    
    Args:
        state: Current conversation state
        
    Returns:
        "end" if out_of_scope, "continue" otherwise
    """
    intent_type = state.get("intent", {}).get("type")
    
    if intent_type == "out_of_scope":
        return "end"
    
    return "continue"


def build_graph() -> StateGraph:
    """Construct and return the compiled LangGraph execution graph.
    
    Graph structure:
        START → Router → [conditional] → Planner → Retriever → Answerer → END
                              ↓
                             END (if out_of_scope)
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Create graph with state schema
    graph = StateGraph(ConversationState)
    
    # Add all nodes
    graph.add_node("router", router_node)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("answerer", answerer_node)
    
    # Set entry point
    graph.set_entry_point("router")
    
    # Add conditional edge after router
    graph.add_conditional_edges(
        "router",
        should_continue_after_router,
        {
            "continue": "planner",
            "end": END
        }
    )
    
    # Add linear edges for main flow
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "answerer")
    graph.add_edge("answerer", END)
    
    # Compile the graph
    return graph.compile()


# Create the compiled agent instance
agent = build_graph()

