"""LangGraph node definitions for the agentic workflow."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ============================================
# Initialize LLM
# ============================================

import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0.1,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=4096
)

# ============================================
# Pydantic Schemas
# ============================================

class Constraints(BaseModel):
    """Extracted constraints from user query"""
    price_max: Optional[float] = Field(None, description="Maximum price in USD")
    price_min: Optional[float] = Field(None, description="Minimum price in USD")
    age: Optional[str] = Field(None, description="Target age or age range")
    gender: Optional[str] = Field(None, description="Target gender: girl, boy, or any")
    brand: Optional[str] = Field(None, description="Specific brand name")
    material: Optional[str] = Field(None, description="Material preference")
    eco_friendly: Optional[bool] = Field(None, description="Eco-friendly requirement")
    educational: Optional[bool] = Field(None, description="Educational toy preference")
    category: Optional[str] = Field(None, description="Specific toy category")
    rating_min: Optional[float] = Field(None, description="Minimum rating (1-5)")


class RouterOutput(BaseModel):
    """Structured output from Router node"""
    intent_type: Literal[
        "product_recommendation",
        "comparison", 
        "filter_extraction",
        "out_of_scope"
    ] = Field(..., description="Classified intent type")
    
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score for intent classification"
    )
    
    constraints: Constraints = Field(
        default_factory=Constraints,
        description="Extracted constraints"
    )
    
    safety_flags: List[str] = Field(
        default_factory=list,
        description="List of safety concerns"
    )
    
    reasoning: str = Field(
        ...,
        description="Brief explanation of the classification"
    )


class SearchParams(BaseModel):
    """Parameters for search execution"""
    top_k: int = Field(default=5, description="Number of results to return")
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Filters to apply (price, rating, etc.)"
    )


class PlannerOutput(BaseModel):
    """Structured output from Planner node"""
    search_strategy: Literal["rag_only", "web_only", "hybrid"] = Field(
        ...,
        description="Which search strategy to use"
    )
    
    plan: List[Literal["rag.search", "web.search"]] = Field(
        ...,
        description="Ordered list of tools to call"
    )
    
    reasoning: str = Field(
        ...,
        description="Explanation for the chosen strategy"
    )
    
    search_params: SearchParams = Field(
        default_factory=SearchParams,
        description="Parameters for search execution"
    )


class Citation(BaseModel):
    """Citation reference for a product"""
    type: Literal["rag", "web"] = Field(..., description="Source type")
    id: Optional[str] = Field(None, description="Product ID for RAG, or URL for web")
    title: str = Field(..., description="Product title")
    price: Optional[float] = Field(None, description="Product price if available")
    rating: Optional[float] = Field(None, description="Product rating if available")


class AnswererOutput(BaseModel):
    """Structured output from Answerer/Critic node"""
    spoken_summary: str = Field(
        ...,
        description="Concise summary suitable for TTS (≤50 words)",
        max_length=300
    )
    
    detailed_analysis: str = Field(
        ...,
        description="Comprehensive analysis with comparisons and trade-offs"
    )
    
    citations: List[Citation] = Field(
        ...,
        description="All sources cited in the answer"
    )
    
    hallucination_check: Literal["passed", "failed"] = Field(
        ...,
        description="Whether the answer is fully grounded in retrieved results"
    )
    
    warnings: List[str] = Field(
        default_factory=list,
        description="Any caveats or warnings for the user"
    )


# ============================================
# System Prompts
# ============================================

ROUTER_SYSTEM_PROMPT = """You are an intent classifier for a voice-based product discovery assistant specializing in Toys & Games.

Your task is to analyze user queries and extract:
1. **Intent Type** - What does the user want?
2. **Constraints** - What are their requirements?
3. **Safety Flags** - Any concerns?

**Intent Types:**
- `product_recommendation`: User wants product suggestions (e.g., "find me a toy for my daughter")
- `comparison`: User wants to compare specific products (e.g., "compare building blocks vs puzzles")
- `filter_extraction`: User is refining search (e.g., "show me ones under $20")
- `out_of_scope`: Query is not about product discovery (e.g., "what's the weather?")

**Constraints to Extract:**
- `price_max`: Maximum price (float)
- `price_min`: Minimum price (float)
- `age`: Target age or age range (string, e.g., "3 years", "3-5 years")
- `gender`: Target gender if specified (string: "girl", "boy", "any")
- `brand`: Specific brand mentioned (string)
- `material`: Material preference (string, e.g., "wood", "plastic")
- `eco_friendly`: Eco-friendly preference (boolean)
- `educational`: Educational toy preference (boolean)
- `category`: Specific category (string, e.g., "building toys", "dolls")
- `rating_min`: Minimum rating (float, 1-5)

**Safety Flags:**
- Flag queries that are inappropriate, harmful, or unrelated to toys

**Output Format:**
Return a JSON object with:
{
  "intent_type": "product_recommendation",
  "confidence": 0.95,
  "constraints": {
    "price_max": 30.0,
    "age": "3 years",
    "gender": "girl"
  },
  "safety_flags": [],
  "reasoning": "User wants toy recommendations for a 3-year-old girl with budget constraint"
}
"""

PLANNER_SYSTEM_PROMPT = """You are a search strategy planner for a product discovery assistant.

Your task is to analyze the user's intent and constraints, then decide:
1. Which tools to call (rag.search and/or web.search)
2. The search strategy to use
3. The execution plan

**Available Tools:**
- `rag.search`: Searches our private product catalog (Amazon 2020 Toys & Games dataset)
  - Use for: General product discovery, finding products by features/category
  - Strengths: Rich product details, ratings, features, ingredients
  - Limitations: Data from 2020, may not have latest prices

- `web.search`: Searches live web for current information
  - Use for: Latest prices, availability, trending products, recent reviews
  - Strengths: Current information, real-time pricing
  - Limitations: Less structured data, may need reconciliation

**Search Strategies:**
- `rag_only`: Only search private catalog
  - Use when: User wants general recommendations, no emphasis on "current/latest/now"
  - Example: "toy for 3 year old girl", "educational building blocks"

- `web_only`: Only search web
  - Use when: User explicitly asks for current info, latest trends, or price comparisons
  - Example: "latest toy trends", "current price of [specific product]"

- `hybrid`: Search both and reconcile results
  - Use when: User wants comprehensive comparison or mentions both features AND current pricing
  - Example: "compare prices", "best rated with current availability"

**Decision Rules:**
1. Default to `rag_only` for standard product recommendations
2. Use `web_only` ONLY if user explicitly mentions: "latest", "current", "now", "today", "trending"
3. Use `hybrid` if user asks for: "compare prices", "best deal", "availability"
4. For `out_of_scope` intent, return empty plan

**Output Format:**
Return a JSON object:
{
  "search_strategy": "rag_only",
  "plan": ["rag.search"],
  "reasoning": "User wants general toy recommendations; private catalog has sufficient data",
  "search_params": {
    "top_k": 5,
    "filters": {"price_max": 25.0, "age": "3 years"}
  }
}
"""

ANSWERER_SYSTEM_PROMPT = """You are a product recommendation assistant that synthesizes search results into concise, helpful answers.

Your task is to:
1. **Synthesize Results**: Create a clear, concise recommendation based on retrieved products
2. **Generate Citations**: Every factual claim must cite its source
3. **Provide Trade-offs**: Help users understand price vs quality vs features
4. **Check Grounding**: Only state facts that are present in the retrieved results

**Output Requirements:**

**Spoken Summary** (for TTS, ≤15 seconds / ~50 words):
- Start with the number of options found
- Highlight 1-2 top picks with key features
- Mention price range
- Natural, conversational tone

**Detailed Analysis** (for screen display):
- Top 3 recommendations with reasoning
- Feature comparisons
- Price vs rating trade-offs
- Any important caveats

**Citations Format:**
- RAG sources: `[RAG:product_id]` (e.g., [RAG:B07KMVJJK7])
- Web sources: `[WEB:url]` (e.g., [WEB:https://example.com])
- Every product mentioned must have a citation

**Hallucination Prevention:**
- NEVER invent product names, prices, or features
- If a detail is not in the results, don't mention it
- If no results found, say so clearly

**Trade-off Analysis:**
- Compare price vs rating
- Highlight unique features
- Note if higher price = better quality or just branding

**Example Output:**
{
  "spoken_summary": "I found 3 great options for you. My top pick is the Educational Building Blocks Set at $24.99 with 4.7 stars—it has 120 colorful pieces and develops motor skills. Also consider the Wooden Puzzle Set at $19.99 if you want something eco-friendly.",
  
  "detailed_analysis": "Based on your requirements for eco-friendly toys under $25 for a 3-year-old:\\n\\n1. Educational Building Blocks Set ($24.99, 4.7★) [RAG:B07KMVJJK7]\\n   - Best overall: 120 BPA-free pieces, compatible with major brands\\n   - Pros: Educational value, motor skill development\\n   - Cons: At upper price limit\\n\\n2. Wooden Puzzle Set ($19.99, 4.6★) [RAG:B07PLMK789]\\n   - Most eco-friendly: Natural beech wood, water-based paint\\n   - Pros: Budget-friendly, 4-pack variety\\n   - Cons: Less interactive than blocks\\n\\nTrade-off: The blocks offer more play value but cost $5 more. The puzzles are more budget-friendly and eco-conscious.",
  
  "citations": [
    {"type": "rag", "id": "B07KMVJJK7", "title": "Educational Building Blocks Set..."},
    {"type": "rag", "id": "B07PLMK789", "title": "Wooden Puzzle Set..."}
  ],
  
  "hallucination_check": "passed",
  "warnings": []
}
"""


# ============================================
# Node Functions
# ============================================

def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Router Node: Classifies intent and extracts constraints.
    
    Args:
        state: Current conversation state with user_query
        
    Returns:
        Updated state with intent, constraints, and safety_flags
    """
    user_query = state["user_query"]
    
    # Add to node logs
    if state.get("node_logs") is None:
        state["node_logs"] = []
    state["node_logs"].append(f"[Router] Processing query: {user_query}")
    
    logger.info(f"Router analyzing: '{user_query}'")
    
    try:
        # Use Claude with structured output
        structured_llm = llm.with_structured_output(RouterOutput)
        
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"Analyze this query: {user_query}")
        ]
        
        # Get structured response
        router_output: RouterOutput = structured_llm.invoke(messages)
        
        # Convert to dict for state
        state["intent"] = {
            "type": router_output.intent_type,
            "confidence": router_output.confidence,
            "reasoning": router_output.reasoning
        }
        
        state["constraints"] = router_output.constraints.model_dump(exclude_none=True)
        state["safety_flags"] = router_output.safety_flags
        
        logger.info(f"Router: {router_output.intent_type} (confidence: {router_output.confidence:.2f})")
        
        state["node_logs"].append(
            f"[Router] Classified as '{router_output.intent_type}' with {len(state['constraints'])} constraints"
        )
        
        # If out of scope, set final answer immediately
        if router_output.intent_type == "out_of_scope":
            state["final_answer"] = {
                "spoken_summary": "I'm sorry, I can only help with toy and game product recommendations. Please ask me about products you'd like to find!",
                "detailed_analysis": "This query is outside my scope. I specialize in helping you discover and compare toys and games.",
                "citations": [],
                "hallucination_check": "passed",
                "warnings": ["Query is out of scope"]
            }
            state["citations"] = []
            state["node_logs"].append("[Router] Out of scope - set default response")
            logger.warning("Out of scope query - will skip to END")
        
    except Exception as e:
        logger.error(f"Router error: {e}")
        # Fallback: treat as generic product recommendation
        state["intent"] = {
            "type": "product_recommendation",
            "confidence": 0.5,
            "reasoning": f"Error during classification: {str(e)}"
        }
        state["constraints"] = {}
        state["safety_flags"] = []
        state["node_logs"].append(f"[Router] Error: {str(e)}")
    
    return state


def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Planner Node: Decides which tools to call and creates execution plan.
    
    Args:
        state: Current conversation state with intent and constraints
        
    Returns:
        Updated state with plan and search_strategy
    """
    intent = state.get("intent", {})
    constraints = state.get("constraints", {})
    user_query = state["user_query"]
    
    state["node_logs"].append(f"[Planner] Creating search plan for intent: {intent.get('type')}")
    
    logger.info(f"Planner analyzing intent: {intent.get('type')}")
    
    # Handle out_of_scope - no search needed
    if intent.get("type") == "out_of_scope":
        state["plan"] = []
        state["search_strategy"] = None
        state["node_logs"].append("[Planner] Out of scope - no search plan needed")
        logger.info("Out of scope - skipping search")
        return state
    
    try:
        # Build context for planner
        planner_context = f"""
User Query: {user_query}
Intent Type: {intent.get('type')}
Confidence: {intent.get('confidence')}
Constraints: {json.dumps(constraints, indent=2)}

Based on this information, determine the search strategy and create an execution plan.
"""
        
        structured_llm = llm.with_structured_output(PlannerOutput)
        
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=planner_context)
        ]
        
        planner_output: PlannerOutput = structured_llm.invoke(messages)
        
        # Update state
        state["search_strategy"] = planner_output.search_strategy
        state["plan"] = planner_output.plan
        
        # Store search params for retriever
        if not state.get("search_params"):
            state["search_params"] = {}
        
        state["search_params"]["top_k"] = planner_output.search_params.top_k
        state["search_params"]["filters"] = planner_output.search_params.filters
        
        logger.info(f"Planner: {planner_output.search_strategy}, Plan: {planner_output.plan}")
        
        state["node_logs"].append(
            f"[Planner] Strategy: {planner_output.search_strategy}, Plan: {planner_output.plan}"
        )
        
    except Exception as e:
        logger.error(f"Planner error: {e}")
        # Fallback: default to rag_only
        state["search_strategy"] = "rag_only"
        state["plan"] = ["rag.search"]
        state["search_params"] = {"top_k": 5, "filters": constraints}
        state["node_logs"].append(f"[Planner] Error: {str(e)} - defaulting to rag_only")
    
    return state


def retriever_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retriever Node: Executes the search plan and reconciles results.
    
    Args:
        state: Current conversation state with plan and search_params
        
    Returns:
        Updated state with rag_results, web_results, and reconciled_results
    """
    plan = state.get("plan", [])
    search_params = state.get("search_params", {})
    user_query = state["user_query"]
    
    state["node_logs"].append(f"[Retriever] Executing plan: {plan}")
    
    logger.info(f"Retriever executing plan: {plan}")
    
    # Import search tools
    from src.mcp.tools.rag_search import rag_search
    from src.mcp.tools.web_search import web_search
    
    # Initialize result storage
    rag_results = []
    web_results = []
    
    # Execute plan
    for tool in plan:
        if tool == "rag.search":
            logger.info("Calling rag.search...")
            try:
                rag_response = rag_search(
                    query=user_query,
                    top_k=search_params.get("top_k", 5)
                )
                rag_results = rag_response.get("results", [])
                state["rag_results"] = rag_results
                state["node_logs"].append(f"[Retriever] RAG returned {len(rag_results)} results")
                logger.info(f"RAG search returned {len(rag_results)} results")
            except Exception as e:
                logger.error(f"RAG search error: {e}")
                state["node_logs"].append(f"[Retriever] RAG error: {str(e)}")
        
        elif tool == "web.search":
            logger.info("Calling web.search...")
            try:
                web_response = web_search(
                    query=user_query,
                    top_k=search_params.get("top_k", 5)
                )
                web_results = web_response.get("results", [])
                state["web_results"] = web_results
                state["node_logs"].append(f"[Retriever] Web returned {len(web_results)} results")
                logger.info(f"Web search returned {len(web_results)} results")
            except Exception as e:
                logger.error(f"Web search error: {e}")
                state["node_logs"].append(f"[Retriever] Web error: {str(e)}")
    
    # Simple reconciliation - just combine results
    if rag_results and web_results:
        state["reconciled_results"] = rag_results + web_results
        state["node_logs"].append(f"[Retriever] Reconciled to {len(state['reconciled_results'])} results")
    elif rag_results:
        state["reconciled_results"] = rag_results
    elif web_results:
        state["reconciled_results"] = web_results
    else:
        state["reconciled_results"] = []
        state["node_logs"].append("[Retriever] No results found")
    
    logger.info(f"Retrieval complete: {len(state.get('reconciled_results', []))} final results")
    
    return state


def answerer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Answerer/Critic Node: Synthesizes final answer with citations.
    
    Args:
        state: Current conversation state with reconciled_results
        
    Returns:
        Updated state with final_answer and citations
    """
    results = state.get("reconciled_results", [])
    intent = state.get("intent", {})
    constraints = state.get("constraints", {})
    user_query = state["user_query"]
    
    state["node_logs"].append(f"[Answerer] Synthesizing answer from {len(results)} results")
    
    logger.info(f"Answerer synthesizing answer from {len(results)} results")
    
    # Handle no results case
    if not results:
        state["final_answer"] = {
            "spoken_summary": "I'm sorry, I couldn't find any products matching your requirements. Try adjusting your constraints like price range or age.",
            "detailed_analysis": "No products found matching your criteria.",
            "citations": [],
            "hallucination_check": "passed",
            "warnings": ["No results found"]
        }
        state["citations"] = []
        state["node_logs"].append("[Answerer] No results to synthesize")
        logger.warning("No results found - returning empty answer")
        return state
    
    try:
        # Build context for answerer
        results_summary = []
        for i, result in enumerate(results[:5], 1):  # Top 5 results
            source_type = result.get("source", "unknown")
            result_text = f"""
Product {i}:
- Title: {result.get('title', 'Unknown')}
- Price: ${result.get('price', 'N/A')}
- Rating: {result.get('rating', 'N/A')} stars
- Brand: {result.get('brand', 'N/A')}
- Features: {str(result.get('features', 'N/A'))[:200]}...
- Source: {source_type}
- ID: {result.get('product_id', 'N/A')}
"""
            if source_type == "web":
                result_text += f"- URL: {result.get('url', 'N/A')}\n"
            
            results_summary.append(result_text)
        
        context = f"""
User Query: {user_query}
Intent: {intent.get('type')}
Constraints: {json.dumps(constraints, indent=2)}

Retrieved Products:
{"".join(results_summary)}

Synthesize a helpful answer that:
1. Provides a concise spoken summary (≤50 words)
2. Gives detailed analysis of top 3 products
3. Includes proper citations for every product mentioned
4. Compares trade-offs (price vs features vs rating)
5. Only states facts present in the retrieved results
"""
        
        structured_llm = llm.with_structured_output(AnswererOutput)
        
        messages = [
            SystemMessage(content=ANSWERER_SYSTEM_PROMPT),
            HumanMessage(content=context)
        ]
        
        answerer_output: AnswererOutput = structured_llm.invoke(messages)
        
        # Store in state
        state["final_answer"] = answerer_output.model_dump()
        state["citations"] = [citation.model_dump() for citation in answerer_output.citations]
        
        logger.info(f"Answerer: Generated answer with {len(answerer_output.citations)} citations")
        logger.info(f"Hallucination check: {answerer_output.hallucination_check}")
        
        state["node_logs"].append(
            f"[Answerer] Generated answer with {len(answerer_output.citations)} citations"
        )
        
    except Exception as e:
        logger.error(f"Answerer error: {e}")
        # Fallback: simple answer
        state["final_answer"] = {
            "spoken_summary": f"I found {len(results)} options for you. Check the screen for details.",
            "detailed_analysis": f"Found {len(results)} products. Error during synthesis: {str(e)}",
            "citations": [],
            "hallucination_check": "failed",
            "warnings": [f"Error: {str(e)}"]
        }
        state["citations"] = []
        state["node_logs"].append(f"[Answerer] Error: {str(e)}")
    
    return state
