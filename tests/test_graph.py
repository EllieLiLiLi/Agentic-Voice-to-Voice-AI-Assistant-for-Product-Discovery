"""Quick test for the agent graph."""
import os

# Set environment variables (replace with your keys)
os.environ['ANTHROPIC_API_KEY'] = 'your-key-here'
os.environ['TAVILY_API_KEY'] = 'your-key-here'  # or skip if using mock

from src.graph.graph import agent

def test_basic_query():
    """Test a basic product recommendation query."""
    initial_state = {
        "user_query": "toy for 3 year old under $25",
        "intent": None,
        "constraints": None,
        "safety_flags": None,
        "plan": None,
        "search_strategy": None,
        "search_params": None,
        "rag_results": None,
        "web_results": None,
        "reconciled_results": None,
        "final_answer": None,
        "citations": None,
        "timestamp": None,
        "node_logs": []
    }
    
    print("Running agent...")
    result = agent.invoke(initial_state)
    
    print("\n✅ Agent executed successfully!")
    print(f"\nIntent: {result['intent']['type']}")
    print(f"Strategy: {result['search_strategy']}")
    print(f"Results: {len(result.get('reconciled_results', []))}")
    print(f"\nAnswer: {result['final_answer']['spoken_summary']}")

if __name__ == "__main__":
    test_basic_query()
