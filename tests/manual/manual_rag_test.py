from src.mcp.tools.rag_search import rag_search

def main():

    query = "toys for 3 year old girl"

    result = rag_search(query, top_k=5)

    print("\n=== RAG Search Test Result ===")
    print(f"Query: {result['query']}")
    print("\nTop Results:\n")
    for i, item in enumerate(result["results"], start=1):
        print(f"{i}. {item}")

if __name__ == "__main__":
    main()
