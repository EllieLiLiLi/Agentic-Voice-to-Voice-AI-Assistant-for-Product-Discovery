# Few-shot Examples Used by Agents

## Router Few-shot
**Example 1 – Query requiring planning + tool use**
User: "Find the best toys for a 3-year-old girl under $30."
Router Output: "planner"

**Example 2 – Query requiring direct answering**
User: "Summarize the key features of Lego DUPLO."
Router Output: "answerer"

---

## Planner Few-shot
**Example**
Task: “Find trending educational toys for toddlers and explain why they are popular.”

Planner Output structure:
1. Call `web.search` with the query  
2. Call `rag.search` for additional product metadata  
3. Synthesize results into final answer  

---

## Answerer Few-shot
**Example Integration**
Inputs:  
- RAG search results (top-K similarities)  
- Web search results (Tavily)  
Answerer combines both sources and generates a grounded explanation.
