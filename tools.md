# MCP Tool Prompt Disclosure

This document discloses how each MCP tool is intended to be used by the LLM,
including its purpose, inputs, outputs, and known limitations.  
This is a **documentation disclosure** for evaluation and transparency, not a runtime system prompt.


## Tool: `rag_search`

### Purpose
Retrieve concrete, purchasable product candidates from a curated internal
product catalog (Amazon 2020 dataset indexed in Chroma).

This tool serves as the **primary authority** for product recommendations
and budget-based filtering.


### When to Use
- The user requests **specific product recommendations**
- The query includes constraints such as:
  - Budget (e.g., “under $25”)
  - Age group
  - Educational intent
  - Toy type or category


### Input Schema
```json
{
  "query": "string",
  "top_k": "integer"
}
```
### Output Schema

Each returned product may include:
```
{
  "product_id": "string",
  "title": "string",
  "price": "number | null",
  "url": "string | null",
  "score": "number"
}
```

### Guarantees

- Returns individual product-level items
- Includes a numeric price field when available
- Suitable for budget comparison and filtering

### Limitations

- Prices come from a historical dataset snapshot and may not reflect current prices
- No ratings, reviews, or real-time availability
- Limited to products present in the internal dataset

## Tool: `web_search`

### Purpose

Provide recent external web context, such as trends, popularity, and general
market discussion.

This tool is not authoritative for final product pricing or availability.

### When to Use

The user asks about:

- Trends
- Popularity
- “What’s popular right now”

- Additional context is needed alongside concrete product recommendations

### Input Schema
```json
{
  "query": "string",
  "top_k": "integer"
}
```
### Output Schema

Each returned product may include:
```
{

  "title": "string",
  "url": "string",
  "snippet": "string",
  "price": "number | null",
  "score": "number"

}
```
