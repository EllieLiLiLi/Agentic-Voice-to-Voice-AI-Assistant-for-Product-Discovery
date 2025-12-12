# Prompt Docs Overview

This directory centralizes the prompts for the product discovery assistant and documents how they connect to the agent graph.

## Prompt Inventory
- **router.md** — Intent classifier that extracts user goals, constraints, and safety flags.
- **planner.md** — Search strategy planner that decides which tools to call and with what parameters.
- **answerer.md** — Product recommendation generator that synthesizes retrieved results into spoken and detailed responses with citations.

## Prompt → Node/Tool Mapping
- **Router node** → Uses `router.md` to classify intent and constraint signals, producing structured metadata for downstream nodes.
- **Planner node** → Uses `planner.md` to select the search strategy and configure tool calls (`rag.search`, `web.search`, or both in hybrid mode`).
- **Answerer node** → Uses `answerer.md` to ground recommendations in retrieved results and format speech + detailed analysis.

## Tool Prompt Disclosure
- **rag.search prompt** — Expects catalog-oriented queries targeting the Amazon 2020 Toys & Games dataset; optimized for feature/category lookups, rating filters, and budget ranges. Planner defaults to this tool for general product discovery and passes structured filters to the retriever.
- **web.search prompt** — Expects live-web information needs such as current prices, availability, or trends. Planner only selects this tool when the user explicitly asks for "latest", "current", "today", or similar recency cues, or combines it with `rag.search` for hybrid comparisons.

## Few-Shot Examples
- **Answerer** includes a worked example showing both spoken and detailed outputs with citations to retrieved products (see `answerer.md`). Use it as a style reference for tone, structure, and citation formatting.
- **Planner** includes example strategy selection JSON that demonstrates the expected structured response schema.

## Additional Notes
- All prompts assume Pydantic-validated structured outputs.
- Citations must follow the RAG (`[RAG:product_id]`) or web (`[WEB:url]`) formats outlined in `answerer.md`.
