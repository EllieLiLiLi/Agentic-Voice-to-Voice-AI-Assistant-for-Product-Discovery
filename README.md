# Agentic Voice-to-Voice AI Assistant for Product Discovery

🎤🧠🔍 *An agentic, multimodal, end-to-end voice assistant for product search and discovery.*

## 🌟 Overview
This project scaffolds a voice-first, multi-agent assistant that helps users discover products using a private slice of the Amazon Product Dataset 2020. The stack combines LangGraph for orchestration, an MCP server for tool exposure, local RAG over a vector index, optional live web search, ASR/TTS for voice I/O, and a Streamlit UI.

Built with:
- OpenAI Realtime API for ASR & TTS  
- LangGraph for multi-agent orchestration  
- RAG pipelines over the Amazon Product Dataset  
- MCP Tools for vector search and web search  
- Streamlit UI for mic capture, logs, and product tables

## ✨ Key Features
- 🎙 Voice Input → Voice Output  
- 🧠 Multi-agent workflow  
- 🔎 Hybrid Retrieval (RAG + Web Search)  
- 📚 Amazon Product Dataset 2020 support  
- 📈 Streamlit UI with transcripts, logs, and citations  
- 🧪 Test scaffolding included

## 📂 Repository Layout

- `src/`
  - `config.py`: Environment-driven configuration bundle for LLMs, audio, and data paths.
  - `pipelines/`: Preprocessing and index-building workflows for the Amazon dataset.
  - `models/`: Data schemas and vector store abstraction used by RAG and MCP tools.
  - `mcp/`: MCP server and tool definitions for `rag.search` and `web.search`.
  - `graph/`: LangGraph nodes and assembly for the agent workflow.
  - `asr_tts/`: ASR and TTS client wrappers.
  - `ui/`: Streamlit application surfaces mic capture, transcripts, agent logs, and product tables.
  - `utils/`: Shared helpers for logging and audio handling.
- `prompts/`: Placeholder prompt files for router, planner, and answerer/critic roles.
- `data/`: Expected location of raw and derived datasets (not versioned).
- `tests/`: Minimal placeholder tests until real coverage is added.
- `.env.example`: Template for required environment variables.
- `requirements.txt`: Python dependencies for the project.

## 🚀 Getting Started

1. Create a virtual environment and install dependencies:
   ```bash
   git clone https://github.com/EllieLiLiLi/Agentic-Voice-to-Voice-AI-Assistant-for-Product-Discovery.git
   cd Agentic-Voice-to-Voice-AI-Assistant-for-Product-Discovery
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and populate provider-specific secrets.
3. Run preprocessing and index building before starting the MCP server or UI:
   ```bash
   python -m src.scripts.build_index --rebuild
   ```
4. Launch the Streamlit UI:
   ```bash
   streamlit run src/ui/app.py
   ```

## ▶️ Running the RAG Index & Web Search Tools Locally

### Installation

1. Clone the repository and enter the project directory.
2. Install dependencies (preferably inside a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

### Required environment variables

- `OPENAI_API_KEY` — required for embedding generation and RAG indexing.
- `WEB_SEARCH_API_KEY` — Tavily API key for the `web.search` MCP tool.

Copy the template and fill in your keys (optional but recommended):
```bash
cp .env.example .env
```

Export the variables in your shell before running tools:
```bash
export OPENAI_API_KEY="your_openai_key"
export WEB_SEARCH_API_KEY="your_tavily_key"
```

### Rebuilding the RAG index

Rebuild the cleaned dataset and Chroma index from the raw CSV:
```bash
python -m src.scripts.build_index --rebuild
```

This command:
- Loads `data/raw/amazon2020.csv`.
- Cleans and writes `data/processed/products_cleaned.parquet`.
- Rebuilds the Chroma index at `data/processed/chroma_index/` (overwrites any existing index).

### Running manual tests (RAG + Web Search)

RAG search smoke test:
```bash
python tests/manual/manual_rag_test.py
```
Expected: prints the top-k similar products for the sample query.

Web search smoke test:
```bash
python tests/manual/manual_web_search_test.py
```
Expected: prints normalized Tavily search results for the sample query.

### Troubleshooting

- RAG returns 0 results → ensure you ran `python -m src.scripts.build_index --rebuild` and that the index exists in `data/processed/chroma_index/`.
- `web.search` returns empty results → confirm `WEB_SEARCH_API_KEY` is set and valid.
- Import errors or module not found → run commands from the project root so Python resolves `src/` correctly.

## 📖 Notes

- Real API keys must be supplied via environment variables or a local `.env` file. Do **not** hardcode secrets.
- TODO comments across the codebase indicate where to plug in concrete implementations for dataset handling, vector search, LangGraph logic, ASR/TTS providers, and MCP plumbing.
