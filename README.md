# Agentic Voice-to-Voice AI Assistant for Product Discovery

This project scaffolds a voice-first, multi-agent assistant that helps users discover products using a private slice of the Amazon Product Dataset 2020. The stack combines LangGraph for orchestration, an MCP server for tool exposure, local RAG over a vector index, optional live web search, ASR/TTS for voice I/O, and a Streamlit UI.

## Repository Layout

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

## Getting Started

1. Create a virtual environment and install dependencies:
   ```bash
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

## Notes

- Real API keys must be supplied via environment variables or a local `.env` file. Do **not** hardcode secrets.
- TODO comments across the codebase indicate where to plug in concrete implementations for dataset handling, vector search, LangGraph logic, ASR/TTS providers, and MCP plumbing.
