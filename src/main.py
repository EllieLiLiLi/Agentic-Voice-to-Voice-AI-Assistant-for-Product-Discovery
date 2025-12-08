"""CLI entrypoint placeholders for project workflows.

This module should expose click/typer commands for preprocessing, index
building, running the MCP server, and launching the Streamlit UI. Keeping
this file minimal ensures other modules remain importable without side
 effects.
"""
from __future__ import annotations

from typing import NoReturn


def main() -> NoReturn:
    """Placeholder main that documents expected commands.

    TODO: Replace with a CLI framework (e.g., Typer) and wire up commands
    such as:
    - `preprocess`: clean and normalize Amazon dataset slices
    - `build-index`: generate embeddings and persist vector store
    - `serve-mcp`: start the MCP server with registered tools
    - `ui`: launch the Streamlit interface
    """

    raise SystemExit(
        "This project uses Streamlit and MCP servers; please run the dedicated\n"
        "scripts instead of invoking main() directly."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
