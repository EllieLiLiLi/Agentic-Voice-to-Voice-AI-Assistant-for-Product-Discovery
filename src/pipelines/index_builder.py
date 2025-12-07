"""Index building pipeline for product retrieval."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.config import AppConfig
from src.models.vector_store import VectorStore


def iter_product_docs(processed_path: Path) -> Iterable[dict]:
    """Yield dictionaries representing product documents.

    TODO: Replace with schema-aware loading that matches the preprocessing
    output. Consider streaming to avoid loading the entire dataset in
    memory when building embeddings.
    """

    # Placeholder to illustrate interface.
    yield {"id": "example", "text": "Example product description."}


def build_index(config: AppConfig) -> Path:
    """Build vector index from processed data and return index path."""

    index_dir = config.data_paths.indexes
    index_dir.mkdir(parents=True, exist_ok=True)

    store = VectorStore(index_dir=index_dir)
    docs = iter_product_docs(config.data_paths.processed)
    store.build(docs)

    return store.index_path


if __name__ == "__main__":  # pragma: no cover
    from src.config import get_config

    cfg = get_config()
    build_index(cfg)
