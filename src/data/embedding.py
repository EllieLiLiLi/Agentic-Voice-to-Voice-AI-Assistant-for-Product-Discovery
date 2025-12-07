"""Embedding utilities for vector indexing.

Supports OpenAI embeddings by default with a "dummy" fallback for offline use.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Callable, List, Literal, Sequence

import numpy as np
import pandas as pd

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:  # pragma: no cover - chromadb should be installed via requirements
    chromadb = None  # type: ignore
    embedding_functions = None  # type: ignore

logger = logging.getLogger(__name__)

EmbeddingBackend = Literal["openai", "dummy"]

# Cleaned dataframe schema is expected to include only these text columns.
TEXT_COLUMNS: Sequence[str] = (
    "title",
    "brand",
    "category",
)


class DummyEmbeddingFunction:
    """Deterministic embedding based on hashing text.

    This keeps the pipeline runnable without network credentials, but should be
    swapped for a real model (e.g., OpenAI or local sentence-transformers) for
    production quality search.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def __call__(self, texts: Sequence[str]) -> List[List[float]]:  # pragma: no cover - simple deterministic behavior
        vectors: List[List[float]] = []
        for text in texts:
            # Create a deterministic hash-based vector
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            repeats = int(np.ceil(self.dim / len(digest)))
            raw = (digest * repeats)[: self.dim]
            vectors.append([b / 255.0 for b in raw])
        return vectors


def get_embedding_function(backend: EmbeddingBackend = "openai") -> Callable[[Sequence[str]], List[List[float]]]:
    """Return an embedding function compatible with Chroma.

    Parameters
    ----------
    backend: "openai" | "dummy"
        Embedding provider to use. OpenAI requires the ``OPENAI_API_KEY`` and
        ``EMBEDDING_MODEL_NAME`` environment variables.
    """

    if backend == "openai":
        if embedding_functions is None:
            raise ImportError("chromadb is required for OpenAI embedding support")
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
        model_name = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY or LLM_API_KEY must be set for OpenAI embeddings.")
        logger.info("Using OpenAI embeddings with model %s", model_name)
        return embedding_functions.OpenAIEmbeddingFunction(model_name=model_name, api_key=api_key)

    logger.warning("Using dummy embeddings; search quality will be poor. Set EMBEDDING_BACKEND=openai for real embeddings.")
    return DummyEmbeddingFunction()


def _make_document(row: pd.Series) -> str:
    """Combine cleaned text fields into a single embedding document.

    The cleaned dataframe is expected to have ``title``, ``brand``, and
    ``category`` columns. Any non-empty values are joined with ``|`` to form a
    compact representation suitable for embedding.
    """

    parts: List[str] = []
    for col in TEXT_COLUMNS:
        if col in row.index:
            val = row[col]
            if pd.notna(val):
                text = str(val).strip()
                if text:
                    parts.append(text)

    return " | ".join(parts).strip()


def build_vector_index(df: pd.DataFrame, index_dir: Path, collection_name: str = "products") -> None:
    """Build and persist a Chroma vector index from the cleaned dataframe."""

    if chromadb is None:  # pragma: no cover - dependency should be installed
        raise ImportError("chromadb is required to build the vector index")

    index_dir.parent.mkdir(parents=True, exist_ok=True)

    embedding_backend: EmbeddingBackend = os.environ.get("EMBEDDING_BACKEND", "openai")  # type: ignore
    embed_fn = get_embedding_function(backend=embedding_backend)

    # Recreate the persistent index directory to avoid stale state
    if index_dir.exists():
        logger.info("Removing existing index directory at %s", index_dir)
        shutil.rmtree(index_dir)

    client = chromadb.PersistentClient(path=str(index_dir))
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embed_fn)

    valid_ids: List[str] = []
    valid_documents: List[str] = []
    valid_metadatas: List[dict] = []

    for idx, row in df.iterrows():
        doc = _make_document(row)
        if not doc:
            continue

        product_id_str = f"prod-{idx}"

        metadata = {
            "title": row.get("title") if pd.notna(row.get("title")) else None,
            "brand": row.get("brand") if pd.notna(row.get("brand")) else None,
            "category": row.get("category") if pd.notna(row.get("category")) else None,
        }

        valid_ids.append(product_id_str)
        valid_documents.append(doc)
        valid_metadatas.append(metadata)

    logger.info("Prepared %d valid documents out of %d cleaned rows", len(valid_documents), len(df))

    if not valid_documents:
        logger.warning("No valid documents to index; skipping Chroma add.")
        return

    assert len(valid_ids) == len(valid_documents) == len(valid_metadatas)

    logger.info("Adding %d documents to Chroma collection '%s'", len(valid_documents), collection_name)
    collection.add(ids=valid_ids, documents=valid_documents, metadatas=valid_metadatas)

    logger.info("Finished building index at %s", index_dir)
