"""Embedding utilities for building and persisting the product vector index.

This module prepares clean documents from the cleaned dataframe schema and
builds a Chroma collection using OpenAI embeddings.
"""
from __future__ import annotations

import logging
import os
from typing import Any, List

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

# Cleaned dataframe schema expected from cleaning pipeline
TEXT_COLUMNS: List[str] = ["title", "brand", "category"]


def get_embedding_function(backend: str = "openai"):
    """Return a Chroma embedding function.

    Currently supports only the OpenAI backend using the
    ``text-embedding-3-small`` model. The API key must be provided via the
    ``OPENAI_API_KEY`` environment variable.
    """

    if backend != "openai":
        raise ValueError("Only 'openai' backend is supported for now.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY must be set for OpenAI embeddings.")

    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
    )


def make_document(row: pd.Series) -> str:
    """Build a single text document from a cleaned row.

    Uses ``title``, ``brand``, and ``category`` if present, ignoring null or
    empty values. Returns a joined string suitable for embedding or an empty
    string if no usable text is available.
    """

    parts: List[str] = []
    for col in TEXT_COLUMNS:
        if col in row.index:
            val = row[col]
            if pd.notna(val):
                text = str(val).strip()
                if text:
                    parts.append(text)

    doc = " | ".join(parts).strip()
    # Truncate to protect against overly long payloads
    return doc[:8000]


def build_vector_index(
    cleaned_df: pd.DataFrame,
    index_dir: str = "data/processed/chroma_index",
    collection_name: str = "products",
) -> None:
    """Build or rebuild a Chroma vector index from the cleaned dataframe.

    Parameters
    ----------
    cleaned_df:
        Dataframe expected to contain ``title``, ``brand``, and ``category``
        columns.
    index_dir:
        Directory where the Chroma persistent index will be stored.
    collection_name:
        Name of the Chroma collection.
    """

    client = chromadb.PersistentClient(path=index_dir)
    embed_fn = get_embedding_function()
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=embed_fn
    )

    valid_ids: List[str] = []
    valid_documents: List[str] = []
    valid_metadatas: List[dict[str, Any]] = []

    for idx, row in cleaned_df.iterrows():
        doc = make_document(row)
        if not doc:
            continue

        product_id = f"prod-{idx}"
        metadata = {
            "title": row.get("title") if pd.notna(row.get("title")) else None,
            "brand": row.get("brand") if pd.notna(row.get("brand")) else None,
            "category": row.get("category") if pd.notna(row.get("category")) else None,
        }

        valid_ids.append(product_id)
        valid_documents.append(doc)
        valid_metadatas.append(metadata)

    logger.info(
        "Prepared %d valid documents out of %d cleaned rows",
        len(valid_documents),
        len(cleaned_df),
    )

    if not valid_documents:
        logger.warning("No valid documents to index; skipping Chroma add.")
        return

    assert len(valid_ids) == len(valid_documents) == len(valid_metadatas)

    logger.info(
        "Adding %d documents to Chroma collection '%s'",
        len(valid_documents),
        collection_name,
    )
    collection.add(
        ids=valid_ids,
        documents=valid_documents,
        metadatas=valid_metadatas,
    )

    logger.info("Finished building index at %s", index_dir)
