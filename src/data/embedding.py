import os
import logging
from typing import List

import pandas as pd
import chromadb
from openai import OpenAI

logger = logging.getLogger(__name__)


def make_document(row: pd.Series) -> str:
    """
    Build a text document from a cleaned row of the dataframe.
    The cleaned dataframe only has: ['title', 'brand', 'category'].
    Use these, ignore nulls, join into one string via ' | '.
    Return an empty string if nothing is usable.
    """

    text_columns: List[str] = ["title", "brand", "category"]
    parts: List[str] = []

    for col in text_columns:
        if col in row.index:
            val = row[col]
            if pd.notna(val):
                text = str(val).strip()
                if text:
                    parts.append(text)

    doc = " | ".join(parts).strip()
    return doc[:8000] if doc else ""


def chunk_list(items: List, chunk_size: int):
    """Yield consecutive chunks of size chunk_size."""

    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def get_openai_client() -> OpenAI:
    """
    Return an OpenAI client and verify OPENAI_API_KEY exists.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY must be set")
    return OpenAI()


def build_vector_index(
    cleaned_df: pd.DataFrame,
    index_dir: str = "data/processed/chroma_index",
    collection_name: str = "products",
) -> None:
    """
    Build a Chroma vector index using OpenAI embeddings.
    The cleaned dataframe has columns: ['title', 'brand', 'category'].
    Steps:
    - Build docs, ids, metadata
    - Compute embeddings manually (batching)
    - Add everything into Chroma
    """

    valid_ids: List[str] = []
    valid_documents: List[str] = []
    valid_metadatas: List[dict] = []

    for idx, row in cleaned_df.iterrows():
        doc = make_document(row)
        if not doc:
            continue

        doc_id = f"prod-{idx}"
        metadata = {
            "title": row.get("title"),
            "brand": row.get("brand"),
            "category": row.get("category"),
        }

        valid_ids.append(doc_id)
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

    client = chromadb.PersistentClient(path=index_dir)
    collection = client.get_or_create_collection(name=collection_name)

    openai_client = get_openai_client()
    batch_size = 256
    total = len(valid_documents)

    for start in range(0, total, batch_size):
        end = start + batch_size
        docs = valid_documents[start:end]
        ids = valid_ids[start:end]
        metas = valid_metadatas[start:end]

        docs = [str(x) for x in docs]

        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=docs,
        )
        embeddings = [item.embedding for item in response.data]

        collection.add(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embeddings,
        )

        logger.info("Indexed batch %d-%d of %d", start, min(end, total), total)

    logger.info(
        "Finished building vector index. Total indexed: %d", len(valid_documents)
    )
