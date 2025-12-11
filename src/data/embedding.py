import os
import logging
from typing import List

import pandas as pd
import chromadb
from openai import OpenAI

logger = logging.getLogger(__name__)


def _build_product_document(row) -> str:
    """Build the text that will be sent to the embedding model for one product."""

    parts = [
        row.get("title", ""),
        f"Brand: {row.get('brand', '')}",
        f"Category: {row.get('category', '')}",
    ]

    price = row.get("price")
    # avoid NaN
    if price is not None and price == price:
        parts.append(f"Price: ${float(price):.2f}")

    desc = row.get("description")
    if isinstance(desc, str) and desc.strip():
        parts.append(f"Details: {desc.strip()}")

    return " | ".join(parts)


def make_document(row: pd.Series) -> str:
    """
    Build a text document from a cleaned row of the dataframe.
    The cleaned dataframe only has: ['title', 'brand', 'category', 'price', 'url', 'description'].
    Build a rich document including core product details.
    Return an empty string if nothing is usable.
    """

    doc = _build_product_document(row)
    doc = doc[:8000].strip()
    return doc if doc else ""


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

    rows = list(cleaned_df.iterrows())
    documents = [_build_product_document(row) for _, row in rows]

    for (idx, row), doc in zip(rows, documents):
        doc = doc.strip()
        if not doc:
            continue

        doc_id = f"prod-{idx}"
        price_val = row.get("price")
        price_clean = None
        if price_val is not None and not pd.isna(price_val):
            try:
                price_clean = float(price_val)
            except (TypeError, ValueError):
                price_clean = None

        url_val = row.get("url", "")
        if pd.isna(url_val):
            url_val = ""

        metadata = {
            "title": row.get("title", ""),
            "brand": row.get("brand", ""),
            "category": row.get("category", ""),
            "price": float(row["price"]) if not pd.isna(row["price"]) else None,
            "url": row.get("url", ""),
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
