import pandas as pd
import logging
from typing import List, Dict, Any
from openai import OpenAI

logger = logging.getLogger(__name__)

client = OpenAI()

EMBED_MODEL = "text-embedding-3-small"

def _combine_text(title: str, price: float) -> str:
    """
    Create a compact text for embedding.
    """
    return f"Product: {title}. Price: {price} USD."

def prepare_documents(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    df should contain: product_id, title, price, url
    """
    docs = []
    for _, row in df.iterrows():
        docs.append({
            "id": str(row["product_id"]),
            "text": _combine_text(row["title"], row["price"]),
            "metadata": {
                "title": row["title"],
                "price": row["price"],
                "url": row["url"],
            },
        })
    logger.info(f"Prepared {len(docs)} documents for embedding")
    return docs

def embed_documents(docs: List[Dict[str, Any]]) -> List[List[float]]:
    texts = [d["text"] for d in docs]
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    embeddings = [e.embedding for e in response.data]
    logger.info(f"Generated {len(embeddings)} embeddings")
    return embeddings
