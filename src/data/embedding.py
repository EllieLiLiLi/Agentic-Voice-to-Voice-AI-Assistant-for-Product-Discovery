import logging
import os
from typing import List, Dict, Any, Tuple

import chromadb
import pandas as pd
from openai import OpenAI

logger = logging.getLogger(__name__)


# ---------- OpenAI client ----------


def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY must be set for OpenAI embeddings."
        )
    return OpenAI(api_key=api_key)


# ---------- Document preparation ----------


def prepare_documents(
    cleaned_df: pd.DataFrame,
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    从清洗后的 dataframe 里构造：
      - documents: 用于做 embedding 的文本（这里用 title 即可）
      - metadatas: 每个向量对应的元数据（id/title/price/url）
      - ids: 作为向量数据库里的 primary key
    期望 cleaned_df 只有四列：product_id, title, price, url
    """
    required_cols = {"product_id", "title", "price", "url"}
    missing = required_cols - set(cleaned_df.columns)
    if missing:
        raise KeyError(f"Cleaned dataframe missing columns: {missing}")

    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for row in cleaned_df.itertuples(index=False):
        product_id = str(row.product_id)
        title = str(row.title).strip()
        price = float(row.price)
        url = str(row.url).strip()

        if not title:
            continue

        text = title  # 目前仅用标题做 embedding

        documents.append(text)
        ids.append(product_id)
        metadatas.append(
            {
                "product_id": product_id,
                "title": title,
                "price": price,
                "url": url,
            }
        )

    if not documents:
        logger.warning("No valid documents prepared from cleaned dataframe.")

    logger.info("Prepared %d documents for embedding", len(documents))
    return documents, metadatas, ids


# ---------- Embedding + Chroma index ----------

client = OpenAI()
def embed_documents(documents, model: str = "text-embedding-3-small"):
    if not documents:
        return []

    texts = [str(d) for d in documents]   # 再保险转一次字符串
    vectors = []

    BATCH_SIZE = 256
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        resp = client.embeddings.create(
            model=model,
            input=batch,   # ✅ 一定是 List[str]
        )
        vectors.extend([item.embedding for item in resp.data])

    return vectors


def build_vector_index(
    cleaned_df: pd.DataFrame,
    index_dir: str,
    collection_name: str = "products",
) -> None:
    """
    构建 / 更新 Chroma 向量索引。
    - documents: title
    - metadata: {product_id, title, price, url}
    - id: product_id
    """
    documents, metadatas, ids = prepare_documents(cleaned_df)

    if not documents:
        logger.warning("No valid documents for embedding; skipping index build.")
        return

    embeddings = embed_documents(documents)
    if not embeddings:
        logger.warning("Embedding call returned empty; skipping index build.")
        return

    client = chromadb.PersistentClient(path=index_dir)
    collection = client.get_or_create_collection(name=collection_name)

    logger.info("Adding %d documents to Chroma collection '%s'", len(ids), collection_name)
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    logger.info("Finished indexing %d products.", len(ids))
