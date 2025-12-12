"""CLI to clean the Amazon 2020 slice and build a vector index.

Usage:
    python -m src.scripts.build_index --rebuild

The script performs the following steps:
1. Load ``data/raw/amazon2020.csv``.
2. Clean and filter rows for a configurable set of categories.
3. Persist the cleaned data to ``data/processed/products_cleaned.parquet``.
4. Build a Chroma vector index combining title and feature/description text.

Defaults assume OpenAI embeddings; set ``EMBEDDING_BACKEND=dummy`` to avoid
network calls during development. TODO: provide your ``OPENAI_API_KEY`` and set
``EMBEDDING_MODEL_NAME`` if you want high-quality embeddings.
"""
import logging
from chromadb import PersistentClient
from src.data.cleaning import load_raw_data, clean_dataframe, save_cleaned_data
from src.data.embedding import prepare_documents, embed_documents

RAW_PATH = "data/raw/amazon2020.csv"
CLEAN_PATH = "data/processed/products_cleaned.parquet"
INDEX_DIR = "data/processed/chroma_index"

logger = logging.getLogger(__name__)

def rebuild_index():
    # Load & clean
    raw_df = load_raw_data(RAW_PATH)
    cleaned_df = clean_dataframe(raw_df)
    save_cleaned_data(cleaned_df, CLEAN_PATH)

    # Prepare embeddings
    docs = prepare_documents(cleaned_df)
    if not docs:
        logger.warning("No valid documents for embedding; skipping index build.")
        return

    embeddings = embed_documents(docs)

    # Build Chroma index
    client = PersistentClient(path=INDEX_DIR)
    collection = client.get_or_create_collection(name="products")

    collection.delete(where={})  # reset index

    collection.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        embeddings=embeddings,
        metadatas=[d["metadata"] for d in docs],
    )

    logger.info(f"Added {len(docs)} items to Chroma index.")

def main():
    rebuild_index()

if __name__ == "__main__":
    main()
