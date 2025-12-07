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
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.data.cleaning import clean_dataframe, load_raw_data
from src.data.embedding import build_vector_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path("data/raw/amazon2020.csv")
CLEANED_OUTPUT_PATH = Path("data/processed/products_cleaned.parquet")
INDEX_DIR = Path("data/processed/chroma_index")
COLLECTION_NAME = "products"

DEFAULT_ALLOWED_KEYWORDS: Sequence[str] = (
    "cleaning",
    "household",
    "laundry",
    "detergent",
    "kitchen",
    "dishwasher",
    "surface",
    "soap",
    "trash",
)


def save_clean_data(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Saved cleaned data to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build vector index for Amazon 2020 slice")
    parser.add_argument("--raw-path", type=Path, default=RAW_DATA_PATH, help="Path to raw amazon2020.csv")
    parser.add_argument("--clean-output", type=Path, default=CLEANED_OUTPUT_PATH, help="Path to save cleaned parquet")
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR, help="Directory to persist Chroma index")
    parser.add_argument(
        "--allowed-keywords",
        nargs="*",
        default=list(DEFAULT_ALLOWED_KEYWORDS),
        help="Category/title keywords to keep (case-insensitive)",
    )
    parser.add_argument("--price-cap-quantile", type=float, default=0.99, help="Quantile for price capping")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index from scratch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.rebuild:
        if args.clean_output.exists():
            logger.info("Removing existing cleaned file %s", args.clean_output)
            args.clean_output.unlink()
        if args.index_dir.exists():
            logger.info("Removing existing index directory %s", args.index_dir)
            shutil.rmtree(args.index_dir)

    raw_df = load_raw_data(args.raw_path)
    cleaned_df = clean_dataframe(raw_df, allowed_keywords=args.allowed_keywords, price_cap_quantile=args.price_cap_quantile)
    logger.info("Cleaned dataframe shape: %s", cleaned_df.shape)

    save_clean_data(cleaned_df, args.clean_output)
    build_vector_index(cleaned_df, index_dir=args.index_dir, collection_name=COLLECTION_NAME)


if __name__ == "__main__":
    main()
