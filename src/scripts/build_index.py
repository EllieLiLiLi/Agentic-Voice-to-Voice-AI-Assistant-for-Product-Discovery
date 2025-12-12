import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd

from src.data.cleaning import load_raw_data, clean_dataframe, save_cleaned_data
from src.data.embedding import build_vector_index

RAW_CSV_PATH = "data/raw/amazon2020.csv"
CLEANED_PARQUET_PATH = "data/processed/products_cleaned.parquet"
CHROMA_INDEX_DIR = "data/processed/chroma_index"


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild product vector index.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Remove existing cleaned data and index before rebuilding.",
    )
    parser.add_argument(
        "--raw-path",
        default=RAW_CSV_PATH,
        help="Path to raw amazon2020.csv",
    )
    parser.add_argument(
        "--cleaned-path",
        default=CLEANED_PARQUET_PATH,
        help="Path to save cleaned parquet.",
    )
    parser.add_argument(
        "--index-dir",
        default=CHROMA_INDEX_DIR,
        help="Directory for Chroma index.",
    )
    parser.add_argument(
        "--price-cap-quantile",
        type=float,
        default=0.95,
        help="Quantile used to cap extreme prices.",
    )
    return parser.parse_args()


def _maybe_reset_outputs(cleaned_path: Path, index_dir: Path) -> None:
    if cleaned_path.exists():
        logger.info("Removing existing cleaned file %s", cleaned_path)
        cleaned_path.unlink()

    if index_dir.exists():
        logger.info("Removing existing index directory %s", index_dir)
        shutil.rmtree(index_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    cleaned_path = Path(args.cleaned_path)
    index_dir = Path(args.index_dir)

    if args.rebuild:
        _maybe_reset_outputs(cleaned_path, index_dir)


    raw_df = load_raw_data(args.raw_path)

 
    cleaned_df = clean_dataframe(raw_df, price_cap_quantile=args.price_cap_quantile)
    logger.info("Cleaned dataframe shape: %s", cleaned_df.shape)
    save_cleaned_data(cleaned_df, str(cleaned_path))

 
    index_dir.mkdir(parents=True, exist_ok=True)
    build_vector_index(cleaned_df, index_dir=str(index_dir))


if __name__ == "__main__":
    main()
