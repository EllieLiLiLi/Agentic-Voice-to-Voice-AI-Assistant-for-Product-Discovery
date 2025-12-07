"""Data preprocessing pipeline for Amazon Product Dataset slices.

Responsibilities:
- Load raw JSON/TSV files from ``data/raw``.
- Normalize product metadata and text fields.
- Persist cleaned outputs to ``data/processed`` for downstream indexing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import AppConfig


def load_raw_files(raw_dir: Path) -> Iterable[pd.DataFrame]:
    """Yield raw dataframes from the raw directory.

    TODO: Implement readers for the specific Amazon Product Dataset 2020
    file formats (e.g., metadata JSON, reviews TSV). Consider streaming
    large files to avoid memory pressure.
    """

    for path in raw_dir.glob("*.json"):  # TODO: broaden file patterns
        yield pd.read_json(path, lines=True)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and clean a single dataframe.

    TODO: Handle missing values, standardize text casing, and derive
    search-friendly fields (title, brand, category, description). This is
    also the place to trim noisy attributes and enforce schema.
    """

    cleaned = df.copy()
    # TODO: Add actual normalization logic.
    return cleaned


def preprocess(config: AppConfig) -> Path:
    """Run the preprocessing pipeline and return path to processed output."""

    processed_dir = config.data_paths.processed
    processed_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for df in load_raw_files(config.data_paths.raw):
        normalized = normalize_dataframe(df)
        output_path = processed_dir / "products.parquet"  # TODO: sharded writes
        normalized.to_parquet(output_path, index=False)
        outputs.append(output_path)

    # TODO: Return multiple paths when sharding is introduced.
    return outputs[-1] if outputs else processed_dir


if __name__ == "__main__":  # pragma: no cover
    from src.config import get_config

    cfg = get_config()
    preprocess(cfg)
