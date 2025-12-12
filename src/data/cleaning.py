"""Data loading and cleaning utilities for the Amazon 2020 slice.

These helpers keep the index building script tidy and allow reuse in tests
or other pipelines. They intentionally avoid hard-coding column names by
probing for the most common fields seen in public Amazon product datasets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence
import logging

import pandas as pd

logger = logging.getLogger(__name__)


CRITICAL_FIELDS: Sequence[str] = ("product_id", "title", "price")


def load_raw_data(raw_path: Path) -> pd.DataFrame:
    """Load the raw CSV and log basic metadata."""
    df = pd.read_csv(raw_path)
    logger.info("Loaded raw data from %s with %d rows and %d columns", raw_path, len(df), df.shape[1])
    logger.info("Columns: %s", list(df.columns))
    return df


def _first_available(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    return next((col for col in candidates if col in df.columns), None)


def select_and_normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Pick useful columns and normalize their names.

    The function is defensive: it searches for multiple common column names and
    only keeps those that are present. Missing optional fields remain absent,
    but critical ones must be present by downstream cleaning steps.
    """

    column_candidates = {
        "product_id": ["asin", "product_id", "sku", "id"],
        "title": ["title", "product_title", "name", "Product Name"],
        "brand": ["brand", "manufacturer", "maker", "Brand Name"],
        "category": ["category", "categories", "parent", "Category"],
        "price": ["price", "list_price", "price_usd", "Price"],
        "rating": ["rating", "star_rating", "average_rating", "stars"],
        "features": ["feature", "features", "description", "bullet_points"],
        "ingredients": ["ingredients", "ingredient"],
    }

    rename_map: dict[str, str] = {}
    selected_columns: List[str] = []

    for normalized, candidates in column_candidates.items():
        chosen = _first_available(df, candidates)
        if chosen:
            rename_map[chosen] = normalized
            selected_columns.append(chosen)
        else:
            logger.debug("No column found for %s", normalized)

    subset = df[selected_columns].rename(columns=rename_map)

    # Combine features/description columns into a single free-text field.
    if "features" in subset.columns:
        subset["features"] = subset["features"].fillna("")

    # Ensure text fields are strings
    text_fields = [field for field in ("title", "brand", "category", "features", "ingredients") if field in subset.columns]
    for field in text_fields:
        subset[field] = subset[field].astype(str)

    return subset


def filter_toys_category(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows whose category contains ``Toys & Games`` (case-insensitive).

    If the filter would remove all rows, log a warning and return the
    unfiltered dataframe to ensure downstream indexing has data to work with.
    """

    if "category" not in df.columns:
        logger.warning("Category column not found; skipping category filter")
        return df

    before = len(df)
    mask = df["category"].str.contains("Toys & Games", case=False, na=False)
    filtered = df[mask]

    if len(filtered) == 0:
        logger.warning(
            "Category filter 'Toys & Games' would remove all rows (%d -> 0); using unfiltered dataframe instead",
            before,
        )
        return df

    logger.info("Filtered rows: %d -> %d using category filter 'Toys & Games'", before, len(filtered))
    return filtered


def _parse_price(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[^0-9.]", "", regex=True)
        .str.strip()
    )
    price = pd.to_numeric(cleaned, errors="coerce")
    return price

# def clean_dataframe(
#     df: pd.DataFrame,
#     allowed_keywords: Optional[Sequence[str]] = None,  # kept for backward compatibility; ignored
#     price_cap_quantile: float = 0.99,
# ) -> pd.DataFrame:
#     """Apply filtering and cleaning rules to the dataframe."""

#     df = select_and_normalize_columns(df)

#     # Drop rows missing critical fields
#     before = len(df)
#     df = df.dropna(subset=[col for col in CRITICAL_FIELDS if col in df.columns])
#     logger.info("Dropped %d rows missing critical fields", before - len(df))

#     # Normalize price to float
#     if "price" in df.columns:
#         df["price"] = pd.to_numeric(df["price"], errors="coerce")
#         before_price = len(df)
#         df = df.dropna(subset=["price"])
#         logger.info("Dropped %d rows with invalid price", before_price - len(df))

#         if price_cap_quantile:
#             cap_value = df["price"].quantile(price_cap_quantile)
#             df.loc[df["price"] > cap_value, "price"] = cap_value
#             logger.info("Capped price at %.2f (quantile %.2f)", cap_value, price_cap_quantile)

#     # Normalize rating to float if present
#     if "rating" in df.columns:
#         df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

#     df = filter_toys_category(df)

#     df = df.reset_index(drop=True)
#     logger.info("Final cleaned dataset has %d rows and columns %s", len(df), list(df.columns))
#     return df

# Add Price & Description:

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Columns: %s", df.columns.tolist())

    mask_toys = df["Category"].astype(str).str.contains(
        TOYS_CATEGORY_KEYWORD, na=False
    )
    df = df[mask_toys]
    logger.info(
        "Filtered rows by category '%s': %d",
        TOYS_CATEGORY_KEYWORD,
        len(df),
    )

    critical_cols = [
        "Product Name",
        "Brand Name",
        "Category",
        "Selling Price",  
    ]
    df = df.dropna(subset=critical_cols)
    logger.info("After dropping NaNs in %s: %d", critical_cols, len(df))

    df["price"] = _parse_price(df["Selling Price"])
    df = df[df["price"].notna()]
    logger.info("After parsing price: %d", len(df))

    text_cols = [
        "Product Description",
        "Product Details",
        "About Product",
        "Product Specification",
    ]
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""

    df[text_cols] = df[text_cols].fillna("")

    df["description"] = (
        df[text_cols]
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    cleaned_df = pd.DataFrame(
        {
            "title": df["Product Name"].astype(str).str.strip(),
            "brand": df["Brand Name"].astype(str).str.strip(),
            "category": df["Category"].astype(str).str.strip(),
            "price": df["price"].astype(float),
            "url": df.get("Product Url", "").astype(str),
            "description": df["description"].astype(str),
        }
    )

    logger.info(
        "Final cleaned dataset has %d rows and columns %s",
        len(cleaned_df),
        list(cleaned_df.columns),
    )

    return cleaned_df
