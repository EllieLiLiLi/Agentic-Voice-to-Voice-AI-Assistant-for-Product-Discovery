"""Data loading and cleaning utilities for the Amazon 2020 slice.

These helpers keep the index building script tidy and allow reuse in tests
or other pipelines. They intentionally avoid hard-coding column names by
probing for the most common fields seen in public Amazon product datasets,
but include the amazon2020.csv toy-slice columns explicitly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# We must at least have these for indexing
CRITICAL_FIELDS: Sequence[str] = ("product_id", "title", "price")


def load_raw_data(raw_path: Path) -> pd.DataFrame:
    """Load the raw CSV and log basic metadata."""
    df = pd.read_csv(raw_path)
    logger.info(
        "Loaded raw data from %s with %d rows and %d columns",
        raw_path,
        len(df),
        df.shape[1],
    )
    logger.info("Columns: %s", list(df.columns))
    return df


def _first_available(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Return the first column name from candidates that exists in df."""
    return next((col for col in candidates if col in df.columns), None)


def select_and_normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Pick useful columns and normalize their names.

    The function is defensive: it searches for multiple common column names and
    only keeps those that are present. Missing optional fields remain absent,
    but critical ones must be present by downstream cleaning steps.
    """

    # ==== IMPORTANT: include amazon2020.csv column names here ====
    column_candidates = {
        # product id: prefer ASIN / SKU when available
        "product_id": [
            "asin",
            "ASIN",
            "Asin",
            "product_id",
            "sku",
            "Sku",
            "SKU",
            "id",
        ],
        # title / name
        "title": ["title", "product_title", "name", "Product Name"],
        # brand
        "brand": ["brand", "manufacturer", "maker", "Brand Name"],
        # category
        "category": ["category", "categories", "parent", "Category"],
        # price: prefer selling price, fall back to list/price columns
        "price": [
            "Selling Price",
            "List Price",
            "price",
            "list_price",
            "price_usd",
            "Price",
        ],
        # rating if we ever have it
        "rating": ["rating", "star_rating", "average_rating", "stars"],
        # long text / description
        "features": [
            "feature",
            "features",
            "description",
            "bullet_points",
            "Product Description",
            "Product Details",
            "About Product",
            "Product Specification",
        ],
        # ingredients (not really used for toys, but kept for generality)
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
    text_fields = [
        field
        for field in ("title", "brand", "category", "features", "ingredients")
        if field in subset.columns
    ]
    for field in text_fields:
        subset[field] = subset[field].astype(str)

    return subset


def filter_by_category(
    df: pd.DataFrame,
    allowed_keywords: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Filter the dataframe to rows that match allowed category keywords.

    Matching is case-insensitive and checks both the category column (if
    present) and the title as a fallback. If ``allowed_keywords`` is empty or
    ``None``, the original dataframe is returned unchanged.

    For our toys slice you can pass something like ["Toys & Games"] from
    the CLI (build_index.py --allowed-keywords "Toys & Games").
    """
    if not allowed_keywords:
        return df

    keywords_lower = [kw.lower() for kw in allowed_keywords]

    def row_matches(row: pd.Series) -> bool:
        haystacks = []
        if "category" in row and pd.notna(row["category"]):
            haystacks.append(str(row["category"]).lower())
        if "title" in row and pd.notna(row["title"]):
            haystacks.append(str(row["title"]).lower())
        combined = " ".join(haystacks)
        return any(keyword in combined for keyword in keywords_lower)

    filtered = df[df.apply(row_matches, axis=1)]
    logger.info(
        "Filtered rows: %d -> %d using keywords %s",
        len(df),
        len(filtered),
        keywords_lower,
    )
    return filtered


def clean_dataframe(
    df: pd.DataFrame,
    allowed_keywords: Optional[Sequence[str]] = None,
    price_cap_quantile: float = 0.99,
) -> pd.DataFrame:
    """Apply filtering and cleaning rules to the dataframe.

    NOTE: the signature MUST stay in sync with build_index.py:
        clean_dataframe(raw_df, allowed_keywords=..., price_cap_quantile=...)
    """

    # 1) Normalize column names & pick the useful subset
    df = select_and_normalize_columns(df)

    # 2) Drop rows missing critical fields
    before = len(df)
    df = df.dropna(subset=[col for col in CRITICAL_FIELDS if col in df.columns])
    logger.info("Dropped %d rows missing critical fields", before - len(df))

    # 3) Normalize price to float (strip currency symbols etc.)
    if "price" in df.columns:
        # Remove anything that's not digit or dot, then cast
        df["price"] = (
            df["price"]
            .astype(str)
            .str.replace(r"[^0-9.]", "", regex=True)
            .str.strip()
        )

        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        before_price = len(df)
        df = df.dropna(subset=["price"])
        logger.info("Dropped %d rows with invalid price", before_price - len(df))

        if price_cap_quantile:
            cap_value = df["price"].quantile(price_cap_quantile)
            df.loc[df["price"] > cap_value, "price"] = cap_value
            logger.info(
                "Capped price at %.2f (quantile %.2f)",
                cap_value,
                price_cap_quantile,
            )

    # 4) Normalize rating to float if present
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # 5) Optional category/title keyword filter (e.g. "Toys & Games")
    df = filter_by_category(df, allowed_keywords=allowed_keywords)

    # 6) Final tidy-up
    df = df.reset_index(drop=True)
    logger.info(
        "Final cleaned dataset has %d rows and columns %s",
        len(df),
        list(df.columns),
    )
    return df
