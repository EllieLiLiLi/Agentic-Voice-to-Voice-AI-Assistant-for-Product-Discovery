import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Columns we keep and rename
COLUMN_MAP = {
    "Uniq Id": "product_id",
    "Product Name": "title",
    "Selling Price": "price",
    "Product Url": "url",
}

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info(f"Loaded raw data with {len(df)} rows and {len(df.columns)} columns")
    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only 4 required columns:
    - product_id
    - title
    - price
    - url
    """
    missing_cols = [c for c in COLUMN_MAP.keys() if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Raw data is missing required columns: {missing_cols}")

    # Select & rename
    df = df[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP)

    # Drop rows with missing critical fields
    df = df.dropna(subset=["product_id", "title", "price", "url"])

    # Convert price to numeric
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])

    # Final log
    logger.info(f"Cleaned dataframe shape: {df.shape}")
    return df

def save_cleaned_data(df: pd.DataFrame, path: str):
    df.to_parquet(path, index=False)
    logger.info(f"Saved cleaned data to {path}")
