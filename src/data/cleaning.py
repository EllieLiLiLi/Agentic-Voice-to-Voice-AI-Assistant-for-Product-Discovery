import logging
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

# 原始列名 → 统一后的列名
COLUMN_MAPPING = {
    "Uniq Id": "product_id",
    "Product Name": "title",
    "Selling Price": "price",
    "Product Url": "url",
}


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Load raw CSV from disk."""
    logger.info("Loading raw data from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded raw data with shape %s", df.shape)
    logger.info("Columns: %s", df.columns.tolist())
    return df


def _normalize_price(series: pd.Series) -> pd.Series:
    """把价格列转成 float；兼容 '$12.99'、'12.99 USD' 等格式."""
    # 先当字符串处理，提取数字部分
    s = (
        series.astype(str)
        .str.replace(r"[,$]", "", regex=True)  # 去掉逗号和美元符号
        .str.extract(r"([\d.]+)", expand=False)  # 提取第一个数字段
    )
    return pd.to_numeric(s, errors="coerce")


def clean_dataframe(
    raw_df: pd.DataFrame,
    price_cap_quantile: float = 0.95,
) -> pd.DataFrame:
    """
    清洗原始数据，只保留 RAG 需要的四个字段：
      - product_id
      - title
      - price
      - url
    且只做“温和”的过滤：
      - 去掉 price/title/url 为空的行
      - price <= 0 的行
      - 价格高于某个分位数时做截断（不丢行，只改数值）
    """
    df = raw_df.copy()

    # 只保留我们关心的四列，并统一列名
    missing_cols = [c for c in COLUMN_MAPPING.keys() if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing expected columns in raw data: {missing_cols}")

    df = df[list(COLUMN_MAPPING.keys())].rename(columns=COLUMN_MAPPING)
    logger.info("Selected columns: %s", df.columns.tolist())

    # 归一化价格
    df["price"] = _normalize_price(df["price"])

    # 基础过滤：title / price / url 必须存在
    before = len(df)
    df = df.dropna(subset=["title", "price", "url"])
    df = df[df["price"] > 0]
    logger.info("Dropped %d rows with invalid title/price/url", before - len(df))

    if df.empty:
        logger.warning("Cleaned dataframe is empty after basic filtering.")
        return df

    # 截断极端高价，避免 embedding / 检索受异常值影响
    price_cap = df["price"].quantile(price_cap_quantile)
    df.loc[df["price"] > price_cap, "price"] = price_cap
    logger.info(
        "Capped price at quantile %.2f (value=%.2f)", price_cap_quantile, price_cap
    )

    logger.info("Final cleaned dataset has %d rows and columns %s", len(df), df.columns.tolist())
    return df


def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """Save cleaned dataframe as Parquet."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Saved cleaned data to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    raw_path = "data/raw/amazon2020.csv"
    cleaned_path = "data/processed/products_cleaned.parquet"

    raw_df = load_raw_data(raw_path)
    cleaned_df = clean_dataframe(raw_df)
    logger.info("Cleaned dataframe shape: %s", cleaned_df.shape)
    save_cleaned_data(cleaned_df, cleaned_path)
