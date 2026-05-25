"""Preprocessing utilities for clickstream data."""

import pandas as pd


RENAME_COLS = {
    "session ID": "session_id",
    "page 1 (main category)": "main_category",
    "page 2 (clothing model)": "clothing_model",
    "model photography": "model_photography",
    "price 2": "price_above_avg",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw dataset columns into cleaner names."""
    return df.rename(columns=RENAME_COLS).copy()


def basic_quality_report(df: pd.DataFrame) -> dict:
    """Return basic quality indicators."""
    return {
        "shape": df.shape,
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_values": df.nunique().to_dict(),
    }
