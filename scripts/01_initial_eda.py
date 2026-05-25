"""
Initial EDA script for Clickstream Behavior Intelligence App.

Run:
    python scripts/01_initial_eda.py
"""

from pathlib import Path
import pandas as pd


RAW_PATH = Path("data/raw/clickstream_raw.csv")
REPORT_TABLE_DIR = Path("reports/tables")


def main() -> None:
    REPORT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Please run: python scripts/00_download_data.py"
        )

    df = pd.read_csv(RAW_PATH, sep=";")

    print("=== Dataset Shape ===")
    print(df.shape)

    print("\n=== Columns ===")
    print(df.columns.tolist())

    print("\n=== Data Types ===")
    print(df.dtypes)

    print("\n=== Missing Values ===")
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing)

    print("\n=== Duplicate Rows ===")
    print(df.duplicated().sum())

    print("\n=== Numeric Description ===")
    print(df.describe())

    print("\n=== Unique Values per Column ===")
    unique_values = df.nunique().sort_values(ascending=False)
    print(unique_values)

    missing.to_csv(REPORT_TABLE_DIR / "missing_values.csv")
    unique_values.to_csv(REPORT_TABLE_DIR / "unique_values.csv")

    print("\nEDA summary tables saved to reports/tables/")


if __name__ == "__main__":
    main()
