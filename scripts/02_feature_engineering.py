"""
Create session-level behavioral dataset from raw clickstream data.

Run:
    python scripts/02_feature_engineering.py
"""

from pathlib import Path
import pandas as pd


RAW_PATH = Path("data/raw/clickstream_raw.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "session_level_dataset.csv"


RENAME_COLS = {
    "session ID": "session_id",
    "page 1 (main category)": "main_category",
    "page 2 (clothing model)": "clothing_model",
    "model photography": "model_photography",
    "price 2": "price_above_avg",
}


def safe_mode(series: pd.Series):
    mode_values = series.mode(dropna=True)
    if len(mode_values) == 0:
        return None
    return mode_values.iloc[0]


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Please run: python scripts/00_download_data.py"
        )

    df = pd.read_csv(RAW_PATH, sep=";")
    df = df.rename(columns=RENAME_COLS)

    required_columns = [
        "session_id",
        "order",
        "main_category",
        "clothing_model",
        "colour",
        "location",
        "model_photography",
        "price",
        "price_above_avg",
        "page",
        "country",
        "month",
        "day",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    session_df = df.groupby("session_id").agg(
        total_clicks=("order", "count"),
        max_order=("order", "max"),
        unique_categories=("main_category", "nunique"),
        unique_products=("clothing_model", "nunique"),
        unique_colours=("colour", "nunique"),
        unique_locations=("location", "nunique"),
        unique_pages=("page", "nunique"),
        avg_price=("price", "mean"),
        max_price=("price", "max"),
        min_price=("price", "min"),
        std_price=("price", "std"),
        premium_click_ratio=("price_above_avg", "mean"),
        dominant_category=("main_category", safe_mode),
        dominant_colour=("colour", safe_mode),
        dominant_page=("page", safe_mode),
        dominant_country=("country", safe_mode),
        month=("month", safe_mode),
        day=("day", safe_mode),
    ).reset_index()

    session_df["std_price"] = session_df["std_price"].fillna(0)

    session_df["category_diversity_ratio"] = (
        session_df["unique_categories"] / session_df["total_clicks"]
    )
    session_df["product_diversity_ratio"] = (
        session_df["unique_products"] / session_df["total_clicks"]
    )
    session_df["colour_diversity_ratio"] = (
        session_df["unique_colours"] / session_df["total_clicks"]
    )
    session_df["page_diversity_ratio"] = (
        session_df["unique_pages"] / session_df["total_clicks"]
    )

    engagement_threshold = session_df["total_clicks"].quantile(0.75)
    session_df["high_engagement"] = (
        session_df["total_clicks"] > engagement_threshold
    ).astype(int)

    session_df["premium_interest"] = (
        session_df["premium_click_ratio"] >= 0.5
    ).astype(int)

    session_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Session-level dataset saved to: {OUTPUT_PATH}")
    print(f"Shape: {session_df.shape}")
    print("\nTarget distribution: high_engagement")
    print(session_df["high_engagement"].value_counts(normalize=True))
    print("\nTarget distribution: premium_interest")
    print(session_df["premium_interest"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
