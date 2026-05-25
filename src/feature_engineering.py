"""Feature engineering utilities for session-level clickstream analysis."""

import pandas as pd


def safe_mode(series: pd.Series):
    mode_values = series.mode(dropna=True)
    if len(mode_values) == 0:
        return None
    return mode_values.iloc[0]


def build_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build session-level features from standardized clickstream data."""
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

    return session_df


def add_engineered_targets(session_df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered classification targets."""
    session_df = session_df.copy()

    engagement_threshold = session_df["total_clicks"].quantile(0.75)
    session_df["high_engagement"] = (
        session_df["total_clicks"] > engagement_threshold
    ).astype(int)

    session_df["premium_interest"] = (
        session_df["premium_click_ratio"] >= 0.5
    ).astype(int)

    return session_df
