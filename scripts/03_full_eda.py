"""
Full EDA for Clickstream Behavior Intelligence App.

Run:
    python scripts/03_full_eda.py

Outputs:
    reports/figures/*.png
    reports/tables/*.csv
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import standardize_columns
from src.feature_engineering import build_session_features


RAW_PATH = Path("data/raw/clickstream_raw.csv")
FIG_DIR = Path("reports/figures")
TABLE_DIR = Path("reports/tables")


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save_hist(series: pd.Series, title: str, xlabel: str, output_path: Path, bins: int = 30) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(series.dropna(), bins=bins, color="#1f77b4", edgecolor="black", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, output_path: Path, rotate: int = 0) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    series.plot(kind="bar", ax=ax, color="#2ca02c")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if rotate:
        plt.xticks(rotation=rotate)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_corr_heatmap(corr: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Session-Level Feature Correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Please place it at data/raw/clickstream_raw.csv"
        )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw dataset...")
    df = pd.read_csv(RAW_PATH, sep=";")
    df = standardize_columns(df)

    numeric_cols = ["order", "price", "price_above_avg", "page", "month", "day"]
    df = coerce_numeric(df, numeric_cols)

    print("Saving event-level tables and figures...")

    if "price" in df.columns:
        save_hist(
            df["price"],
            "Price Distribution (Event Level)",
            "Price",
            FIG_DIR / "event_price_distribution.png",
            bins=40,
        )

    if "country" in df.columns:
        top_countries = df["country"].value_counts().head(10)
        top_countries.to_frame(name="count").to_csv(TABLE_DIR / "top_countries.csv")
        save_bar(
            top_countries,
            "Top 10 Countries by Clicks",
            "Country",
            "Clicks",
            FIG_DIR / "event_top_countries.png",
            rotate=45,
        )

    if "main_category" in df.columns:
        top_categories = df["main_category"].value_counts().head(10)
        top_categories.to_frame(name="count").to_csv(TABLE_DIR / "top_categories.csv")
        save_bar(
            top_categories,
            "Top 10 Main Categories by Clicks",
            "Main Category",
            "Clicks",
            FIG_DIR / "event_top_categories.png",
            rotate=45,
        )

    if "month" in df.columns:
        clicks_by_month = df.groupby("month").size().sort_index()
        clicks_by_month.to_frame(name="clicks").to_csv(
            TABLE_DIR / "clicks_by_month.csv"
        )
        save_bar(
            clicks_by_month,
            "Clicks by Month",
            "Month",
            "Clicks",
            FIG_DIR / "event_clicks_by_month.png",
        )

    if "day" in df.columns:
        clicks_by_day = df.groupby("day").size().sort_index()
        clicks_by_day.to_frame(name="clicks").to_csv(TABLE_DIR / "clicks_by_day.csv")
        save_bar(
            clicks_by_day,
            "Clicks by Day",
            "Day",
            "Clicks",
            FIG_DIR / "event_clicks_by_day.png",
        )

    if "page" in df.columns:
        clicks_by_page = df.groupby("page").size().sort_index()
        clicks_by_page.to_frame(name="clicks").to_csv(TABLE_DIR / "clicks_by_page.csv")
        save_bar(
            clicks_by_page,
            "Clicks by Page",
            "Page",
            "Clicks",
            FIG_DIR / "event_clicks_by_page.png",
        )

    print("Building session-level dataset for deeper EDA...")
    session_df = build_session_features(df)

    if "total_clicks" in session_df.columns:
        save_hist(
            session_df["total_clicks"],
            "Total Clicks per Session",
            "Total Clicks",
            FIG_DIR / "session_total_clicks_distribution.png",
            bins=50,
        )

    if "unique_products" in session_df.columns:
        save_hist(
            session_df["unique_products"],
            "Unique Products per Session",
            "Unique Products",
            FIG_DIR / "session_unique_products_distribution.png",
            bins=40,
        )

    if "avg_price" in session_df.columns:
        save_hist(
            session_df["avg_price"],
            "Average Price per Session",
            "Average Price",
            FIG_DIR / "session_avg_price_distribution.png",
            bins=40,
        )

    session_summary = session_df.describe(include="all")
    session_summary.to_csv(TABLE_DIR / "session_feature_summary.csv")

    numeric_session = session_df.select_dtypes(include="number")
    if not numeric_session.empty:
        corr = numeric_session.corr()
        corr.to_csv(TABLE_DIR / "session_feature_correlation.csv")
        save_corr_heatmap(corr, FIG_DIR / "session_correlation_heatmap.png")

    print("Full EDA completed.")
    print("Figures saved to reports/figures/")
    print("Tables saved to reports/tables/")


if __name__ == "__main__":
    main()
