"""
Build assets for UX/Product insights.

Run:
    python scripts/06_build_insight_assets.py

Outputs:
    data/processed/session_level_with_clusters.csv
    reports/tables/cluster_profiles_kmeans_k4.csv
    reports/tables/cluster_sizes_kmeans_k4.csv
    reports/tables/session_action_list.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/processed/session_level_dataset.csv")
OUTPUT_DATASET = Path("data/processed/session_level_with_clusters.csv")

CLUSTER_PROFILE_PATH = Path("reports/tables/cluster_profiles_kmeans_k4.csv")
CLUSTER_SIZE_PATH = Path("reports/tables/cluster_sizes_kmeans_k4.csv")
SESSION_ACTION_PATH = Path("reports/tables/session_action_list.csv")

NUMERIC_FEATURES = [
    "total_clicks",
    "max_order",
    "unique_categories",
    "unique_products",
    "unique_colours",
    "unique_locations",
    "unique_pages",
    "avg_price",
    "max_price",
    "min_price",
    "std_price",
    "premium_click_ratio",
    "category_diversity_ratio",
    "product_diversity_ratio",
    "colour_diversity_ratio",
    "page_diversity_ratio",
]

SCORE_FEATURES = {
    "total_clicks": 0.35,
    "unique_products": 0.2,
    "unique_categories": 0.15,
    "page_diversity_ratio": 0.15,
    "avg_price": 0.15,
}


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - series.mean()) / std


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Run: python scripts/02_feature_engineering.py"
        )

    df = pd.read_csv(DATA_PATH)

    missing = [col for col in NUMERIC_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required numeric features: {missing}")

    X = df[NUMERIC_FEATURES].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)

    df = df.copy()
    df["cluster_kmeans_k4"] = labels

    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DATASET, index=False)

    CLUSTER_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLUSTER_SIZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSION_ACTION_PATH.parent.mkdir(parents=True, exist_ok=True)

    cluster_sizes = df["cluster_kmeans_k4"].value_counts().sort_index()
    cluster_sizes.to_frame(name="count").to_csv(CLUSTER_SIZE_PATH)

    profile = df.groupby("cluster_kmeans_k4")[NUMERIC_FEATURES].mean()
    profile.insert(0, "size", cluster_sizes)
    profile.to_csv(CLUSTER_PROFILE_PATH)

    score = pd.Series(0.0, index=df.index)
    for feature, weight in SCORE_FEATURES.items():
        if feature in df.columns:
            score += zscore(df[feature]) * weight

    df["engagement_score"] = score

    action_cols = [
        "session_id",
        "cluster_kmeans_k4",
        "engagement_score",
        "total_clicks",
        "unique_products",
        "unique_categories",
        "page_diversity_ratio",
        "avg_price",
    ]
    action_cols = [col for col in action_cols if col in df.columns]

    action_list = df[action_cols].sort_values(
        by="engagement_score", ascending=False
    ).head(50)
    action_list.to_csv(SESSION_ACTION_PATH, index=False)

    print("Insight assets saved:")
    print(f"- {OUTPUT_DATASET}")
    print(f"- {CLUSTER_PROFILE_PATH}")
    print(f"- {CLUSTER_SIZE_PATH}")
    print(f"- {SESSION_ACTION_PATH}")


if __name__ == "__main__":
    main()
