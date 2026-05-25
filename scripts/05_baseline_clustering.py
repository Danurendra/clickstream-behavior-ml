"""
Baseline clustering for session-level clickstream dataset.

Run:
    python scripts/05_baseline_clustering.py

Outputs:
    reports/tables/clustering_baseline_metrics.csv
    reports/tables/clustering_cluster_sizes.csv
    reports/figures/clustering_kmeans_silhouette.png
"""

from __future__ import annotations

from pathlib import Path
import os
import sys
import warnings

# Use logical cores to avoid joblib core-detection warning on Windows.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"joblib\.externals\.loky\.backend\.context",
)

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


DATA_PATH = Path("data/processed/session_level_dataset.csv")
OUTPUT_METRICS = Path("reports/tables/clustering_baseline_metrics.csv")
OUTPUT_SIZES = Path("reports/tables/clustering_cluster_sizes.csv")
OUTPUT_SILHOUETTE = Path("reports/figures/clustering_kmeans_silhouette.png")


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


def safe_silhouette(X: pd.DataFrame, labels: pd.Series) -> float | None:
    if len(set(labels)) < 2:
        return None
    return silhouette_score(X, labels)


def safe_calinski(X: pd.DataFrame, labels: pd.Series) -> float | None:
    if len(set(labels)) < 2:
        return None
    return calinski_harabasz_score(X, labels)


def safe_davies(X: pd.DataFrame, labels: pd.Series) -> float | None:
    if len(set(labels)) < 2:
        return None
    return davies_bouldin_score(X, labels)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in NUMERIC_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required numeric features: {missing}")

    X = df[NUMERIC_FEATURES].copy()
    return X


def evaluate_clustering(name: str, labels: pd.Series, X_scaled: pd.DataFrame) -> dict:
    return {
        "model": name,
        "clusters": len(set(labels)) - (1 if -1 in labels else 0),
        "noise_points": int((labels == -1).sum()) if (-1 in labels) else 0,
        "silhouette": safe_silhouette(X_scaled, labels),
        "calinski_harabasz": safe_calinski(X_scaled, labels),
        "davies_bouldin": safe_davies(X_scaled, labels),
    }


def plot_kmeans_silhouette(k_values: list[int], scores: list[float | None]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, [s if s is not None else 0 for s in scores], marker="o")
    ax.set_title("KMeans Silhouette Scores")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette")
    fig.tight_layout()
    OUTPUT_SILHOUETTE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_SILHOUETTE, dpi=150)
    plt.close(fig)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Run: python scripts/02_feature_engineering.py"
        )

    df = pd.read_csv(DATA_PATH)
    X = prepare_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    OUTPUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SIZES.parent.mkdir(parents=True, exist_ok=True)

    metrics: list[dict] = []

    k_values = [2, 3, 4, 5, 6, 7, 8]
    kmeans_scores: list[float | None] = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X_scaled)
        score = safe_silhouette(X_scaled, labels)
        kmeans_scores.append(score)
        metrics.append(evaluate_clustering(f"kmeans_k{k}", labels, X_scaled))

    plot_kmeans_silhouette(k_values, kmeans_scores)

    agglo = AgglomerativeClustering(n_clusters=4)
    agglo_labels = agglo.fit_predict(X_scaled)
    metrics.append(evaluate_clustering("agglomerative_k4", agglo_labels, X_scaled))

    dbscan = DBSCAN(eps=0.7, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    metrics.append(evaluate_clustering("dbscan_eps0.7", dbscan_labels, X_scaled))

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUTPUT_METRICS, index=False)

    size_records: list[dict] = []
    for name, labels in [
        ("kmeans_k4", KMeans(n_clusters=4, random_state=42, n_init="auto").fit_predict(X_scaled)),
        ("agglomerative_k4", agglo_labels),
        ("dbscan_eps0.7", dbscan_labels),
    ]:
        series = pd.Series(labels, name="cluster")
        counts = series.value_counts().sort_index()
        for cluster_id, count in counts.items():
            size_records.append({
                "model": name,
                "cluster": int(cluster_id),
                "count": int(count),
            })

    pd.DataFrame(size_records).to_csv(OUTPUT_SIZES, index=False)

    print("Baseline clustering completed.")
    print(f"Metrics saved to: {OUTPUT_METRICS}")
    print(f"Cluster sizes saved to: {OUTPUT_SIZES}")
    print(f"Silhouette plot saved to: {OUTPUT_SILHOUETTE}")


if __name__ == "__main__":
    main()
