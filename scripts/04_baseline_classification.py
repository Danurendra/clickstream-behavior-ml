"""
Baseline classification for session-level clickstream dataset.

Run:
    python scripts/04_baseline_classification.py

Outputs:
    reports/tables/classification_baseline_results.csv
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.evaluation import evaluate_binary_classifier, results_to_dataframe


DATA_PATH = Path("data/processed/session_level_dataset.csv")
OUTPUT_TABLE = Path("reports/tables/classification_baseline_results.csv")

TARGETS = ["high_engagement", "premium_interest"]

LEAKAGE_DROP = {
    # Target engineered from total_clicks (max_order mirrors total_clicks)
    "high_engagement": ["total_clicks", "max_order"],
    # Target engineered from premium_click_ratio
    "premium_interest": ["premium_click_ratio"],
}


def get_feature_groups(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )


def get_models() -> dict[str, object]:
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }


def evaluate_model(
    model_name: str,
    model: object,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
) -> dict:
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    start_time = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - start_time

    y_pred = clf.predict(X_test)

    y_score = None
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_test)
        if proba.shape[1] > 1:
            y_score = proba[:, 1]
    elif hasattr(clf, "decision_function"):
        y_score = clf.decision_function(X_test)

    metrics = evaluate_binary_classifier(y_test, y_pred, y_score)
    metrics.update({
        "model": model_name,
        "train_time_sec": round(train_time, 4),
    })

    return metrics


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Run: python scripts/02_feature_engineering.py"
        )

    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    results: list[dict] = []

    for target in TARGETS:
        if target not in df.columns:
            print(f"Skipping target '{target}' (column missing).")
            continue

        drop_cols = ["session_id", target] + LEAKAGE_DROP.get(target, [])
        feature_df = df.drop(columns=[col for col in drop_cols if col in df.columns])

        numeric_cols, categorical_cols = get_feature_groups(feature_df)
        if not numeric_cols and not categorical_cols:
            print(f"No usable features for target '{target}'.")
            continue

        X = feature_df
        y = df[target]

        if y.nunique() < 2:
            print(f"Skipping target '{target}' (only one class present).")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        models = get_models()

        print(f"\nTraining baseline models for target: {target}")
        print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
        print(y.value_counts(normalize=True).rename("share").to_string())

        for name, model in models.items():
            metrics = evaluate_model(
                name, model, X_train, X_test, y_train, y_test, preprocessor
            )
            metrics["target"] = target
            results.append(metrics)

        print("\nClassification report (best by F1 will be ranked in table):")
        best_model_name = max(
            [r for r in results if r["target"] == target],
            key=lambda r: r["f1_score"],
        )["model"]
        print(f"Best model: {best_model_name}")

    if not results:
        print("No results generated.")
        return

    results_df = results_to_dataframe(results)
    results_df.to_csv(OUTPUT_TABLE, index=False)
    print(f"\nBaseline results saved to: {OUTPUT_TABLE}")


if __name__ == "__main__":
    main()
