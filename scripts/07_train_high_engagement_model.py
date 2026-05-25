"""
Train a high_engagement prediction model and save it.

Run:
    python scripts/07_train_high_engagement_model.py

Outputs:
    models/high_engagement_pipeline.joblib
    reports/tables/high_engagement_model_metrics.csv
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.evaluation import evaluate_binary_classifier


DATA_PATH = Path("data/processed/session_level_dataset.csv")
BASELINE_RESULTS = Path("reports/tables/classification_baseline_results.csv")
MODEL_PATH = Path("models/high_engagement_pipeline.joblib")
METRICS_PATH = Path("reports/tables/high_engagement_model_metrics.csv")

TARGET = "high_engagement"
LEAKAGE_DROP = ["total_clicks", "max_order"]

MODEL_FACTORY = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=42),
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


def choose_model() -> tuple[str, object]:
    default_name = "random_forest"
    if BASELINE_RESULTS.exists():
        results = pd.read_csv(BASELINE_RESULTS)
        subset = results[results["target"] == TARGET]
        if not subset.empty:
            best_name = subset.sort_values(by="f1_score", ascending=False).iloc[0][
                "model"
            ]
            if best_name in MODEL_FACTORY:
                return best_name, MODEL_FACTORY[best_name]()

    return default_name, MODEL_FACTORY[default_name]()


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Run: python scripts/02_feature_engineering.py"
        )

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Target column missing: {TARGET}")

    y = df[TARGET]
    if y.nunique() < 2:
        raise ValueError("Target has only one class. Revisit target definition.")

    drop_cols = ["session_id", TARGET] + LEAKAGE_DROP
    feature_df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    numeric_cols, categorical_cols = get_feature_groups(feature_df)
    if not numeric_cols and not categorical_cols:
        raise ValueError("No usable features for training.")

    X_train, X_test, y_train, y_test = train_test_split(
        feature_df, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model_name, model = choose_model()

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    start_time = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - start_time

    y_pred = pipeline.predict(X_test)
    y_score = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X_test)
        if proba.shape[1] > 1:
            y_score = proba[:, 1]
    elif hasattr(pipeline, "decision_function"):
        y_score = pipeline.decision_function(X_test)

    metrics = evaluate_binary_classifier(y_test, y_pred, y_score)
    metrics.update(
        {
            "model": model_name,
            "train_time_sec": round(train_time, 4),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)
    pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
