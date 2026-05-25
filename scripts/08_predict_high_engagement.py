"""
Generate high_engagement predictions from the trained model.

Run:
    python scripts/08_predict_high_engagement.py

Outputs:
    reports/tables/high_engagement_predictions.csv
    reports/tables/high_engagement_top50.csv
"""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


DATA_PATH = Path("data/processed/session_level_dataset.csv")
MODEL_PATH = Path("models/high_engagement_pipeline.joblib")
OUTPUT_PRED = Path("reports/tables/high_engagement_predictions.csv")
OUTPUT_TOP = Path("reports/tables/high_engagement_top50.csv")

TARGET = "high_engagement"
LEAKAGE_DROP = ["total_clicks", "max_order"]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Run: python scripts/02_feature_engineering.py"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run: python scripts/07_train_high_engagement_model.py"
        )

    df = pd.read_csv(DATA_PATH)
    session_ids = df.get("session_id")

    drop_cols = ["session_id", TARGET] + LEAKAGE_DROP
    feature_df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    pipeline = joblib.load(MODEL_PATH)

    pred_label = pipeline.predict(feature_df)

    pred_proba = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(feature_df)
        if proba.shape[1] > 1:
            pred_proba = proba[:, 1]
    elif hasattr(pipeline, "decision_function"):
        pred_proba = pipeline.decision_function(feature_df)

    output = pd.DataFrame(
        {
            "session_id": session_ids if session_ids is not None else range(len(df)),
            "predicted_label": pred_label,
        }
    )

    if pred_proba is not None:
        output["predicted_proba"] = pred_proba

    OUTPUT_PRED.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PRED, index=False)

    top_df = output.sort_values(
        by="predicted_proba" if "predicted_proba" in output.columns else "predicted_label",
        ascending=False,
    ).head(50)
    top_df.to_csv(OUTPUT_TOP, index=False)

    print(f"Predictions saved to: {OUTPUT_PRED}")
    print(f"Top sessions saved to: {OUTPUT_TOP}")


if __name__ == "__main__":
    main()
