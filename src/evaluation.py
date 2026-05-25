"""Evaluation utilities."""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_binary_classifier(y_true, y_pred, y_proba=None) -> dict:
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        result["roc_auc"] = roc_auc_score(y_true, y_proba)

    return result


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(results).sort_values(by="f1_score", ascending=False)
