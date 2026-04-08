"""
Classification metrics and JSON export.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray | None = None,
) -> dict:
    """Accuracy, precision, recall, F1 (weighted), and confusion matrix."""
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels.astype(int).tolist(),
    }


def save_metrics(payload: dict, path: Path | None = None) -> Path:
    path = path or (Path(__file__).resolve().parents[1] / "results" / "metrics.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
