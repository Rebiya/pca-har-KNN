"""
KNN classifier training and prediction with timing.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = 5,
    n_jobs: int = -1,
) -> tuple[KNeighborsClassifier, float]:
    """
    Fit KNeighborsClassifier and return (model, training_time_seconds).

    Note: sklearn's KNN ``fit`` mainly stores data; time is still reported
    as required for pipeline comparison (prediction cost differs by dimension).
    """
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    return clf, train_time


def predict_knn(clf: KNeighborsClassifier, X: np.ndarray) -> np.ndarray:
    return clf.predict(X)


def predict_time_knn(clf: KNeighborsClassifier, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Predict and measure wall-clock prediction time (useful for efficiency story)."""
    t0 = time.perf_counter()
    y_hat = clf.predict(X)
    elapsed = time.perf_counter() - t0
    return y_hat, elapsed
