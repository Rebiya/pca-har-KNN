"""
Feature scaling for PCA and KNN.

PCA finds directions of maximum variance; if features use different scales,
large-magnitude features dominate the covariance structure and the principal
components no longer reflect meaningful structure across all variables.

KNN uses distance (e.g. Euclidean); without scaling, features with larger
numerical range contribute disproportionately to the distance, biasing
neighbors toward those dimensions.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def fit_transform_train_test(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit StandardScaler on training data only, then transform train and test.

    This avoids leakage: test statistics must not influence scaling.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler
