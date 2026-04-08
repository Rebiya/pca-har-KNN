"""
End-to-end pipeline: load HAR, scale, KNN baseline vs KNN + PCA, plots and metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from src.data_loader import load_har_splits
from src.evaluation import compute_metrics, save_metrics
from src.model import predict_knn, predict_time_knn, train_knn
from src.pca_module import apply_pca_2d, apply_pca_variance, explained_variance_plot
from src.preprocessing import fit_transform_train_test
from src.visualization import (
    plot_accuracy_comparison,
    plot_confusion_matrix_heatmap,
    plot_cumulative_variance,
    plot_explained_variance_bar,
    plot_pca_2d_scatter,
)

RESULTS = Path(__file__).resolve().parent / "results"
PLOTS = RESULTS / "plots"
RNG = 42


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)

    # 1. Load
    X_train, X_test, y_train, y_test = load_har_splits()
    n_features_orig = X_train.shape[1]

    # 2. Scale
    X_train_s, X_test_s, _scaler = fit_transform_train_test(X_train, X_test)

    label_universe = np.arange(1, 7)

    # --- WITHOUT PCA ---
    clf_base, t_train_base = train_knn(X_train_s, y_train)
    y_pred_base, t_pred_base = predict_time_knn(clf_base, X_test_s)
    metrics_base = compute_metrics(y_test, y_pred_base, labels=label_universe)

    # --- WITH PCA (95% variance); fit on train only ---
    X_train_pca, pca95, n_comp = apply_pca_variance(X_train_s, variance=0.95, random_state=RNG)
    X_test_pca = pca95.transform(X_test_s)

    clf_pca, t_train_pca = train_knn(X_train_pca, y_train)
    y_pred_pca, t_pred_pca = predict_time_knn(clf_pca, X_test_pca)
    metrics_pca = compute_metrics(y_test, y_pred_pca, labels=label_universe)

    # Explained variance plot from the 95% PCA (per-component ratios available)
    explained_variance_plot(pca95)

    # PCA for variance curve (enough components to show cumulative shape)
    n_for_curve = min(100, X_train_s.shape[1])
    pca_curve = PCA(n_components=n_for_curve, random_state=RNG)
    pca_curve.fit(X_train_s)
    plot_explained_variance_bar(pca_curve, max_components=min(40, n_for_curve))
    plot_cumulative_variance(pca_curve, max_components=min(80, n_for_curve))

    # --- PCA 2D visualization (fit on train) ---
    X_train_2d, pca2 = apply_pca_2d(X_train_s, random_state=RNG)
    plot_pca_2d_scatter(X_train_2d, y_train)

    # Accuracy bar chart
    plot_accuracy_comparison(
        ["KNN (no PCA)", f"KNN + PCA ({n_comp} comps, ~95% var.)"],
        [metrics_base["accuracy"], metrics_pca["accuracy"]],
    )

    # Confusion matrices
    plot_confusion_matrix_heatmap(
        metrics_base["confusion_matrix"],
        metrics_base["labels"],
        "Confusion matrix — KNN without PCA",
        PLOTS / "confusion_matrix_no_pca.png",
    )
    plot_confusion_matrix_heatmap(
        metrics_pca["confusion_matrix"],
        metrics_pca["labels"],
        "Confusion matrix — KNN with PCA (95% variance)",
        PLOTS / "confusion_matrix_pca_95.png",
    )

    payload = {
        "reproducibility": {"random_state_pca": RNG},
        "feature_counts": {
            "original": int(n_features_orig),
            "pca_95_variance": int(n_comp),
        },
        "knn_without_pca": {
            "train_time_sec": t_train_base,
            "predict_time_sec": t_pred_base,
            "metrics": {k: v for k, v in metrics_base.items() if k != "confusion_matrix"},
            "confusion_matrix": metrics_base["confusion_matrix"],
            "labels": metrics_base["labels"],
        },
        "knn_with_pca_95": {
            "train_time_sec": t_train_pca,
            "predict_time_sec": t_pred_pca,
            "metrics": {k: v for k, v in metrics_pca.items() if k != "confusion_matrix"},
            "confusion_matrix": metrics_pca["confusion_matrix"],
            "labels": metrics_pca["labels"],
        },
    }
    save_metrics(payload, RESULTS / "metrics.json")

    # 11. Summary prints
    print("\n=== PCA + KNN on UCI HAR ===\n")
    print(f"Features: {n_features_orig} → {n_comp} (PCA retaining 95% variance)")
    print(f"Training time (fit):  no PCA {t_train_base:.6f}s  |  PCA {t_train_pca:.6f}s")
    print(f"Prediction time:      no PCA {t_pred_base:.6f}s  |  PCA {t_pred_pca:.6f}s")
    print(f"Accuracy:             no PCA {metrics_base['accuracy']:.4f}  |  PCA {metrics_pca['accuracy']:.4f}")
    print(f"\nMetrics saved to: {RESULTS / 'metrics.json'}")
    print(f"Plots saved under:  {PLOTS}\n")


if __name__ == "__main__":
    main()
