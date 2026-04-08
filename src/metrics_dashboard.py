"""
Create a timestamped summary visualization from results/metrics.json.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "results" / "metrics.json"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"


def load_metrics(path: Path = METRICS_PATH) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def timestamped_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PLOTS_DIR / f"metrics_dashboard_{timestamp}.png"


def create_metrics_dashboard(metrics_data: dict, save_path: Path | None = None) -> Path:
    save_path = save_path or timestamped_output_path()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    without_pca = metrics_data["knn_without_pca"]
    with_pca = metrics_data["knn_with_pca_95"]

    model_names = ["KNN", "KNN + PCA (95%)"]
    colors = ["#1f77b4", "#ff7f0e"]

    accuracy_values = [
        without_pca["metrics"]["accuracy"],
        with_pca["metrics"]["accuracy"],
    ]
    f1_values = [
        without_pca["metrics"]["f1_weighted"],
        with_pca["metrics"]["f1_weighted"],
    ]
    precision_values = [
        without_pca["metrics"]["precision_weighted"],
        with_pca["metrics"]["precision_weighted"],
    ]
    recall_values = [
        without_pca["metrics"]["recall_weighted"],
        with_pca["metrics"]["recall_weighted"],
    ]
    train_times = [
        without_pca["train_time_sec"],
        with_pca["train_time_sec"],
    ]
    predict_times = [
        without_pca["predict_time_sec"],
        with_pca["predict_time_sec"],
    ]
    feature_counts = [
        metrics_data["feature_counts"]["original"],
        metrics_data["feature_counts"]["pca_95_variance"],
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("HAR KNN Metrics Summary", fontsize=16, fontweight="bold")

    metric_names = ["Accuracy", "F1 (weighted)", "Precision", "Recall"]
    metric_series = [accuracy_values, f1_values, precision_values, recall_values]
    metric_ax = axes[0, 0]
    x = np.arange(len(metric_names))
    width = 0.36

    metric_ax.bar(x - width / 2, [series[0] for series in metric_series], width=width, color=colors[0], label=model_names[0])
    metric_ax.bar(x + width / 2, [series[1] for series in metric_series], width=width, color=colors[1], label=model_names[1])
    metric_ax.set_xticks(x)
    metric_ax.set_xticklabels(metric_names, rotation=10)
    metric_ax.set_ylim(0, 1.05)
    metric_ax.set_ylabel("Score")
    metric_ax.set_title("Model quality comparison")
    metric_ax.legend()

    for idx, values in enumerate(metric_series):
        metric_ax.text(idx - width / 2, values[0] + 0.015, f"{values[0]:.3f}", ha="center", fontsize=9)
        metric_ax.text(idx + width / 2, values[1] + 0.015, f"{values[1]:.3f}", ha="center", fontsize=9)

    time_ax = axes[0, 1]
    x = np.arange(len(model_names))
    time_width = 0.35
    time_ax.bar(x - time_width / 2, train_times, width=time_width, color="#2ca02c", label="Train time")
    time_ax.bar(x + time_width / 2, predict_times, width=time_width, color="#d62728", label="Predict time")
    time_ax.set_xticks(x)
    time_ax.set_xticklabels(model_names, rotation=10)
    time_ax.set_ylabel("Seconds")
    time_ax.set_title("Runtime comparison")
    time_ax.legend()

    for idx, value in enumerate(train_times):
        time_ax.text(idx - time_width / 2, value + max(train_times + predict_times) * 0.02, f"{value:.3f}", ha="center", fontsize=9)
    for idx, value in enumerate(predict_times):
        time_ax.text(idx + time_width / 2, value + max(train_times + predict_times) * 0.02, f"{value:.3f}", ha="center", fontsize=9)

    feature_ax = axes[1, 0]
    feature_bars = feature_ax.bar(model_names, feature_counts, color=colors, edgecolor="black", linewidth=0.5)
    feature_ax.set_ylabel("Number of features")
    feature_ax.set_title("Feature reduction")

    for bar, value in zip(feature_bars, feature_counts):
        feature_ax.text(bar.get_x() + bar.get_width() / 2, value + max(feature_counts) * 0.02, str(value), ha="center", fontsize=10)

    summary_ax = axes[1, 1]
    summary_ax.axis("off")
    speedup = predict_times[0] / predict_times[1] if predict_times[1] else float("inf")
    feature_reduction = 1 - (feature_counts[1] / feature_counts[0])
    summary_lines = [
        "Key takeaways",
        f"Accuracy change: {accuracy_values[1] - accuracy_values[0]:+.4f}",
        f"Weighted F1 change: {f1_values[1] - f1_values[0]:+.4f}",
        f"Prediction speedup with PCA: {speedup:.2f}x",
        f"Feature reduction with PCA: {feature_reduction:.1%}",
        f"Train time saved: {train_times[0] - train_times[1]:.4f} sec",
    ]
    summary_ax.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=12,
        bbox={"boxstyle": "round", "facecolor": "#f7f7f7", "edgecolor": "#cccccc"},
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main() -> None:
    metrics_data = load_metrics()
    save_path = create_metrics_dashboard(metrics_data)
    print(f"Saved metrics dashboard to: {save_path}")


if __name__ == "__main__":
    main()
