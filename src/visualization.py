"""
Figures for PCA exploration and model comparison.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA


def _plots_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "results" / "plots"


def plot_pca_2d_scatter(
    X_2d: np.ndarray,
    y: np.ndarray,
    save_path: Path | None = None,
    title: str = "PCA projection (2 components) — training set",
) -> Path:
    save_path = save_path or (_plots_dir() / "pca_2d_scatter.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    activity_names = {
        1: "WALKING",
        2: "WALKING_UPSTAIRS",
        3: "WALKING_DOWNSTAIRS",
        4: "SITTING",
        5: "STANDING",
        6: "LAYING",
    }
    labels_sorted = sorted(np.unique(y))
    palette = sns.color_palette("husl", n_colors=len(labels_sorted))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, lab in enumerate(labels_sorted):
        m = y == lab
        name = activity_names.get(int(lab), str(lab))
        ax.scatter(
            X_2d[m, 0],
            X_2d[m, 1],
            s=8,
            alpha=0.6,
            label=name,
            color=palette[i],
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(markerscale=2, frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_explained_variance_bar(pca: PCA, save_path: Path | None = None, max_components: int = 40) -> Path:
    """Bar chart of explained variance ratio (first ``max_components`` components)."""
    save_path = save_path or (_plots_dir() / "viz_explained_variance.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ratios = pca.explained_variance_ratio_[:max_components]
    n = len(ratios)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(1, n + 1), ratios, color="teal", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title(f"Explained variance ratio (first {n} components)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_cumulative_variance(pca: PCA, save_path: Path | None = None, max_components: int = 80) -> Path:
    """Cumulative explained variance vs number of components."""
    save_path = save_path or (_plots_dir() / "cumulative_variance.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cum = np.cumsum(pca.explained_variance_ratio_)
    cum = cum[:max_components]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(1, len(cum) + 1), cum, marker="o", ms=3, color="darkgreen")
    ax.axhline(0.95, color="crimson", ls="--", label="95% variance")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("Cumulative explained variance")
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_accuracy_comparison(
    names: list[str],
    accuracies: list[float],
    save_path: Path | None = None,
) -> Path:
    save_path = save_path or (_plots_dir() / "accuracy_comparison.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(names))
    ax.bar(x, accuracies, color=["#4472c4", "#ed7d31"], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("KNN: without PCA vs with PCA (95% variance)")
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(accuracies):
        ax.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_confusion_matrix_heatmap(
    cm: list | np.ndarray,
    class_labels: list | np.ndarray,
    title: str,
    save_path: Path,
) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
