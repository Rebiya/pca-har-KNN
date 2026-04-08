"""
PCA utilities: 2D projection, variance retention, and explained-variance plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def _plots_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "results" / "plots"


def apply_pca_2d(X: np.ndarray, random_state: int = 42) -> tuple[np.ndarray, PCA]:
    """
    Reduce X to 2 principal components (for visualization).

    Returns
    -------
    X_2d : ndarray of shape (n_samples, 2)
    pca : fitted PCA instance
    """
    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X)
    return X_2d, pca


def apply_pca_variance(
    X: np.ndarray, variance: float = 0.95, random_state: int = 42
) -> tuple[np.ndarray, PCA, int]:
    """
    Reduce X keeping at least ``variance`` fraction of total variance.

    Returns
    -------
    X_reduced : ndarray
    pca : fitted PCA
    n_components : int
        Number of components selected by sklearn.
    """
    pca = PCA(n_components=variance, random_state=random_state)
    X_reduced = pca.fit_transform(X)
    n_components = pca.n_components_
    return X_reduced, pca, int(n_components)


def explained_variance_plot(pca: PCA, save_path: Path | None = None) -> Path:
    """
    Bar plot of per-component explained variance ratio (for fitted PCA).

    Saves under results/plots/ by default.
    """
    save_path = save_path or (_plots_dir() / "explained_variance_ratio.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ratios = pca.explained_variance_ratio_
    # Cap bars for readability when many components are retained
    max_show = min(len(ratios), 60)
    ratios = ratios[:max_show]
    n = len(ratios)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(1, n + 1), ratios, color="steelblue", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("Explained variance ratio per component " f"(showing first {n})")
    ax.set_xticks(np.arange(1, n + 1, max(1, n // 15)))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
