# PCA on High-Dimensional Human Activity Recognition (HAR)

This report accompanies the `pca-har-KNN` project: a KNN baseline on the UCI HAR dataset (561 features) compared with KNN after PCA-based dimensionality reduction.

---

## 1. What is PCA? (Simple explanation)

**Principal Component Analysis (PCA)** finds a new set of axes for your data. The first axis (first *principal component*) is the direction along which the data varies the most; the second is the direction of next-most variation *orthogonal* to the first, and so on.  

You can use PCA to:

- **Visualize** high-dimensional data in 2D or 3D by keeping only the first few components.
- **Compress** the data by keeping only the top components that explain most of the total variance, which often speeds up distance-based models (like KNN) and can reduce noise.

PCA is *unsupervised*: it does not use class labels; it only uses the feature values (after scaling, which matters a lot).

---

## 2. Why PCA helps with high-dimensional data

Real sensor/feature pipelines (like HAR) often produce **many correlated** measurements. That leads to:

- **Redundancy**: several features carry overlapping information.
- **Curse of dimensionality**: in high dimensions, distances become less informative and models can be slower and more sensitive to noise.

PCA replaces hundreds of original features with a smaller set of **uncorrelated** directions ordered by variance. Retaining, for example, 95% of total variance often needs far fewer than 561 dimensions, which:

- Shortens vectors for distance computations (often faster prediction for KNN).
- Can **denoise** by dropping very low-variance directions that are often dominated by noise.

Trade-offs appear when you discard variance: you may lose subtle discriminative information, so accuracy can stay similar, improve slightly, or drop slightly depending on the data and classifier.

---

## 3. Eigenvalues and eigenvectors (from scratch)

For a square matrix **A**, a non-zero vector **v** is an **eigenvector** if multiplying **A** by **v** only *scales* **v**:

\[
A v = \lambda v
\]

The scalar **λ** is the **eigenvalue** associated with **v**.

### Example matrix

\[
A = \begin{bmatrix} 4 & 2 \\ 2 & 3 \end{bmatrix}
\]

### Characteristic equation \(\det(A - \lambda I) = 0\)

\[
A - \lambda I = \begin{bmatrix} 4-\lambda & 2 \\ 2 & 3-\lambda \end{bmatrix}
\]

\[
\det(A - \lambda I) = (4-\lambda)(3-\lambda) - 4 = \lambda^2 - 7\lambda + 8 = 0
\]

### Solve for eigenvalues

Using the quadratic formula:

\[
\lambda = \frac{7 \pm \sqrt{49 - 32}}{2} = \frac{7 \pm \sqrt{17}}{2}
\]

So:

\[
\lambda_1 = \frac{7 + \sqrt{17}}{2},\quad \lambda_2 = \frac{7 - \sqrt{17}}{2}
\]

(Numerically: \(\lambda_1 \approx 5.56\), \(\lambda_2 \approx 1.44\).)

### Eigenvectors (conceptually)

For each \(\lambda\), solve \((A - \lambda I) v = 0\) to get **v** up to scaling.  

Those directions are the same *idea* as principal components when PCA is applied to the **covariance** matrix of centered data: the eigenvectors of the covariance matrix are the principal directions, and the eigenvalues tell you how much variance lies along each direction.

**In simple words:** eigenvalues say *how much stretching* happens along each special direction (eigenvector). PCA picks the directions with the largest variance (largest eigenvalues of the covariance matrix) as the main “axes” of the data cloud.

---

## 4. Results interpretation (this project)

After running `python main.py` from the project root:

- **`results/metrics.json`** stores accuracy, precision, recall, F1, confusion matrices, and **train/predict times** for KNN with and without PCA.
- **`results/plots/`** holds the 2D PCA scatter, explained/cumulative variance plots, accuracy comparison bar chart, and confusion matrix heatmaps.

### Before vs after PCA

- **Before**: KNN uses all **561** standardized features.
- **After**: PCA keeps enough components to retain **~95%** of variance. On this dataset that is often **on the order of 100** components (not as small as ~20), because many features carry complementary information—see the printed count in `main.py` output and `results/metrics.json`.

### Trade-offs

| Aspect | Without PCA | With PCA (~95% variance) |
|--------|-------------|---------------------------|
| Dimensionality | High (561) | Much lower |
| Speed | Often slower **prediction** (longer vectors) | Often faster **prediction** |
| Accuracy | Baseline | Often similar; may change slightly |

PCA does not guarantee higher accuracy; the goal here is to show **large compression** with **competitive** performance and **clear efficiency gains** on distance-based inference, plus **2D plots** that are impossible in the original 561-dimensional space without projection.

---

*Generated as part of the pca-har-KNN portfolio project.*
