# HAR-KNN with PCA Optimization

This project demonstrates the application of **Principal Component Analysis (PCA)** to a **K-Nearest Neighbors (KNN)** classifier for Human Activity Recognition (HAR). It optimizes model efficiency by reducing the feature space from **561 to 102 dimensions** while maintaining a high accuracy of **0.874**.

---

## Folder Structure

* **`data/`**: Contains the **UCI HAR Dataset**, including raw signal data and activity labels.
* **`src/`**: Modularized Python scripts for data loading, preprocessing, PCA implementation, and model evaluation.
* **`notebooks/`**: An `experiment.ipynb` file for interactive data exploration and visualization.
* **`results/`**: Stores generated plots (e.g., 2D scatter plots, cumulative variance) and performance `metrics.json`.
* **`report/`**: Contains the `pca_report.md` providing in-depth mathematical analysis and findings.
* **`main.py`**: The entry point to run the entire end-to-end pipeline.

---

## Setup and Installation

1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd pca-har-KNN
    ```

2.  **Set up Virtual Environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## How to Run

### 1. Run the Full Pipeline
To execute the data preprocessing, PCA transformation, KNN training, and evaluation, run:
```bash
python main.py
```

### 2. Run Experiments
To view step-by-step experimentation, launch Jupyter and open the notebook:
```bash
jupyter notebook notebooks/experiment.ipynb
```

### 3. View Results
After execution, check the **`results/plots/`** folder for performance dashboards and the **`report/`** folder for the final technical summary.