"""
Load UCI Human Activity Recognition (HAR) dataset.

Expected layout after download (see `download_uci_har_if_needed`):
    data/UCI HAR Dataset/train/X_train.txt
    data/UCI HAR Dataset/train/y_train.txt
    data/UCI HAR Dataset/test/X_test.txt
    data/UCI HAR Dataset/test/y_test.txt
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

# Official UCI mirror (classic path; dataset is also on archive.ics.uci.edu).
UCI_HAR_ZIP_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/"
    "UCI%20HAR%20Dataset.zip"
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_data_dir() -> Path:
    return _project_root() / "data"


def download_uci_har_if_needed(data_dir: Path | None = None) -> Path:
    """
    Download and extract UCI HAR if the expected folder is missing.

    Returns
    -------
    Path
        Path to the root folder containing ``train`` and ``test`` subdirs.
    """
    data_dir = data_dir or _default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    inner = data_dir / "UCI HAR Dataset"
    if (inner / "train" / "X_train.txt").is_file():
        return inner

    print(f"Downloading UCI HAR dataset to {data_dir} ...")
    with urlopen(UCI_HAR_ZIP_URL, timeout=120) as resp:
        raw = resp.read()

    zf = zipfile.ZipFile(io.BytesIO(raw))
    zf.extractall(data_dir)
    zf.close()

    if not (inner / "train" / "X_train.txt").is_file():
        raise FileNotFoundError(
            f"Extracted archive but could not find {inner / 'train' / 'X_train.txt'}. "
            "Manually download UCI HAR and place under data/."
        )
    print("Download complete.")
    return inner


def load_har_splits(data_root: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load train/test feature matrices and label vectors.

    Labels are returned as integers in {1, ..., 6} (UCI encoding).
    """
    root = data_root or download_uci_har_if_needed()

    train_dir = root / "train"
    test_dir = root / "test"

    X_train = pd.read_csv(train_dir / "X_train.txt", sep=r"\s+", header=None, engine="python").values
    X_test = pd.read_csv(test_dir / "X_test.txt", sep=r"\s+", header=None, engine="python").values

    y_train = pd.read_csv(train_dir / "y_train.txt", header=None).values.ravel()
    y_test = pd.read_csv(test_dir / "y_test.txt", header=None).values.ravel()

    y_train = np.asarray(y_train, dtype=np.int64).ravel()
    y_test = np.asarray(y_test, dtype=np.int64).ravel()

    # UCI files use 1..6; ensure plain integers
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    return X_train, X_test, y_train, y_test
