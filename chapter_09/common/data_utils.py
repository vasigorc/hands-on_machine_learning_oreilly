from numpy.typing import NDArray
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import numpy as np


def load_split_olivetti_dataset():
    olivetti = fetch_olivetti_faces()
    X_temp, X_test, y_temp, y_test = train_test_split(
        olivetti.data,
        olivetti.target,
        stratify=olivetti.target,
        test_size=0.2,
        random_state=42,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_temp, y_temp, stratify=y_temp, test_size=0.2, random_state=42
    )
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def get_modified_faces(
    X: NDArray[np.float64], y: NDArray[np.int32]
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """
    In accordance with the tasks from  exercises 12 and 13,  we modify some images (e.g. rotate, flip,
    darken)

    Args:
        X: Array of face images, each row is a flattened image of 64x64 pixels
        y: Array of labels corresponding to the face images

    Returns:
        X_bad_faces: Array of modified face images (rotated, flipped, darkened)
        y_bad: Array of labels for the modified images

    """
    # Re-using author's solution for this part (from: https://colab.research.google.com/github/ageron/handson-ml3/blob/main/09_unsupervised_learning.ipynb)
    n_rotated = 4
    rotated = np.transpose(X[:n_rotated].reshape(-1, 64, 64), axes=[0, 2, 1])
    rotated = rotated.reshape(-1, 64 * 64)
    y_rotated = y[:n_rotated]

    n_flipped = 3
    flipped = X[:n_flipped].reshape(-1, 64, 64)[:, ::-1]
    flipped = flipped.reshape(-1, 64 * 64)
    y_flipped = y[:n_flipped]

    n_darkened = 3
    darkened = X[:n_darkened].copy()
    darkened[:, 1:-1] *= 0.3
    y_darkened = y[:n_darkened]

    X_bad_faces = np.r_[rotated, flipped, darkened]
    y_bad = np.concatenate([y_rotated, y_flipped, y_darkened])
    return X_bad_faces, y_bad
