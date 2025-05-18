from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from numpy.typing import NDArray
import numpy as np


def compute_silhouette(k: int, X: NDArray[any]) -> float:
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    return silhouette_score(X, labels)


def find_optimal_clusters(
    k_values: np.ndarray, X: NDArray[any], n_jobs=-1
) -> tuple[np.ndarray, list[float]]:
    """Parallel silhouette score computation for multiple k values."""
    scores = Parallel(n_jobs=n_jobs)(
        delayed(compute_silhouette)(k, X) for k in k_values
    )
    return k_values, scores
