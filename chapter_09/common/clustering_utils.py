from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from numpy.typing import NDArray
from typing import Any, Callable
import numpy as np
from sklearn.mixture import GaussianMixture


def compute_silhouette(k: int, X: NDArray[Any]) -> float:
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    return silhouette_score(X, labels)


def compute_aic(k: int, X: NDArray[Any]) -> float:
    """
    Fits a Gaussian Mixture Model with k components to the data X and returns the AIC score.

        Args:
        k: Number of mixture components (clusters) for the GaussianMixture model.
            X: Data to fit the model on (2D array: samples x features).

    Returns:
    The Akaike Information Criterion (AIC) score for the fitted model on X.
    """
    gm = GaussianMixture(n_components=k, n_init=10, random_state=42)
    gm.fit(X)
    return gm.aic(X)


def find_optimal_clusters(
    k_values: np.ndarray,
    X: NDArray[Any],
    score_func: Callable[[int, NDArray[Any]], float],
    n_jobs=-1,
) -> tuple[np.ndarray, list[float]]:
    """
    Evaluates a given scoring function (in parallel) across multiple values of k for clustering.

    Args:
        k_values: Array of candidate 'k' (number of clusters/components) values to evaluate.
        X: Data to cluster (2D array: samples x features).
        score_func: Function that takes (k, X) and returns a float score (e.g., silhouette or AIC).
        n_jobs: Number of parallel jobs (-1 uses all available processors).

    Returns:
        Tuple of (k_values, scores) where scores is a list of floats from score_func for each k.
    """
    scores = Parallel(n_jobs=n_jobs)(delayed(score_func)(k, X) for k in k_values)
    return k_values, scores
