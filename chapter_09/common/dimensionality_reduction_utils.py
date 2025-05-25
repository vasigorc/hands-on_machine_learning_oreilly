import numpy as np
from sklearn.decomposition import PCA
from numpy.typing import NDArray


def determine_optimal_pca_components(
    X_train: NDArray[np.floating], variance_threshold: float = 0.95
) -> int:
    """
    Calculate the minimum number of principal components needed to retain a specified percentage of the variance.

    Args:
        X_train: 2D array-like, shape (n_samples, n_features), the training data.
        variance_threshold: Float between 0 and 1, the target fraction of variance to retain (default: 0.95).

    Returns:
        The minimum number of PCA components required to preserve at least the desired variance.
    """
    pca = PCA().fit(X_train)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= variance_threshold) + 1
    print(
        f"Optimal number of PCA components to keept at least variance {variance_threshold:.0%} is {d}"
    )
    return d
