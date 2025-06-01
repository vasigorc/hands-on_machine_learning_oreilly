"""
Exercise 13.

Some dimensionality reduction techniques can also be used for anomaly detection. For example, take the
Olivetti faces dataset, and reduce it with PCA, preserving 99% of the variance. Then compute the reconstruction
error for each image. Next, take some of the modified images you built in the previous exercise and
look at their reconstruction error: notice how much larger it is. If you plot a reconstructed image,
you will see why: it tries to reconstruct a normal face.
"""

from matplotlib.pyplot import axis
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np

from chapter_09.common.dimensionality_reduction_utils import (
    determine_optimal_pca_components,
)
from chapter_09.common.data_utils import get_modified_faces


def main():
    # Step 1. Fetch & split the original dataset
    olivetti = fetch_olivetti_faces()
    X_train, _, y_train, _ = train_test_split(
        olivetti.data,
        olivetti.target,
        stratify=olivetti.target,
        test_size=0.2,
        random_state=42,
    )

    # Step 2. Reduce dataset's dimensionality with PCA
    n_components = determine_optimal_pca_components(X_train, 0.99)
    pca = PCA(n_components=n_components)
    X_train_reduced = pca.fit_transform(X_train)

    # Step 3. Compute the reconstruction error for each image
    X_train_reconstructed = pca.inverse_transform(X_train_reduced)

    """
    We can compute the reconstruction error for each image as the mean squared error (MSE) per image 
    or the total squared error per image. The exercise doesn't specify, but since it's an image, we 
    might compute the MSE.
  
    We could equally use `sklearn.metrics.mean_squared_error`, but that would return us the average 
    over all samples (images) and all features. Since we're asked for the error per image, we would 
    have to use it in a loop, hence vectorized `np.mean(..., axis=1)` seems more efficient and straightforward.

    Why `axis=1`? The array `X_train` has shape `(n_samples, n_features)`. When we compute `(X_train 
    X_train_reconstructed) ** 2`, we get a matrix of squared errors of the same shape. Then, `np.mean(..., 
    axis=1)` takes the mean across the columns (i.e., for each row, which represents one image). So 
    we get one mean squared error per image.
    """
    reconstruction_errors = np.mean(np.square(X_train - X_train_reconstructed), axis=1)
    print(f"Train errors (AVG): {np.mean(reconstruction_errors)}")

    # Step 4. Compute the reconstruction error for modified images
    # Load a subset (10) of modified images
    X_bad_faces, y_bad_faces = get_modified_faces(X_train, y_train)
    X_bad_faces_reconstructed = pca.inverse_transform(pca.transform(X_bad_faces))
    reconstruction_errors_modified = np.mean(
        np.square(X_bad_faces - X_bad_faces_reconstructed), axis=1
    )
    print(f"Modified images' errors (AVG): {np.mean(reconstruction_errors_modified)}")


if __name__ == "__main__":
    main()
"""
The program prints:

    python3 -m chapter_09.ex13.main
    Optimal number of PCA components to keept at least variance 99% is 222
    Train errors (AVG): 0.0001925208343891427
    Modified images' errors (AVG): 0.00438636913895607
"""
