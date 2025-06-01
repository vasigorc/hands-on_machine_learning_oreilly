"""
Exercise 12.

Train a Gaussian mixture model on the Olivetti faces dataset. To speed up the algorithm, you should
probably reduce the dataset's dimensionality (e.g. use PCA, preserving 99% of variance). Use the model
to generate some new faces (using the `sample()` method), and visualize them (if you used PCA, you will
need to use its `inverse_transform()` method). Try to modify some images (e.g. rotate, flip, darken)
and see if the model can detect the anomalies (i.e. compare the output of the `score_samples()` method
for normal images and for anomalies).
"""

from os import wait
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from chapter_09.common.clustering_utils import compute_aic, find_optimal_clusters
from chapter_09.common.plot_utils import plot_faces
from chapter_09.common.dimensionality_reduction_utils import (
    determine_optimal_pca_components,
)
from chapter_09.common.data_utils import get_modified_faces


def main():
    # Step 1. Split the dataset into training and test sets
    olivetti = fetch_olivetti_faces()
    X_train, _, y_train, _ = train_test_split(
        olivetti.data,
        olivetti.target,
        stratify=olivetti.target,
        test_size=0.2,
        random_state=42,
    )

    # Step 2. Reduce dataset's dimensionality with PCA
    # establishes as 222
    n_components = determine_optimal_pca_components(X_train, 0.99)
    pca = PCA(n_components=n_components)
    X_train_reduced = pca.fit_transform(X_train)

    # Step 3. Select the number of cluster using aic() or `BayesianGaussianMixture`
    (k_candidates, k_scores) = find_optimal_clusters(
        np.arange(30, 60, 3), X_train_reduced, compute_aic
    )
    optimal_k = k_candidates[np.argmax(k_scores)]
    # establishes as 57
    print(f"Best k for max score is {optimal_k}")

    # Step 4. Train a Gaussian mixture on the reduced dataset
    gm = GaussianMixture(n_components=optimal_k, n_init=10, random_state=42)
    gm.fit(X_train_reduced)
    print(f"GM converged {gm.converged_} in {gm.n_iter_} iterations")

    # Step 5. Sample a few new faces and visualize them
    X_new_reduced, y_new_reduced = gm.sample(4)
    X_new = pca.inverse_transform(X_new_reduced)

    plot_faces(
        X_new,
        y_new_reduced,
        n_cols=4,
        display_handler=lambda fig: fig.savefig("chapter_09/ex12/generated_images.png"),
    )

    # Step 6. Modify some other images and see if the model can detect anomalies
    # For this task we want to use unseen data: pristine normal data, but not the
    # data that the model was trained on.
    X_bad_faces, y_bad = get_modified_faces(X_train, y_train)

    plot_faces(
        X_bad_faces,
        y_bad,
        n_cols=4,
        display_handler=lambda fig: fig.savefig("chapter_09/ex12/modified_images.png"),
    )
    X_bad_faces_pca = pca.transform(X_bad_faces)
    # The bad faces are all considered highly unlikely by the Gaussian Mixture
    print(
        f"Score samples of the modified (bad) images: {gm.score_samples(X_bad_faces_pca)}"
    )
    # Compare them with the scores of some training instances
    print(
        f"Score samples of a subset of original (well clustered) images: {gm.score_samples(X_train_reduced[:10])}"
    )


if __name__ == "__main__":
    main()
    """
    The program prints:

        Optimal number of PCA components to keept at least variance 99% is 222
        Best k for max score is 57
        GM converged True in 2 iterations
        Score samples of the modified (bad) images: [-4.00156091e+07 -5.44773597e+07 -3.28416563e+07 -5.08251726e+07
        -4.36970226e+07 -3.21515949e+07 -3.64219250e+07 -1.02726072e+08
        -8.43133990e+07 -8.82647910e+07]
        Score samples of a subset of original (well clustered) images: [1271.21386197 1293.38579325 1263.92216384 1262.20262781 1286.22643668
        1286.16343563 1257.1409456  1292.32617308 1292.9149134  1271.21386161]
    """
